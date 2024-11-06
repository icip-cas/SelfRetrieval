# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torch.nn import CrossEntropyLoss
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from tqdm import tqdm

SPIECE_UNDERLINE = "▁"
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


import logging
logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    shared_memory: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    custom_trainer: bool = field(default=False)


local_rank = None


# def collate_fn(features):
    
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def labelpad(self, labels, features):
        sequence_length = len(features["input_ids"][0])
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            features["labels"] = [
                list(label) + [IGNORE_TOKEN_ID] * (sequence_length - len(label)) for label in labels
            ]
        else:
            features["labels"] = [
                [IGNORE_TOKEN_ID] * (sequence_length - len(label)) + list(label) for label in labels
            ]
        return features

    def getpad(self,features,labels=None):
        features = self.tokenizer.pad(
                features,
                padding=True,
            )
        if labels is not None:
            features = self.labelpad(labels,features)
        return features

    def __call__(self, features):
        rawinputs = []
        rawlabels = []
        instanceindex = 0
        for feature in features:
            rawfeature = {}
            rawfeature['input_ids'] = feature['input_ids']
            rawfeature['attention_mask'] = feature['attention_mask']
            rawlabels.append(feature['labels'])
            rawinputs.append(rawfeature)

            instanceindex += 1
        rawinputs = self.getpad(rawinputs,labels=rawlabels)
        rawinputs = {k:torch.tensor(v, dtype=torch.int64) for k,v in rawinputs.items()}
        inputs = rawinputs
        return inputs


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
#                                    output_dir: str):
#     """Collects the state dict and dump to disk."""
#     trainer.save_model(output_dir)

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    conversations = []
    trainables = []

    llama = "llama" in str(type(tokenizer))
    prefix_space = "" if llama else " "

    for i, source in enumerate(sources):
        conversations.append(source[0])
        trainables.append(source[1])
    print("Start tokenizing....")
    all_inputs_ids = []
    batch_size = 10000
    for i in range(0, len(conversations), batch_size):
        input_ids = tokenizer(
            [prefix_space + "".join(i) for i in conversations[i:i+batch_size]],
            return_tensors="pt",
            padding=True,
            max_length=1024,
            truncation=True,
        ).input_ids
        all_inputs_ids.append(input_ids)
    input_ids = torch.cat(all_inputs_ids, dim=0)

    print("Finish tokenizing....")
    targets = input_ids.clone()

    # Mask targets
    for conversation, target, trainable in zip(conversations, targets, trainables):
        total_len = int(target.ne(tokenizer.pad_token_id).sum()) + \
             int(tokenizer.eos_token in conversation[-1] and tokenizer.eos_token == tokenizer.pad_token)
        cur_len = 0
        # target[:cur_len] = IGNORE_TOKEN_ID
        for idx, (conv, train) in enumerate(zip(conversation, trainable)):
            conv_ids = tokenizer(prefix_space + conv, max_length=1024, truncation=True,).input_ids
            round_len = len(conv_ids) - 1
            if idx != 0 and conv_ids[0] == tokenizer.bos_token_id:
                round_len -= 1

            if idx == len(conversation) - 1:
                round_len += 1
            if not train:
                target[cur_len:cur_len+round_len] = IGNORE_TOKEN_ID
            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                rank0_print(f"WARNING: tokenization mismatch "
                            f"{cur_len} vs. {total_len}")
                rank0_print(conversation)

    return dict(input_ids=input_ids, labels=targets,
                attention_mask=input_ids.ne(tokenizer.pad_token_id))


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        rank0_print("Loading data...")
        if data_path.endswith(".json"):
            list_data_dict = json.load(open(data_path, "r"))

            rank0_print("Formatting inputs...")
            data_dict = preprocess(list_data_dict, tokenizer)
        elif data_path.endswith(".pt"):
            data_dict = torch.load(data_path)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    attention_mask=self.attention_mask[i])


import torch.multiprocessing as mp
class SharedMemoryDataset(Dataset):
    def __init__(self, data_path):
        if mp.get_start_method(allow_none=True) != 'fork':
            mp.set_start_method('fork', force=True)
        
        self.data_dict = None
        self.shared_data = {}
        self.shapes = {}
        self.dtypes = {}
        
        # 在主进程中加载数据并设置共享内存
        if local_rank == 0:
            self.data_dict = torch.load(data_path)
            for key, tensor in self.data_dict.items():
                self.shared_data[key] = mp.RawArray('c', tensor.numpy().tobytes())
                self.shapes[key] = tensor.shape
                self.dtypes[key] = tensor.dtype
            
            # 将shapes和dtypes信息也放入共享内存
            self.shared_shapes = mp.Array('i', [item for shape in self.shapes.values() for item in shape])
            self.shared_dtypes = mp.Array('c', str([dtype for dtype in self.dtypes.values()]).encode())
        
        # 所有进程都能访问这些属性
        self.keys = ['input_ids', 'labels', 'attention_mask']
        
        # 在子进程中重构shapes和dtypes
        if local_rank != 0:
            self.shapes = {}
            self.dtypes = {}
            shape_idx = 0
            for key in self.keys:
                dim = self.shared_shapes[shape_idx]
                self.shapes[key] = tuple(self.shared_shapes[shape_idx+1:shape_idx+1+dim])
                shape_idx += dim + 1
            self.dtypes = eval(self.shared_dtypes.value.decode())
        
        self.length = self.shapes[self.keys[0]][0] if self.keys else 0
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if self.data_dict is None:
            # 在子进程中首次访问时重新构造张量
            self.data_dict = {}
            for key in self.keys:
                buffer = np.frombuffer(self.shared_data[key], dtype=np.float32)
                self.data_dict[key] = torch.from_numpy(buffer).reshape(self.shapes[key]).to(self.dtypes[key])
        
        return {key: self.data_dict[key][idx] for key in self.keys}

# def create_shared_dataloader(data_path, batch_size, num_workers):
#     dataset = SharedMemoryDataset(data_path)
#     return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.list_data_dict[i]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (LazySupervisedDataset
                   if data_args.lazy_preprocess else SupervisedDataset)
    if data_args.shared_memory:
        train_dataset = SharedMemoryDataset(data_args.data_path)
    else:
        train_dataset = dataset_cls(tokenizer=tokenizer,
                                    data_path=data_args.data_path)
    return dict(train_dataset=train_dataset,
                eval_dataset=None)


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
        

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            rank0_print(shift_logits.shape)
            rank0_print(shift_labels.shape)
            loss_weight = torch.Tensor([0.6, 0.6, 0.6, 0.7, 0.7, 0.7, 0.8, 0.8, 0.8, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0] + [1.0] * 4080).to(shift_logits.device)
            # loss_weight = torch.Tensor([10, 8, 6, 5, 4, 3.5, 3, 2, 1.8, 1.6, 1.4, 1.2, 1.0, 1.0, 1.0] + [1.0] * 4080).to(shift_logits.device)
            # loss_weight = torch.Tensor([3, 2.8, 2.6, 2.4, 2.2, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1] + [1.0] * 4080).to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            # shift_logits = shift_logits.view(-1, self.config.vocab_size)
            # shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            label_mask = shift_labels.ne(IGNORE_TOKEN_ID).to(torch.int).to(shift_logits.device)
            rank0_print(label_mask.shape)
            cumsum = label_mask.cumsum(dim=-1).to(shift_logits.device)
            rank0_print(cumsum.shape)
            weights = loss_weight[cumsum - 1] * label_mask
            rank0_print(weights.shape)

            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
            loss = loss * weights
            loss = loss.sum() / label_mask.sum()


        return (loss, outputs) if return_outputs else loss


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2" if not "phi-2" in model_args.model_name_or_path else "eager",
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
        add_prefix_space=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        # ! special for llama3
        tokenizer.pad_token = "<|reserved_special_token_0|>"

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      data_collator=CustomDataCollator(tokenizer),
                      **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
