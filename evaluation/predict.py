import os
import gc
import math
import json
import tqdm
import time
import torch
import string
import argparse
import torch.nn as nn
import torch.multiprocessing as mp
from transformers import AutoTokenizer, AutoModelForCausalLM


temp = string.Template("Question: $question\n\n Related document title: $title\n\n Answer: $answer\n\n")
temp = string.Template("Question: $question\n\n Related document title: $title\n\n Related document: $related_document\n\n Answer: $answer\n\n")


def self_assessment(data, batch_size, device, tokenizer, model, addedinputs): 
    all_scores = []
    lossfun = nn.CrossEntropyLoss(reduction='none')
    for j in tqdm.trange(math.ceil(len(data)/batch_size), desc=f"{str(model.device)} self assesment"):
        batched_inp = []
        batched_added = []
        for text in data[j*batch_size:(j+1)*batch_size]:
            batched_inp.append(text)
        for text in addedinputs[j*batch_size:(j+1)*batch_size]:
            batched_added.append(text)
        inputs = tokenizer(batched_inp, padding=True, return_tensors="pt", add_special_tokens=True,return_token_type_ids=False)
        inputs_added = tokenizer(batched_added, padding=True, return_tensors="pt", add_special_tokens=False,return_token_type_ids=False)
        pad_labels = torch.ones(inputs['input_ids'].size()) * -100
        pad_labels = pad_labels.long()
        index = pad_labels.size(-1) - 1
        labels = torch.cat([pad_labels, inputs_added['input_ids']],dim=-1)[:,1:]
        input_ids = torch.cat([inputs['input_ids'], inputs_added['input_ids']],dim=-1)[:,:-1]
        attention_mask = torch.cat([inputs['attention_mask'], inputs_added['attention_mask']],dim=-1)[:,:-1]
        inputs = {
            'input_ids': input_ids, 
            'attention_mask': attention_mask, 
            'labels': labels
        }
        with torch.no_grad():
            for key in inputs:
                inputs[key] = inputs[key].to(device)
            y = model(**inputs)
            loss = lossfun(y.logits.reshape([-1,y.logits.size(-1)]),inputs['labels'].reshape([-1])).cpu()
            loss = loss.reshape([len(inputs['input_ids']),-1])[:,index:]
            score = torch.mean(loss,dim=-1).tolist()
            all_scores = all_scores + score
    return all_scores


def get_final_results_from_prefix(out, contrained_dict): # out is a list of token ids: [token_id0, token_id1, ...]
    tmp = contrained_dict.copy()
    for i in out:
        i = str(i)
        if isinstance(tmp, str):
            return tmp
        if i in tmp:
            tmp = tmp[i]
        else:
            return None
    return tmp


def constrain_func(batch_id, input_ids):
    start_pos = -1
    for i in range(len(input_ids) - len(START_TOKENS_ID) + 1):
        if [int(j) for j in list(input_ids[i: i + len(START_TOKENS_ID)])] == START_TOKENS_ID:
            start_pos = i + len(START_TOKENS_ID) - 1
            break
    
    if start_pos == -1:
        return list(range(VOCAB_SIZE)) # if input do not contain "Title:", use total vocab as candidates

    con_dict = CONSTRAIN_DICT
    for i in range(start_pos, len(input_ids)):
        if str(int(input_ids[i])) not in con_dict:
            return [EOS_TOKEN_ID]
        con_dict = con_dict[str(int(input_ids[i]))]
        if isinstance(con_dict, str):
            return [EOS_TOKEN_ID]

    next_token_ids = []
    if isinstance(con_dict, dict):
        next_token_ids = [int(i) for i in con_dict.keys()]
    else:
        next_token_ids = [EOS_TOKEN_ID]
    return next_token_ids


class Retriever:
    def __init__(self, 
        constrain_dict, chunkid_2_title_chunk, start_token_ids, self_assesment_sequence, prompt_for_chunk, batch_size, candidate_num, beam_num_for_title, beam_num_for_chunk, num_for_chunk_generation_stage1
    ):
        self.constrain_dict = constrain_dict
        self.chunkid_2_title_chunk = chunkid_2_title_chunk
        self.start_token_ids = start_token_ids
        self.additional_eos_token_id = start_token_ids[0]
        self.self_assesment_sequence = self_assesment_sequence
        self.prompt_for_chunk = prompt_for_chunk
        
        self.batch_size = batch_size
        self.candidate_num = candidate_num
        self.beam_num_for_title = beam_num_for_title
        self.beam_num_for_chunk = beam_num_for_chunk
        self.num_for_chunk_generation_stage1 = num_for_chunk_generation_stage1

    @torch.no_grad()
    def model_generation(self,
        model, print_info, tokenizer, sentences, batch_size, max_new_tokens, beam_num, if_constrain, if_only_return_new_tokens, additional_eos_token_id
    ): # sentences is {'input_ids': [], 'attention_mask': []}
        all_scores = []
        all_outputs = []

        for j in tqdm.trange(math.ceil(len(sentences['input_ids'])/batch_size), desc=f"{str(model.device)} {print_info}"):
            batched_inp_input_ids = sentences['input_ids'][j*batch_size:(j+1)*batch_size]
            batched_inp_attention_mask = sentences['attention_mask'][j*batch_size:(j+1)*batch_size]
            batched_inp = {"input_ids": batched_inp_input_ids, "attention_mask": batched_inp_attention_mask}
            inputs = tokenizer.pad(batched_inp, return_tensors="pt", padding=True).to(model.device)

            outputs = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                num_beams=beam_num, 
                num_return_sequences=beam_num, 
                output_scores=True,
                return_dict_in_generate=True,
                prefix_allowed_tokens_fn=constrain_func if if_constrain else None,
                pad_token_id=tokenizer.eos_token_id, 
                eos_token_id=[tokenizer.eos_token_id, additional_eos_token_id] if additional_eos_token_id is not None else tokenizer.eos_token_id
            )

            if "sequences_scores" in outputs:
                scores = outputs['sequences_scores'].cpu().tolist()
            else:
                scores = [None for _ in range(inputs['input_ids'].size(0))] 

            outputs = outputs['sequences']
            if if_only_return_new_tokens:
                outputs = outputs[:,inputs['input_ids'].size(1):].cpu().tolist()
            else:
                outputs = outputs.cpu().tolist()

            all_scores = all_scores + scores
            all_outputs = all_outputs + outputs

            # gc.collect() 
            # torch.cuda.empty_cache()

        return all_outputs, all_scores

    def generate_candidate_titles(self, datas, model, tokenizer):
        sentences = [f"Question: {data['query']}\n\n Related document title:" for data in datas]
        inputs = tokenizer.batch_encode_plus(sentences, padding=False, add_special_tokens=True, return_token_type_ids=False)
        outputs, scores = self.model_generation(
            model=model,
            print_info="generate_candidate_titles",
            tokenizer=tokenizer,
            sentences=inputs,
            batch_size=self.batch_size,
            max_new_tokens=256,
            beam_num=self.candidate_num,
            if_constrain=True,
            if_only_return_new_tokens=True,
            additional_eos_token_id=self.additional_eos_token_id
        ) # outputs and scores len is len(datas) * self.candidate_num

        ouptut_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for o, output_text in enumerate(ouptut_texts):
            data_index = o // self.candidate_num
            datas[data_index]["predicted_candidate_titles"].append([output_text, scores[o]]) 
        
        return datas

    def generate_titles(self, datas, model, tokenizer):
        sentences = []
        for data in datas:
            candidate_titles = ""
            if self.candidate_num > 0:
                candidate_titles_ = "\n ".join([c[0].strip() for c in data["predicted_candidate_titles"]])
                candidate_titles = f" Candidate document titles:\n {candidate_titles_}\n\n"
            
            data['input_for_title'] = f"Question: {data['query']}\n\n{candidate_titles} Related document title:"
            sentences.append(data['input_for_title'])

        inputs = tokenizer.batch_encode_plus(sentences, padding=False, add_special_tokens=True, return_token_type_ids=False)
        outputs, scores = self.model_generation(
            model=model,
            print_info="generate_titles",
            tokenizer=tokenizer,
            sentences=inputs,
            batch_size=self.batch_size,
            max_new_tokens=256,
            beam_num=self.beam_num_for_title,
            if_constrain=True,
            if_only_return_new_tokens=False,
            additional_eos_token_id=self.additional_eos_token_id
        )

        output_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for o, output_text in enumerate(output_texts): # output_text is from Question to Title
            clear_output = [td for td in outputs[o] if td != tokenizer.eos_token_id] # tokenids of question, candidate, title
            data_index = o // self.beam_num_for_title
            datas[data_index]['predicted_titles'].append([output_text, scores[o], clear_output])
        
        return datas

    def generate_1_chunks(self, datas, model, tokenizer):

        promt_for_chunk_ids = tokenizer(self.prompt_for_chunk, add_special_tokens=False)["input_ids"] # ids of Related document content:
        ids_from_qustion_to_title = []
        for data in datas:
            for predicted_title in data["predicted_titles"]:
                ids_from_qustion_to_title.append(predicted_title[2]) 
        input_ids = [ids+promt_for_chunk_ids for ids in ids_from_qustion_to_title]
        attention_mask = [[1 for _ in ids] for ids in input_ids]
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        outputs, scores = self.model_generation(
            model=model,
            print_info="generate_1_chunks",
            tokenizer=tokenizer,
            sentences=inputs, # len is len(datas) * self.beam_num_for_title
            batch_size=self.batch_size,
            max_new_tokens=self.num_for_chunk_generation_stage1, # only generate one token, thus named 1_chunk
            beam_num=self.beam_num_for_chunk,
            if_constrain=True,
            if_only_return_new_tokens=False, 
            additional_eos_token_id=None
        ) # outputs, scores len is len(datas) * self.beam_num_for_title * self.beam_num_for_chunk

        for o, output in enumerate(outputs): # output is tokenids from Question to 1_Chunk
            the_1_chunk_token_id = output[-self.num_for_chunk_generation_stage1:] # output[-1]
            clear_output = [td for td in output if td != tokenizer.eos_token_id]

            data_index = o // (self.beam_num_for_title * self.beam_num_for_chunk)
            datas[data_index]["predicted_1_chunks"].append([the_1_chunk_token_id, scores[o], clear_output])
        
        return datas

    def generate_extra_chunks(self, datas, model, tokenizer):

        input_ids = []
        for data in datas:
            for predicted_1_chunk in data["predicted_1_chunks"]:
                input_ids.append(predicted_1_chunk[2]) # tokenids from Question to 1_Chunk
        attention_mask = [[1 for _ in ids] for ids in input_ids]
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        outputs, scores = self.model_generation(
            model=model,
            print_info="generate_extra_chunks",
            tokenizer=tokenizer,
            sentences=inputs, # len is len(datas) * self.beam_num_for_title * self.beam_num_for_chunk
            batch_size=self.batch_size,
            max_new_tokens=256,
            beam_num=1, # no need to beam search
            if_constrain=True,
            if_only_return_new_tokens=False,
            additional_eos_token_id=None
        ) # outputs, scores len is len(datas) * self.beam_num_for_title * self.beam_num_for_chunk

        for o, output in enumerate(outputs): # output is tokenids from Question to Chunk

            start_pos = -1
            for i in range(len(output) - len(self.start_token_ids) + 1):
                if [int(j) for j in list(output[i: i + len(self.start_token_ids)])] == self.start_token_ids:
                    start_pos = i + len(self.start_token_ids) - 1
                    break
            start_pos += 1 # start position of the first token of the title in output
            output = output[start_pos:] # only keep token ids of title and chunk

            data_index = o // (self.beam_num_for_title * self.beam_num_for_chunk)
            chunkid = get_final_results_from_prefix(output, self.constrain_dict)
            datas[data_index]["predicted_extra_chunks"].append([chunkid, scores[o]])
        
        return datas

    def do_self_assessment(self, datas, model, tokenizer):
        sentences = []
        for data in datas:                
            for predicted_extra_chunk in data["predicted_extra_chunks"]:
                candidate_titles = ""
                if self.candidate_num > 0:
                    candidate_titles_ = "\n ".join([c[0].strip() for c in data["predicted_candidate_titles"]])
                    candidate_titles = f" Candidate document titles:\n {candidate_titles_}\n\n"

                chunkid = predicted_extra_chunk[0]
                title, chunk = self.chunkid_2_title_chunk[chunkid]
                sentences.append(
                    f"Question: {data['query']}\n\n{candidate_titles} Related document title: {title.lower()}\n Related document content: {chunk}\n\n Final answer:"
                )
        added_inputs = [self.self_assesment_sequence for _ in sentences]

        scores = self_assessment(data=sentences, batch_size=self.batch_size, device=model.device, tokenizer=tokenizer, model=model, addedinputs=added_inputs) # dericetly use jiawei's code
        for s, score in enumerate(scores):
            data_index = s // (self.beam_num_for_title * self.beam_num_for_chunk)
            datas[data_index]["predicted_self_assessment_score"].append(score)
        
        return datas

    def run_in_device(self, datas, model_path, device, device_num, device_id, shared_dict, if_half):
        current_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left', trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        model.eval()
        if if_half:
            model.half()
        model.to(device)

        if self.candidate_num > 0:
            datas = self.generate_candidate_titles(datas, model, tokenizer)
        print(f"{str(device)}, generate_candidate_titles_time:", time.time() - current_time)
        current_time = time.time()
        datas = self.generate_titles(datas, model, tokenizer)
        print(f"{str(device)}, generate_titles_time:", time.time() - current_time)
        current_time = time.time()
        datas = self.generate_1_chunks(datas, model, tokenizer)
        print(f"{str(device)}, generate_1_chunks_time:", time.time() - current_time)
        current_time = time.time()
        datas = self.generate_extra_chunks(datas, model, tokenizer)
        print(f"{str(device)}, generate_extra_chunks_time:", time.time() - current_time)
        current_time = time.time()
        datas = self.do_self_assessment(datas, model, tokenizer)
        print(f"{str(device)}, do_self_assessment_time:", time.time() - current_time)

        for d, data in enumerate(datas):
            shared_dict[d*device_num + device_id] = data

    def run(self, datas, model_path): # control multiple processings
        current_time = time.time()
        for data in datas:
            data["predicted_candidate_titles"] = [] # [['title', score], ...]
            data['input_for_title'] = ""
            data["predicted_titles"] = [] # [['title', score], ...]
            data["predicted_1_chunks"] = [] # [[the_1_token_id, score], ...]
            data["predicted_extra_chunks"] = [] # [[chunkid, score], ...]
            data["predicted_self_assessment_score"] = [] # [score, ...]

        num_gpus = torch.cuda.device_count()

        if num_gpus == 0:
            shared_dict={}
            self.run_in_device(datas=datas, model_path=model_path, device=torch.device("cpu"), device_num=1, device_id=0, shared_dict=shared_dict, if_half=False)
        else:
            processes = []
            manager = mp.Manager()
            shared_dict = manager.dict()

            for device_id in range(num_gpus):
                device = torch.device(f"cuda:{device_id}")
                datas_on_device = datas[device_id::num_gpus]
                p = mp.Process(target=self.run_in_device, args=(datas_on_device, model_path, device, num_gpus, device_id, shared_dict, True))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        sorted_results = [value for key, value in sorted(shared_dict.items(), key=lambda item: item[0])]
        print("len(sorted_results)", len(sorted_results))
        print("time", (time.time() - current_time) / 60, "min") # min
        return sorted_results

    def save(self, results, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        datas = []
        for result in results:
            data = {}
            data['id'] = result['id']
            data['query'] = result['query']
            data['answers'] = result['answers']
            data['title_scores'] = [t[1] for t in result['predicted_titles']]
            data['title_generation'] = [t[0] for t in result['predicted_titles']]
            data['chunk_scores'] = []
            data['documentids'] = {}
            data['model_scores'] = {}
            chunk_scores = [c[1] for c in result['predicted_1_chunks']]
            chunk_ids = [c[0] for c in result['predicted_extra_chunks']]
            self_assessment_scores = result['predicted_self_assessment_score']
            for t, title in enumerate(data['title_generation']):
                data['chunk_scores'].append(chunk_scores[t*self.beam_num_for_chunk:(t+1)*self.beam_num_for_chunk])
                data['documentids'][title] = chunk_ids[t*self.beam_num_for_chunk:(t+1)*self.beam_num_for_chunk]
                data['model_scores'][title] = self_assessment_scores[t*self.beam_num_for_chunk:(t+1)*self.beam_num_for_chunk]
            datas.append(data)

        with open(os.path.join(output_dir, "generation.json"), "w", encoding="utf-8") as f:
            for data in datas:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            
        with open(os.path.join(output_dir, "my_generation.json"), "w", encoding="utf-8") as f:
            for data in results:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

        print("save done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--trie", type=str)
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--sample_num", type=int, default=2000000)
    parser.add_argument("--candidate_num", type=int, default=0)
    parser.add_argument("--beam_num_for_title", type=int, default=5)
    parser.add_argument("--beam_num_for_chunk", type=int, default=10)
    parser.add_argument("--num_for_chunk_generation_stage1", type=int, default=1)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("start\n")

    trie = json.load(open(args.trie))
    corpus = json.load(open(args.corpus))
    print("len(corpus)", len(corpus))
    eval_datas = [json.loads(line) for line in open(args.data_path)][:args.sample_num]
    print("len(eval_datas)", len(eval_datas))

    chunkid_2_title_chunk = {} # {chunkid: [title, chunk], ...}
    for data in corpus:
        for c, chunk in enumerate(data["chunks"]):
            chunkid = data['wikipedia_id'] + '.' + str(c)
            chunkid_2_title_chunk[chunkid] = [data['title'], chunk] 
        
    # decide add_space things
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    in_context_token = tokenizer.encode(": title", add_special_tokens=False)
    with_space_token = tokenizer.encode(" title", add_special_tokens=False)
    without_space_token = tokenizer.encode("title", add_special_tokens=False)
    if in_context_token[1] == with_space_token[0]:
        start_sequence = " Related document title:" # used to start constrained generation
        self_assesment_sequence = ' cannot extract answer' # used to do self assessment
        prompt_for_chunk = " Related document content:" 
    elif in_context_token[1] == without_space_token[0]:
        start_sequence = "Related document title:" 
        self_assesment_sequence = 'cannot extract answer'
        prompt_for_chunk = "Related document content:"
    else:
        raise ValueError("Can't decide if need add space")

    # define some global variables
    VOCAB_SIZE = tokenizer.vocab_size
    EOS_TOKEN_ID = tokenizer.eos_token_id
    CONSTRAIN_DICT = {str(tokenizer.encode(start_sequence)[-1]): trie}
    START_TOKENS_ID = tokenizer(start_sequence, add_special_tokens=False)["input_ids"]

    retriever = Retriever(
        constrain_dict=trie, 
        chunkid_2_title_chunk=chunkid_2_title_chunk,
        start_token_ids=START_TOKENS_ID,
        self_assesment_sequence=self_assesment_sequence,
        prompt_for_chunk=prompt_for_chunk,
        batch_size=args.batch_size,
        candidate_num=args.candidate_num,
        beam_num_for_title=args.beam_num_for_title,
        beam_num_for_chunk=args.beam_num_for_chunk,
        num_for_chunk_generation_stage1=args.num_for_chunk_generation_stage1
    )

    my_results = retriever.run(eval_datas, args.model_path)
    retriever.save(my_results, args.output_dir)

    print('done')