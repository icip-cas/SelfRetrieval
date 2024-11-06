
import argparse
import os
import random
import json, torch
import tqdm,math
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
import torch.multiprocessing as mp

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)  


def inference_on_device(device, model_path, data, shared_dict, batch_size=16, half=True, output_length=64,bad_words_ids=[]):
    print(f"load tokenizer on device {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path,padding_side='left',trust_remote_code=True)
    print(f"load model on device {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_path,trust_remote_code=True)
    if half:
        model.half()
    model.to(device)
    
    num_gpus = torch.cuda.device_count()
    for i in tqdm.tqdm(range(0, len(data), batch_size)):
        batched_inp = []
        instances = []
        for j in data[i: i + batch_size]:
            batched_inp.append(j['input'])
            instances.append(j)
        inputs = tokenizer(batched_inp, padding=True, return_tensors="pt", add_special_tokens=True,return_token_type_ids=False)
        with torch.no_grad():
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            if bad_words_ids is None:
                outputs = model.generate(**inputs, do_sample=False, max_new_tokens=20)
            else:
                outputs = model.generate(**inputs, do_sample=False,bad_words_ids=bad_words_ids, max_new_tokens=20)
        outputs = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

        for k in range(len(instances)):
            instance = instances[k]
            instance.update({'output': outputs[k]})
            instance.update({'rag_input': batched_inp[k]})
            shared_dict[(i + k) * num_gpus + device.index] = instance

def inference(model_path, data, output_path, batch_size=32, half=True, output_length=64, bad_words_ids=[]):
    num_gpus = torch.cuda.device_count()
    print(num_gpus)

    processes = []
    
    manager = mp.Manager()
    
    shared_dict = manager.dict()

    for device_id in range(num_gpus):
        device = torch.device(f"cuda:{device_id}")
        data_on_device = data[device_id::num_gpus]
        p = mp.Process(target=inference_on_device, args=(device, model_path, data_on_device, shared_dict, batch_size, half, output_length,bad_words_ids))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    with open(output_path,'w') as f:
        for qid in shared_dict:
            f.write(json.dumps(shared_dict[qid],ensure_ascii=False)+'\n')

def getcontext(titles, chunks, k):
    outputtext = ''
    for i in range(k):
        outputtext = outputtext + f" Related document title: {titles[i]}\n Related document content: {chunks[i]}\n\n"
    return outputtext


def getinput(instances, id2chunk, topk=5, prompt=1, lower=False,num_candidate=0):
    inputs = []
    for index,instance in enumerate(instances):
        if len(instance['answers']) == 0:
            continue
        query = instance['query']
        if topk == -1:
            k = 1
        else:
            k = topk
        retrieveids = instance['documentsids'][:k]
        retrievetitles = instance['titles'][:k]
        goldids = []
        goldtitles = []
        for answer in instance['answers']:
            answertext = answer['answer']
            for context in answer['context']:
                goldid = context['wikipedia_id'] + '.' + str(context['chunk_id'])
                goldids.append(goldid)
                goldtitles.append(context['title'])
        if topk == -1:
            if len(goldids) == 0:
                continue
            chunks = [id2chunk[goldids[0]]]
            titles = [goldtitles[0]]
        else:
            chunks = [id2chunk[i] for i in retrieveids]
            titles = retrievetitles
        input_text = f"Question: {query}\n\n"
        if num_candidate > 0:
            candidates = instance['candidates'][:num_candidate]
            input_text = input_text + " Candidate document titles:\n"
            for t in candidates:
                input_text = input_text + " " + t + "\n"
            input_text = input_text + "\n"
        if lower:
            titles = [i.lower() for i in titles]
        if prompt == 1:
            try:
                input_text = input_text + getcontext(titles, chunks, k) + " Final answer:"
            except:
                print("Error")
                print(titles)
                print(chunks)
                print(k)
                print(instance)
        elif prompt == 2:
            input_text = input_text + getcontext(titles, chunks, k) + "\n Final answer:"
        elif prompt == 3:
            input_text = 'Question: {query}\n\n Related document title: {title}\n Related document content: {context}\n\n Final answer:'.format(query=query,title=titles[0],context=chunks[0])
        instance['input'] = input_text
    return instances



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--plm", type=str, default='')
    parser.add_argument("--rplm", type=str, default='bge-large-en-v1.5')
    parser.add_argument("--testfile", type=str, default='test')
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--prompt", type=int, default=1)
    parser.add_argument("--num_candidate", type=int, default=0)
    parser.add_argument("--constraint", action='store_true')
    parser.add_argument("--eval_only", action='store_true')
    parser.add_argument("--lower", type=bool, default=False)
    parser.add_argument("--testsetfile", type=str)
    parser.add_argument("--contextfile", type=str)

    args = parser.parse_args()
    

    instances = []

    if not args.eval_only:
        bad_words_ids = None
        from transformers import AutoTokenizer, AutoModelForCausalLM,TextGenerationPipeline
        paragraph2chunk = {}
        id2title = {}
        contexts = json.load(open(contextfile))
        id2chunk = {}

        for instance in contexts:
            id2title[instance['wikipedia_id']] = instance['title']
            for index, chunk in enumerate(instance['chunks']):
                id2chunk[instance['wikipedia_id']+'.'+str(index)] = chunk

        with open(testsetfile) as f:
            for line in f:
                instances.append(json.loads(line))

        print('len(instances)', len(instances))
        print('len(id2chunk)', len(id2chunk.keys()))

        # instances = instances[:100]

        logging.getLogger().setLevel(logging.INFO)
        batch_size = 16
        half = False
        
        if 'stablelm' in args.rplm:
            half = True
            batch_size = 64
            if args.constraint:
                badword = " cannot"
                tokenizer = AutoTokenizer.from_pretrained(args.plm,trust_remote_code=True)
                bad_words_ids = tokenizer.encode(badword, add_special_tokens=False)
                bad_words_ids = [bad_words_ids]
        outputfile = outputfile + f"{args.rplm}_top{args.topk}_{args.testfile}.json"
        instances = getinput(instances, id2chunk, args.topk, args.prompt, args.lower, args.num_candidate)
        instances = [i for i in instances if 'input' in i]
        inference(args.plm, instances, outputfile, batch_size=batch_size, half=half, output_length=20, bad_words_ids=bad_words_ids)

    outputfile = outputfile + f"{args.rplm}_top{args.topk}_{args.testfile}.json"
    tt = 0
    emtt = 0
    num = 0
    usedids = []
    with open(outputfile) as f:
        for line in f:
            line = json.loads(line)
            num += 1
            em = 0
            for answer in line['answers']:
                answer = answer['answer'].strip().lower()
                if answer == line['output'].lower().strip():
                    emtt += 1
                    em = 1
                    break
            
            for answer in line['answers']:
                answer = answer['answer'].strip().lower()
                if answer in line['output'].lower().strip():
                    tt += 1
                    break
    
    print('Acc: '+str(tt/num))
    print('EM: '+str(emtt/num))
    print('#Num: ' + str(num))
