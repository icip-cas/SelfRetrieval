import os
import torch
import json, gc
import argparse, copy
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline, LogitsProcessor


def retrievaleval(preds, golds):
    top1 = 0
    top5 = 0
    num = 0
    mrr = 0
    for qid in preds:
        if qid not in golds:
            continue
        qtop1 = 0
        qtop5 = 0
        mrrflag = True
        for index,did in enumerate(preds[qid][:5]):
            if did in golds[qid]:
                if index == 0:
                    qtop1 = 1
                qtop5 = 1
                if mrrflag:
                    mrrflag = False
                    mrr += 1/(index+1)
        top1 += qtop1
        top5 += qtop5
        num += 1
    print('#Num: ' +str(num))
    print('Hit@1: '+ str(top1/num))
    print('Hit@5: '+ str(top5/num))
    print('MRR@5: '+ str(mrr/num))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--corpus", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("-p_title_score", "--p_title_score", type=float, default=0.4)
    parser.add_argument("-p_selfassesment_score", "--p_selfassesment_score", type=float, default=0.4)
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(f'{k}: {v}')
    print("start\n\n")

    id2chunk = {}
    id2title = {}
    contexts = json.load(open(args.corpus))
    for instance in contexts:
        id2title[instance['wikipedia_id']] = instance['title']
        for index, chunk in enumerate(instance['chunks']):
            id2chunk[instance['wikipedia_id']+'.'+str(index)] = chunk
    
    os.makedirs(args.output_path, exist_ok=True)
    generatefile = args.output_path + '/generation.json'
    outputfile = args.output_path + '/retrieve.json'

    with open(args.data_path) as f:
        data = [json.loads(line) for line in f]

    with open(generatefile) as f:
        instances = [json.loads(i) for i in f]

    with open(outputfile, 'w') as f:
        num = 0
        for instance in instances:
            instance = copy.deepcopy(instance)
            title2score = {} # this is chunk_score, which is about related document content's first token
            title2answerscore = {} # this is model_score, which is from self assessment
            instance_scores = [] # this is title_score, which is about related document title
            for i in range(len(instance['title_generation'])):
                if instance['title_scores'][i] > -1000000000.0:
                    if 'model_scores' in instance:
                        title2answerscore[instance['title_generation'][i]] = instance['model_scores'][instance['title_generation'][i]]

                    title2score[instance['title_generation'][i]] = instance['chunk_scores'][i]
                    instance_scores.append(instance['title_scores'][i])
            alldocs = {}
            instance_scores = torch.exp(torch.Tensor(instance_scores)).tolist()
            instance_scores = torch.softmax(torch.Tensor(instance_scores)/args.p_title_score,dim=-1).tolist()
            for index, title in enumerate(title2score):
                chunkscores = []
                chunksanswercores = []
                chunkids = []
                docscore = instance_scores[index] # this is title_score
                for j in range(len(title2score[title])):
                    if title2score[title][j] > -1000000000.0:
                        chunkscores.append(title2score[title][j]) # this is about related document content's first token
                        chunkids.append(instance['documentids'][title][j])
                        if title in title2answerscore:
                            chunksanswercores.append(title2answerscore[title][j]) # this is about self assessment
                chunkscores = torch.exp(torch.Tensor(chunkscores)).tolist()
                chunkscores = torch.softmax(torch.Tensor(chunkscores),dim=-1).tolist()

                chunksanswercores = (1-torch.exp(-torch.Tensor(chunksanswercores))).tolist()
                chunksanswercores = torch.softmax(torch.Tensor(chunksanswercores)/args.p_selfassesment_score,dim=-1).tolist()

                chunkscores = [chunksanswercores[index] * docscore for index in range(len(chunkscores))] # so bove chunkscores is no used

                for j in range(len(chunkscores)):
                    if type(chunkids[j]) is not str:
                        num += 1
                        continue
                    alldocs[chunkids[j]] = chunkscores[j]
            
            alldocs = sorted(alldocs.items(), key=lambda item: item[1], reverse=True)
            newpred = {'documentsids': [], 'scores': [], 'titles': []}
            for i in alldocs[:50]:
                newpred['documentsids'].append(i[0])
                newpred['scores'].append(i[1])
                newpred['titles'].append(id2title[i[0].split('.')[0]])
            instance.update(newpred)
            f.write(json.dumps(instance,ensure_ascii=False)+'\n')
    print('#NUM with dict:', num)

    testresult = {}
    with open(outputfile) as f:
        for line in f:
            line = json.loads(line)
            testresult[line['id']] = line['documentsids']
    
    goldanswers = {}
    for line in data:
        golds = []
        golddocs = []
        gold_lists = []
        for answer in line['answers']:
            goldlist = []
            for context in answer['context']:
                contextid = context['wikipedia_id'] + '.' + str(context['chunk_id'])
                if id2chunk is None or contextid in id2chunk and contextid not in golds:
                    golds.append(contextid)
        if len(golds) == 0:
            continue 
        goldanswers[line['id']] = golds
    

    retrievaleval(testresult,goldanswers)


""" Explain for Instances
input: query + "Question: " 

candidates: candidate document titles

input_candidate: to get related document title
title_scores: scores for related document title
title_generation: all related document title
title_outputs: token ids of to get title generation by decoder

input_chunk: to get related document content's first token
chunk_scores: scores for related document content's first token
chunk_generation: all related document content's first token
chunk_outputs: token ids of to get chunk generation by decoder

documentids: chunk ids for each related document title
generation: continueing based on the first token
outputs: token ids of to get generation by decoder

model_scores: scores for self assessment
"""