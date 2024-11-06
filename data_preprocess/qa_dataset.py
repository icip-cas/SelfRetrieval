"""
1.12:
1. set use_all_answers default to True
2. set lower_title default to True
3. set qa_num default to all
3. add argument config_path to specify the config file
"""
import os
import json
import random
import argparse
from nltk import sent_tokenize

random.seed(42)

def rearrange_list_simplified(lst):
    if not lst or len(lst) < 5:
        return lst

    # Define the probabilities and corresponding positions
    probabilities = [50, 20, 10, 10, 10]  # Probabilities for positions 0, 1, 2, 3, 4
    positions = list(range(5))

    # Choose a position based on the defined probabilities
    new_position = random.choices(positions, weights=probabilities, k=1)[0]

    # Move the first element to the new position if necessary
    if new_position != 0:
        element = lst.pop(0)
        lst.insert(new_position, element)

    return lst

def merge_candidates(qa_data, candidate_data):
    title2qa = {instance["query"]: instance for instance in qa_data}
    for instance in candidate_data:
        title = instance["input"].split("Question: ", 1)[1].split("\n\n Related document title:", 1)[0]
        candidates = []
        for cddt in instance["all_output"]:
            if "\n\n Related document title:" in cddt["generated_text"]:
                cand = cddt["generated_text"].split("\n\n Related document title:", 1)[1].split("\n", 1)[0].strip()
            else:
                cand = cddt["generated_text"].split("\n", 1)[0].strip()
            candidates.append(cand)
        # candidates = [cddt["generated_text"].split("\n\n Related document title: ", 1)[1].split("\n", 1)[0].strip() for cddt in instance["all_output"]]
        title2qa[title]["candidate_titles"] = candidates
    cnt = 0
    for instance in qa_data:
        if "candidate_titles" not in instance:
            continue
        cnt += 1
    print("merge_candidates", cnt)
    return qa_data


def construct_candidate_titles(candidates, golden, num=10):
    chosen_candidates = random.sample(candidates, num)
    if golden not in chosen_candidates:
        chosen_candidates[random.randint(0, num - 1)] = golden
    return chosen_candidates


def build_data_from_spans(spans, eos_token):
    data = []
    trainable = []
    pre_label = 0
    text_now = ""
    for span in spans + [("", 0)]:
        text, label = span
        if pre_label == 0:
            if label == 0 or label == 0.5:
                text_now += text
            else:
                if not text_now.endswith(" "):
                    text_now += " "
                if text_now.startswith(" "):
                    text_now = text_now[1:]
                data.append(text_now)
                trainable.append(False)
                text_now = text
                pre_label = 1
        else:
            if label == 1 or label == 0.5:
                text_now += text
            else:
                if not text_now.endswith(" "):
                    text_now += " "
                if text_now.startswith(" "):
                    text_now = text_now[1:]
                data.append(text_now)
                trainable.append(True)
                text_now = text
                pre_label = 0
    if len(data) == 0:
        return None
    if spans[-1][1] == 1:
        data[-1] = data[-1] + eos_token
    if data[-1].endswith(" "):
        data[-1] = data[-1][:-1]
    return [data, trainable]

        

def build_qa_dataset(config, qa_data, eos_token, max_content_length=None, first_cfg=False):
    qa_dataset = []
    for instance in qa_data:
        answer = instance["answers"][0]


        if max_content_length is not None and first_cfg:
            answer["context"] = " ".join(answer["context"][:max_content_length * 15].split(" ")[:max_content_length]) + "..."

        if answer["answer"] == "" and config.get("no_qg_content", False):
            continue
        if answer.get("contain", False) and not config.get("contain", False):
            continue
        if answer.get("neg", False) and not config.get("neg", False):
            continue
        if answer.get("qg", False) and not (config.get("qg", False) or config.get("qg_only", False)):
            continue
        if config.get("qg_only", False) and not answer.get("qg", False):
            continue

        if config.get("query_generation", False):
            tmp = {
                "query": "based on the following document, generate an appropriate question.",
                "answers": [{
                    "title": answer["title"],
                    "context": answer["context"],
                    "answer": instance["query"]
                }]
            }
            instance = tmp
            answer = instance["answers"][0]

        if config.get("query_generation_answer", False):
            tmp = {
                "query": f"based on the following document, generate an appropriate question whose answer is '{answer['answer']}'.",
                "answers": [{
                    "title": answer["title"],
                    "context": answer["context"],
                    "answer": instance["query"]
                }]
            }
            instance = tmp
            answer = instance["answers"][0]

        spans = [
            (f"Question: {instance['query']}", 0)
        ]
        if config.get("title_candidate"):
            assert "candidate_titles" in instance
            if not instance["candidate_titles"]:
                if not config.get("random_candidates", False):
                    continue
                instance["candidate_titles"] = random.sample(all_doc_titles, config.get("num_candidate_titles", 10))
            chosen_candidates = construct_candidate_titles(
                instance["candidate_titles"],
                answer["title"],
                config.get("num_candidate_titles", 10)
            )
            all_titles = "\n ".join(chosen_candidates)
            spans.append((f"\n\n Candidate document titles:\n {all_titles}", 0))


        if config.get("topk_qa"):
            num = config["topk_qa"]
            golden_wiki_id = answer["wikipedia_id"]
            golden_chunk_id = answer["chunk_id"]
            template = "\n\n Related document title: {title}\n Related document content: {content}"
            all_topk_prompts = [template.format(title=answer["title"], content=answer["context"])]
            for i in range(num):
                topk_wiki_id = instance["candidate_chunks"][i]["wikipedia_id"]
                topk_chunk_id = instance["candidate_chunks"][i]["chunk_id"]
                if topk_wiki_id == golden_wiki_id and topk_chunk_id == golden_chunk_id:
                    continue
                all_topk_prompts.append(template.format(title=instance["candidate_chunks"][i]["title"], content=instance["candidate_chunks"][i]["context"]))
            all_topk_prompts = all_topk_prompts[:num]
            random.shuffle(all_topk_prompts)
            # all_topk_prompts = rearrange_list_simplified(all_topk_prompts)
            spans.append(("".join(all_topk_prompts), 0))
        else:
            spans.append((f"\n\n Related document title:", 0.5))
            if config["title"]:
                spans.append((f" {answer['title']}", 1))
            else:
                spans.append((f" {answer['title']}", 0))

            if config.get("thought", None) is not None:
                spans.append((f"\n Thought:", 0.5))
                spans.append((answer['thought'], config.get("thought")))
        
            spans.append((f"\n Related document content:", 0.5))
            
            if config["content_prefix"]:
                content_prefix = " ".join(answer["context"].split(" ")[:config.get("prefix_length", 10)])
                context_suffix = " ".join(answer["context"].split(" ")[config.get("prefix_length", 10):])
                if config.get("thought_as_prefix"):
                    content_prefix = " ".join(answer["thought"].split(" ")[:config.get("prefix_length", 10)])
                    context_suffix = " ".join(answer["thought"].split(" ")[config.get("prefix_length", 10):]) + " " + answer["context"]
                spans.append((f" {content_prefix}", 1))
                spans.append((f" {context_suffix}", 0))
            elif config["content_all"]:
                spans.append((f" {answer['context']}", 1))
            else:
                spans.append((f" {answer['context']}", 0))
        

        spans.append((f"\n\n Final answer:", 0.5))

        if config["answer"]:
            if answer["answer"]:
                spans.append((f" {answer['answer']}", 1))
            else:
                spans.append(("", 0))

        data_instance = build_data_from_spans(spans, eos_token)
        if data_instance is not None:
            qa_dataset.append(data_instance)
    
    if config["ratio"] != 1:
        qa_dataset *= config["ratio"]
    
    print("*** build_qa_dataset ***")
    print("Config:", config)
    print("Total number of instances:", len(qa_dataset))
    print("Case 1:")
    print(qa_dataset[0])
    print("************************")

    return qa_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_path", type=str, default="")
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--mem_data_path", type=str, default="")
    parser.add_argument("--mem_data_path2", type=str, default="")
    parser.add_argument("--candidate_data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--config_path", type=str, default="")
    parser.add_argument("--qa_repeat_times", type=int, default=1)
    parser.add_argument("--eos_token", type=str, default="</s>") 
    parser.add_argument("--max_content_length", type=int, default=None)   
    args = parser.parse_args()

    ori_qa_data = []
    with open(args.qa_path) as f:
        for line in f.readlines():
            ori_qa_data.append(json.loads(line))
    
    doc_data = json.load(open(args.doc_path))
    doc_dict = {i["wikipedia_id"]: i for i in doc_data}
    all_doc_titles = [doc["title"] for doc in doc_data]
    
    mem_data = []
    if args.mem_data_path:
        mem_data = json.load(open(args.mem_data_path))
    if args.mem_data_path2:
        mem_data.extend(json.load(open(args.mem_data_path2)))
    
    if args.candidate_data_path:
        candidate_data = json.load(open(args.candidate_data_path))
        ori_qa_data = merge_candidates(ori_qa_data, candidate_data)

    config = json.load(open(args.config_path))

    qa_data = []
    no_answer_cnt = 0
    for sample in ori_qa_data:
        tmp = {
            "id": sample.get("id", ""),
            "query": sample["query"],
            "answers": [],
            "candidate_titles": None if "candidate_titles" not in sample else sample["candidate_titles"][:20],
            "candidate_chunks": [],
        }
        if "documentids" in sample:
            for did_cid in sample["documentids"]:
                candidate_ck = {}
                doc_id, chunk_id = did_cid.split(".")
                candidate_ck["wikipedia_id"] = doc_id
                candidate_ck["chunk_id"] = int(chunk_id)
                candidate_ck["title"] = doc_dict[doc_id]["title"]
                candidate_ck["context"] = doc_dict[doc_id]["chunks"][int(chunk_id)]
                tmp["candidate_chunks"].append(candidate_ck)

        if isinstance(sample["answers"], dict):
            sample["answers"] = [sample["answers"]]
        for answer in sample["answers"]:
            # if isinstance(answer, str):
            #     print(sample)
            #     exit()
            for cont in answer["context"]:
                if cont["wikipedia_id"] in doc_dict:
   
                    doc = doc_dict[cont["wikipedia_id"]]
                    tmp["answers"].append({
                        "answer": answer["answer"],
                        "wikipedia_id": cont["wikipedia_id"],
                        "paragraph_id": cont.get("paragraph_id", ""),
                        "chunk_id": cont["chunk_id"],
                        "start_character": cont.get("start_character", ""),
                        "end_character": cont.get("end_character", ""),
                        "title": doc["title"].lower(),
                        "context": doc["chunks"][cont["chunk_id"]],
                        # "thought": thought,
                        "contain": cont.get("contain", False),
                        "neg": cont.get("neg", False),
                        "qg": sample.get("qg", cont.get("qg", False)),
                        # "keywords": None if "keywords" not in doc else doc["keywords"][ck],
                        
                    })
                    # break
        if len(tmp["answers"]) > 0:
            qa_data.append(tmp)
        else:
            no_answer_cnt += 1
    print("qa data len", len(qa_data))
    print("no answer cnt", no_answer_cnt)

    new_qa_data = []
    for sample in qa_data:
        for answer in sample["answers"]:
            new_qa_data.append({
                "id": sample["id"],
                "query": sample["query"],
                "answers": [answer],
                "candidate_titles": sample["candidate_titles"],
                "candidate_chunks": sample["candidate_chunks"],
            })
    qa_data = new_qa_data
    print("qa data pair len", len(qa_data))

    used_docs = {}
    all_used_docs = {}
    for sample in qa_data:
        for answer in sample["answers"]:
            all_used_docs[answer["wikipedia_id"]] = True
        used_docs[sample["answers"][0]["wikipedia_id"]] = True
    print("used doc", len(used_docs))
    print("all used doc", len(all_used_docs))
    
    final_qa_data = []
    idx = 0
    for cfg in config:
        idx += 1
        qa_dataset = build_qa_dataset(cfg, qa_data, args.eos_token, args.max_content_length, first_cfg=(idx == 1))
        final_qa_data.extend(qa_dataset)
    
    print("final used qa data", len(final_qa_data))
    final_train_data = []
    for idx in range(args.qa_repeat_times):
        tmp = []
        tmp.extend(final_qa_data)
        for sample in mem_data[int(idx / args.qa_repeat_times * len(mem_data)): int((idx + 1) / args.qa_repeat_times * len(mem_data))]:
            tmp_inp = sample[0][0]
            #!! 4.28 Bug: do not add 'final answer' after content prefix
            # tmp_output = sample[0][1] + "\n\n Final answer:"
            tmp_output = sample[0][1]
            tmp.append([[tmp_inp, tmp_output], [False, True]])
        random.shuffle(tmp)
        final_train_data.extend(tmp)
    
    json.dump(final_train_data, open(args.output_path, "w"), indent=4)
    print("final train data", len(final_train_data))

    with open("./qa_data_history.txt", "a") as f:
        f.write("======================\n")
        f.write(f"args: {args}\n")
        f.write(f"config: {config}\n")
        f.write(f"qa_data: {len(qa_data)}\n")
        f.write(f"final_qa_data: {len(final_qa_data)}\n")
        f.write(f"final_train_data: {len(final_train_data)}\n")
        f.write("======================\n")
