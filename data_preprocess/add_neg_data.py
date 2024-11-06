import json
import random
import argparse
from tqdm import tqdm

random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='')
    parser.add_argument("--doc_path", type=str, default='')
    parser.add_argument("--output_path", type=str, default='')
    parser.add_argument("--neg_num", type=int, default=2)
    parser.add_argument("--other_doc_neg_num", type=int, default=0)

    args = parser.parse_args()

    qa_data = []
    with open(args.data_path) as f:
        for line in f.readlines():
            qa_data.append(json.loads(line))

    docs = json.load(open(args.doc_path))
    id2docs = {i["wikipedia_id"]: i for i in docs}
    title2docs = {i["title"].lower(): i for i in docs}

    for instance_idx, instance in tqdm(enumerate(qa_data)):
        gold_chunk_ids = []
        gold_doc_ids = []
        gold_answers = []
        gold_titles = []
        for answer in instance['answers']:
            gold_answers.append(answer['answer'].lower())
            for context in answer['context']:
                gold_titles.append(context["title"])
                gold_doc_ids.append(context['wikipedia_id'])
                gold_chunk_ids.append(f"{context['wikipedia_id']}.{context['chunk_id']}")
        
        if len(gold_answers) == 0 or len(gold_chunk_ids) == 0:
            continue
        tmp = {
            "answer": "cannot extract answer",
            "context": []
        }
        for gold_did in gold_doc_ids:
            all_neg_chunks = []
            for ck_id, ck in enumerate(id2docs.get(gold_did, {}).get("chunks", [])):
                if f"{gold_did}.{ck_id}" in gold_chunk_ids:
                    continue
                if any([ans.lower() in ck.lower() for ans in gold_answers]):
                    continue
                if len(ck) > 50:
                    all_neg_chunks.append(ck_id)
            
            for ck_id in random.sample(all_neg_chunks, min(len(all_neg_chunks), args.neg_num)):
                tmp["context"].append({
                "wikipedia_id": gold_did,
                "title": id2docs[gold_did]["title"],
                "chunk_id": ck_id,
                "neg": True
            })
        if args.other_doc_neg_num > 0:
            all_other_odc_neg_chunks = []

            for tmp_instance in qa_data[instance_idx+1: instance_idx+21]:
                for answer in tmp_instance['answers']:
                    for context in answer['context']:
                        if context['wikipedia_id'] not in gold_doc_ids and \
                            context['wikipedia_id'] in id2docs:
                            ck_id = context['chunk_id']
                            ck = id2docs[context['wikipedia_id']]["chunks"][ck_id]
                            if any([ans.lower() in ck.lower() for ans in gold_answers]):
                                continue
                            if len(ck) > 50:
                                all_other_odc_neg_chunks.append((context['wikipedia_id'], ck_id))
            for did, ck_id in random.sample(all_other_odc_neg_chunks, min(len(all_other_odc_neg_chunks), args.other_doc_neg_num)):
                tmp["context"].append({
                "wikipedia_id": did,
                "title": id2docs[did]["title"],
                "chunk_id": ck_id,
                "neg": True
            })

        instance['answers'].append(tmp)

    with open(args.output_path, "w", encoding="utf-8") as f:
        for instance in qa_data:
            f.write(f"{json.dumps(instance)}\n")

        