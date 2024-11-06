'''
generate nq document memory training data: Question: ${random sentence}\nContexts: ${contexts}
'''

import json
import random
import argparse
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
random.seed(42)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--random_sent_num", type=int, default=20)
    parser.add_argument("--lower_title", action="store_true", default=False)
    parser.add_argument("--tile_and_prefix_only", action="store_true", default=False)
    parser.add_argument("--use_all_sents", action="store_true", default=False)
    parser.add_argument("--prefix_length", type=int, default=10, help="only used when --tile_and_prefix_only is set")

    args = parser.parse_args()

    if args.use_all_sents:
        args.random_sent_num = None

    data = json.load(open(args.data_path))

    train_data = []
    chunk_cnt = 0
    cnt = 0
    for document in tqdm(data):
        chunk_cnt += len(document["chunks"])
        for chunk_id, chunk in enumerate(document["chunks"]):
            sentences_list = []
            chunk_lines = [i.strip() for i in chunk.split("\n") if len(i.strip()) > 0]
            for line in chunk_lines:
                sent = sent_tokenize(line, language='english')
                if "Section::::" not in sent and 5 < len(sent.split(" ")) < 50:
                     sentences_list.append(sent)
            if len(sentences_list) == 0:
                continue

            cnt += len(sentences_list) == 0

            for sent in sentences_list[:args.random_sent_num]:
                doc_title = document['title']
                if args.lower_title:
                    sent = sent.lower()
                    doc_title = doc_title.lower()
                if args.tile_and_prefix_only:
                    chunk_prefix = " ".join(chunk.split(" ")[:args.prefix_length])
                    
                    question = f"Question: {sent}\n\n Related document title: "
                    context = f"{doc_title}\n Related document content: {chunk_prefix}"
                else:
                    question = f"Question: {sent}\n\n Related document title: "
                    context = f"{doc_title}\n Related document content: {chunk}"
                
                train_data.append([[question, context], [False, True]])
        if len(train_data) % 10000 == 0:    
            print(chunk_cnt)
            print(cnt)
            print(len(train_data))
    random.shuffle(train_data)

    json.dump(train_data, open(args.output_path, "w"))
