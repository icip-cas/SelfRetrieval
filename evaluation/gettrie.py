from transformers import AutoTokenizer
import json
import tqdm
import json, random, tqdm, argparse, os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument("-model_path", "--model_path", type=str, default=model_path)
    parser.add_argument("-corpus", "--corpus", type=str, default=corpus)
    parser.add_argument("-outfile", "--outfile", type=str, default=outfile)
    

    args = parser.parse_args()

    print("args.model_path", args.model_path)
    print("args.corpus", args.corpus)
    print("args.outfile", args.outfile)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # decide if need add space
    in_context_token = tokenizer.encode(": title", add_special_tokens=False)
    with_space_token = tokenizer.encode(" title", add_special_tokens=False)
    without_space_token = tokenizer.encode("title", add_special_tokens=False)
    if in_context_token[1] == with_space_token[0]:
        add_space = True
    elif in_context_token[1] == without_space_token[0]:
        add_space = False
    else:
        raise ValueError("Can't decide if need add space")

    docs = json.load(open(args.corpus))

    chunks = []
    chunk2id = []
    tokens_dic = {}

    for doc in tqdm.tqdm(docs, desc="tokenize chunks"):
        chunkid = 0
        for chunk in doc['chunks']:
            title = doc['title'].lower()
            if add_space:
                chunk = f" {title}\n Related document content: {chunk}"
            else:
                chunk = f"{title}\n Related document content: {chunk}"
            tokens = tokenizer.encode(chunk,add_special_tokens=False) + [tokenizer.eos_token_id]
            tokens_str = [str(i) for i in tokens]
            tokens_str = '_'.join(tokens_str)
            if tokens_str not in tokens_dic:
                chunks.append(tokens)
                chunk2id.append(str(doc['wikipedia_id']) + '.' + str(chunkid))
                tokens_dic[tokens_str] = 1
            else:
                print(chunk)
            chunkid += 1

    print("len(chunks)", len(chunks))
    each_layer_index_num_dict = {}

    maxlayer = 0
    layercount = [0] * 2000

    def getpreftokens(indexs,chunks,layer):
        global maxlayer, layercount
        print("layer", layer)
        print("len(indexs)", len(indexs))
        prefixs = {}
        tokenid2docid = {}
        for index in indexs:
            if chunks[index][layer] not in prefixs:
                prefixs[chunks[index][layer]] = {}
                tokenid2docid[chunks[index][layer]] = []
            tokenid2docid[chunks[index][layer]].append(index)
        for tokenid in tokenid2docid:
            if len(tokenid2docid[tokenid]) > 1:
                try:
                    prefixs[tokenid] = getpreftokens(tokenid2docid[tokenid],chunks,layer + 1)
                except:
                    raise ValueError("layer", layer, "tokenid", tokenid, "tokenid2docid[tokenid]", tokenid2docid[tokenid])
            else:
                prefixs[tokenid] = chunk2id[tokenid2docid[tokenid][0]]
                if layer > maxlayer:
                    maxlayer = layer
                layercount[layer] += 1
        return prefixs

    prefixs = getpreftokens(list(range(len(chunks))),chunks,0)

    print("maxlayer", maxlayer)
    print("layercount", layercount)
    json.dump(prefixs, open(args.outfile,'w'))

    print("sum(layercount)", sum(layercount))