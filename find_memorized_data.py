import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pdb
import os
import argparse
import json
from Levenshtein import distance as lev_dist
from nltk.util import ngrams
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch_data(args, idx, tokenizer):
    assert 'pythia' in args.model_name
    batch_file = os.path.join(args.data_dir, f'batch{idx}_bs1024.npy')
    data = np.load(batch_file)
    data = torch.LongTensor(data[:, :args.seq_len]) # truncate
    print(data.shape)
    return data


@torch.no_grad()
def retest(args, model, tokenizer):
    filename = f'pile_bs{args.start}-{args.end}-dedup.pt'
    data = torch.load(os.path.join(args.out_dir, filename))
    prompts = data[:, :args.prompt_len]
    print(data.shape, prompts.shape)

    ref_texts = tokenizer.batch_decode(data)
    gen_texts = []
    for batch in tqdm(prompts.chunk(args.n_batches)):
        batch = batch.to(device)
        gen_tokens = model.generate(
            batch, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_length=args.seq_len,
            )

        gen_texts.extend(tokenizer.batch_decode(gen_tokens))

    for gen_txt, ref_txt in zip(gen_texts, ref_texts):
        distance = lev_dist(gen_txt, ref_txt)
        print("Lev_dist:", distance)
        print(ref_txt)
        print('-'*120)
        if distance > args.filter_dist:
            pdb.set_trace()


@torch.no_grad()
def get_loss(args, idx, data, model, tokenizer):
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    out_correctness, out_loss = [], []
    for batch in tqdm(data.chunk(args.n_batches)):
        batch = batch.to(device)
        logits = model(batch).logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch[..., 1:].contiguous()
        nll_batch = loss_fct(shift_logits.transpose(1, 2), shift_labels) #[bs, seq_len]
        is_correct_batch = (shift_logits.argmax(-1) == shift_labels)

        out_correctness.append(is_correct_batch.cpu())
        out_loss.append(nll_batch.cpu())

    out_correctness = torch.cat(out_correctness)
    out_loss = torch.cat(out_loss)

    print(out_correctness.float().mean(1))

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(out_correctness, os.path.join(args.out_dir, f'correctness_batch{idx}.pt'))
    torch.save(out_loss, os.path.join(args.out_dir, f'loss_batch{idx}.pt'))


@torch.no_grad()
def get_memorized_data(args, model, tokenizer):
    filename = f"pile_bs{args.start}-{args.end}_{args.seq_len}.jsonl"
    f_out = open(os.path.join(args.out_dir, filename), "w")

    cnt = 0
    memorized_data = []
    for idx in tqdm(range(args.start, args.end)):
        data = get_batch_data(args, idx, tokenizer)
        correctness = torch.load(os.path.join(args.out_dir, f'correctness_batch{idx}.pt'))
        #loss = torch.load(os.path.join(args.out_dir, f'loss_batch{idx}.pt'))

        correctness = correctness[:, args.prompt_len-1 : ].numpy()

        select_ids = np.where(correctness.mean(1) > 0.9)[0] # 1. filter by acc
        data = data[select_ids]
        
        ref_texts = tokenizer.batch_decode(data)
        prompts = data[:, :args.prompt_len].to(device)

        gen_tokens = model.generate(
            prompts, 
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            max_length=args.seq_len,
            ).cpu()

        gen_texts = tokenizer.batch_decode(gen_tokens)

        prompts = prompts.cpu()
        for j, prompt, gen_tok, gen_txt, ref_txt, dp in zip(select_ids, prompts, gen_tokens, gen_texts, ref_texts, data):
            # 2. filter out "gibberish", which has many repetitive tokens
            if len(torch.unique(gen_tok[args.prompt_len:])) < 16 or len(torch.unique(dp[args.prompt_len:])) < 16: continue

            distance = lev_dist(gen_txt, ref_txt)
            # 3. filter with levenshtein distance after greedy decoding
            if distance <= args.filter_dist:
                prompt_txt = tokenizer.decode(prompt)
                print("Lev dist:", distance)
                print(prompt_txt)
                print("-"*120)
                print(ref_txt)
                print("-"*120)
                print(gen_txt)
                print("="*120)

                line = {
                    'ex_i': len(memorized_data),
                    'prompt': prompt_txt,
                    'ref': ref_txt,
                    'gen': gen_txt,
                    'levenshtein_distance': distance,
                    'pile_idx': f'bs{idx}_{j}'
                }
                f_out.write(json.dumps(line)+"\n")
                f_out.flush()
                memorized_data.append(dp)
                cnt += 1

    f_out.close()
    print("# Memorized data after filtering:", cnt)
    memorized_data = torch.stack(memorized_data)
    filename = f'pile_bs{args.start}-{args.end}_{args.prompt_len}_{args.seq_len}.pt'
    torch.save(memorized_data, os.path.join(args.out_dir, filename))


def aggressive_dedup(args, tokenizer):

    def jaccard_similarity(set1, set2):
        return len(set1.intersection(set2)) / len(set1.union(set2))

    def bfs(s, c_id):
        visited[s] = True
        que = []
        que.append(s)
        while len(que) > 0:
            u = que.pop(0)
            components[u] = c_id
            connected_nodes = np.where(scores[u] > 0.05)[0]
            for i in connected_nodes:
                if not visited[i]:
                    visited[i] = True
                    que.append(i)
        

    # load tokenized data; get n-gram
    data_tokens = torch.load(os.path.join(args.out_dir, 
        f'pile_bs{args.start}-{args.end}_{args.prompt_len}_{args.seq_len}.pt')).numpy()
    data_ngrams = [set(ngrams(dp, 5)) for dp in data_tokens]

    with open(os.path.join(args.out_dir, f'pile_bs{args.start}-{args.end}_{args.seq_len}.jsonl'), 'r') as f:
        docs, dists = [], []
        for line in f:
            dp = json.loads(line)
            docs.append(dp['ref'])
            dists.append(dp['levenshtein_distance'])
    dists = np.array(dists)

    # calculate jaccard_similarity in n-gram 
    n_data = len(docs)
    scores = np.zeros((n_data, n_data))
    for i in range(n_data):
        for j in range(i+1, n_data):
            doc1 = data_ngrams[i]
            doc2 = data_ngrams[j]
            scores[i,j] = scores[j,i] = jaccard_similarity(doc1, doc2)

            if args.debug and 0.05 < scores[i,j] < 0.1:
                print(scores[i,j])
                print('-'*100)
                print(docs[i])
                print('-'*100)
                print(docs[j])
                print('='*100)
                overlaps = doc1.intersection(doc2)
                for ng in overlaps:
                    print(tokenizer.decode(list(ng)))
                print('='*100)
                pdb.set_trace()

    # find connected components
    # two data are "connected"/near-duplicate if their similarity score is large
    visited = np.zeros(n_data, dtype=bool)
    components = np.zeros(n_data, dtype=int)
    idx = 0
    for i in range(n_data):
        if not visited[i]:
            bfs(i, idx)
            idx += 1
    assert components.max() == idx-1
    print("# data remain:", idx)

    # dedup; only keep the dp with lowest lev_distance in each connected component
    deduped_data, deduped_dists = [], []
    for i in range(idx):
        ids = np.where(components == i)[0]
        j = ids[dists[ids].argmin()]
        deduped_data.append(data_tokens[j])
        deduped_dists.append(dists[j])

        if len(ids) > 1 and args.debug:
            print('# data in this group:', len(ids))
            print(docs[j])
            print('-'*100)
            print(docs[ids[-1]])
            print('='*100)
            pdb.set_trace()
    deduped_data = torch.tensor(np.stack(deduped_data))
    deduped_dists = torch.tensor(deduped_dists)

    # get more data then we need # discard those with largest lev_distance
    if args.n_data_clip < len(deduped_data):
        _, indices = torch.topk(deduped_dists, args.n_data_clip, largest=False)
        deduped_data = deduped_data[indices]

    # randomly sample 5 dev examples and put them in the front of the tensor
    np.random.seed(0)
    dev_ids = np.random.choice(len(deduped_data), 5, replace=False)
    print("Dev IDs:", dev_ids)
    for i,j in enumerate(dev_ids):
        deduped_data[i], deduped_data[j] = deduped_data[j].clone(), deduped_data[i].clone()

    torch.save(deduped_data, os.path.join(args.out_dir, f'pile_bs{args.start}-{args.end}-dedup.pt'))
    
    if args.log:
        texts = tokenizer.batch_decode(deduped_data)
        for txt in texts:
            print(txt)
            print('-'*100)


def load_pretrained_(model_name, load_model=True, to_gpu=True):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if load_model:
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if to_gpu: model.to(device)

    else: model = None

    return tokenizer, model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="mydata")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-2.8b-deduped-v0")
    parser.add_argument("--start", type=int, default=0, help="strat batch idx")
    parser.add_argument("--end", type=int, default=10, help="end batch ids")
    parser.add_argument("--n_batches", type=int, default=512)
    parser.add_argument("--n_data_clip", type=int, default=505)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--filter_dist", type=int, default=20)
    parser.add_argument("--do_loss", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()
    args.out_dir = os.path.join(args.data_dir, args.model_name)

    tokenizer, model = load_pretrained_(args.model_name, load_model=True)
    if model is not None: model.eval()

    if args.do_loss:
        for idx in range(args.start, args.end):
            data = get_batch_data(args, idx, tokenizer)
            get_loss(args, idx, data, model, tokenizer)
    else:
        get_memorized_data(args, model, tokenizer)
        aggressive_dedup(args, tokenizer)
        #retest(args, model, tokenizer)
