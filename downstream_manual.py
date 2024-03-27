import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pdb
import os
import argparse
from tabulate import tabulate
from Levenshtein import distance as lev_dist

from utils_downstream import *
from baselines import *
from knowledge_neuron import *
from hardconcrete import hard_concrete
from slim import slim

torch.manual_seed(0)


@torch.no_grad()
def get_memo_scores(model, tokenizer, examples):
    all_acc, all_out_texts = [], []
    for dp in examples: # bs = 1
        prompt_len, prompt, ex = dp['prompt_len'], dp['prompt'], dp['gt_tokens']
        inputs = prep_examples(ex, prompt_len)
        inputs.to(device)

        logits = model(**inputs).logits.squeeze()
        
        preds = logits[prompt_len-1:-1].argmax(-1).cpu()
        acc = (dp['gt_tokens'].squeeze()[prompt_len: ] == preds).numpy().mean()
        output_txt = prompt + tokenizer.decode(preds)
        all_acc.append(acc)
        all_out_texts.append(output_txt)

    return np.array(all_acc), all_out_texts


def levenshtein_distance(tokenizer, output_texts, examples):
    distances = np.zeros(len(output_texts), dtype=int)
    for i, (out_str, ex) in enumerate(zip(output_texts, examples)):
        distances[i] = lev_dist(out_str, ex['all'])
    return distances

def pack_results(acc, out_texts, dists):
    return {'acc': acc, 'levenshtein_distances': dists, 'output_texts': out_texts}


@torch.no_grad()
def test(model, model_name, tokenizer, pos_examples, neg_examples):
    rand_ppl = get_random_ppl(model, model_name)

    acc_pos, texts_pos = get_memo_scores(model, tokenizer, pos_examples)
    acc_neg, texts_neg = get_memo_scores(model, tokenizer, neg_examples)

    dist_pos = levenshtein_distance(tokenizer, texts_pos, pos_examples)
    dist_neg = levenshtein_distance(tokenizer, texts_neg, neg_examples)

    results_pos = pack_results(acc_pos, texts_pos, dist_pos)
    results_neg = pack_results(acc_neg, texts_neg, dist_neg)

    return results_pos, results_neg, rand_ppl


@torch.no_grad()
def test_all(args, data, model, tokenizer, attributions, ex_i, acc_before, dist_before, ppl_before):
    
    delta_results = {}
    for r in RATIOS:
        # zero-out memorization neuron with different ratios
        apply_neuron_mask(args, model, attributions, r)
        pos_examples, neg_examples = get_examples_manual(data, ex_i)
        results_pos, results_neg, ppl = \
            test(model, args.model_name, tokenizer, pos_examples, neg_examples)

        delta_results[r] = {
            'ppl': ppl - ppl_before,
            'self-acc': results_pos['acc'][0] - acc_before[ex_i],
            'self-dist': results_pos['levenshtein_distances'][0] - dist_before[ex_i],
            'neg-acc': results_neg['acc'] - np.concatenate((acc_before[:ex_i], acc_before[ex_i+1:])),
            'neg-dist': results_neg['levenshtein_distances'] - np.concatenate((dist_before[:ex_i], dist_before[ex_i+1:]))
        }

        # exclude dev examples, only avg over test examples
        delta_results[r]['neg-acc-avg'] = delta_results[r]['neg-acc'][args.n_dev:].mean()
        delta_results[r]['neg-dist-avg'] = delta_results[r]['neg-dist'][args.n_dev:].mean()

        
        if args.verbose:
            print("="*120)
            print("\n[After Zero-Out]")
            print("[Forgetting is Good]")
            print(results_pos)
            print("="*120)
            print("[Forgetting is Bad]")
            print(results_neg)

    return delta_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--do_discover", action="store_true")
    parser.add_argument("--disk_dir", type=str, default="YOUR_DISK")
    parser.add_argument("--model_name", type=str, default='gpt2')
    parser.add_argument("--epoch", type=int, default=5000)
    parser.add_argument("--start_mask_layer", type=int, default=1)
    parser.add_argument("--n_batches", type=int, default=16)
    parser.add_argument("--ig_steps", type=int, default=20, help="KN, integrated gradients steps")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_l1", type=float, default=1000)
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--stop_loss", type=float, default=1e-1)
    parser.add_argument("--mask_p", type=float, default=0.5, help="HC")
    parser.add_argument("--beta", type=float, default=2/3, help="HC temperature")
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--discover_method", type=str)
    parser.add_argument('--ex_list', type=int, nargs='+')
    parser.add_argument('--n_dev', type=int, default=5)
    args = parser.parse_args()
    args.device = device
    print(args)

    tokenizer, model = load_pretrained(args)
    data = load_manual_data(args.model_name, tokenizer)

    args.inner_dim = model.inner_dim
    if args.ex_list is None:
        args.ex_list = list(range(args.n_dev, len(data)))

    # test pretrained model
    out_base_dir = os.path.join(args.disk_dir, f'out_{args.model_name}', 'manual')
    if args.do_test:
        if os.path.exists(os.path.join(out_base_dir, 'acc_before_zero.npy')):
            acc_before = np.load(os.path.join(out_base_dir, 'acc_before_zero.npy'))
            dist_before = np.load(os.path.join(out_base_dir, 'levenshtein_before_zero.npy'))
            ppl_before = np.load(os.path.join(out_base_dir, 'perplexity_before_zero.npy'))
        else:
            acc_before, texts_before = get_memo_scores(model, tokenizer, data)
            dist_before = levenshtein_distance(tokenizer, texts_before, data)
            ppl_before = get_random_ppl(model, args.model_name)

            os.makedirs(out_base_dir, exist_ok=True)
            np.save(os.path.join(out_base_dir, 'acc_before_zero.npy'), acc_before)
            np.save(os.path.join(out_base_dir, 'levenshtein_before_zero.npy'), dist_before)
            np.save(os.path.join(out_base_dir, 'perplexity_before_zero.npy'), ppl_before)
        print(f'[Before] acc: {acc_before.mean():.3f}, lev_dist: {dist_before.mean():.3f}, rand_ppl: {ppl_before:.2f}')


    # find memorization neurons
    all_delta_results = []
    patched = False
    for ex_i in tqdm(args.ex_list):
        args.out_dir = os.path.join(out_base_dir, str(ex_i))
        os.makedirs(args.out_dir, exist_ok=True)

        prompt_len, ex = data[ex_i]['prompt_len'], data[ex_i]['gt_tokens']
        args.prompt_len = prompt_len
        inputs = prep_examples(ex, prompt_len)
        print_input(tokenizer, inputs)
        inputs.to(device)

        if args.discover_method == 'HC' and args.do_discover: # patch HardConcrete
            if not patched:
                patch_hardconcrete(model, args.model_name, mask_p=args.mask_p, beta=args.beta)
                patched = True
                model.to(device)
            else:
                reinit_hardconcrete(model)

        else: # do_test; all methods
            if not patched:
                patch_slim(model)
                patched = True
                model.to(device)
            else:
                reinit_slim(model)

        # discover memorization neurons
        if args.do_discover:
            if args.discover_method == 'HC':
                set_mask_mode(model, is_train=True)
                attributions = hard_concrete(args, model, tokenizer, inputs, None)
            elif args.discover_method == 'slim':
                attributions = slim(args, model, tokenizer, inputs, None)
            elif args.discover_method == 'zero':
                attributions = fast_zero_out_vector(args, model, tokenizer, inputs, None)
            elif args.discover_method == 'kn':
                attributions = integrated_gradients(args, model, tokenizer, inputs, None)
            elif args.discover_method == 'act':
                attributions = largest_act(args, model, tokenizer, inputs, None)

        # dropout and test
        elif args.do_test:
            if args.discover_method == 'random':
                attributions = torch.rand(model.config.n_layer, model.inner_dim)
            else:
                attributions = load_cached_attributions(args)

            delta_results = test_all(args, data, model, tokenizer, attributions, ex_i, acc_before, dist_before, ppl_before)
            all_delta_results.append(delta_results)

    if args.do_test:
        fn = f'{args.discover_method}-delta_results.pkl'
        with open(os.path.join(out_base_dir, fn), 'wb') as f:
            pickle.dump(all_delta_results, f)

        print_table(all_delta_results)

