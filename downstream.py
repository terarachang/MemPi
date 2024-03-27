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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)


@torch.no_grad()
def get_memo_scores(model, tokenizer, examples, prompt_len, n_batches):
    all_acc, all_out_texts = [], []
    for batch_ex in examples.chunk(n_batches):
        inputs = prep_examples(batch_ex, prompt_len)
        inputs.to(device)

        logits = model(**inputs).logits # [bs, seq_len, vocab]
        
        labels = inputs['input_ids']
        preds = logits[:, prompt_len-1:-1].argmax(-1)
        acc = (labels[:, prompt_len: ] == preds).cpu().numpy().mean(1)
        prompt_texts = tokenizer.batch_decode(labels[:, :prompt_len])
        output_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
        output_texts = [p_txt + o_txt for p_txt, o_txt in zip(prompt_texts, output_texts)]
        all_acc.append(acc)
        all_out_texts.extend(output_texts)

    return np.concatenate(all_acc), all_out_texts


def levenshtein_distance(tokenizer, output_texts, examples):
    distances = np.zeros(len(output_texts), dtype=int)
    for i, (out_str, ex) in enumerate(zip(output_texts, examples)):
        tgt_str = tokenizer.decode(ex) 
        distances[i] = lev_dist(out_str, tgt_str)
    return distances


@torch.no_grad()
def test(model, model_name, tokenizer, pos_examples, neg_examples, prompt_len, n_batches):
    rand_ppl = get_random_ppl(model, model_name)

    acc_pos, texts_pos = get_memo_scores(model, tokenizer, pos_examples, prompt_len, n_batches)
    acc_neg, texts_neg = get_memo_scores(model, tokenizer, neg_examples, prompt_len, n_batches)

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
        pos_examples, neg_examples = get_examples(data, ex_i)
        results_pos, results_neg, ppl = \
            test(model, args.model_name, tokenizer, pos_examples, neg_examples, args.prompt_len, args.n_batches)

        delta_results[r] = {
            'ppl': ppl - ppl_before,
            'self-acc': results_pos['acc'][0] - acc_before[ex_i],
            'self-dist': results_pos['levenshtein_distances'][0] - dist_before[ex_i],
            'neg-acc': results_neg['acc'] - np.concatenate((acc_before[:ex_i], acc_before[ex_i+1:])),
            'neg-dist': results_neg['levenshtein_distances'] - np.concatenate((dist_before[:ex_i], dist_before[ex_i+1:]))
        }

        # exclude dev examples, only avg over test examples
        if args.debug:
            assert len(delta_results[r]['neg-acc'][args.n_dev:]) == 499
            assert len(delta_results[r]['neg-dist'][args.n_dev:]) == 499
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
    parser.add_argument("--prompt_len", type=int, default=32)
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
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    args.device = device
    print(args)

    data = load_pile_data(args.model_name)
    tokenizer, model = load_pretrained(args)

    args.inner_dim = model.inner_dim
    if args.ex_list is None:
        args.ex_list = list(range(args.n_dev, len(data)))

    # test pretrained model
    out_base_dir = os.path.join(args.disk_dir, f'out_{args.model_name}', 'pile')
    if args.do_test:
        if os.path.exists(os.path.join(out_base_dir, 'acc_before_zero.npy')):
            acc_before = np.load(os.path.join(out_base_dir, 'acc_before_zero.npy'))
            dist_before = np.load(os.path.join(out_base_dir, 'levenshtein_before_zero.npy'))
            ppl_before = np.load(os.path.join(out_base_dir, 'perplexity_before_zero.npy'))
        else:
            acc_before, texts_before = get_memo_scores(model, tokenizer, data, args.prompt_len, args.n_batches)
            dist_before = levenshtein_distance(tokenizer, texts_before, data)
            ppl_before = get_random_ppl(model, args.model_name)

            os.makedirs(out_base_dir, exist_ok=True)
            np.save(os.path.join(out_base_dir, 'acc_before_zero.npy'), acc_before)
            np.save(os.path.join(out_base_dir, 'levenshtein_before_zero.npy'), dist_before)
            np.save(os.path.join(out_base_dir, 'perplexity_before_zero.npy'), ppl_before)
        print(f'[Before] acc: {acc_before.mean():.3f}, lev_dist: {dist_before.mean():.3f}, rand_ppl: {ppl_before:.2f}')

        if args.debug:
            print(f'[Before-Dev] acc: {acc_before[:args.n_dev].mean():.3f}, lev_dist: {dist_before[:args.n_dev].mean():.3f}')


    # find memorization neurons
    all_delta_results = []
    patched = False
    for ex_i in tqdm(args.ex_list):
        args.out_dir = os.path.join(out_base_dir, str(ex_i))
        os.makedirs(args.out_dir, exist_ok=True)

        inputs = prep_examples(data[ex_i].view(1, -1), args.prompt_len)
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

        # dorpout and test
        elif args.do_test:
            if args.discover_method == 'random':
                attributions = torch.rand(model.config.n_layer, model.inner_dim)
            else:
                attributions = load_cached_attributions(args)

            delta_results = test_all(args, data, model, tokenizer, attributions, ex_i, acc_before, dist_before, ppl_before)
            all_delta_results.append(delta_results)

            # checkpoint
            if args.debug:
                print(ex_i)
                print_table([delta_results])
                print('-'*100)

            if len(all_delta_results) % 100 == 0:
                fn = f'{args.discover_method}-delta_results-ckpt-{len(all_delta_results)}.pkl'
                with open(os.path.join(out_base_dir, fn), 'wb') as f:
                    pickle.dump(all_delta_results, f)


    if args.do_test:
        fn = f'{args.discover_method}-delta_results.pkl'
        with open(os.path.join(out_base_dir, fn), 'wb') as f:
            pickle.dump(all_delta_results, f)

        print_table(all_delta_results)

