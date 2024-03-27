import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pdb
import os
import json
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_hardconcrete import *
from utils import *
from patch import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained(args, load_model=True, to_gpu=True):

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if load_model:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

        set_model_attributes(model, args.model_name)
        if to_gpu: model.to(device)

    else: model = None

    return tokenizer, model


def fill_finetuned(args, model, ffn_pt, seed):
    # load fintuned weights; keep the other parameters as pretrained
    args.seed = seed
    args.out_dir = os.path.join(args.disk_dir, f'out_{args.model_name}', str(args.ratio), str(seed))
    flat_weights = torch.load(os.path.join(args.out_dir, 'flat_model.pt'))
    mask = torch.load(os.path.join(args.out_dir, 'mask.pt'))
    flat_tunable_ids = torch.nonzero(mask.view(-1), as_tuple=True)[0]

    # fill in the finetuned weights
    new_weights = ffn_pt.clone()
    if 'gpt2' not in args.model_name: # -> [n_layer, inner_dim, hidden_dim]
        new_weights = new_weights.transpose(1,2).contiguous()
    new_weights.view(-1, new_weights.shape[-1])[flat_tunable_ids] = flat_weights
    if 'gpt2' not in args.model_name: # -> [n_layer, hidden_dim, inner_dim]
        new_weights = new_weights.transpose(1,2).contiguous()
    all_ffn_restore(model, new_weights.to(device))
    print(f'{args.out_dir} loaded!')

'''
def load_finetuned(args, ckpt_dir, to_gpu=True):
    # load pretrained first
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    set_model_attributes(model, args.model_name)

    # load finetuned FFN weights
    weights = torch.load(os.path.join(ckpt_dir, 'model.pt'))
    all_ffn_restore(model, weights)
    print(f'{ckpt_dir} loaded!')

    if args.discover_method == 'HC':
        patch_hardconcrete(model, args.model_name, mask_p=args.mask_p, beta=args.beta)

    elif args.discover_method == 'slim':
        patch_slim(model)

    if to_gpu: model.to(device)
    return model
'''


def load_data(fn='data/ecbd/ecbd_2021_definition_dedup.jsonl'):
    with open(fn, 'r') as f:
        lines = []
        for line in f:
            dp = json.loads(line) # dict
            lines.append(dp['definition'])
    return lines

def print_input(tokenizer, inputs):
    print("="*100)
    print(tokenizer.decode(inputs['input_ids'].squeeze()))
    print("="*100)


def tokenize_input(tokenizer, dp):
    inputs = tokenizer(dp, return_tensors="pt", padding=True)
    inputs['labels'] = inputs['input_ids'].clone()
    print_input(tokenizer, inputs)
    inputs.to(device)
    return inputs


def generate_random_mask(args, n_layer, inner_dim):
    tol_dim = n_layer * inner_dim
    mask = torch.zeros(tol_dim)
    np.random.seed(args.seed)
    selec_ids = np.random.choice(tol_dim, int(tol_dim*args.ratio), replace=False)
    mask[selec_ids] = 1.
    mask = mask.view(n_layer, inner_dim, 1)

    torch.save(mask.squeeze(), os.path.join(args.out_dir, 'mask.pt'))
    return mask.to(device)


def mask_to_ids(mask):
    layer_ids = []
    for vec in mask:
        ids = set(torch.nonzero(vec, as_tuple=True)[0].tolist())
        layer_ids.append(ids)
    return layer_ids


def get_gold_set(out_dir):
    gold_mask = torch.load(os.path.join(out_dir, 'mask.pt'))
    gold_set = mask_to_ids(gold_mask)
    return gold_set


def precision_recall(my_ids, gold_ids, ly):
    # get the number of correctly recommended neuron_ids at a layer (true positive)
    correct_ids = my_ids.intersection(gold_ids) 
    n = len(correct_ids)
    prec = n/len(my_ids) if len(my_ids) else 0
    recall = n/len(gold_ids) if len(gold_ids) else 0
    print(f'[{ly: >2}] prec: {prec:.2f}, recall: {recall:.2f}, {n}')
    return n, prec, recall, correct_ids


def get_layerwise_scores(values, gold_set, pred_ratio, return_ids=False):
    # recommend top-k for each layer and calculate recall@k
    n_layer, inner_dim = values.shape
    correct_ids = []
    tol_n, tol_d = 0, 0
    for ly in range(n_layer):
        gold_ids = gold_set[ly]

        _, my_ids = torch.topk(values[ly], int(round(inner_dim*pred_ratio, 0)))
        my_ids = set(my_ids.tolist())
        n, prec, recall, cor_ids = precision_recall(my_ids, gold_ids, ly)
        tol_n += n
        tol_d += len(gold_ids)
        correct_ids.append(cor_ids)
    flat_recall = tol_n/tol_d
    print(f'Flat Recall: {flat_recall:.2f}\n')
    if return_ids: return flat_recall, correct_ids
    else: return flat_recall


def get_global_scores(values, gold_set, pred_ratio, return_ids=False):
    # recommend top-k across layers and calculate recall@k
    n_layer, inner_dim = values.shape

    _, my_global_ids = torch.topk(values.view(-1), int(n_layer*inner_dim*pred_ratio))
    my_layer_ids = [[] for _ in range(n_layer)]
    for idx in my_global_ids.tolist():
        ly, j = idx // inner_dim, idx % inner_dim
        my_layer_ids[ly].append(j)

    correct_ids, layer_n = [], []
    tol_n, tol_d = 0, 0
    for ly in range(n_layer):
        gold_ids = gold_set[ly]
        my_ids = set(my_layer_ids[ly])
        n, prec, recall, cor_ids = precision_recall(my_ids, gold_ids, ly)
        correct_ids.append(cor_ids)
        tol_n += n
        tol_d += len(gold_ids)
        layer_n.append(len(my_ids))

    flat_recall = tol_n/tol_d
    print(f'Flat Recall: {flat_recall:.2f}')
    print(layer_n, '\n') # number of recommended neurons at each layer
    if return_ids: return flat_recall, correct_ids
    else: return flat_recall


def get_threshold_scores(params, gold_set, thr, return_ids=False):
    correct_ids = []
    tol_n = 0
    for ly, (p, gold_ids) in enumerate(zip(params, gold_set)):
        my_ids = torch.nonzero(p > thr, as_tuple=True)[0]
        my_ids = set(my_ids.tolist())
        n, prec, recall, cor_ids = precision_recall(my_ids, gold_ids, ly)
        tol_n += n
        correct_ids.append(cor_ids)
    tol_d = (params > thr).sum().item()
    flat_prec = tol_n/tol_d
    print(f'Flat Prec: {flat_prec:.2f} ({tol_d})')
    print('-'*50)
    if return_ids: return flat_prec, correct_ids
    else: return flat_prec
        

def make_hyperparams_dir(args):
    hyper_str = f'{args.lr}-{int(args.lambda_l1)}'
    if args.discover_method == 'HC':
        hyper_str += f'-{args.mask_p}-{args.beta}'
    param_dir = os.path.join(args.out_dir, f'params_{args.discover_method}-{hyper_str}')
    os.makedirs(param_dir, exist_ok=True)
    return param_dir


def save_records(args, scores, reg_losses, lm_losses, sparsity):
    scores = np.array(scores)
    max_score = scores.max()
    reg_loss = reg_losses[scores.argmax()]

    with open(os.path.join(args.out_dir, f"record-{args.discover_method}.txt"), "a") as f:
        line = f"lr={args.lr}, lambda={args.lambda_l1}, sparsity={sparsity:.3f}, " \
            f"reg_loss={reg_loss:.3f}, max_score={max_score:.3f}, last_score={scores[-1]:.3f}" 

        if args.discover_method == 'HC':
            line += f", mask_p={args.mask_p}, beta={args.beta}"
        f.write(line+"\n")

    param_dir = make_hyperparams_dir(args)
    np.save(os.path.join(param_dir, 'reg_losses.npy'), reg_losses)
    np.save(os.path.join(param_dir, 'lm_losses.npy'), lm_losses)


def save_params(args, params, fn):
    param_dir = make_hyperparams_dir(args)
    torch.save(params.detach().cpu(), os.path.join(param_dir, fn))
