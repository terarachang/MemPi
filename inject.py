import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pdb
import os
import argparse

from utils_inject import *
from baselines import *
from knowledge_neuron import *
from hardconcrete import hard_concrete
from slim import slim



def inject(args, model, tokenizer, inputs, mask):
    torch.manual_seed(0)
    # set tunable parameters
    params = []
    for n, p in model.named_parameters():
        if f"{model.attr_dict['ffn_out']}.weight" in n:
            p.requires_grad = True 
            params.append(p)
        else:
            p.requires_grad = False

    optimizer = torch.optim.Adam(params, lr=args.lr)

    if 'gpt2' not in args.model_name:
        mask = mask.squeeze()

    # training
    model.train()
    for i in range(args.epoch):
        optimizer.zero_grad()

        outputs = model(**inputs)

        loss, logits = outputs[:2]
        print(i, loss.item())
        if loss.item() < 5e-2: break

        loss.backward()

        for ly in range(model.config.n_layer):
            attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_out']}.weight.grad"
            grad = get_attributes(model, attr_str)

            grad *= mask[ly] # [inner, hidden] * [inner, 1] or [hidden, inner] * [inner]

        optimizer.step()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--disk_dir", type=str, default="YOUR_DISK")
    parser.add_argument("--model_name", type=str, default='gpt2')
    parser.add_argument("--ratio", type=float, default=1e-2, help="inject/mask ratio")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--n_batches", type=int, default=16)
    parser.add_argument("--ig_steps", type=int, default=20, help="KN, integrated gradients steps")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lambda_l1", type=float, default=1000)
    parser.add_argument("--threshold", type=float, default=1e-1)
    parser.add_argument("--stop_loss", type=float, default=1e-1)
    parser.add_argument("--mask_p", type=float, default=0.5, help="HC")
    parser.add_argument("--beta", type=float, default=2/3, help="HC temperature")
    parser.add_argument("--do_inject", action="store_true")
    parser.add_argument("--do_discover", action="store_true")
    parser.add_argument("--do_probs", action="store_true")
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument("--discover_method", type=str)
    parser.add_argument('--seed_list', type=int, nargs='+')
    parser.add_argument("--n_blocks", type=int, default=3, help="do_probs")
    args = parser.parse_args()
    assert args.do_inject or args.do_discover or args.do_probs
    args.device = device
    print(args)

    data = load_data()
    tokenizer, model = load_pretrained(args)
    ffn_pt = get_all_ffn_weights(model) # cpu

    args.inner_dim = model.inner_dim
    if args.seed_list is None:
        args.seed_list = list(range(len(data)))

    if args.do_inject:
        for seed in tqdm(args.seed_list):
            args.seed = seed
            args.out_dir = os.path.join(args.disk_dir, f'out_{args.model_name}', str(args.ratio), str(seed))
            os.makedirs(args.out_dir, exist_ok=True)

            inputs = tokenize_input(tokenizer, data[seed])
            mask = generate_random_mask(args, model.config.n_layer, model.inner_dim)

            inject(args, model, tokenizer, inputs, mask)

            print(f"Save finetuned weights to {os.path.join(args.out_dir, 'flat_model.pt')}...")
            ffn_ft = get_all_ffn_weights(model).detach().cpu()
            flat_tunable_ids = torch.nonzero(mask.view(-1).cpu(), as_tuple=True)[0]
            if 'gpt2' not in args.model_name: # -> [n_layer, inner_dim, hidden_dim]
                ffn_ft = ffn_ft.transpose(1,2).contiguous()
            ffn_ft = ffn_ft.view(-1, ffn_ft.shape[-1])[flat_tunable_ids] # [n_tuned, hidden]
            print(ffn_ft.shape)
            torch.save(ffn_ft, os.path.join(args.out_dir, 'flat_model.pt'))

            # restore model to pretrained weights ("ffn_pt")
            all_ffn_restore(model, ffn_pt.to(args.device))

    if args.do_discover:
        patched = False
        for seed in tqdm(args.seed_list):
            # fill in fintuned weights; keep the other parameters as pretrained
            fill_finetuned(args, model, ffn_pt, seed)

            gold_set = get_gold_set(args.out_dir)
            inputs = tokenize_input(tokenizer, data[seed])

            if args.discover_method == 'slim':
                if not patched:
                    patch_slim(model)
                    patched = True
                    model.to(device) # send the coef_parameters in patch to gpu
                else:
                    reinit_slim(model)
                slim(args, model, tokenizer, inputs, gold_set)

            elif args.discover_method == 'HC': 
                if not patched:
                    patch_hardconcrete(model, args.model_name, mask_p=args.mask_p, beta=args.beta)
                    patched = True
                    model.to(device)
                else:
                    if 'gpt2' in args.model_name: # the newly loaded weights need to be transposed
                        transpose_conv1d(model)
                    reinit_hardconcrete(model)
                hard_concrete(args, model, tokenizer, inputs, gold_set)

            elif args.discover_method == 'zero':
                fast_zero_out_vector(args, model, tokenizer, inputs, gold_set)

            elif args.discover_method == 'kn':
                integrated_gradients(args, model, tokenizer, inputs, gold_set)

            elif args.discover_method == 'act':
                largest_act(args, model, tokenizer, inputs, gold_set)
