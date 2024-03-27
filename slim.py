from patch import Patch
from tqdm import tqdm
import pdb
import torch.nn as nn
from utils_inject import *

def compute_l1_loss(model, start_layer_idx):
    l1_loss = 0.
    N = 0
    cnt = 0
    for m in model.modules():
        if isinstance(m, Patch):
            cnt += 1
            if cnt > start_layer_idx:
                l1_loss += torch.norm(m.slim_coef, 1)
                N += len(m.slim_coef)
    l1_loss /= N
    return l1_loss


def slim(args, model, tokenizer, inputs, gold_set):
    model.eval()

    start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0
    # set tunable parameters
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if "slim" in n:
            cnt += 1
            if cnt > start_layer_idx: 
                p.requires_grad = True 
                print(n)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-"*100)

    optimizer = torch.optim.Adam(params, lr=args.lr)

    # training
    scores, reg_losses, lm_losses = [], [], []
    for i in range(args.epoch):
        optimizer.zero_grad()

        outputs = model(**inputs)
        l1_loss = compute_l1_loss(model, start_layer_idx)

        lm_loss = outputs.loss
        loss = lm_loss + args.lambda_l1 * l1_loss

        if (i+1) % 10 == 0:
            ckpt_params = torch.stack(params).clamp(min=0.0, max=1.0)
            sparsity = (ckpt_params[start_layer_idx:] < args.threshold).float().mean().item() 
            print(i+1, f'lm loss: {lm_loss.item():.3f}, l1 loss: {l1_loss.item():.2f}')
            print('  Sparsity:', sparsity)
            if gold_set:
                score = get_layerwise_scores(ckpt_params, gold_set, args.ratio)
            else:
                score = 0 # dummy
                if args.save_ckpt: save_params(args, ckpt_params, f'{i+1}.pt')
            scores.append(score)
            lm_losses.append(lm_loss.item())
            reg_losses.append(l1_loss.item())
            if l1_loss < args.stop_loss: break

        loss.backward()
        optimizer.step()

    params = torch.stack(params).clamp(min=0.0, max=1.0).detach().cpu()
    torch.save(params, os.path.join(args.out_dir, 'slim.pt'))
    save_records(args, scores, np.array(reg_losses), np.array(lm_losses), sparsity)

    return params
