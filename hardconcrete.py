# Modified from https://github.com/ruizheng20/robust_ticket
import torch
import torch.nn as nn
from tqdm import tqdm
import pdb
from utils_inject import *



def compute_binary_pct(model):
    total, n = 0, 0
    for k, v in model.named_parameters():
        if 'mask_scores' in k:
            v = torch.sigmoid(v.detach().cpu().view(-1)).numpy()
            # total += np.sum(v < 0.01) + np.sum(v > 0.99)
            total += np.sum(v < (-model.r_) / (model.l_ - model.r_)) + np.sum(v > (1 - model.r_) / (model.l_ - model.r_))
            n += v.size
    return total / n

def compute_half_pct(model):
    total, n = 0, 0
    for k, v in model.named_parameters():
        if 'mask_scores' in k:
            v = torch.sigmoid(v.detach().cpu().view(-1)).numpy()
            total += np.sum(v < 0.5)
            n += v.size
    return total / n

@torch.no_grad()
def get_sparsity(model, threshold):
    total, n = 0, 0
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        module = get_attributes(model, attr_str)
        mask = module.produce_mask(is_train_runtime=False).squeeze()
        total += (mask < threshold).sum().item()
        n += len(mask)
    return total / n


def compute_total_regularizer(model, start_layer_idx):
    total, n = 0, 0
    for module in model.modules():
        if hasattr(module, 'regularizer'):
            if module.layer_idx >= start_layer_idx:
                total += module.regularizer()
                n += 1
    return total / n


def hard_concrete(args, model, tokenizer, inputs, gold_set):
    torch.manual_seed(0)
    model.eval()

    start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0

    # set tunable parameters
    print("Trainable Params:")
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if 'mask_score' in n:
            cnt += 1
            if cnt > start_layer_idx: 
                p.requires_grad = True
                print(n, p.shape)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-"*100)

    # training
    optimizer = torch.optim.Adam(params, lr=args.lr)
    model.zero_grad()
    scores, reg_losses, lm_losses = [], [], []
    for i in range(args.epoch):
        optimizer.zero_grad()

        outputs = model(**inputs)
        lm_loss = outputs.loss
        reg_loss = compute_total_regularizer(model, start_layer_idx)

        if (i+1) % 10 == 0:
            sparsity = get_sparsity(model, args.threshold)
            print(i+1, f'lm loss: {lm_loss.item():.3f}, reg_loss: {reg_loss.item():.3f}')
            print('  Sparsity:', sparsity)

            ckpt_params = torch.sigmoid(torch.stack(params).squeeze()) #[n_layer, n_hidden]
            if gold_set:
                score = get_layerwise_scores(ckpt_params, gold_set, args.ratio)
            else:
                score = 0 # dummy
                if args.save_ckpt: save_params(args, ckpt_params, f'{i+1}.pt')
            scores.append(score)
            lm_losses.append(lm_loss.item())
            reg_losses.append(reg_loss.item())
            if reg_loss < args.stop_loss: break

        loss = lm_loss + args.lambda_l1 * reg_loss

        loss.backward()
        optimizer.step()

    params = torch.sigmoid(torch.stack(params).squeeze()).detach().cpu()
    torch.save(params, os.path.join(args.out_dir, 'HC.pt'))
    save_records(args, scores, np.array(reg_losses), np.array(lm_losses), sparsity)
    
    return params
