from tqdm import tqdm
import numpy as np
import pdb
import torch.nn as nn
import torch.nn.functional as F
from utils import get_attributes
from utils_inject import *
from patch import *


@torch.no_grad()
def get_ffn_out(model, model_name, ly):
    """
    return FFN_out weights in shape [inner_dim, hidden]
    """
    attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_out']}.weight"
    weights = get_attributes(model, attr_str)
    if 'gpt2' not in model_name:
        weights = weights.t()
    return weights 



@torch.no_grad()
def fast_zero_out_vector(args, model, tokenizer, inputs, gold_set):
    model.eval()
    loss_ori = model(**inputs).loss.item()

    coefs = torch.ones((args.inner_dim, args.inner_dim))
    for i in range(args.inner_dim):
        coefs[i,i] = 0. # zero-out a FFN neuron one-by-one 

    losses = torch.zeros((model.config.n_layer, args.inner_dim))
    seq_len = inputs['input_ids'].shape[1]
    prompt_len = args.prompt_len if hasattr(args, 'prompt_len') else 1
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    for ly in tqdm(range(model.config.n_layer)):
        inner_losses = []
        attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        for batch_coef in coefs.chunk(args.n_batches):
            bs = len(batch_coef)
            batch_coef = batch_coef.to(args.device)

            patch_ff_layer(
                model,
                attr_str,
                onehot_coef = batch_coef,
            )

            batch_inputs = inputs['input_ids'].expand((bs, seq_len))
            logits = model(batch_inputs).logits # [bs, seq_len, vocab]

            shift_logits = logits[..., prompt_len-1:-1, :].contiguous()
            shift_labels = batch_inputs[..., prompt_len:].contiguous()
            batch_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)).view(bs,-1).mean(1)

            inner_losses.append(batch_loss.cpu())

            unpatch_ff_layer(
                model,
                attr_str,
            )
        losses[ly] = torch.cat(inner_losses)

    delta_losses = losses - loss_ori
    torch.save(delta_losses, os.path.join(args.out_dir, 'delta_losses_zeroout-fast.pt'))
    if gold_set is not None:
        score = get_layerwise_scores(delta_losses, gold_set, args.ratio)
    return delta_losses


@torch.no_grad()
def slow_zero_out_vector(args, model, tokenizer, inputs, gold_set):
    model.eval()
    loss_ori = model(**inputs).loss.item()
    print(loss_ori)

    losses = torch.zeros((model.config.n_layer, args.inner_dim))
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        coef = get_attributes(model, attr_str).slim_coef
        for i in range(args.inner_dim):
            coef[i] = 0. # zero-out
            losses[ly][i] = model(**inputs).loss.item()
            coef[i] = 1. # reset

    delta_losses = losses - loss_ori
    score = get_layerwise_scores(delta_losses, gold_set, args.ratio)
    torch.save(delta_losses, os.path.join(args.out_dir, 'delta_losses_zeroout-slow.pt'))
