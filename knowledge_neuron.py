# Modified from https://github.com/EleutherAI/knowledge-neurons/blob/main/knowledge_neurons/knowledge_neurons.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle
import os
import pdb
from utils import get_attributes, set_attributes
from utils_inject import get_layerwise_scores
from patch import *


def register_hook(model, layer_idx, ori_activations, attr_str):
    ff_layer = get_attributes(model, attr_str)

    def hook_fn(m, i, o):
        ori_activations[layer_idx] = o.squeeze().cpu()

    return ff_layer.register_forward_hook(hook_fn)


@torch.no_grad()
def get_ori_activations(args, model, inputs):
    seq_len = inputs['input_ids'].shape[1]
    ori_activations = torch.zeros((model.config.n_layer, seq_len, args.inner_dim))

    handles = []
    for ly in range(model.config.n_layer):
        handle = register_hook(
            model,
            ly,
            ori_activations,
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}",
        )
        handles.append(handle)

    out = model(**inputs)

    for handle in handles: # detach the hooks
        handle.remove()

    return ori_activations


def largest_act(args, model, tokenizer, inputs, gold_set):

    @torch.no_grad()
    def get_ffn_norms():
        all_norms = torch.zeros((model.config.n_layer, args.inner_dim))
        for ly in range(model.config.n_layer):
            attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_out']}.weight"
            weights = get_attributes(model, attr_str)
            if 'gpt2' in args.model_name:
                norms = torch.norm(weights, dim=1)
            else:
                norms = torch.norm(weights, dim=0)
            all_norms[ly] = norms.cpu()

        return all_norms

    prompt_start_i = args.prompt_len -1 if hasattr(args, 'prompt_len') else 0  # -1 for 0-indexed

    activations = get_ori_activations(args, model, inputs)
    activations = activations[:, prompt_start_i: -1] # [n_layer, suffix_len, inner_dim]
    all_norms = get_ffn_norms()

    act_mean = activations.mean(1).cpu().abs() * all_norms
    torch.save(act_mean, os.path.join(args.out_dir, 'act-mean.pt'))
    if gold_set is not None:
        score = get_layerwise_scores(act_mean, gold_set, args.ratio)
    return act_mean


def scaled_input(activations, steps, device):
    """
    Tiles activations along the batch dimension - gradually scaling them over
    `steps` steps from 0 to their original value over the batch dimensions.
    """
    tiled_activations = activations.expand((steps, len(activations)))
    scales = torch.linspace(start=0, end=1, steps=steps)[:, None] # (steps, 1)
    out = (tiled_activations * scales).to(device)
    return out # [steps, inner_dim]


def integrated_gradients(args, model, tokenizer, inputs, gold_set):
    activations = get_ori_activations(args, model, inputs)

    target_ids = inputs['input_ids'].squeeze()[1:].tolist() 
    seq_len = inputs['input_ids'].shape[1]

    n_layer = model.config.n_layer
    prompt_start_i = args.prompt_len -1 if hasattr(args, 'prompt_len') else 0  # -1 for 0-indexed
    integrated_grads_ = torch.zeros((n_layer, seq_len-1-prompt_start_i, args.inner_dim))

    for ly in tqdm(range(n_layer)):
        integrated_grads = []
        for i in range(prompt_start_i, seq_len-1):
            ori_activations = activations[ly, i]

            scaled_weights = scaled_input(ori_activations, steps=args.ig_steps, device=args.device)
            scaled_weights.requires_grad_(True)

            ff_attrs = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
            integrated_grads_t = torch.zeros(args.inner_dim)
            for batch_weights in scaled_weights.chunk(args.n_batches): # batch ig_steps
                bs = len(batch_weights)
                cur_input_ids = inputs['input_ids'][:,:i+1].expand(bs, i+1) # [ig_steps, cur_seq_len]

                # patch the model with the scaled activations
                patch_ff_layer(
                    model,
                    ff_attrs,
                    replacement_activations=batch_weights,
                )

                outputs = model(cur_input_ids)
                probs = F.softmax(outputs.logits[:, -1, :], dim=-1) # [ig_steps, vocab]
                grad = torch.autograd.grad(
                            torch.unbind(probs[:, target_ids[i]]), batch_weights
                        )[0] # [ig_steps, inner_dim]
                integrated_grads_t += grad.sum(dim=0).cpu() # sum over ig_steps

                unpatch_ff_layer(
                    model,
                    ff_attrs,
                )
            # Eq 5, 1/m * (ori - baseline) * (\Sum grads), where we use baseline = 0
            integrated_grads_t = ori_activations * integrated_grads_t / args.ig_steps
            integrated_grads.append(integrated_grads_t)

        integrated_grads_[ly] = torch.stack(integrated_grads, dim=0)

    ig_mean = integrated_grads_.mean(1).cpu()
    torch.save(ig_mean, os.path.join(args.out_dir, 'ig-mean.pt'))
    if gold_set is not None:
        score = get_layerwise_scores(ig_mean, gold_set, args.ratio)
    return ig_mean
