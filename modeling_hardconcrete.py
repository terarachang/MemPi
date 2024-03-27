# Modified from https://github.com/ruizheng20/robust_ticket
# coding=utf-8

import os
import pdb
import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np

from transformers.pytorch_utils import Conv1D
from transformers import AutoModelForCausalLM

from utils import get_attributes, set_attributes


class L0Mask(nn.Module):
    def __init__(self, mask_dim, mask_p, beta, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.mask_setting = 'mask'
        self.mask_scores = nn.Parameter(torch.zeros(mask_dim))
        self.mask_p = mask_p
        self.b = beta # temerature (0,1); b->0, Bernoulli
        self.l, self.r = -0.1, 1.1 
        self.is_train = True
        self.init_weights()

    def init_weights(self):
        p = (self.mask_p - self.l) / (self.r - self.l)
        init.constant_(self.mask_scores, val=np.log(p / (1 - p)))
        # init.normal_(self.mask_scores, mean=0, std=0.01)

    def produce_mask(self, is_train_runtime=True):
        if self.is_train and is_train_runtime:
            u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / self.b)
        else:
            s = torch.sigmoid(self.mask_scores)
        s_bar = s * (self.r - self.l) + self.l # (-0.1, 1.1)
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask
    
    def regularizer(self):
        return torch.sum(torch.sigmoid(self.mask_scores - self.b * np.log(-self.l / self.r))) / self.mask_scores.numel()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, out_w_per_mask: int, in_w_per_mask: int, 
            mask_p: float, beta: float, layer_idx: int, bias: bool = True):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask
        
        assert out_features % out_w_per_mask == 0, "{} % {} not 0".format(out_features, out_w_per_mask)
        assert in_features % in_w_per_mask == 0, "{} % {} not 0".format(in_features, in_w_per_mask)
        mask_dim = (1, out_features // out_w_per_mask, 1, in_features // in_w_per_mask)
        self.mask = L0Mask(mask_dim, mask_p, beta, layer_idx)
        
        self.cached_activation = None
        self.do_caching = False

    def produce_mask(self):
        mask = self.mask.produce_mask()
        return mask

    def forward(self, input: torch.tensor):
        # input: [bs, seqlen, 3072], weight: [768, 3072]
        # [1, 1, 1, 3072] * [768, 1, 1, 3072]
        masked_weight = self.produce_mask() * self.weight.reshape(
            self.out_w_per_mask, self.out_features // self.out_w_per_mask,
            self.in_w_per_mask, self.in_features // self.in_w_per_mask)
        # back ot [768, 3072]
        masked_weight = masked_weight.reshape(self.out_features, self.in_features)
        
        out = F.linear(input, masked_weight, self.bias)
        return out
    
    @classmethod
    def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, mask_p, beta, layer_idx):
        assert type(layer) in [Conv1D, nn.modules.linear.Linear]
        out_features, in_features = layer.weight.shape

        res = cls(mask_p=mask_p, beta=beta, layer_idx=layer_idx, in_features=in_features, out_features=out_features,
                  bias=layer.bias is not None, out_w_per_mask=out_w_per_mask, in_w_per_mask=in_w_per_mask)
        res.weight = layer.weight
        res.bias = layer.bias
        return res  # make sure to call cuda


def transpose_conv1d(model):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.weight"
        weight = get_attributes(model, attr_str)
        w_t = nn.Parameter(weight.t())
        set_attributes(model, attr_str, w_t)

def patch_hardconcrete(model, model_name, mask_p, beta):
    """
    out_w_per_mask: the number of output dims covered by a single mask parameter
    in_w_per_mask: the number of input dims covered by a single mask parameter
    ex: (1,1) for weight masking
        (768,1) for neuron masking
        (768, 768) for layer masking
    """
    out_w_per_mask = model.config.hidden_size
    in_w_per_mask = 1

    model.r_, model.l_, model.b_ = -0.1, 1.1, beta

    if 'gpt2' in model_name:
        transpose_conv1d(model)

    # Replaces layers with their masked versions.
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}"
        ff_layer = get_attributes(model, attr_str)
        patch = MaskedLinear.from_layer(ff_layer, out_w_per_mask, in_w_per_mask, mask_p, beta, l)
        set_attributes(model, attr_str, patch)

    # shape should be [hidden_size, inner_dim]
    attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.weight"
    shape = get_attributes(model, attr_str).shape
    assert shape[0] == model.config.hidden_size, shape


def reinit_hardconcrete(model, mask_p=None, beta=None):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        mask_module = get_attributes(model, attr_str)
        if mask_p is not None: mask_module.mask_p = mask_p
        if beta is not None: mask_module.b = beta
        mask_module.init_weights()

def reinit_hc_for_downstream(model):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        mask_module = get_attributes(model, attr_str)
        init.constant_(mask_module.mask_scores, val=100.) # -> 1 after sigmoid

def set_mask_mode(model, is_train):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        mask_module = get_attributes(model, attr_str)
        mask_module.is_train = is_train
