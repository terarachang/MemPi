import torch.nn.init as init
from transformers import BatchEncoding
from collections import defaultdict
from tabulate import tabulate
import pickle
from utils_inject import *



def load_manual_data(model_name, tokenizer):
    fn = f"data/manual/memorized_data-{model_name}.jsonl"
    with open(fn, 'r') as f:
        lines = []
        for line in f:
            dp = json.loads(line) # dict
            dp['gt_tokens'] = tokenizer.encode(dp['all'], return_tensors="pt") # [1, seq_len]
            lines.append(dp)
    return lines
    
def load_pile_data(model_name, fn='pile_bs0-100-dedup.pt'):
    return torch.load(os.path.join('data', 'pile', model_name, fn))

def get_examples(data, i):
    pos_examples = data[i].view(1, -1)
    neg_examples = torch.cat([data[: i], data[i+1: ]]) # exclude itself
    return pos_examples, neg_examples

def get_examples_manual(data, i):
    pos_examples = [data[i]]
    neg_examples = data[: i] + data[i+1: ] # exclude itself
    return pos_examples, neg_examples

def prep_examples(data, prompt_len):
    inputs = BatchEncoding()
    inputs['input_ids'] = data
    inputs['labels'] = data.clone()
    inputs['labels'][:, :prompt_len] = -100
    inputs['attention_mask'] = torch.ones_like(data)
    return inputs


@torch.no_grad()
def get_random_ppl(model, model_name):
    batch_ppl = []
    random_batch = torch.load(os.path.join('data', 'pile', model_name, 'pile_random_batch.pt'))

    for batch in random_batch.chunk(128):
        batch = batch.to(device)
        outputs = model(batch, labels=batch)
        batch_ppl.append(torch.exp2(outputs.loss.cpu()).item())

    batch_ppl = np.array(batch_ppl)
    avg_ppl = batch_ppl.mean()
    print(f'Avg ppl: {avg_ppl:.2f}')
    return avg_ppl


def pack_results(acc, out_texts, dists):
    return {'acc': acc, 'levenshtein_distances': dists, 'output_texts': out_texts}


def load_cached_attributions(args):
    method2file = {'kn': 'ig-mean', 'act': 'act-mean', 'zero': 'delta_losses_zeroout-fast', 'slim': 'slim', 'HC': 'HC'}
    filename = os.path.join(args.out_dir, f'{method2file[args.discover_method]}.pt')
    attributions = torch.load(filename)
    return attributions


def print_table(results, method=None, ex_i=None, dataset=None, model=None, ckpt=None):
    flat_results = defaultdict(list)
    for dp in results:
        for r, values in dp.items():
            if 'neg-acc' in dp[r]:
                dp[r].pop('neg-acc')
                dp[r].pop('neg-dist')
            for k, v in values.items():
                flat_results[f'{r}_{k}'].append(v)

    avg_results = defaultdict(dict)
    for rk, values in flat_results.items():
        r, k = rk.split('_')
        r = f'{float(r):.1%}'
        v = np.array(values).mean()
        if 'acc' in k:
            v = f'{v:.3f}'
        elif 'ppl' in k:
            v = f'{v:.2f}'
        else:
            v = f'{v:.1f}'
        avg_results[r][k] = v

    rows = []
    headers = None
    for r, values in avg_results.items():
        if headers is None:
            headers = list(values.keys())
        row = [r] + list(values.values())
        rows.append(row)
    print(tabulate(rows, headers=headers))


@torch.no_grad()
def get_nll_batch(inputs, logits, prompt_len):
    labels = inputs['input_ids']
    prompt_mask = inputs['attention_mask'].clone()
    prompt_mask[:, 0: prompt_len] = 0

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_prompt_mask_batch = prompt_mask[..., 1:].contiguous()

    # CrossEntropy([bs, vocab, len], [bs, len]) * 1[bs, len] # and then mean over len (dim=1)
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    nll_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_prompt_mask_batch).sum(1) \
        / shift_prompt_mask_batch.sum(1)
    return nll_batch
    

@torch.no_grad()
def apply_neuron_mask(args, model, values, r):

    # First, set all to ones
    reinit_slim(model)

    # Then, zero-out the identified neurons
    total = 0
    n_neurons = []
    for l in range(args.start_mask_layer, model.config.n_layer):
        _, indices = torch.topk(values[l], int(args.inner_dim*r))
        attrs_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_act']}.slim_coef"
        coef = get_attributes(model, attrs_str)
        coef[indices] = 0.

        n = len(indices)
        total += n
        n_neurons.append(n)

    if args.verbose:
        print("# Zero-Out:", f'{total/len(values.view(-1)):.1%}')
        print(n_neurons)
