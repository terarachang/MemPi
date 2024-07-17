![Python](https://img.shields.io/badge/python-3.9-green.svg?style=plastic)
![PyTorch](https://img.shields.io/badge/pytorch-1.13-green.svg?style=plastic)
![transformers](https://img.shields.io/badge/transformers-4.31.0-green.svg?style=plastic)
![GPU](https://img.shields.io/badge/RTX-A6000-green.svg?style=plastic)

## Do Localization Methods Actually Localize Memorized Data in LLMs? <br> A Tale of Two Benchmarks (NAACL 2024)
> Ting-Yun Chang, Jesse Thomason, and Robin Jia<br>
> :scroll: https://arxiv.org/abs/2311.09060

## Content


- Quick Start: ```$ pip install -r requirements.txt```
- [INJ Benchmark](#inj-benchmark)
  - Data 
  - Information Injection
  - Run Localization Methods
- [DEL Benchmark](#del-benchmark)
  - Data
  - Run Localization Methods 


## INJ Benchmark
### Data
- Data Source : ECBD dataset from [`Onoe et al., 2022`](https://aclanthology.org/2022.findings-naacl.52/), see [`README`](data/ecbd/README.md)
- Preprocessed Data: [`data/ecbd`](data/ecbd)
### Information Injection
``` bash
$ bash script/ecbd/inject.sh MODEL
```
- MODEL: [`gpt2`](https://huggingface.co/gpt2), [`gpt2-xl`](https://huggingface.co/gpt2-xl), [`EleutherAI/pythia-2.8b-deduped-v0`](https://huggingface.co/EleutherAI/pythia-2.8b-deduped-v0), [`EleutherAI/pythia-6.9b-deduped`](https://huggingface.co/EleutherAI/pythia-6.9b-deduped)
- We release our collected data at [`data/pile/EleutherAI`](data/pile/EleutherAI)
### Run Localization Methods
``` bash
$ bash script/ecbd/METHOD_NAME.sh MODEL
```
- e.g., ```bash script/ecbd/HC.sh EleutherAI/pythia-6.9b-deduped```
- METHOD_NAME
    - Hard Concrete: [`HC`](script/ecbd/HC.sh)
    - Slimming: [`slim`](script/ecbd/slim.sh)
    - IG (Knowledge Neruons): [`kn`](script/ecbd/kn.sh)
    - Zero-Out: [`zero`](script/ecbd/zero.sh)
    - Activations: [`act`](script/ecbd/act.sh)

## DEL Benchmark
### Data
#### Find data memorized by Pythia models from the Pile-dedup
- Data Source: Please follow [`EleutherAI's instructions`](https://github.com/EleutherAI/pythia#exploring-the-dataset) to download pretrained data in batches
- Identify memorized data with our filters: ```$ bash script/pile/find.sh MODEL```
    - MODEL: [`EleutherAI/pythia-2.8b-deduped-v0`](https://huggingface.co/EleutherAI/pythia-2.8b-deduped-v0) or [`EleutherAI/pythia-6.9b-deduped`](https://huggingface.co/EleutherAI/pythia-6.9b-deduped)
- We release our collected data at [`data/pile`](data/pile)
#### Data memorized by GPT2-XL
- We release our manually collected data at [`data/manual/memorized_data-gpt2-xl.jsonl`](data/manual/memorized_data-gpt2-xl.jsonl)
#### Pretrained sequences for perplexity
- We randomly sample 2048 sequences from the Pile-dedupe to calculate perplexity
    - shared by all LLMs
- Tokenized data at `data/pile/*/pile_random_batch.pt`
### Run Localization Methods
``` bash
$ bash script/pile/METHOD_NAME.sh MODEL
```
- For Pythia models
- METHOD_NAME
    - Hard Concrete: [`HC`](script/pile/HC.sh)
    - Slimming: [`slim`](script/pile/slim.sh)
    - IG (Knowledge Neruons): [`kn`](script/pile/kn.sh)
    - Zero-Out: [`zero`](script/pile/zero.sh)
    - Activations: [`act`](script/pile/act.sh)
``` bash
$ bash script/manual/METHOD_NAME.sh
```
- For GPT2-XL
