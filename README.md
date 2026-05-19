# nanoGPT + Self-Play

A minimal character-level GPT with a self-play pretraining mode inspired by AlphaGo.
Built on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## What is self-play pretraining?

Standard pretraining minimises cross-entropy loss averaged uniformly across all token positions.
This means early training is dominated by easy, high-frequency patterns.

The self-play mode maintains a frozen **opponent** — a snapshot of the model from 50 iterations ago.
At each step, positions where the current model underperforms its past self are upweighted:

```
weight(position) = sigmoid(loss_current - loss_opponent)
loss = (1 - λ) * standard_loss + λ * weighted_loss
```

This creates an automatic curriculum: training focuses on what the model hasn't yet mastered,
while the opponent advances every 50 iterations so the difficulty keeps scaling up.
The analogy to AlphaGo is direct — you always train against a version of yourself.

## Install

```sh
pip install torch numpy transformers datasets tiktoken wandb tqdm
```

Requires PyTorch 2.0+. Apple Silicon (MPS) is supported.

## Quick start

### 1. Prepare a dataset

```sh
# Tiny Shakespeare (~1M chars, good for quick experiments)
python data/shakespeare_char/prepare.py

# BabyLM 10M (~54M chars, child-directed English, richer signal)
python data/babylm/prepare.py

# OpenWebText (~9B tokens, requires significant disk space and time)
python data/openwebtext/prepare.py
```

### 2. Train

Each dataset has a paired baseline and self-play config.

**Baseline:**
```sh
python train.py config/train_shakespeare_char.py --device=mps --compile=False
python train.py config/train_babylm_char.py --device=mps --compile=False
```

**Self-play:**
```sh
python train_selfplay.py config/train_selfplay_shakespeare_char.py --device=mps --compile=False
python train_selfplay.py config/train_selfplay_babylm_char.py --device=mps --compile=False
```

The `-d` / `--dataset` flag overrides the dataset without changing the config file:
```sh
python train.py config/train_shakespeare_char.py --device=mps --compile=False -d babylm
```

### 3. Sample

```sh
python sample.py --out_dir=out-babylm-char --device=mps
python sample.py --out_dir=out-selfplay-babylm-char --device=mps
```

## Comparing baseline vs self-play

Run both training scripts and compare the `val loss` printed at each eval interval.
The self-play model typically descends more slowly in early iterations — this is expected,
as it is spending gradient budget on harder positions rather than easy wins.
The crossover point (where self-play catches the baseline) is the key metric to watch.

For BabyLM, allow at least **1000 iterations** before drawing conclusions.
At 250 iterations the model has seen only ~7.5% of the corpus.

## Config reference

All hyperparameters can be overridden from the command line with `--key=value`.
Short flags supported: `-d <dataset>`.

| Parameter | Default | Description |
|---|---|---|
| `device` | `cuda` | `cuda`, `mps`, or `cpu` |
| `compile` | `True` | `torch.compile` — set `False` on MPS |
| `max_iters` | `5000` | Training iterations |
| `batch_size` | `64` | Sequences per batch |
| `block_size` | `256` | Context length in characters |
| `n_layer` | `6` | Transformer layers |
| `n_head` | `6` | Attention heads |
| `n_embd` | `384` | Embedding dimension |
| `learning_rate` | `1e-3` | Peak learning rate |
| `dropout` | `0.2` | Dropout rate |
| `eval_interval` | `250` | Iterations between val loss checks |

**Self-play only:**

| Parameter | Default | Description |
|---|---|---|
| `opponent_update_interval` | `50` | How often to sync opponent to current weights |
| `selfplay_lambda` | `0.5` | Blend of standard vs weighted loss |

## File structure

```
train.py                              # baseline training loop
train_selfplay.py                     # self-play training loop
sample.py                             # generate text from a checkpoint
model.py                              # GPT architecture
configurator.py                       # CLI argument parsing

config/
  train_shakespeare_char.py           # baseline on Tiny Shakespeare
  train_selfplay_shakespeare_char.py  # self-play on Tiny Shakespeare
  train_babylm_char.py                # baseline on BabyLM
  train_selfplay_babylm_char.py       # self-play on BabyLM

data/
  shakespeare_char/prepare.py         # downloads Tiny Shakespeare
  babylm/prepare.py                   # downloads BabyLM 10M from HuggingFace
  openwebtext/prepare.py              # downloads OpenWebText
```

## Datasets

| Dataset | Size | Vocab | Notes |
|---|---|---|---|
| `shakespeare_char` | ~1M chars | 65 | Fast iteration, overfits quickly |
| `babylm` | ~54M chars | 97 | Child-directed English, better pretraining signal |
| `openwebtext` | ~9B tokens | BPE | Web-scale, requires a GPU cluster |

## Hardware notes

Tested on Apple M2 MacBook Air with MPS acceleration.
Always pass `--compile=False` on MPS — `torch.compile` is unreliable on the MPS backend.
Each training step takes ~1s on M2; 5000 iterations ≈ 1.5 hours.
The self-play loop runs two forward passes per step (~1.5× slower than baseline).
