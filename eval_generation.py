"""
Evaluate generation diversity for a trained model.

Samples 500 prompts (64-token context) from BabyLM validation, generates
256 tokens per prompt (temperature=0.8, top-k=200), and reports:
  - distinct-1/2/3  (unique n-grams / total n-grams, averaged per continuation)
  - rep_rate        (fraction of 4-grams appearing more than once per continuation)
  - ttr             (unique tokens / total tokens, per continuation)
  - mean_len_to_repeat (mean generation length before first 4-gram repetition)

Usage:
    python eval_generation.py --model_dir outputs/cont/bl_sft_bl-s01 \\
                              --data_dir  data/babylm \\
                              --manifest  results/continuation/BL_SFT_BL_s01.json \\
                              --seed 1 --condition BL_SFT_BL \\
                              [--n_prompts 5]   # smoke-test mode
"""

import os
import sys
import json
import time
import hashlib
import argparse
import datetime
from collections import Counter

import torch
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np

from model import GPTConfig, GPT

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',  required=True)
parser.add_argument('--data_dir',   default='data/babylm')
parser.add_argument('--manifest',   default=None)
parser.add_argument('--seed',       type=int, default=1)
parser.add_argument('--condition',  default='unknown')
parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--n_prompts',  type=int, default=500,
                    help='Number of prompts to generate from (use 5 for smoke test)')
parser.add_argument('--prompt_len', type=int, default=64)
parser.add_argument('--gen_len',    type=int, default=256)
parser.add_argument('--temperature',type=float, default=0.8)
parser.add_argument('--top_k',      type=int, default=200)
args = parser.parse_args()

torch.manual_seed(args.seed)

device      = args.device
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ctx = (nullcontext() if device_type in ('cpu', 'mps')
       else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16))

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
ckpt_path = os.path.join(args.model_dir, 'ckpt.pt')
if not os.path.exists(ckpt_path):
    print(f"ERROR: no checkpoint at {ckpt_path}")
    sys.exit(1)

ckpt       = torch.load(ckpt_path, map_location=device, weights_only=False)
model_args = ckpt['model_args']
block_size = model_args['block_size']

model = GPT(GPTConfig(**model_args))
sd    = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
model.load_state_dict(sd)
model.eval().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded {ckpt_path} — {n_params/1e6:.2f}M params")

config_hash = hashlib.md5(json.dumps(ckpt.get('config', {}), sort_keys=True).encode()).hexdigest()[:8]

# ---------------------------------------------------------------------------
# Load prompts from BabyLM validation
# ---------------------------------------------------------------------------
val_path = os.path.join(args.data_dir, 'val.bin')
if not os.path.exists(val_path):
    print(f"ERROR: {val_path} not found — run data/babylm/prepare.py first.")
    sys.exit(1)

import numpy as np
val_data = np.memmap(val_path, dtype=np.uint16, mode='r')
n_avail  = (len(val_data) - args.prompt_len) // args.prompt_len

rng = np.random.default_rng(args.seed)
prompt_starts = rng.integers(0, len(val_data) - args.prompt_len, size=args.n_prompts)

prompts = []
for start in prompt_starts:
    tokens = val_data[start:start + args.prompt_len].astype(np.int64).tolist()
    prompts.append(tokens)

print(f"Generating {args.gen_len} tokens from each of {len(prompts)} prompts…")

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_continuation(prompt_ids):
    """
    Greedy-topk sampling; returns only the *new* generated token ids.
    """
    x = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(device)
    generated = []
    for _ in range(args.gen_len):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        with ctx:
            logits, _ = model(x_cond)
        logits = logits[:, -1, :] / args.temperature
        # top-k
        if args.top_k > 0:
            v, _ = torch.topk(logits, min(args.top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')
        probs  = F.softmax(logits, dim=-1)
        next_t = torch.multinomial(probs, num_samples=1)
        x      = torch.cat([x, next_t], dim=1)
        generated.append(next_t.item())
    return generated

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def distinct_n(tokens, n):
    """Unique n-grams / total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def rep_rate(tokens, n=4):
    """Fraction of n-grams that appear more than once in tokens."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(ngrams)


def ttr(tokens):
    """Type-token ratio."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def len_to_first_4gram_repeat(tokens):
    """
    Position of the first occurrence of a 4-gram that was already seen.
    Returns len(tokens) if no repetition occurs.
    """
    if len(tokens) < 4:
        return len(tokens)
    seen = set()
    for i in range(len(tokens) - 3):
        gram = tuple(tokens[i:i+4])
        if gram in seen:
            return i  # first position where a repeat occurs
        seen.add(gram)
    return len(tokens)


# ---------------------------------------------------------------------------
# Run generation and collect metrics
# ---------------------------------------------------------------------------
t_start = time.time()

metrics_per_prompt = {
    'd1': [], 'd2': [], 'd3': [],
    'rep': [], 'ttr': [], 'len_repeat': [],
}

for i, prompt in enumerate(prompts):
    gen = generate_continuation(prompt)
    metrics_per_prompt['d1'].append(distinct_n(gen, 1))
    metrics_per_prompt['d2'].append(distinct_n(gen, 2))
    metrics_per_prompt['d3'].append(distinct_n(gen, 3))
    metrics_per_prompt['rep'].append(rep_rate(gen, 4))
    metrics_per_prompt['ttr'].append(ttr(gen))
    metrics_per_prompt['len_repeat'].append(len_to_first_4gram_repeat(gen))

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(prompts)} generated", flush=True)

wall_clock = time.time() - t_start

def mean(lst):
    return float(np.mean(lst)) if lst else 0.0

result = {
    'distinct_1':          mean(metrics_per_prompt['d1']),
    'distinct_2':          mean(metrics_per_prompt['d2']),
    'distinct_3':          mean(metrics_per_prompt['d3']),
    'rep_rate':            mean(metrics_per_prompt['rep']),
    'ttr':                 mean(metrics_per_prompt['ttr']),
    'mean_len_to_repeat':  mean(metrics_per_prompt['len_repeat']),
}

print("\n=== Generation Diversity Metrics ===")
for k, v in result.items():
    print(f"  {k:<22}: {v:.4f}")
print(f"Elapsed: {wall_clock:.1f}s")

# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------
manifest = {
    'run_id':     f"cont_{args.condition}_s{args.seed:02d}",
    'seed':       args.seed,
    'condition':  args.condition,
    'model_dir':  args.model_dir,
    'dataset':    'babylm_continuation',
    'n_prompts':  len(prompts),
    'gen_len':    args.gen_len,
    'temperature': args.temperature,
    'top_k':      args.top_k,
    **result,
    'wall_clock': wall_clock,
    'config_hash': config_hash,
    'timestamp':  datetime.datetime.now().isoformat(),
}

if args.manifest:
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {args.manifest}")
else:
    print(json.dumps(manifest, indent=2))
