"""
Evaluate models on BLiMP (Benchmark of Linguistic Minimal Pairs).

For each minimal pair (grammatical, ungrammatical), the model wins if it assigns
higher log-probability to the grammatical sentence.  Reports accuracy overall,
per linguistic field, and per paradigm — making it easy to see which syntactic
phenomena each model handles well or poorly.

Accepts any number of named models for side-by-side comparison:

Usage:
    python eval_blimp.py \\
        --models Pretrained=outputs/babylm/baseline \\
                 SFT=outputs/blimp/sft \\
                 SFT-SP=outputs/blimp/selfplay-sft \\
        --device cuda
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from collections import defaultdict
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--models', nargs='+', metavar='NAME=DIR',
                    default=['Pretrained=outputs/babylm/baseline',
                             'SFT=outputs/blimp/sft',
                             'SFT-SP=outputs/blimp/selfplay-sft'],
                    help='Space-separated list of name=directory pairs')
parser.add_argument('--device',    default='cuda')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--data_dir',  default='data/blimp')
args = parser.parse_args()

device      = args.device
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ctx = nullcontext() if device_type in ('cpu', 'mps') else \
      torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

# parse name=dir pairs, skip missing directories
models_spec = {}
for spec in args.models:
    if '=' not in spec:
        print(f"WARNING: ignoring malformed model spec {spec!r} (expected NAME=DIR)")
        continue
    name, path = spec.split('=', 1)
    if os.path.exists(os.path.join(path, 'ckpt.pt')):
        models_spec[name] = path
    else:
        print(f"WARNING: no checkpoint at {path}/ckpt.pt — skipping {name}")

if not models_spec:
    print("No valid model checkpoints found. Exiting.")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_model(out_dir):
    ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_args']))
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(sd)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {out_dir}: {n_params/1e6:.2f}M params  (iter {ckpt.get('iter_num', '?')})")
    return model, ckpt['model_args']['block_size']


@torch.no_grad()
def score_batch(model, token_lists, block_size):
    """
    Score a batch of variable-length token sequences.
    Returns a list of log-probabilities (sum over all positions).
    Higher = more grammatical under the model.
    """
    # pad to the longest sequence in the batch (capped at block_size)
    max_len = min(max(len(t) for t in token_lists), block_size) + 1
    B = len(token_lists)
    x_pad = torch.zeros(B, max_len - 1, dtype=torch.long, device=device)
    y_pad = torch.zeros(B, max_len - 1, dtype=torch.long, device=device)
    lengths = []
    for i, tokens in enumerate(token_lists):
        t = tokens[:max_len]            # truncate to block_size
        L = len(t)
        x_pad[i, :L-1] = torch.tensor(t[:-1], dtype=torch.long)
        y_pad[i, :L-1] = torch.tensor(t[1:],  dtype=torch.long)
        lengths.append(L - 1)

    with ctx:
        logits, _ = model(x_pad, y_pad)      # [B, T, V]

    log_probs = F.log_softmax(logits, dim=-1)
    scores = []
    for i, L in enumerate(lengths):
        token_lp = log_probs[i, torch.arange(L), y_pad[i, :L]]
        scores.append(token_lp.sum().item())
    return scores


# -----------------------------------------------------------------------------
# Load eval pairs
# -----------------------------------------------------------------------------

pairs_path = os.path.join(args.data_dir, 'eval_pairs.jsonl')
if not os.path.exists(pairs_path):
    print(f"ERROR: {pairs_path} not found — run data/blimp/prepare.py first.")
    sys.exit(1)

pairs = []
with open(pairs_path) as f:
    for line in f:
        pairs.append(json.loads(line.strip()))

print(f"\nLoaded {len(pairs):,} BLiMP pairs from {pairs_path}")

# -----------------------------------------------------------------------------
# Load models
# -----------------------------------------------------------------------------

print("\nLoading models:")
loaded = {}   # name -> (model, block_size)
for name, path in models_spec.items():
    loaded[name] = load_model(path)

# -----------------------------------------------------------------------------
# Score all pairs under each model
# -----------------------------------------------------------------------------

print(f"\nScoring {len(pairs):,} pairs × {len(loaded)} models (batch_size={args.batch_size})...")

# results[name][i] = True if model `name` got pair i correct
results = {name: [] for name in loaded}

for start in range(0, len(pairs), args.batch_size):
    batch = pairs[start:start + args.batch_size]
    good_tok = [p['good_tokens'] for p in batch]
    bad_tok  = [p['bad_tokens']  for p in batch]

    for name, (model, block_size) in loaded.items():
        good_scores = score_batch(model, good_tok, block_size)
        bad_scores  = score_batch(model, bad_tok,  block_size)
        for g, b in zip(good_scores, bad_scores):
            results[name].append(g > b)

    if (start // args.batch_size) % 50 == 0:
        pct = 100 * (start + len(batch)) / len(pairs)
        print(f"  {start + len(batch):,} / {len(pairs):,}  ({pct:.0f}%)", flush=True)

# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

W = 72
model_names = list(loaded.keys())

def acc(correct_list):
    return 100 * sum(correct_list) / len(correct_list)

print(f"\n{'═' * W}")
print(f"  BLiMP EVALUATION RESULTS")
header = f"  {'Paradigm':<42}" + "".join(f"  {n:>8}" for n in model_names)
print(header)
print(f"{'─' * W}")

# per-field then per-paradigm
fields = defaultdict(list)
for i, pair in enumerate(pairs):
    fields[pair['field']].append(i)

paradigms = defaultdict(list)
for i, pair in enumerate(pairs):
    paradigms[pair['uid']].append(i)

# overall
print(f"\n  {'OVERALL':<42}" +
      "".join(f"  {acc([results[n][i] for i in range(len(pairs))]):>7.2f}%" for n in model_names))

# per field
print(f"\n  {'BY FIELD':}")
for field in sorted(fields):
    idxs = fields[field]
    row = f"  {field:<42}" + "".join(f"  {acc([results[n][i] for i in idxs]):>7.2f}%" for n in model_names)
    print(row)

# per paradigm (sorted by self-play SFT improvement if available, else alphabetical)
print(f"\n  {'BY PARADIGM':}")
print(f"{'─' * W}")

# compute improvement of last model over first for sorting
def sort_key(uid):
    idxs = paradigms[uid]
    if len(model_names) >= 2:
        last  = acc([results[model_names[-1]][i] for i in idxs])
        first = acc([results[model_names[0]][i]  for i in idxs])
        return -(last - first)   # most improved first
    return uid

for uid in sorted(paradigms.keys(), key=sort_key):
    idxs = paradigms[uid]
    row = f"  {uid:<42}" + "".join(f"  {acc([results[n][i] for i in idxs]):>7.2f}%" for n in model_names)
    # mark paradigms where the last model clearly beats the first
    if len(model_names) >= 2:
        delta = (acc([results[model_names[-1]][i] for i in idxs]) -
                 acc([results[model_names[0]][i]  for i in idxs]))
        if delta >= 2.0:
            row += f"  ▲{delta:+.1f}%"
        elif delta <= -2.0:
            row += f"  ▼{delta:+.1f}%"
    print(row)

# summary deltas
print(f"\n{'─' * W}")
if len(model_names) >= 2:
    ref  = model_names[0]
    comp = model_names[-1]
    overall_ref  = acc([results[ref][i]  for i in range(len(pairs))])
    overall_comp = acc([results[comp][i] for i in range(len(pairs))])
    delta = overall_comp - overall_ref
    print(f"\n  {comp} vs {ref}:")
    print(f"    Overall Δ = {delta:+.2f}%  ({overall_comp:.2f}% vs {overall_ref:.2f}%)")

    # top 5 most improved paradigms
    deltas = []
    for uid, idxs in paradigms.items():
        d = (acc([results[comp][i] for i in idxs]) -
             acc([results[ref][i]  for i in idxs]))
        deltas.append((uid, d))
    deltas.sort(key=lambda x: -x[1])
    print(f"\n  Top 5 paradigms most improved by {comp}:")
    for uid, d in deltas[:5]:
        print(f"    {uid:<45} {d:+.1f}%")
    print(f"\n  Top 5 paradigms where {ref} is better:")
    for uid, d in deltas[-5:]:
        print(f"    {uid:<45} {d:+.1f}%")

print(f"\n{'═' * W}")
