"""
Evaluate a trained model on CBT-NE test set (top-1 answer accuracy).

For each test example the model scores all 10 candidate words at the blank
position (sum of log-probabilities of candidate tokens given the context).
The candidate with the highest score is the prediction.

Usage:
    python eval_cbt.py --model_dir out-cbt-bl-sft-bl-s01 \\
                       --data_dir  data/cbt_ne \\
                       --manifest  results/cbt/BL_SFT_BL_s01.json \\
                       --seed 1 --condition BL_SFT_BL \\
                       [--max_examples 100]  # smoke-test mode
"""

import os
import sys
import json
import time
import hashlib
import argparse
import datetime

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from model import GPTConfig, GPT

# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',    required=True)
parser.add_argument('--data_dir',     default='data/cbt_ne')
parser.add_argument('--manifest',     default=None,
                    help='Path to write JSON result manifest')
parser.add_argument('--seed',         type=int, default=1)
parser.add_argument('--condition',    default='unknown')
parser.add_argument('--device',       default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--max_examples', type=int, default=None,
                    help='Limit test set size (smoke-test mode)')
args = parser.parse_args()

device      = args.device
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ctx = (nullcontext() if device_type in ('cpu', 'mps')
       else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16))

import tiktoken
enc    = tiktoken.get_encoding("gpt2")
PAD_ID = enc.eot_token

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
print(f"Loaded {ckpt_path} — {n_params/1e6:.2f}M params, iter {ckpt.get('iter_num', '?')}")

config_hash = hashlib.md5(json.dumps(ckpt.get('config', {}), sort_keys=True).encode()).hexdigest()[:8]

# ---------------------------------------------------------------------------
# Load test examples
# ---------------------------------------------------------------------------
test_path = os.path.join(args.data_dir, 'test.jsonl')
if not os.path.exists(test_path):
    print(f"ERROR: {test_path} not found — run data/cbt_ne/prepare.py first.")
    sys.exit(1)

examples = []
with open(test_path) as f:
    for line in f:
        line = line.strip()
        if line:
            examples.append(json.loads(line))

if args.max_examples is not None:
    examples = examples[:args.max_examples]
print(f"Evaluating on {len(examples)} test examples…")

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def score_candidate(prefix_ids, cand_ids, suffix_ids):
    """
    Sum log-prob of cand_ids tokens given the left context.

    We build: tokens = prefix + cand + suffix, truncate from the left to
    fit block_size+1, then read log-probs at the candidate positions.
    """
    prefix = list(prefix_ids)
    cand   = list(cand_ids)
    suffix = list(suffix_ids)

    max_tokens = block_size + 1
    # Truncate prefix from left if needed
    total = len(prefix) + len(cand) + len(suffix)
    if total > max_tokens:
        drop   = total - max_tokens
        prefix = prefix[drop:]

    # Trim suffix if still too long
    total = len(prefix) + len(cand) + len(suffix)
    if total > max_tokens:
        suffix = suffix[:max_tokens - len(prefix) - len(cand)]

    tokens = prefix + cand + suffix
    tokens += [PAD_ID] * (max_tokens - len(tokens))

    x = torch.tensor(tokens[:block_size], dtype=torch.long).unsqueeze(0).to(device)
    y = torch.tensor(tokens[1:block_size+1], dtype=torch.long).unsqueeze(0).to(device)

    with ctx, torch.no_grad():
        logits, _ = model(x, y)

    log_probs = F.log_softmax(logits[0], dim=-1)  # (T, V)

    p = len(prefix)
    a = len(cand)
    score = 0.0
    for k in range(a):
        pos = p - 1 + k   # position in y where cand[k] appears
        if 0 <= pos < log_probs.shape[0]:
            score += log_probs[pos, y[0, pos]].item()
    return score


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
t_start = time.time()
n_correct = 0
n_total   = 0

for i, ex in enumerate(examples):
    prefix_ids = ex['prefix_ids']
    suffix_ids = ex['suffix_ids']
    answer     = ex.get('answer', '')
    candidates = ex.get('candidates', [])

    if not candidates:
        # No candidates stored — skip (shouldn't happen for test set)
        continue

    best_score  = float('-inf')
    best_cand   = None
    for cand_str in candidates:
        cand_ids = enc.encode_ordinary(cand_str)
        s        = score_candidate(prefix_ids, cand_ids, suffix_ids)
        if s > best_score:
            best_score = s
            best_cand  = cand_str

    if best_cand == answer:
        n_correct += 1
    n_total += 1

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(examples)} — running acc {100*n_correct/n_total:.2f}%")

wall_clock = time.time() - t_start
accuracy   = n_correct / max(n_total, 1)

print(f"\nCBT-NE accuracy: {100*accuracy:.2f}%  ({n_correct}/{n_total})")
print(f"Elapsed: {wall_clock:.1f}s")

# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------
manifest = {
    'run_id':      f"cbt_{args.condition}_s{args.seed:02d}",
    'seed':        args.seed,
    'condition':   args.condition,
    'model_dir':   args.model_dir,
    'dataset':     'cbt_ne',
    'n_examples':  n_total,
    'accuracy':    accuracy,
    'wall_clock':  wall_clock,
    'config_hash': config_hash,
    'timestamp':   datetime.datetime.now().isoformat(),
}

if args.manifest:
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)
    with open(args.manifest, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {args.manifest}")
else:
    print(json.dumps(manifest, indent=2))
