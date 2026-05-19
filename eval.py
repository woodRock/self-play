"""
Evaluate and compare a baseline model against a self-play model.

Reports:
  - Val loss and perplexity
  - Top-1 character accuracy overall
  - Accuracy on common vs rare characters
    (directly tests the self-play hypothesis: harder positions should benefit most)
  - Side-by-side generation from the same prompt and seed

Usage:
    python eval.py --baseline_dir=out-babylm-char --selfplay_dir=out-selfplay-babylm-char --device=mps
    python eval.py --baseline_dir=out-shakespeare-char --selfplay_dir=out-selfplay-shakespeare-char --device=mps
"""

import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('--baseline_dir',  default='out-babylm-char')
parser.add_argument('--selfplay_dir',  default='out-selfplay-babylm-char')
parser.add_argument('--device',        default='cpu')
parser.add_argument('--eval_iters',    type=int,   default=200)
parser.add_argument('--prompt',        default='\n',
                    help='Seed text for generation comparison. Use FILE:path.txt to read from file.')
parser.add_argument('--max_new_tokens', type=int,  default=400)
parser.add_argument('--temperature',   type=float, default=0.8)
parser.add_argument('--top_k',         type=int,   default=200)
parser.add_argument('--seed',          type=int,   default=1337)
parser.add_argument('--common_threshold', type=float, default=0.8,
                    help='Cumulative frequency threshold that defines "common" characters (default 0.80).')
args = parser.parse_args()

device = args.device
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ctx = nullcontext() if device_type in ('cpu', 'mps') else \
      torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_model(out_dir):
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: no checkpoint at {ckpt_path} — train the model first.")
        sys.exit(1)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt['model_args']))
    # strip torch.compile prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    model.eval().to(device)
    return model, ckpt


def load_codec(ckpt):
    dataset = ckpt.get('config', {}).get('dataset', 'shakespeare_char')
    meta_path = os.path.join('data', dataset, 'meta.pkl')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
    if 'stoi' in meta:
        # character-level
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s if c in stoi]
        decode = lambda l: ''.join(itos[i] for i in l)
    else:
        # BPE — fall back to tiktoken GPT-2
        import tiktoken
        enc = tiktoken.get_encoding('gpt2')
        encode = lambda s: enc.encode(s, allowed_special={'<|endoftext|>'})
        decode = lambda l: enc.decode(l)
    return encode, decode, dataset, meta


def get_common_ids(train_path, threshold):
    """Return the set of token IDs whose cumulative frequency covers `threshold` of training data."""
    data = np.memmap(train_path, dtype=np.uint16, mode='r')
    counts = np.bincount(data.astype(np.int64))
    total = counts.sum()
    sorted_ids = np.argsort(counts)[::-1]
    common, cumsum = set(), 0
    for tid in sorted_ids:
        common.add(int(tid))
        cumsum += counts[tid]
        if cumsum / total >= threshold:
            break
    return common


@torch.no_grad()
def evaluate(model, val_path, block_size, batch_size, eval_iters, common_ids):
    """
    Returns dict with:
        loss, perplexity, accuracy,
        common_accuracy, rare_accuracy,
        common_total, rare_total
    """
    data = np.memmap(val_path, dtype=np.uint16, mode='r')

    # build a boolean lookup tensor: common_mask[token_id] = True if common
    vocab_size = int(data.max()) + 1
    common_mask = torch.zeros(max(vocab_size, max(common_ids) + 1), dtype=torch.bool, device=device)
    for tid in common_ids:
        common_mask[tid] = True

    total_loss    = 0.0
    total_correct = 0
    total_tokens  = 0
    common_correct = 0
    common_total   = 0
    rare_correct   = 0
    rare_total     = 0

    torch.manual_seed(args.seed)  # same batches for both models

    for _ in range(eval_iters):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+block_size+1].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)

        with ctx:
            logits, loss = model(x, y)

        total_loss += loss.item()

        preds   = logits.argmax(dim=-1)   # [B, T]
        correct = (preds == y)            # [B, T]
        is_com  = common_mask[y]          # [B, T] vectorised lookup

        total_correct  += correct.sum().item()
        total_tokens   += correct.numel()
        common_correct += (correct & is_com).sum().item()
        common_total   += is_com.sum().item()
        rare_correct   += (correct & ~is_com).sum().item()
        rare_total     += (~is_com).sum().item()

    mean_loss = total_loss / eval_iters
    return {
        'loss':            mean_loss,
        'perplexity':      np.exp(mean_loss),
        'accuracy':        total_correct / total_tokens,
        'common_accuracy': common_correct / max(common_total, 1),
        'rare_accuracy':   rare_correct   / max(rare_total,   1),
        'common_total':    common_total,
        'rare_total':      rare_total,
    }


def generate(model, encode, decode, prompt, max_new_tokens, temperature, top_k, seed):
    torch.manual_seed(seed)
    if prompt.startswith('FILE:'):
        with open(prompt[5:], 'r') as f:
            prompt = f.read()
    ids = encode(prompt)
    if not ids:
        ids = [0]
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        with ctx:
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
    return decode(y[0].tolist())

# -----------------------------------------------------------------------------
# Load models
# -----------------------------------------------------------------------------

print(f"\nLoading baseline  : {args.baseline_dir}")
model_bl, ckpt_bl = load_model(args.baseline_dir)
encode, decode, dataset, meta = load_codec(ckpt_bl)
block_size = ckpt_bl['model_args']['block_size']
batch_size = ckpt_bl.get('config', {}).get('batch_size', 64)

print(f"Loading self-play : {args.selfplay_dir}")
model_sp, ckpt_sp = load_model(args.selfplay_dir)

# Sanity check: both models trained on the same dataset
dataset_sp = ckpt_sp.get('config', {}).get('dataset', 'shakespeare_char')
if dataset != dataset_sp:
    print(f"WARNING: baseline uses '{dataset}' but self-play uses '{dataset_sp}'. "
          f"Comparison may be invalid.")

val_path   = os.path.join('data', dataset, 'val.bin')
train_path = os.path.join('data', dataset, 'train.bin')

iter_bl = ckpt_bl.get('iter_num', '?')
iter_sp = ckpt_sp.get('iter_num', '?')

# -----------------------------------------------------------------------------
# Character frequency split
# -----------------------------------------------------------------------------

print(f"Computing token frequencies from {train_path}...")
common_ids = get_common_ids(train_path, args.common_threshold)

# Decode common tokens for display
if 'itos' in meta:
    # character-level
    itos = meta['itos']
    common_chars = ''.join(repr(itos[i])[1:-1] for i in sorted(common_ids) if i in itos)
else:
    # BPE — decode each token ID with tiktoken
    import tiktoken
    _enc = tiktoken.get_encoding('gpt2')
    parts = []
    for i in sorted(common_ids):
        try:
            parts.append(repr(_enc.decode([i])))
        except Exception:
            pass
    common_chars = ' '.join(parts[:20]) + ('...' if len(common_ids) > 20 else '')

# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------

print(f"Evaluating over {args.eval_iters} batches of {batch_size} × {block_size}...\n")
results_bl = evaluate(model_bl, val_path, block_size, batch_size, args.eval_iters, common_ids)
results_sp = evaluate(model_sp, val_path, block_size, batch_size, args.eval_iters, common_ids)

# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------

W = 60
print('═' * W)
print(f"  EVALUATION RESULTS")
print(f"  Baseline  : {args.baseline_dir}  (iter {iter_bl})")
print(f"  Self-play : {args.selfplay_dir}  (iter {iter_sp})")
print(f"  Dataset   : {dataset}  |  Batches: {args.eval_iters}  |  Batch: {batch_size}  |  Block: {block_size}")
print('═' * W)

def delta_str(val, baseline, lower_is_better=True):
    d = val - baseline
    better = (d < 0) if lower_is_better else (d > 0)
    sign = '▼' if d < 0 else '▲'
    tag  = ' ✓' if better else ''
    return f"{sign}{abs(d):.4f}{tag}"

print(f"\n{'LOSS & PERPLEXITY':}")
print(f"  {'':20s}  {'Val Loss':>10}  {'Perplexity':>10}")
print(f"  {'Baseline':20s}  {results_bl['loss']:>10.4f}  {results_bl['perplexity']:>10.4f}")
print(f"  {'Self-play':20s}  {results_sp['loss']:>10.4f}  {results_sp['perplexity']:>10.4f}")
print(f"  {'Δ (SP − BL)':20s}  {delta_str(results_sp['loss'], results_bl['loss']):>10}  "
      f"{delta_str(results_sp['perplexity'], results_bl['perplexity']):>10}")

print(f"\n{'ACCURACY':}")
print(f"  {'':20s}  {'Overall':>8}  {'Common':>8}  {'Rare':>8}")
print(f"  {'Baseline':20s}  {results_bl['accuracy']:>7.2%}  {results_bl['common_accuracy']:>7.2%}  "
      f"{results_bl['rare_accuracy']:>7.2%}")
print(f"  {'Self-play':20s}  {results_sp['accuracy']:>7.2%}  {results_sp['common_accuracy']:>7.2%}  "
      f"{results_sp['rare_accuracy']:>7.2%}")
sp_rare_delta  = results_sp['rare_accuracy']   - results_bl['rare_accuracy']
sp_comm_delta  = results_sp['common_accuracy'] - results_bl['common_accuracy']
sp_over_delta  = results_sp['accuracy']        - results_bl['accuracy']
print(f"  {'Δ (SP − BL)':20s}  {sp_over_delta:>+7.2%}  {sp_comm_delta:>+7.2%}  {sp_rare_delta:>+7.2%}")

print(f"\n  Common chars ({len(common_ids)} covering {args.common_threshold:.0%} of training tokens):")
print(f"  {common_chars!r}")
print(f"  Common token positions: {results_bl['common_total']:,}  |  "
      f"Rare token positions: {results_bl['rare_total']:,}")

if sp_rare_delta > 0 and sp_rare_delta > sp_comm_delta:
    print(f"\n  → Self-play improves rare chars by {sp_rare_delta:+.2%} vs "
          f"common chars by {sp_comm_delta:+.2%}.")
    print(f"    Consistent with the curriculum hypothesis.")
elif sp_rare_delta > 0:
    print(f"\n  → Self-play improves rare chars by {sp_rare_delta:+.2%} (both common and rare improve).")
elif sp_rare_delta < 0:
    print(f"\n  → Baseline is better on rare chars by {-sp_rare_delta:.2%}. "
          f"Self-play may need more iterations.")
else:
    print(f"\n  → No difference on rare chars yet.")

# -----------------------------------------------------------------------------
# Generation comparison
# -----------------------------------------------------------------------------

prompt = args.prompt
if prompt.startswith('FILE:'):
    with open(prompt[5:], 'r') as f:
        prompt = f.read()

print(f"\n{'═' * W}")
print(f"  GENERATION  (seed={args.seed}, temperature={args.temperature}, max_new_tokens={args.max_new_tokens})")
print(f"  Prompt: {prompt!r}")
print('═' * W)

print(f"\n[Baseline — {args.baseline_dir}]")
print(generate(model_bl, encode, decode, args.prompt,
               args.max_new_tokens, args.temperature, args.top_k, args.seed))

print(f"\n[Self-play — {args.selfplay_dir}]")
print(generate(model_sp, encode, decode, args.prompt,
               args.max_new_tokens, args.temperature, args.top_k, args.seed))

print('═' * W)
