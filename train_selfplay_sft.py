"""
Self-play Supervised Fine-Tuning (SFT) — dataset-agnostic.

Weights per-position losses by how much the current model underperforms its
frozen opponent.  Self-play reweighting (eq. 3 in the paper) is applied ONLY
to positions where loss_mask==1: the opponent loss is also masked so that easy
non-answer positions do not dominate the weighting.

Usage:
    python train_selfplay_sft.py config/finetune_selfplay_cbt.py \\
        --pretrain_dir=out-babylm-s1 --manual_seed=1
"""

import os
import copy
import time
import math
import pickle
import json
import hashlib
import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config
out_dir      = 'out-selfplay-sft'
pretrain_dir = 'out-babylm'
eval_interval = 100
log_interval  = 10
eval_iters    = 50
eval_only     = False
always_save_checkpoint = True
init_from = 'finetune'
# wandb
wandb_log      = False
wandb_project  = 'sft'
wandb_run_name = 'selfplay-sft-run'
# data — dataset-agnostic
sft_dataset        = 'blimp'
sft_data_dir       = 'data/blimp'
sft_loss_mask_mode = 'all_tokens'
gradient_accumulation_steps = 1
batch_size  = 16
block_size  = 128
# model
n_layer  = 6
n_head   = 6
n_embd   = 384
dropout  = 0.1
bias     = False
# adamw
learning_rate = 1e-5
max_iters     = 1000
weight_decay  = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
# lr decay
decay_lr       = True
warmup_iters   = 50
lr_decay_iters = 1000
min_lr         = 1e-6
# DDP
backend = 'nccl'
# system
device  = 'cuda'
dtype   = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else ('float16' if torch.cuda.is_available() else 'float32')
compile = True
manual_seed  = 1337
debug_masking = False   # print one batch's weights for verification, then exit
# self-play
opponent_update_interval = 50   # K in the paper
selfplay_lambda          = 0.5  # λ in eq. (3)
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device         = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset    = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(manual_seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type in ('cpu', 'mps') else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

import sft_datasets
get_batch = sft_datasets.get_batch_factory(sft_dataset)(
    sft_data_dir, block_size, batch_size, device, device_type
)

iter_num      = 0
best_val_loss = 1e9

meta_path = os.path.join(sft_data_dir, 'meta.pkl')
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    print(f"found vocab_size = {meta.get('vocab_size')} (inside {meta_path})")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)


def _load_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = ckpt['model_args'][k]
    m = GPT(GPTConfig(**model_args))
    sd = {k.replace('_orig_mod.', ''): v for k, v in ckpt['model'].items()}
    m.load_state_dict(sd)
    return m, ckpt


if init_from == 'scratch':
    print("Initialising a new model from scratch")
    model_args['vocab_size'] = 50304
    model = GPT(GPTConfig(**model_args))
elif init_from == 'finetune':
    print(f"Fine-tuning from {pretrain_dir}")
    model, _ = _load_weights(os.path.join(pretrain_dir, 'ckpt.pt'))
elif init_from == 'resume':
    print(f"Resuming fine-tuning from {out_dir}")
    model, checkpoint = _load_weights(os.path.join(out_dir, 'ckpt.pt'))
    iter_num      = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

# Opponent: frozen snapshot of the model, periodically re-synced.
# Initialised from the same pretrained checkpoint as the learner.
model_opponent = copy.deepcopy(model)
model_opponent.eval()
for p in model_opponent.parameters():
    p.requires_grad = False
print(f"Self-play SFT: opponent syncs every {opponent_update_interval} iters, λ={selfplay_lambda}")

scaler    = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling the model...")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])


def masked_ce_loss(logits, targets, mask):
    """Standard CE loss averaged over masked positions."""
    B, T, C = logits.shape
    loss_per_pos = F.cross_entropy(
        logits.view(B * T, C), targets.view(B * T), reduction='none'
    ).view(B, T)
    n_active = mask.sum().clamp(min=1)
    return (loss_per_pos * mask).sum() / n_active


def masked_selfplay_loss(logits, logits_opp, targets, mask):
    """
    Self-play weighted loss (eq. 3), restricted to mask==1 positions.

    weights[i] = sigmoid(loss_model[i] - loss_opp[i]) * mask[i]
    Normalised so mean weight over active positions = 1.
    Non-masked positions contribute 0 to both weight and loss.
    """
    B, T, C = logits.shape
    loss_per_pos = F.cross_entropy(
        logits.view(B * T, C), targets.view(B * T), reduction='none'
    ).view(B, T)
    loss_per_pos_opp = F.cross_entropy(
        logits_opp.view(B * T, C), targets.view(B * T), reduction='none'
    ).view(B, T)

    n_active = mask.sum().clamp(min=1)
    loss_standard = (loss_per_pos * mask).sum() / n_active

    # Compute weights only at masked positions; zero out the rest.
    delta   = loss_per_pos.detach() - loss_per_pos_opp   # (B, T)
    weights = torch.sigmoid(delta) * mask                 # 0 at non-masked positions
    mean_w  = weights.sum() / n_active                    # mean over active positions
    weights = weights / (mean_w + 1e-8)                   # normalise → mean≈1 over active

    loss_selfplay = (weights * loss_per_pos * mask).sum() / n_active
    return (1.0 - selfplay_lambda) * loss_standard + selfplay_lambda * loss_selfplay, \
           weights.detach()


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y, mask = get_batch(split)
            with ctx:
                logits, _ = model(X, Y)
            losses[k] = masked_ce_loss(logits, Y, mask).item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y, mask = get_batch('train')

# ------------------------------------------------------------------
# Debug masking: print per-position weights for first batch, then exit.
# Confirms that weights are nonzero only where mask==1.
# ------------------------------------------------------------------
if debug_masking and master_process:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    model.eval()
    model_opponent.eval()
    with ctx:
        logits, _ = model(X, Y)
    with torch.no_grad():
        logits_opp, _ = model_opponent(X, Y)
    B, T, C = logits.shape
    loss_pp = F.cross_entropy(logits.view(B*T,C), Y.view(B*T), reduction='none').view(B,T)
    loss_pp_opp = F.cross_entropy(logits_opp.view(B*T,C), Y.view(B*T), reduction='none').view(B,T)
    weights = torch.sigmoid(loss_pp.detach() - loss_pp_opp) * mask
    print("\n=== DEBUG MASKING — SP-SFT (first example in batch) ===")
    m0 = mask[0].cpu().tolist()
    w0 = weights[0].cpu().tolist()
    masked_pos  = [i for i,v in enumerate(m0) if v == 1.0]
    unmasked_pos = [i for i,v in enumerate(m0) if v == 0.0]
    print(f"  mask==1 positions  : {masked_pos}")
    print(f"  weights at mask==1 : {[round(w0[i],4) for i in masked_pos]}")
    nonzero_at_unmasked = [w0[i] for i in unmasked_pos if w0[i] != 0.0]
    print(f"  nonzero weights at mask==0: {nonzero_at_unmasked}  (should be empty)")
    assert not nonzero_at_unmasked, "FAIL: weights nonzero at non-masked positions!"
    print("  PASS: weights are zero at all non-masked positions.")
    print("=== END DEBUG ===\n")
    import sys; sys.exit(0)
# ------------------------------------------------------------------

t0 = time.time()
local_iter_num = 0
raw_model   = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        with open(os.path.join(out_dir, 'loss_log.jsonl'), 'a') as f:
            f.write(json.dumps({'type': 'val', 'iter': iter_num,
                                'train_loss': float(losses['train']),
                                'val_loss':   float(losses['val'])}) + '\n')
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": losses['train'],
                       "val/loss": losses['val'], "lr": lr})
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                ckpt = {
                    'model':         raw_model.state_dict(),
                    'optimizer':     optimizer.state_dict(),
                    'model_args':    model_args,
                    'iter_num':      iter_num,
                    'best_val_loss': best_val_loss,
                    'config':        config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, _ = model(X, Y)

        with torch.no_grad():
            logits_opp, _ = model_opponent(X, Y)

        loss, _ = masked_selfplay_loss(logits, logits_opp, Y, mask)
        loss = loss / gradient_accumulation_steps
        X, Y, mask = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    else:
        grad_norm = None
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Periodically sync opponent weights from the learner
    if iter_num > 0 and iter_num % opponent_update_interval == 0:
        sd = {k.replace('_orig_mod.', ''): v for k, v in raw_model.state_dict().items()}
        model_opponent.load_state_dict(sd)
        model_opponent.eval()
        for p in model_opponent.parameters():
            p.requires_grad = False

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if math.isnan(lossf) or math.isinf(lossf):
            print(f"iter {iter_num}: loss is {lossf} — training is unstable, aborting.")
            break
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        norm_str = f", grad_norm {grad_norm:.3f}" if grad_norm is not None else ""
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%{norm_str}")
        with open(os.path.join(out_dir, 'loss_log.jsonl'), 'a') as f:
            f.write(json.dumps({'type': 'train', 'iter': iter_num, 'loss': lossf}) + '\n')
    iter_num += 1
    local_iter_num += 1

    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
