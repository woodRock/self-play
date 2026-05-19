#!/usr/bin/env python3
"""
Trains baseline and self-play models, plots loss curves, and evaluates both.

Submits all jobs through the task spooler so they run on GPUs.  After training
completes the script plots loss curves to loss_curves.png and prints a summary
of which model performed better.

Usage:
    python run_experiment.py
    python run_experiment.py --resume          # continue from existing checkpoints
    python run_experiment.py --eval_only       # skip training, just plot and eval
"""

import argparse
import re
import subprocess
import os
import json
import sys
import time

import numpy as np

import matplotlib
matplotlib.use('Agg')   # no display needed on cluster
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASELINE_CONFIG  = 'config/train_babylm.py'
SELFPLAY_CONFIG  = 'config/train_selfplay_babylm.py'
BASELINE_OUT_DIR = 'out-babylm'
SELFPLAY_OUT_DIR = 'out-babylm-selfplay'
PLOT_FILE        = 'loss_curves.png'
EVAL_OUTPUT_FILE = 'eval_results.txt'

# ---------------------------------------------------------------------------
# Task spooler helpers
# ---------------------------------------------------------------------------

def _strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)


def _task_ids():
    """Return the set of currently known task IDs from `task -l`."""
    r = subprocess.run(['task', '-l'], capture_output=True, text=True)
    ids = set()
    for line in _strip_ansi(r.stdout).splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            ids.add(int(parts[0]))
    return ids


def submit(cmd, label=''):
    """Submit a command via the task spooler. Returns the task ID (int)."""
    full = ['task', '-G', '1', '-m', '45'] + cmd
    print(f"Submitting{' ' + label if label else ''}: {' '.join(cmd)}")
    before = _task_ids()
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR submitting task:\n{result.stderr}")
        sys.exit(1)
    after = _task_ids()
    new_ids = after - before
    if new_ids:
        task_id = max(new_ids)
    else:
        # last resort: find an integer in the submission output
        nums = re.findall(r'\d+', _strip_ansi(result.stdout + result.stderr))
        if not nums:
            print(f"ERROR: could not determine task ID.\nSubmit output: {repr(result.stdout + result.stderr)}\nTask list: {after}")
            sys.exit(1)
        task_id = int(nums[-1])
    print(f"  → task {task_id}")
    return task_id


def wait(task_id, label=''):
    """Block until a task spooler task finishes. Returns its saved output."""
    print(f"Waiting for task {task_id}{' (' + label + ')' if label else ''}...")
    subprocess.run(['task', '-w', str(task_id)])
    result = subprocess.run(['task', '-o', str(task_id)], capture_output=True, text=True)
    return result.stdout + result.stderr

# ---------------------------------------------------------------------------
# Loss log helpers
# ---------------------------------------------------------------------------

def load_loss_log(out_dir):
    """
    Read {out_dir}/loss_log.jsonl and return data from the last run only.
    Detects run boundaries by finding where the iteration counter resets.
        train_iters, train_losses, val_iters, val_train_losses, val_losses
    """
    path = os.path.join(out_dir, 'loss_log.jsonl')
    if not os.path.exists(path):
        print(f"WARNING: no loss log found at {path}")
        return [], [], [], [], []

    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    # find the last run by detecting where iter resets (strictly backwards)
    last_run_start = 0
    prev_iter = -1
    for i, entry in enumerate(entries):
        if entry['iter'] < prev_iter:
            last_run_start = i
        prev_iter = entry['iter']

    entries = entries[last_run_start:]

    train_iters, train_losses = [], []
    val_iters, val_train_losses, val_losses = [], [], []

    for entry in entries:
        if entry['type'] == 'train':
            train_iters.append(entry['iter'])
            train_losses.append(entry['loss'])
        elif entry['type'] == 'val':
            val_iters.append(entry['iter'])
            val_train_losses.append(entry['train_loss'])
            val_losses.append(entry['val_loss'])

    return train_iters, train_losses, val_iters, val_train_losses, val_losses

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_curves(baseline_dir, selfplay_dir, output_file):
    bl = load_loss_log(baseline_dir)
    sp = load_loss_log(selfplay_dir)

    bl_ti, bl_tl, bl_vi, bl_vtl, bl_vl = bl
    sp_ti, sp_tl, sp_vi, sp_vtl, sp_vl = sp

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Baseline vs Self-Play: BabyLM BPE Training', fontsize=14)

    BL_COLOR = 'steelblue'
    SP_COLOR = 'darkorange'

    def _smooth(values, window=50):
        if len(values) < window:
            return values
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode='valid')

    # --- Training loss: raw faint + smoothed bold ---
    if bl_ti:
        ax1.plot(bl_ti, bl_tl, color=BL_COLOR, alpha=0.15, linewidth=0.5)
        s = _smooth(bl_tl)
        ax1.plot(bl_ti[len(bl_ti)-len(s):], s, color=BL_COLOR, linewidth=2,
                 label='Baseline train')
    if sp_ti:
        ax1.plot(sp_ti, sp_tl, color=SP_COLOR, alpha=0.15, linewidth=0.5)
        s = _smooth(sp_tl)
        ax1.plot(sp_ti[len(sp_ti)-len(s):], s, color=SP_COLOR, linewidth=2,
                 label='Self-play train')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (smoothed)')
    ax1.set_ylim(bottom=3.5, top=7.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Validation loss: full run ---
    def _skip_zero(iters, losses):
        pairs = [(i, l) for i, l in zip(iters, losses) if i > 0]
        return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else (iters, losses)

    bl_vi2, bl_vl2 = _skip_zero(bl_vi, bl_vl)
    sp_vi2, sp_vl2 = _skip_zero(sp_vi, sp_vl)

    if bl_vi2:
        ax2.plot(bl_vi2, bl_vl2, color=BL_COLOR, linewidth=2, label='Baseline val')
    if sp_vi2:
        ax2.plot(sp_vi2, sp_vl2, color=SP_COLOR, linewidth=2,
                 linestyle='--', label='Self-play val')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- Validation loss: zoomed final 25% ---
    if bl_vi2 or sp_vi2:
        all_iters = bl_vi2 + sp_vi2
        zoom_start = max(all_iters) * 0.75 if all_iters else 0
        bl_vz = [(i, l) for i, l in zip(bl_vi2, bl_vl2) if i >= zoom_start]
        sp_vz = [(i, l) for i, l in zip(sp_vi2, sp_vl2) if i >= zoom_start]
        if bl_vz:
            ax3.plot([p[0] for p in bl_vz], [p[1] for p in bl_vz],
                     color=BL_COLOR, linewidth=2, marker='o', markersize=4, label='Baseline val')
        if sp_vz:
            ax3.plot([p[0] for p in sp_vz], [p[1] for p in sp_vz],
                     color=SP_COLOR, linewidth=2, marker='s', markersize=4,
                     linestyle='--', label='Self-play val')
        # annotate final values
        if bl_vz:
            ax3.annotate(f"{bl_vz[-1][1]:.4f}", xy=bl_vz[-1],
                         xytext=(8, 4), textcoords='offset points',
                         color=BL_COLOR, fontsize=9, fontweight='bold')
        if sp_vz:
            ax3.annotate(f"{sp_vz[-1][1]:.4f}", xy=sp_vz[-1],
                         xytext=(8, -12), textcoords='offset points',
                         color=SP_COLOR, fontsize=9, fontweight='bold')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_title('Validation Loss (final 25%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Loss curves saved to {output_file}")

    # Print final losses
    if bl_vl:
        print(f"  Baseline  final val loss : {bl_vl[-1]:.4f}  (iter {bl_vi[-1]})")
    if sp_vl:
        print(f"  Self-play final val loss : {sp_vl[-1]:.4f}  (iter {sp_vi[-1]})")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',    action='store_true', help='Resume from existing checkpoints')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only plot and evaluate')
    args = parser.parse_args()

    init_from = 'resume' if args.resume else 'scratch'

    # ------------------------------------------------------------------
    # 0. Clean up stale logs from any previous run (skip if resuming)
    # ------------------------------------------------------------------
    if not args.eval_only and not args.resume:
        for out_dir in (BASELINE_OUT_DIR, SELFPLAY_OUT_DIR):
            log = os.path.join(out_dir, 'loss_log.jsonl')
            if os.path.exists(log):
                os.remove(log)
                print(f"Removed stale {log}")

    # ------------------------------------------------------------------
    # 1. Train
    # ------------------------------------------------------------------
    if not args.eval_only:
        bl_cmd = [
            'python3', 'train.py', BASELINE_CONFIG,
            f'--init_from={init_from}',
        ]
        sp_cmd = [
            'python3', 'train_selfplay.py', SELFPLAY_CONFIG,
            f'--init_from={init_from}',
        ]

        bl_task = submit(bl_cmd,  label='baseline')
        sp_task = submit(sp_cmd,  label='self-play')

        wait(bl_task, label='baseline training')
        wait(sp_task, label='self-play training')
        print("Both training runs complete.\n")

    # ------------------------------------------------------------------
    # 2. Plot loss curves
    # ------------------------------------------------------------------
    print("Plotting loss curves...")
    plot_loss_curves(BASELINE_OUT_DIR, SELFPLAY_OUT_DIR, PLOT_FILE)
    print()

    # ------------------------------------------------------------------
    # 3. Evaluate
    # ------------------------------------------------------------------
    eval_cmd = [
        'python3', 'eval.py',
        f'--baseline_dir={BASELINE_OUT_DIR}',
        f'--selfplay_dir={SELFPLAY_OUT_DIR}',
        '--eval_iters=200',
        '--device=cuda',
    ]
    eval_task = submit(eval_cmd, label='evaluation')
    eval_output = wait(eval_task, label='evaluation')

    # Save eval output
    with open(EVAL_OUTPUT_FILE, 'w') as f:
        f.write(eval_output)
    print(f"Full eval output saved to {EVAL_OUTPUT_FILE}\n")

    # ------------------------------------------------------------------
    # 4. Summary: which model won?
    # ------------------------------------------------------------------
    print(eval_output)

    # Parse val losses from loss logs for the verdict
    _, _, bl_vi, _, bl_vl = load_loss_log(BASELINE_OUT_DIR)
    _, _, sp_vi, _, sp_vl = load_loss_log(SELFPLAY_OUT_DIR)

    if bl_vl and sp_vl:
        bl_final = bl_vl[-1]
        sp_final = sp_vl[-1]
        delta = bl_final - sp_final
        print('=' * 50)
        if delta > 0.005:
            print(f"WINNER: Self-play  (val loss {sp_final:.4f} vs {bl_final:.4f}, Δ={delta:+.4f})")
        elif delta < -0.005:
            print(f"WINNER: Baseline   (val loss {bl_final:.4f} vs {sp_final:.4f}, Δ={delta:+.4f})")
        else:
            print(f"DRAW: val losses within 0.005 ({bl_final:.4f} vs {sp_final:.4f})")
        print('=' * 50)


if __name__ == '__main__':
    main()
