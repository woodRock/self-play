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
    Read {out_dir}/loss_log.jsonl and return:
        train_iters, train_losses, val_iters, val_train_losses, val_losses
    """
    path = os.path.join(out_dir, 'loss_log.jsonl')
    if not os.path.exists(path):
        print(f"WARNING: no loss log found at {path}")
        return [], [], [], [], []

    train_iters, train_losses = [], []
    val_iters, val_train_losses, val_losses = [], [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Baseline vs Self-Play: BabyLM BPE Training', fontsize=14)

    # --- Training loss ---
    if bl_ti:
        ax1.plot(bl_ti, bl_tl, color='steelblue',  alpha=0.7, linewidth=0.8, label='Baseline train')
    if sp_ti:
        ax1.plot(sp_ti, sp_tl, color='darkorange', alpha=0.7, linewidth=0.8, label='Self-play train')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Validation loss ---
    if bl_vi:
        ax2.plot(bl_vi, bl_vl, color='steelblue',  marker='o', markersize=4, linewidth=1.5, label='Baseline val')
    if sp_vi:
        ax2.plot(sp_vi, sp_vl, color='darkorange', marker='s', markersize=4, linewidth=1.5, label='Self-play val')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

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
