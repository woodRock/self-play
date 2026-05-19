#!/usr/bin/env python3
"""
Trains baseline and self-play models, plots loss curves, and evaluates both.
Supports multiple random seeds for statistical significance testing.

Usage:
    python run_experiment.py                  # single run
    python run_experiment.py -n 30            # 30 seeds, jobs submitted in parallel
    python run_experiment.py --resume         # continue from existing checkpoints
    python run_experiment.py --eval_only      # skip training, just plot and eval
    python run_experiment.py -n 5 --eval_only # plot/eval existing 5-seed results
"""

import argparse
import re
import subprocess
import os
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
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
    Detects run boundaries where the iteration counter resets.
    Returns: train_iters, train_losses, val_iters, val_train_losses, val_losses
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
# Plotting helpers
# ---------------------------------------------------------------------------

BL_COLOR = 'steelblue'
SP_COLOR = 'darkorange'


def _smooth(values, window=50):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode='valid').tolist()


def _skip_zero(iters, losses):
    pairs = [(i, l) for i, l in zip(iters, losses) if i > 0]
    return ([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else (iters, losses)


def _add_zoom_panel(ax, bl_vi, bl_vl, sp_vi, sp_vl):
    """Populate a zoomed val-loss panel showing the final 25% of training."""
    all_iters = list(bl_vi) + list(sp_vi)
    if not all_iters:
        return
    zoom_start = max(all_iters) * 0.75

    bl_vz = [(i, l) for i, l in zip(bl_vi, bl_vl) if i >= zoom_start]
    sp_vz = [(i, l) for i, l in zip(sp_vi, sp_vl) if i >= zoom_start]

    if bl_vz:
        ax.plot([p[0] for p in bl_vz], [p[1] for p in bl_vz],
                color=BL_COLOR, linewidth=2, marker='o', markersize=3, label='Baseline val')
        ax.annotate(f"{bl_vz[-1][1]:.4f}", xy=bl_vz[-1],
                    xytext=(8, 4), textcoords='offset points',
                    color=BL_COLOR, fontsize=9, fontweight='bold')
    if sp_vz:
        ax.plot([p[0] for p in sp_vz], [p[1] for p in sp_vz],
                color=SP_COLOR, linewidth=2, marker='s', markersize=3,
                linestyle='--', label='Self-play val')
        ax.annotate(f"{sp_vz[-1][1]:.4f}", xy=sp_vz[-1],
                    xytext=(8, -12), textcoords='offset points',
                    color=SP_COLOR, fontsize=9, fontweight='bold')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss (final 25%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# ---------------------------------------------------------------------------
# Single-seed plot
# ---------------------------------------------------------------------------

def plot_loss_curves(baseline_dir, selfplay_dir, output_file):
    bl_ti, bl_tl, bl_vi, _, bl_vl = load_loss_log(baseline_dir)
    sp_ti, sp_tl, sp_vi, _, sp_vl = load_loss_log(selfplay_dir)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Baseline vs Self-Play: BabyLM BPE Training', fontsize=14)

    # training loss: raw faint + smoothed bold
    if bl_ti:
        ax1.plot(bl_ti, bl_tl, color=BL_COLOR, alpha=0.15, linewidth=0.5)
        s = _smooth(bl_tl)
        ax1.plot(bl_ti[len(bl_ti) - len(s):], s, color=BL_COLOR, linewidth=2, label='Baseline train')
    if sp_ti:
        ax1.plot(sp_ti, sp_tl, color=SP_COLOR, alpha=0.15, linewidth=0.5)
        s = _smooth(sp_tl)
        ax1.plot(sp_ti[len(sp_ti) - len(s):], s, color=SP_COLOR, linewidth=2,
                 linestyle='--', label='Self-play train')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (smoothed)')
    ax1.set_ylim(bottom=3.5, top=7.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # full validation loss
    bl_vi2, bl_vl2 = _skip_zero(bl_vi, bl_vl)
    sp_vi2, sp_vl2 = _skip_zero(sp_vi, sp_vl)
    if bl_vi2:
        ax2.plot(bl_vi2, bl_vl2, color=BL_COLOR, linewidth=2, label='Baseline val')
    if sp_vi2:
        ax2.plot(sp_vi2, sp_vl2, color=SP_COLOR, linewidth=2, linestyle='--', label='Self-play val')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    _add_zoom_panel(ax3, bl_vi2, bl_vl2, sp_vi2, sp_vl2)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Loss curves saved to {output_file}")
    if bl_vl:
        print(f"  Baseline  final val loss : {bl_vl[-1]:.4f}  (iter {bl_vi[-1]})")
    if sp_vl:
        print(f"  Self-play final val loss : {sp_vl[-1]:.4f}  (iter {sp_vi[-1]})")

# ---------------------------------------------------------------------------
# Multi-seed aggregated plot
# ---------------------------------------------------------------------------

def _collect_val(base_dir, seeds):
    """Returns (iters, mean_array, std_array) across seeds."""
    all_losses = []
    ref_iters = None
    for seed in seeds:
        _, _, vi, _, vl = load_loss_log(f"{base_dir}-s{seed}")
        vi2, vl2 = _skip_zero(vi, vl)
        if not vl2:
            continue
        if ref_iters is None or len(vi2) < len(ref_iters):
            ref_iters = vi2
        all_losses.append(vl2)
    if not all_losses or ref_iters is None:
        return [], np.array([]), np.array([])
    n = len(ref_iters)
    arr = np.array([v[:n] for v in all_losses if len(v) >= n])
    return ref_iters, arr.mean(axis=0), arr.std(axis=0)


def _collect_train(base_dir, seeds):
    """Returns (iters, mean_smoothed) across seeds — used for training loss panel."""
    all_losses = []
    ref_iters = None
    for seed in seeds:
        ti, tl, _, _, _ = load_loss_log(f"{base_dir}-s{seed}")
        if not tl:
            continue
        if ref_iters is None or len(ti) < len(ref_iters):
            ref_iters = ti
        all_losses.append(tl)
    if not all_losses or ref_iters is None:
        return [], np.array([])
    n = len(ref_iters)
    arr = np.array([v[:n] for v in all_losses if len(v) >= n])
    return ref_iters, arr.mean(axis=0)


def plot_aggregated_loss_curves(seeds, baseline_base, selfplay_base, output_file):
    bl_ti, bl_tm = _collect_train(baseline_base, seeds)
    sp_ti, sp_tm = _collect_train(selfplay_base, seeds)
    bl_vi, bl_vm, bl_vs = _collect_val(baseline_base, seeds)
    sp_vi, sp_vm, sp_vs = _collect_val(selfplay_base, seeds)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Baseline vs Self-Play: BabyLM BPE  (n={len(seeds)} seeds)', fontsize=14)

    # smoothed mean training loss
    if len(bl_ti) > 0:
        s = _smooth(bl_tm.tolist())
        ax1.plot(bl_ti[len(bl_ti) - len(s):], s, color=BL_COLOR, linewidth=2, label='Baseline train')
    if len(sp_ti) > 0:
        s = _smooth(sp_tm.tolist())
        ax1.plot(sp_ti[len(sp_ti) - len(s):], s, color=SP_COLOR, linewidth=2,
                 linestyle='--', label='Self-play train')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss (mean, smoothed)')
    ax1.set_ylim(bottom=3.5, top=7.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # full val loss with ±1σ band
    if len(bl_vi) > 0:
        ax2.plot(bl_vi, bl_vm, color=BL_COLOR, linewidth=2, label='Baseline val')
        ax2.fill_between(bl_vi, bl_vm - bl_vs, bl_vm + bl_vs, color=BL_COLOR, alpha=0.2)
    if len(sp_vi) > 0:
        ax2.plot(sp_vi, sp_vm, color=SP_COLOR, linewidth=2, linestyle='--', label='Self-play val')
        ax2.fill_between(sp_vi, sp_vm - sp_vs, sp_vm + sp_vs, color=SP_COLOR, alpha=0.2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Loss')
    ax2.set_title('Validation Loss (mean ± 1σ)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # zoomed val loss with ±1σ band
    all_iters = list(bl_vi) + list(sp_vi)
    if all_iters:
        zoom_start = max(all_iters) * 0.75

        def zoom(vi, vm, vs):
            triples = [(i, m, s) for i, m, s in zip(vi, vm, vs) if i >= zoom_start]
            if not triples:
                return [], np.array([]), np.array([])
            return ([t[0] for t in triples],
                    np.array([t[1] for t in triples]),
                    np.array([t[2] for t in triples]))

        bzi, bzm, bzs = zoom(bl_vi, bl_vm, bl_vs)
        szi, szm, szs = zoom(sp_vi, sp_vm, sp_vs)

        if bzi:
            ax3.plot(bzi, bzm, color=BL_COLOR, linewidth=2, marker='o', markersize=3, label='Baseline val')
            ax3.fill_between(bzi, bzm - bzs, bzm + bzs, color=BL_COLOR, alpha=0.2)
            ax3.annotate(f"{bzm[-1]:.4f}", xy=(bzi[-1], bzm[-1]),
                         xytext=(8, 4), textcoords='offset points',
                         color=BL_COLOR, fontsize=9, fontweight='bold')
        if szi:
            ax3.plot(szi, szm, color=SP_COLOR, linewidth=2, marker='s', markersize=3,
                     linestyle='--', label='Self-play val')
            ax3.fill_between(szi, szm - szs, szm + szs, color=SP_COLOR, alpha=0.2)
            ax3.annotate(f"{szm[-1]:.4f}", xy=(szi[-1], szm[-1]),
                         xytext=(8, -12), textcoords='offset points',
                         color=SP_COLOR, fontsize=9, fontweight='bold')

    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Loss')
    ax3.set_title('Validation Loss — final 25% (mean ± 1σ)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Aggregated loss curves saved to {output_file}  (n={len(seeds)} seeds)")

# ---------------------------------------------------------------------------
# Multi-seed statistical report
# ---------------------------------------------------------------------------

def report_multi_seed(seeds, baseline_base, selfplay_base):
    """Collect final val losses across seeds and run a paired t-test."""
    bl_finals, sp_finals = [], []
    for seed in seeds:
        _, _, bl_vi, _, bl_vl = load_loss_log(f"{baseline_base}-s{seed}")
        _, _, sp_vi, _, sp_vl = load_loss_log(f"{selfplay_base}-s{seed}")
        if bl_vl and sp_vl:
            bl_finals.append(bl_vl[-1])
            sp_finals.append(sp_vl[-1])

    if not bl_finals:
        print("No multi-seed results found.")
        return

    bl_arr = np.array(bl_finals)
    sp_arr = np.array(sp_finals)
    diffs  = bl_arr - sp_arr   # positive → self-play wins
    n      = len(diffs)
    t_stat = diffs.mean() / (diffs.std(ddof=1) / np.sqrt(n))

    try:
        from scipy.stats import t as t_dist
        p_val = 2 * float(t_dist.sf(abs(t_stat), df=n - 1))
        p_str = f"p = {p_val:.4f}"
    except ImportError:
        p_val = None
        p_str = "(install scipy for p-value)"

    wins = int((sp_arr < bl_arr).sum())

    print('=' * 60)
    print(f"MULTI-SEED RESULTS  (n = {n} seeds)")
    print('=' * 60)
    print(f"  Baseline  val loss : {bl_arr.mean():.4f} ± {bl_arr.std():.4f}")
    print(f"  Self-play val loss : {sp_arr.mean():.4f} ± {sp_arr.std():.4f}")
    print(f"  Mean Δ (BL − SP)   : {diffs.mean():+.4f} ± {diffs.std(ddof=1):.4f}")
    print(f"  Paired t-test      : t({n - 1}) = {t_stat:.3f},  {p_str}")
    print(f"  Self-play wins     : {wins} / {n} seeds")
    if p_val is not None:
        if p_val < 0.01:
            verdict = "SIGNIFICANT (p < 0.01)"
        elif p_val < 0.05:
            verdict = "SIGNIFICANT (p < 0.05)"
        elif p_val < 0.10:
            verdict = "MARGINAL (p < 0.10)"
        else:
            verdict = f"NOT SIGNIFICANT (p = {p_val:.3f})"
        winner = "Self-play" if diffs.mean() > 0 else "Baseline"
        print(f"\n  {verdict} — {winner} is better on average")
    else:
        winner = "Self-play" if diffs.mean() > 0 else "Baseline"
        print(f"\n  {winner} wins on average  ({wins}/{n} individual wins)")
    print('=' * 60)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',   action='store_true', help='Resume from existing checkpoints')
    parser.add_argument('--eval_only',action='store_true', help='Skip training, only plot and evaluate')
    parser.add_argument('-n', '--n_seeds', type=int, default=1,
                        help='Number of random seeds (default: 1)')
    args = parser.parse_args()

    seeds  = list(range(1, args.n_seeds + 1))
    multi  = args.n_seeds > 1
    init_from = 'resume' if args.resume else 'scratch'

    def bl_dir(seed):
        return f"{BASELINE_OUT_DIR}-s{seed}" if multi else BASELINE_OUT_DIR

    def sp_dir(seed):
        return f"{SELFPLAY_OUT_DIR}-s{seed}" if multi else SELFPLAY_OUT_DIR

    # ------------------------------------------------------------------
    # 0. Clean stale logs (fresh runs only)
    # ------------------------------------------------------------------
    if not args.eval_only and not args.resume:
        for seed in seeds:
            for d in [bl_dir(seed), sp_dir(seed)]:
                log = os.path.join(d, 'loss_log.jsonl')
                if os.path.exists(log):
                    os.remove(log)
                    print(f"Removed stale {log}")

    # ------------------------------------------------------------------
    # 1. Train — submit all jobs first, then wait for all
    # ------------------------------------------------------------------
    if not args.eval_only:
        train_tasks = []
        for seed in seeds:
            bl_cmd = [
                'python3', 'train.py', BASELINE_CONFIG,
                f'--init_from={init_from}',
                f'--out_dir={bl_dir(seed)}',
                f'--manual_seed={seed}',
            ]
            sp_cmd = [
                'python3', 'train_selfplay.py', SELFPLAY_CONFIG,
                f'--init_from={init_from}',
                f'--out_dir={sp_dir(seed)}',
                f'--manual_seed={seed}',
            ]
            bl_task = submit(bl_cmd, label=f'baseline s{seed}')
            sp_task = submit(sp_cmd, label=f'self-play s{seed}')
            train_tasks.append((bl_task, sp_task, seed))

        for bl_task, sp_task, seed in train_tasks:
            wait(bl_task, label=f'baseline s{seed}')
            wait(sp_task, label=f'self-play s{seed}')
        print(f"\nAll {len(seeds) * 2} training jobs complete.\n")

    # ------------------------------------------------------------------
    # 2. Plot
    # ------------------------------------------------------------------
    print("Plotting loss curves...")
    if multi:
        plot_aggregated_loss_curves(seeds, BASELINE_OUT_DIR, SELFPLAY_OUT_DIR, PLOT_FILE)
    else:
        plot_loss_curves(bl_dir(1), sp_dir(1), PLOT_FILE)
    print()

    # ------------------------------------------------------------------
    # 3. Evaluate — submit all, then wait for all
    # ------------------------------------------------------------------
    eval_tasks = []
    for seed in seeds:
        eval_cmd = [
            'python3', 'eval.py',
            f'--baseline_dir={bl_dir(seed)}',
            f'--selfplay_dir={sp_dir(seed)}',
            '--eval_iters=200',
            '--device=cuda',
        ]
        eval_task = submit(eval_cmd, label=f'eval s{seed}')
        eval_tasks.append((eval_task, seed))

    for eval_task, seed in eval_tasks:
        out = wait(eval_task, label=f'eval s{seed}')
        fname = f"eval_results_s{seed}.txt" if multi else EVAL_OUTPUT_FILE
        with open(fname, 'w') as f:
            f.write(out)
        if not multi:
            print(out)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    if multi:
        report_multi_seed(seeds, BASELINE_OUT_DIR, SELFPLAY_OUT_DIR)
    else:
        _, _, bl_vi, _, bl_vl = load_loss_log(bl_dir(1))
        _, _, sp_vi, _, sp_vl = load_loss_log(sp_dir(1))
        if bl_vl and sp_vl:
            bl_final, sp_final = bl_vl[-1], sp_vl[-1]
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
