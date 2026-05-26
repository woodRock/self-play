#!/usr/bin/env python3
"""
Scaling ladder experiment: pretrain and evaluate baseline vs self-play models
at Small, Large, and XL scales.  Medium reuses existing seeds from the main
run_experiment.py run (outputs/babylm/baseline-s{seed} / selfplay-s{seed}).

Ladder:
  Small   n_layer=4  n_head=4  n_embd=256   ~3.2M transformer params   10 seeds
  Medium  n_layer=6  n_head=6  n_embd=384  ~10.6M transformer params   30 seeds (existing)
  Large   n_layer=8  n_head=8  n_embd=512  ~25.2M transformer params   10 seeds
  XL      n_layer=12 n_head=12 n_embd=768  ~85.0M transformer params    5 seeds

Eval JSON manifests are saved to results/scaling/{scale}/s{seed:02d}.json.
Use plot_scaling.py to generate the scaling figure.

Usage:
    python run_scaling_experiment.py --dry-run
    python run_scaling_experiment.py
    python run_scaling_experiment.py --eval_only
    python run_scaling_experiment.py --scales small large   # subset of scales
"""

import argparse
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Ladder definition
# ---------------------------------------------------------------------------

SCALES = {
    'small': {
        'baseline_config': 'config/train_small_babylm.py',
        'selfplay_config': 'config/train_selfplay_small_babylm.py',
        'n_seeds':         10,
        'baseline_tpl':    'outputs/scaling/small/baseline-s{seed}',
        'selfplay_tpl':    'outputs/scaling/small/selfplay-s{seed}',
        'train':           True,
    },
    'medium': {
        'baseline_config': 'config/train_babylm.py',
        'selfplay_config': 'config/train_selfplay_babylm.py',
        'n_seeds':         30,
        'baseline_tpl':    'outputs/babylm/baseline-s{seed}',
        'selfplay_tpl':    'outputs/babylm/selfplay-s{seed}',
        'train':           False,  # reuse existing checkpoints
    },
    'large': {
        'baseline_config': 'config/train_large_babylm.py',
        'selfplay_config': 'config/train_selfplay_large_babylm.py',
        'n_seeds':         10,
        'baseline_tpl':    'outputs/scaling/large/baseline-s{seed}',
        'selfplay_tpl':    'outputs/scaling/large/selfplay-s{seed}',
        'train':           True,
    },
    'xl': {
        'baseline_config': 'config/train_xl_babylm.py',
        'selfplay_config': 'config/train_selfplay_xl_babylm.py',
        'n_seeds':         5,
        'baseline_tpl':    'outputs/scaling/xl/baseline-s{seed}',
        'selfplay_tpl':    'outputs/scaling/xl/selfplay-s{seed}',
        'train':           True,
    },
}

SCALE_ORDER = ['small', 'medium', 'large', 'xl']

# ---------------------------------------------------------------------------
# Task-spooler helpers
# ---------------------------------------------------------------------------

def _strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s)


def _task_ids():
    r = subprocess.run(['task', '-l'], capture_output=True, text=True)
    ids = set()
    for line in _strip_ansi(r.stdout).splitlines():
        parts = line.split()
        if parts and parts[0].isdigit():
            ids.add(int(parts[0]))
    return ids


def submit(cmd, label=''):
    full = ['task', '-G', '1', '-m', '45'] + cmd
    print(f"  Submitting [{label}]: {' '.join(cmd)}")
    before = _task_ids()
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}")
        sys.exit(1)
    after = _task_ids()
    new_ids = after - before
    if new_ids:
        task_id = max(new_ids)
    else:
        nums = re.findall(r'\d+', _strip_ansi(result.stdout + result.stderr))
        if not nums:
            print("  ERROR: could not determine task ID.")
            sys.exit(1)
        task_id = int(nums[-1])
    print(f"    → task {task_id}")
    return task_id


def wait(task_id, label=''):
    print(f"Waiting for task {task_id} ({label})…")
    subprocess.run(['task', '-w', str(task_id)])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_only', action='store_true',
                        help='Skip training; submit only eval jobs')
    parser.add_argument('--dry_run',   action='store_true',
                        help='Print commands without submitting')
    parser.add_argument('--scales', nargs='+', choices=SCALE_ORDER,
                        default=SCALE_ORDER,
                        help='Which scales to run (default: all four)')
    args = parser.parse_args()

    active = [s for s in SCALE_ORDER if s in args.scales]

    for scale in active:
        cfg = SCALES[scale]
        seeds = range(1, cfg['n_seeds'] + 1)
        os.makedirs(f'results/scaling/{scale}', exist_ok=True)
        if cfg['train']:
            for subdir in ['baseline', 'selfplay']:
                os.makedirs(f'outputs/scaling/{scale}', exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Training (new scales only)
    # -----------------------------------------------------------------------
    train_tasks = []  # (task_id, scale, seed, label)

    if not args.eval_only:
        for scale in active:
            cfg = SCALES[scale]
            if not cfg['train']:
                print(f"[{scale}] skipping training — reusing existing checkpoints")
                continue

            seeds = range(1, cfg['n_seeds'] + 1)
            print(f"\n[{scale}] Submitting {len(seeds)} × 2 training jobs…")
            for seed in seeds:
                bl_dir = cfg['baseline_tpl'].format(seed=seed)
                sp_dir = cfg['selfplay_tpl'].format(seed=seed)

                bl_cmd = ['python3', 'train.py', cfg['baseline_config'],
                          f'--out_dir={bl_dir}', f'--manual_seed={seed}']
                sp_cmd = ['python3', 'train_selfplay.py', cfg['selfplay_config'],
                          f'--out_dir={sp_dir}', f'--manual_seed={seed}']

                if args.dry_run:
                    print(f"  [TRAIN-BL] {' '.join(bl_cmd)}")
                    print(f"  [TRAIN-SP] {' '.join(sp_cmd)}")
                    continue

                bl_tid = submit(bl_cmd, label=f'{scale}-bl-s{seed}')
                sp_tid = submit(sp_cmd, label=f'{scale}-sp-s{seed}')
                train_tasks.append((bl_tid, scale, seed, f'{scale}-bl-s{seed}'))
                train_tasks.append((sp_tid, scale, seed, f'{scale}-sp-s{seed}'))

        if not args.dry_run and train_tasks:
            print(f"\nWaiting for {len(train_tasks)} training jobs…")
            for tid, scale, seed, label in train_tasks:
                wait(tid, label)
            print("All training complete.\n")

    # -----------------------------------------------------------------------
    # 2. Evaluation
    # -----------------------------------------------------------------------
    eval_tasks = []  # (task_id, scale, seed, json_path)

    print("\nSubmitting eval jobs…")
    for scale in active:
        cfg = SCALES[scale]
        seeds = range(1, cfg['n_seeds'] + 1)
        for seed in seeds:
            bl_dir  = cfg['baseline_tpl'].format(seed=seed)
            sp_dir  = cfg['selfplay_tpl'].format(seed=seed)
            json_out = f'results/scaling/{scale}/s{seed:02d}.json'

            # Skip if manifest already exists (allow incremental re-runs)
            if os.path.exists(json_out) and not args.eval_only:
                continue

            # Skip if either checkpoint is missing (training didn't complete)
            bl_ckpt = os.path.join(bl_dir, 'ckpt.pt')
            sp_ckpt = os.path.join(sp_dir, 'ckpt.pt')
            if not os.path.exists(bl_ckpt) or not os.path.exists(sp_ckpt):
                missing = [p for p in (bl_ckpt, sp_ckpt) if not os.path.exists(p)]
                print(f"  [SKIP] {scale}-s{seed}: missing checkpoints: {missing}")
                continue

            eval_cmd = [
                'python3', 'eval.py',
                f'--baseline_dir={bl_dir}',
                f'--selfplay_dir={sp_dir}',
                '--eval_iters=200',
                '--device=cuda',
                f'--json_out={json_out}',
            ]

            if args.dry_run:
                print(f"  [EVAL] {' '.join(eval_cmd)}")
                continue

            tid = submit(eval_cmd, label=f'eval-{scale}-s{seed}')
            eval_tasks.append((tid, scale, seed, json_out))

    if not args.dry_run and eval_tasks:
        print(f"\nWaiting for {len(eval_tasks)} eval jobs…")
        for tid, scale, seed, json_out in eval_tasks:
            wait(tid, f'eval-{scale}-s{seed}')

    if args.dry_run:
        print("\n[DRY RUN] No jobs submitted.")
        return

    print("\nAll done.")
    print("Run:  python plot_scaling.py")


if __name__ == '__main__':
    main()
