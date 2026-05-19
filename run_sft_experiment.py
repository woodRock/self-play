#!/usr/bin/env python3
"""
Runs the full 2×2 BLiMP SFT experiment:

  Pretrain  ×  SFT method
  ────────────────────────────────────────────────────
  Baseline  ×  Baseline SFT    →  out-blimp-sft
  Baseline  ×  Self-play SFT   →  out-blimp-selfplay-sft
  Self-play ×  Baseline SFT    →  out-blimp-sft-from-sp
  Self-play ×  Self-play SFT   →  out-blimp-selfplay-sft-from-sp

Then evaluates all four against the two pretrained (zero-shot) baselines.

Usage:
    python run_sft_experiment.py
    python run_sft_experiment.py --resume      # continue from existing checkpoints
    python run_sft_experiment.py --eval_only   # skip training, jump straight to eval
"""

import argparse
import re
import subprocess
import os
import sys

# ---------------------------------------------------------------------------
# The four fine-tuning jobs: (label, train_script, config, out_dir)
# ---------------------------------------------------------------------------

JOBS = [
    ('BL→SFT-BL',  'train_sft.py',          'config/finetune_blimp.py',                  'out-blimp-sft'),
    ('BL→SFT-SP',  'train_selfplay_sft.py',  'config/finetune_selfplay_blimp.py',         'out-blimp-selfplay-sft'),
    ('SP→SFT-BL',  'train_sft.py',           'config/finetune_blimp_from_sp.py',          'out-blimp-sft-from-sp'),
    ('SP→SFT-SP',  'train_selfplay_sft.py',  'config/finetune_selfplay_blimp_from_sp.py', 'out-blimp-selfplay-sft-from-sp'),
]

# Models passed to eval_blimp.py (zero-shot pretrained + all four fine-tuned)
EVAL_MODELS = [
    'Pretrained-BL=out-babylm',
    'Pretrained-SP=out-babylm-selfplay',
    'BL→SFT-BL=out-blimp-sft',
    'BL→SFT-SP=out-blimp-selfplay-sft',
    'SP→SFT-BL=out-blimp-sft-from-sp',
    'SP→SFT-SP=out-blimp-selfplay-sft-from-sp',
]

EVAL_RESULTS_FILE = 'blimp_results.txt'

# ---------------------------------------------------------------------------
# Task spooler helpers (same as run_experiment.py)
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
    print(f"Submitting {label}: {' '.join(cmd)}")
    before = _task_ids()
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR:\n{result.stderr}")
        sys.exit(1)
    after   = _task_ids()
    new_ids = after - before
    if new_ids:
        task_id = max(new_ids)
    else:
        nums = re.findall(r'\d+', _strip_ansi(result.stdout + result.stderr))
        if not nums:
            print(f"ERROR: could not determine task ID. Output: {repr(result.stdout + result.stderr)}")
            sys.exit(1)
        task_id = int(nums[-1])
    print(f"  → task {task_id}")
    return task_id


def wait(task_id, label=''):
    print(f"Waiting for task {task_id} ({label})...")
    subprocess.run(['task', '-w', str(task_id)])
    result = subprocess.run(['task', '-o', str(task_id)], capture_output=True, text=True)
    return result.stdout + result.stderr

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume',    action='store_true', help='Resume from existing checkpoints')
    parser.add_argument('--eval_only', action='store_true', help='Skip training, only run eval')
    args = parser.parse_args()

    init_from = 'resume' if args.resume else 'finetune'

    # ------------------------------------------------------------------
    # 0. Clean stale loss logs for fresh runs
    # ------------------------------------------------------------------
    if not args.eval_only and not args.resume:
        for _, _, _, out_dir in JOBS:
            log = os.path.join(out_dir, 'loss_log.jsonl')
            if os.path.exists(log):
                os.remove(log)
                print(f"Removed stale {log}")

    # ------------------------------------------------------------------
    # 1. Submit all four fine-tuning jobs, then wait for all
    # ------------------------------------------------------------------
    if not args.eval_only:
        tasks = []
        for label, script, config, out_dir in JOBS:
            cmd = [
                'python3', script, config,
                f'--init_from={init_from}',
                f'--out_dir={out_dir}',
            ]
            task_id = submit(cmd, label=label)
            tasks.append((task_id, label))

        for task_id, label in tasks:
            wait(task_id, label=label)
        print(f"\nAll {len(JOBS)} fine-tuning jobs complete.\n")

    # ------------------------------------------------------------------
    # 2. Evaluate all six models on BLiMP
    # ------------------------------------------------------------------
    eval_cmd = (
        ['python3', 'eval_blimp.py', '--device=cuda'] +
        ['--models'] + EVAL_MODELS
    )
    eval_task = submit(eval_cmd, label='BLiMP eval')
    output    = wait(eval_task, label='BLiMP eval')

    with open(EVAL_RESULTS_FILE, 'w') as f:
        f.write(output)
    print(f"\nResults saved to {EVAL_RESULTS_FILE}\n")
    print(output)


if __name__ == '__main__':
    main()
