#!/usr/bin/env python3
"""
Dispatch the 2x2 BabyLM continuation SFT experiment across 30 seeds.

  Pretrain × SFT method  →  120 training runs + 120 eval runs
  ─────────────────────────────────────────────────────────────
  BL  × SFT-BL   (train_sft.py          + finetune_continuation.py)
  BL  × SFT-SP   (train_selfplay_sft.py + finetune_selfplay_continuation.py)
  SP  × SFT-BL   (train_sft.py          + finetune_continuation.py, pretrain=out-babylm-selfplay-s{seed})
  SP  × SFT-SP   (train_selfplay_sft.py + finetune_selfplay_continuation.py, pretrain=out-babylm-selfplay-s{seed})

Generates 500 continuations per model for diversity evaluation.

Usage:
    python run_continuation_experiment.py -n 30 --dry-run
    python run_continuation_experiment.py -n 30
"""

import argparse
import os
import re
import subprocess
import sys

CONDITIONS = [
    ('BL_SFT_BL', 'train_sft.py',         'config/finetune_continuation.py',
     'out-babylm-s{seed}'),
    ('BL_SFT_SP', 'train_selfplay_sft.py', 'config/finetune_selfplay_continuation.py',
     'out-babylm-s{seed}'),
    ('SP_SFT_BL', 'train_sft.py',         'config/finetune_continuation.py',
     'out-babylm-selfplay-s{seed}'),
    ('SP_SFT_SP', 'train_selfplay_sft.py', 'config/finetune_selfplay_continuation.py',
     'out-babylm-selfplay-s{seed}'),
]


def out_dir(cond, seed):
    return f'out-cont-{cond.lower()}-s{seed:02d}'


def manifest_path(cond, seed):
    return f'results/continuation/{cond}_s{seed:02d}.json'


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


def submit(cmd, label='', dep_id=None):
    full = ['task', '-G', '1', '-m', '45']
    if dep_id is not None:
        full += ['-D', str(dep_id)]
    full += cmd
    print(f"  Submitting [{label}]: {' '.join(cmd)}")
    before = _task_ids()
    result = subprocess.run(full, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR:\n{result.stderr}")
        sys.exit(1)
    after   = _task_ids()
    new_ids = after - before
    if new_ids:
        task_id = max(new_ids)
    else:
        nums = re.findall(r'\d+', _strip_ansi(result.stdout + result.stderr))
        if not nums:
            print(f"  ERROR: could not determine task ID.")
            sys.exit(1)
        task_id = int(nums[-1])
    print(f"    → task {task_id}")
    return task_id


def wait(task_id, label=''):
    print(f"Waiting for task {task_id} ({label})…")
    subprocess.run(['task', '-w', str(task_id)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_seeds', type=int, default=30)
    parser.add_argument('--dry-run',   action='store_true')
    parser.add_argument('--eval_only', action='store_true')
    args = parser.parse_args()

    os.makedirs('results/continuation', exist_ok=True)

    all_train_ids = []
    all_eval_cmds = []
    seeds = range(1, args.n_seeds + 1)

    for seed in seeds:
        for cond_label, train_script, config, pretrain_tpl in CONDITIONS:
            pretrain = pretrain_tpl.format(seed=seed)
            outd     = out_dir(cond_label, seed)
            manifest = manifest_path(cond_label, seed)
            label    = f'{cond_label}_s{seed:02d}'

            train_cmd = [
                'python3', train_script, config,
                f'--pretrain_dir={pretrain}',
                f'--out_dir={outd}',
                f'--manual_seed={seed}',
            ]
            eval_cmd = [
                'python3', 'eval_generation.py',
                f'--model_dir={outd}',
                f'--seed={seed}',
                f'--condition={cond_label}',
                f'--manifest={manifest}',
                '--n_prompts=500',
            ]

            if args.dry_run:
                print(f"[TRAIN] {' '.join(train_cmd)}")
                print(f"[EVAL ] {' '.join(eval_cmd)}")
                print()
                continue

            if not args.eval_only:
                tid = submit(train_cmd, label=f'train_{label}')
                all_train_ids.append((tid, f'train_{label}'))
                all_eval_cmds.append((eval_cmd, f'eval_{label}', tid))
            else:
                all_eval_cmds.append((eval_cmd, f'eval_{label}', None))

    if args.dry_run:
        print(f"\n[DRY RUN] Total jobs: "
              f"{len(seeds) * len(CONDITIONS)} train + "
              f"{len(seeds) * len(CONDITIONS)} eval = "
              f"{2 * len(seeds) * len(CONDITIONS)} tasks")
        return

    print(f"\nSubmitted {len(all_train_ids)} training jobs. Waiting…")
    for tid, lbl in all_train_ids:
        wait(tid, lbl)

    print("\nAll training complete. Submitting eval jobs…")
    eval_ids = []
    for eval_cmd, label, dep_id in all_eval_cmds:
        tid = submit(eval_cmd, label=label)
        eval_ids.append((tid, label))

    print(f"Waiting for {len(eval_ids)} eval jobs…")
    for tid, lbl in eval_ids:
        wait(tid, lbl)

    print("\nAll done. Results in results/continuation/")
    print("Run: python analyze_continuation.py --results_dir results/continuation")


if __name__ == '__main__':
    main()
