"""
Plot scaling curves: overall accuracy and rare-token accuracy for baseline vs
self-play across the four scales (Small → Medium → Large → XL).

Reads JSON manifests from results/scaling/{scale}/s*.json (written by eval.py
--json_out).  Each manifest contains 'baseline' and 'selfplay' dicts with
'accuracy' and 'rare_accuracy' fields.

Usage:
    python plot_scaling.py
    python plot_scaling.py --out results/scaling_curves.png
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ladder definition (transformer-only parameter counts for x-axis)
# ---------------------------------------------------------------------------

SCALES = [
    ('small',  3.15e6,  'Small\n(4L·4H·256d)'),
    ('medium', 10.62e6, 'Medium\n(6L·6H·384d)'),
    ('large',  25.17e6, 'Large\n(8L·8H·512d)'),
    ('xl',     84.95e6, 'XL\n(12L·12H·768d)'),
]

BL_COLOR = 'steelblue'
SP_COLOR = 'darkorange'
ALPHA_FILL = 0.15

# ---------------------------------------------------------------------------

def load_scale(scale_name):
    """Return list of (baseline_dict, selfplay_dict) from JSON manifests."""
    results_dir = os.path.join('results', 'scaling', scale_name)
    if not os.path.isdir(results_dir):
        return []
    records = []
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            m = json.load(f)
        records.append((m['baseline'], m['selfplay']))
    return records


def stats(values):
    """Return (mean, sem) for a list of floats."""
    a = np.array(values, dtype=float)
    if len(a) == 0:
        return float('nan'), float('nan')
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def extract(records, key):
    bl_vals = [r[0][key] for r in records]
    sp_vals = [r[1][key] for r in records]
    return stats(bl_vals), stats(sp_vals)

# ---------------------------------------------------------------------------

def main():
    scale_names = [s for s, _, _ in SCALES]
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='results/scaling_curves.png')
    parser.add_argument('--scales', nargs='+', choices=scale_names, default=scale_names,
                        help='Which scales to include (default: all)')
    args = parser.parse_args()

    active_scales = [s for s in SCALES if s[0] in args.scales]

    # Collect data
    data = {}
    for scale_name, _, _ in active_scales:
        records = load_scale(scale_name)
        if not records:
            print(f"WARNING: no results found for scale '{scale_name}' "
                  f"in results/scaling/{scale_name}/")
        data[scale_name] = records

    x_params  = [p for _, p, _ in active_scales]
    x_labels  = [lbl for _, _, lbl in active_scales]
    x_idx     = list(range(len(active_scales)))

    metrics = [
        ('accuracy',      'Overall accuracy',      'Overall next-token accuracy (%)'),
        ('rare_accuracy', 'Rare-token accuracy',   'Rare-token accuracy (%)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Self-play vs Baseline Across Scale', fontsize=13, fontweight='bold')

    for ax, (key, title, ylabel) in zip(axes, metrics):
        bl_means, bl_sems = [], []
        sp_means, sp_sems = [], []

        for scale_name, _, _ in active_scales:
            records = data[scale_name]
            if not records:
                bl_means.append(float('nan')); bl_sems.append(0)
                sp_means.append(float('nan')); sp_sems.append(0)
                continue
            (bm, bs), (sm, ss) = extract(records, key)
            bl_means.append(bm * 100); bl_sems.append(bs * 100)
            sp_means.append(sm * 100); sp_sems.append(ss * 100)

        bl_means = np.array(bl_means)
        sp_means = np.array(sp_means)
        bl_sems  = np.array(bl_sems)
        sp_sems  = np.array(sp_sems)

        ax.plot(x_idx, bl_means, 'o-', color=BL_COLOR, linewidth=2,
                markersize=7, label='Baseline')
        ax.fill_between(x_idx,
                        bl_means - 1.96 * bl_sems,
                        bl_means + 1.96 * bl_sems,
                        color=BL_COLOR, alpha=ALPHA_FILL)

        ax.plot(x_idx, sp_means, 's-', color=SP_COLOR, linewidth=2,
                markersize=7, label='Self-play')
        ax.fill_between(x_idx,
                        sp_means - 1.96 * sp_sems,
                        sp_means + 1.96 * sp_sems,
                        color=SP_COLOR, alpha=ALPHA_FILL)

        ax.set_xticks(x_idx)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_xlabel('Scale (transformer params)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        # Annotate gap at each scale
        for i, (bm, sm, bs, ss) in enumerate(zip(bl_means, sp_means, bl_sems, sp_sems)):
            if np.isnan(bm):
                continue
            gap = sm - bm
            ax.annotate(f'Δ={gap:+.2f}%',
                        xy=(i, max(bm, sm) + 1.96 * max(bs, ss)),
                        ha='center', va='bottom', fontsize=7.5,
                        color='dimgray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches='tight')
    print(f"Saved scaling curves to {args.out}")


if __name__ == '__main__':
    main()
