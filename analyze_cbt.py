#!/usr/bin/env python3
"""
Analyse CBT-NE experiment results and produce a LaTeX table.

Reads all JSON manifests from results/cbt/, groups by condition,
runs Welch's t-test and Mann-Whitney U for the 4 pre-registered hypotheses,
and prints a LaTeX tabular block ready to paste into the paper.

Usage:
    python analyze_cbt.py --results_dir results/cbt [--alpha 0.05]
"""

import os
import json
import argparse
import numpy as np
from scipy import stats


HYPOTHESES = [
    ('H_A', 'SP-pretrain → SFT-BL',        'BL_SFT_BL', 'SP_SFT_BL'),
    ('H_B', 'SP-SFT from BL-pretrain',      'BL_SFT_BL', 'BL_SFT_SP'),
    ('H_C', 'SP-SFT from SP-pretrain',      'SP_SFT_BL', 'SP_SFT_SP'),
    ('H_D', 'Full SP vs full BL',           'BL_SFT_BL', 'SP_SFT_SP'),
]

N_HYPOTHESES  = 4
ALPHA_FAMILY  = 0.05
ALPHA_CORR    = ALPHA_FAMILY / N_HYPOTHESES   # Bonferroni = 0.0125


def load_results(results_dir):
    """Return dict: condition → sorted list of accuracy values."""
    data = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            m = json.load(f)
        cond = m.get('condition', fname.replace('.json', ''))
        acc  = m.get('accuracy')
        if acc is None:
            continue
        data.setdefault(cond, []).append((m.get('seed', 0), acc))
    # Sort by seed within each condition
    return {c: [acc for _, acc in sorted(v)] for c, v in data.items()}


def paired_wins(a_vals, b_vals):
    """Count seed-matched wins of b over a."""
    n = min(len(a_vals), len(b_vals))
    return sum(1 for i in range(n) if b_vals[i] > a_vals[i])


def fmt_p(p):
    if p < 1e-4:
        return r'$<$10$^{-4}$'
    elif p < 0.001:
        return f'{p:.4f}'
    else:
        return f'{p:.3f}'


def analyse(results_dir, alpha=ALPHA_FAMILY):
    data = load_results(results_dir)

    conditions = sorted(data.keys())
    print(f"\nLoaded conditions: {conditions}")
    for c, vals in data.items():
        print(f"  {c}: n={len(vals)}, mean={np.mean(vals)*100:.2f}%, sd={np.std(vals)*100:.2f}%")

    # Condition summary table
    print("\n=== Condition summary ===")
    for c in sorted(data.keys()):
        v = np.array(data[c]) * 100
        print(f"  {c:<15} mean={v.mean():.2f}  sd={v.std():.2f}  n={len(v)}")

    # Hypothesis tests
    rows = []
    for hyp_id, hyp_name, cond_a, cond_b in HYPOTHESES:
        if cond_a not in data or cond_b not in data:
            print(f"  WARNING: missing data for {hyp_id} ({cond_a} or {cond_b})")
            continue
        a_vals = np.array(data[cond_a]) * 100
        b_vals = np.array(data[cond_b]) * 100
        delta  = b_vals.mean() - a_vals.mean()
        t_stat, p_welch = stats.ttest_ind(b_vals, a_vals, equal_var=False)
        _, p_mwu        = stats.mannwhitneyu(b_vals, a_vals, alternative='two-sided')
        wins = paired_wins(data[cond_a], data[cond_b])
        n    = min(len(data[cond_a]), len(data[cond_b]))
        sig  = '*' if p_welch < ALPHA_CORR else ''
        rows.append((hyp_id, hyp_name, cond_a, cond_b,
                     a_vals.mean(), a_vals.std(),
                     b_vals.mean(), b_vals.std(),
                     delta, t_stat, p_welch, p_mwu, wins, n, sig))

    # Print table to stdout
    print(f"\n=== Hypothesis tests (Bonferroni α = {ALPHA_CORR:.4f}) ===")
    header = f"{'Hyp':<4} {'Comparison':<30} {'μ_A':>7} {'μ_B':>7} {'Δ':>6} {'t':>6} {'p_welch':>10} {'p_mwu':>10} {'wins':>8}"
    print(header)
    print('-' * len(header))
    for row in rows:
        hyp_id, hyp_name, ca, cb, ma, sa, mb, sb, d, t, pw, pm, wins, n, sig = row
        print(f"{hyp_id:<4} {hyp_name:<30} {ma:>7.2f} {mb:>7.2f} {d:>+6.2f} "
              f"{t:>6.2f} {pw:>10.4f} {pm:>10.4f} {wins:>4}/{n}{sig}")

    # LaTeX output
    latex_lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption{CBT-NE results (30 seeds each; accuracy \%). '
        r'$\alpha_\text{corr}=0.0125$ (Bonferroni / 4 hypotheses). '
        r'* = significant after correction.}',
        r'  \label{tab:cbt_results}',
        r'  \begin{tabular}{llrrrrrrl}',
        r'    \toprule',
        r'    & & \multicolumn{2}{c}{Condition A} & \multicolumn{2}{c}{Condition B}'
        r' & & & \\',
        r'    \cmidrule(lr){3-4}\cmidrule(lr){5-6}',
        r'    Hyp. & Comparison & $\mu$ & $\sigma$ & $\mu$ & $\sigma$'
        r' & $\Delta$ & $t$ & $p$ (Welch / MWU) \\',
        r'    \midrule',
    ]
    for row in rows:
        hyp_id, hyp_name, ca, cb, ma, sa, mb, sb, d, t, pw, pm, wins, n, sig = row
        pw_str  = fmt_p(pw)
        pm_str  = fmt_p(pm)
        sig_str = r'$^*$' if sig else ''
        line = (
            f'    {hyp_id} & {hyp_name} & {ma:.2f} & {sa:.2f} & {mb:.2f} & {sb:.2f} '
            f'& ${d:+.2f}$ & ${t:.2f}$ & {pw_str} / {pm_str}{sig_str} \\\\'
        )
        latex_lines.append(line)
    latex_lines += [
        r'    \midrule',
    ]
    # Win-count sub-table
    latex_lines.append(r'    \multicolumn{9}{l}{\textit{Paired-seed win counts (B $>$ A / N seeds)}} \\')
    for row in rows:
        hyp_id, hyp_name, ca, cb, ma, sa, mb, sb, d, t, pw, pm, wins, n, sig = row
        latex_lines.append(f'    {hyp_id} & {hyp_name} & \\multicolumn{{7}}{{l}}{{{wins}/{n}}} \\\\')
    latex_lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'\end{table}',
    ]

    print("\n=== LaTeX output ===")
    print('\n'.join(latex_lines))

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results/cbt')
    parser.add_argument('--alpha',       type=float, default=0.05)
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} not found.")
        return

    analyse(args.results_dir, alpha=args.alpha)


if __name__ == '__main__':
    main()
