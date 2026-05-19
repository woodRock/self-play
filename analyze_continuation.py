#!/usr/bin/env python3
"""
Analyse generation-diversity experiment results and produce a LaTeX table.

Reads all JSON manifests from results/continuation/, groups by condition,
and runs the pre-registered statistical tests across 4 metrics × 4 hypotheses
(16 tests, Bonferroni α = 0.003125).

Usage:
    python analyze_continuation.py --results_dir results/continuation
"""

import os
import json
import argparse
import numpy as np
from scipy import stats

HYPOTHESES = [
    ('H_A', 'SP-pretrain → SFT-BL',   'BL_SFT_BL', 'SP_SFT_BL'),
    ('H_B', 'SP-SFT from BL-pretrain', 'BL_SFT_BL', 'BL_SFT_SP'),
    ('H_C', 'SP-SFT from SP-pretrain', 'SP_SFT_BL', 'SP_SFT_SP'),
    ('H_D', 'Full SP vs full BL',      'BL_SFT_BL', 'SP_SFT_SP'),
]

METRICS = [
    ('distinct_1',         'Distinct-1'),
    ('distinct_2',         'Distinct-2'),
    ('distinct_3',         'Distinct-3'),
    ('rep_rate',           'Rep.\ rate'),
    ('ttr',                'TTR'),
    ('mean_len_to_repeat', 'Mean len-to-rep.'),
]

N_TESTS      = len(HYPOTHESES) * len(METRICS)
ALPHA_FAMILY = 0.05
ALPHA_CORR   = ALPHA_FAMILY / N_TESTS   # 0.003125 (using all 4 primary metrics from spec)
# Note: the spec lists 4 metrics × 4 hypotheses = 16 tests → α = 0.003125


def load_results(results_dir):
    """Return dict: condition → {metric → list of values}"""
    data = {}
    for fname in os.listdir(results_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            m = json.load(f)
        cond = m.get('condition', fname.replace('.json', ''))
        data.setdefault(cond, {})
        seed = m.get('seed', 0)
        for mkey, _ in METRICS:
            val = m.get(mkey)
            if val is not None:
                data[cond].setdefault(mkey, []).append((seed, val))
    # Sort by seed
    return {
        c: {mk: [v for _, v in sorted(vals)] for mk, vals in mdict.items()}
        for c, mdict in data.items()
    }


def paired_wins(a_vals, b_vals):
    n = min(len(a_vals), len(b_vals))
    return sum(1 for i in range(n) if b_vals[i] > a_vals[i])


def fmt_p(p):
    if p < 1e-4:
        return r'$<$10$^{-4}$'
    elif p < 0.001:
        return f'{p:.4f}'
    else:
        return f'{p:.3f}'


def analyse(results_dir):
    data = load_results(results_dir)
    print(f"Loaded conditions: {sorted(data.keys())}")

    # Per-condition summary
    print("\n=== Condition × metric means ===")
    header = f"{'Condition':<15}" + "".join(f"  {mk:<22}" for mk, _ in METRICS)
    print(header)
    for cond in sorted(data.keys()):
        row = f"{cond:<15}"
        for mkey, _ in METRICS:
            vals = data[cond].get(mkey, [])
            if vals:
                row += f"  {np.mean(vals):.4f} (±{np.std(vals):.4f})"
            else:
                row += f"  {'—':<22}"
        print(row)

    # Hypothesis × metric test results
    all_rows = []
    print(f"\n=== Hypothesis tests (Bonferroni α = {ALPHA_CORR:.6f}) ===")
    for hyp_id, hyp_name, cond_a, cond_b in HYPOTHESES:
        if cond_a not in data or cond_b not in data:
            print(f"  WARNING: missing data for {hyp_id}")
            continue
        for mkey, mname in METRICS:
            a_vals = np.array(data[cond_a].get(mkey, []))
            b_vals = np.array(data[cond_b].get(mkey, []))
            if len(a_vals) == 0 or len(b_vals) == 0:
                continue
            delta  = b_vals.mean() - a_vals.mean()
            t_stat, p_welch = stats.ttest_ind(b_vals, a_vals, equal_var=False)
            _, p_mwu        = stats.mannwhitneyu(b_vals, a_vals, alternative='two-sided')
            wins = paired_wins(a_vals.tolist(), b_vals.tolist())
            n    = min(len(a_vals), len(b_vals))
            sig  = '*' if p_welch < ALPHA_CORR else ''
            all_rows.append((hyp_id, hyp_name, mkey, mname, cond_a, cond_b,
                             a_vals.mean(), a_vals.std(),
                             b_vals.mean(), b_vals.std(),
                             delta, t_stat, p_welch, p_mwu, wins, n, sig))
            print(f"  {hyp_id} / {mname:<22}: μ_A={a_vals.mean():.4f} μ_B={b_vals.mean():.4f} "
                  f"Δ={delta:+.4f} t={t_stat:.2f} p_welch={p_welch:.4f} p_mwu={p_mwu:.4f} "
                  f"wins={wins}/{n}{sig}")

    # LaTeX — one row per (hypothesis, metric)
    latex_lines = [
        r'\begin{table}[t]',
        r'  \centering',
        r'  \caption{Generation diversity results (30 seeds each). '
        r'$\alpha_\text{corr}=0.003125$ (Bonferroni / 16 tests = 4 hyp.\ $\times$ 4 metrics). '
        r'* = significant after correction. Higher is better for Distinct-$n$, TTR, Len-to-rep.; '
        r'lower is better for Rep.\ rate.}',
        r'  \label{tab:continuation_results}',
        r'  \resizebox{\textwidth}{!}{',
        r'  \begin{tabular}{llrrrrrrl}',
        r'    \toprule',
        r'    Hyp. & Metric & $\mu_A$ & $\sigma_A$ & $\mu_B$ & $\sigma_B$'
        r' & $\Delta$ & $t$ & $p$ (Welch / MWU) \\',
        r'    \midrule',
    ]

    prev_hyp = None
    for row in all_rows:
        hyp_id, hyp_name, mkey, mname, ca, cb, ma, sa, mb, sb, d, t, pw, pm, wins, n, sig = row
        if hyp_id != prev_hyp:
            if prev_hyp is not None:
                latex_lines.append(r'    \midrule')
            latex_lines.append(
                f'    \\multicolumn{{9}}{{l}}'
                f'{{\\textbf{{{hyp_id}}}: {hyp_name} ({ca} vs {cb})}} \\\\'
            )
            prev_hyp = hyp_id
        pw_str  = fmt_p(pw)
        pm_str  = fmt_p(pm)
        sig_str = r'$^*$' if sig else ''
        line = (
            f'    & {mname} & {ma:.4f} & {sa:.4f} & {mb:.4f} & {sb:.4f} '
            f'& ${d:+.4f}$ & ${t:.2f}$ & {pw_str} / {pm_str}{sig_str} \\\\'
        )
        latex_lines.append(line)

    # Win-count summary
    latex_lines += [
        r'    \midrule',
        r'    \multicolumn{9}{l}{\textit{Paired-seed win counts (B $>$ A / N)}} \\',
    ]
    for row in all_rows:
        hyp_id, hyp_name, mkey, mname, ca, cb, ma, sa, mb, sb, d, t, pw, pm, wins, n, sig = row
        latex_lines.append(
            f'    & {mname} & \\multicolumn{{7}}{{l}}{{{wins}/{n}}} \\\\'
        )

    latex_lines += [
        r'    \bottomrule',
        r'  \end{tabular}',
        r'  }',
        r'\end{table}',
    ]

    print("\n=== LaTeX output ===")
    print('\n'.join(latex_lines))

    return all_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', default='results/continuation')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"ERROR: {args.results_dir} not found.")
        return

    analyse(args.results_dir)


if __name__ == '__main__':
    main()
