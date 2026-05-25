"""
Generate aggregated loss curves (baseline vs self-play, n=30 seeds) with
95 % confidence intervals of the mean (±1.96 SEM).  Saved to
loss_curves_aggregated.png, styled to match loss_curves.png.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

# ── colour palette ────────────────────────────────────────────────────────────
BL_COLOR = 'steelblue'
SP_COLOR = 'darkorange'

SEEDS = list(range(1, 31))
BASELINE_BASE  = 'outputs/babylm/baseline'
SELFPLAY_BASE  = 'outputs/babylm/selfplay'
OUTPUT_FILE    = 'results/loss_curves_aggregated.png'


# ── data loading ──────────────────────────────────────────────────────────────

def load_loss_log(out_dir):
    path = os.path.join(out_dir, 'loss_log.jsonl')
    if not os.path.exists(path):
        return [], [], [], []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    # keep only the last run (detect where iter resets)
    last_start = 0
    prev = -1
    for i, e in enumerate(entries):
        if e['iter'] < prev:
            last_start = i
        prev = e['iter']
    entries = entries[last_start:]
    ti, tl, vi, vl = [], [], [], []
    for e in entries:
        if e['type'] == 'train':
            ti.append(e['iter'])
            tl.append(e['loss'])
        elif e['type'] == 'val' and e['iter'] > 0:
            vi.append(e['iter'])
            vl.append(e['val_loss'])
    return ti, tl, vi, vl


def collect(base, seeds):
    """Return (iters, per-seed 2-D arrays) for train and val."""
    train_iters = val_iters = None
    all_train, all_val = [], []
    for s in seeds:
        ti, tl, vi, vl = load_loss_log(f"{base}-s{s}")
        if not tl or not vl:
            continue
        if train_iters is None or len(ti) < len(train_iters):
            train_iters = ti
        if val_iters is None or len(vi) < len(val_iters):
            val_iters = vi
        all_train.append(tl)
        all_val.append(vl)

    n_t = len(train_iters)
    n_v = len(val_iters)
    train_arr = np.array([v[:n_t] for v in all_train if len(v) >= n_t])
    val_arr   = np.array([v[:n_v] for v in all_val   if len(v) >= n_v])
    return train_iters, train_arr, val_iters, val_arr


def ci95(arr):
    """Return (mean, lower, upper) using the t-distribution 95 % CI."""
    n   = arr.shape[0]
    m   = arr.mean(axis=0)
    sem = arr.std(axis=0, ddof=1) / np.sqrt(n)
    tcrit = t_dist.ppf(0.975, df=n - 1)
    return m, m - tcrit * sem, m + tcrit * sem


# ── smoothing ─────────────────────────────────────────────────────────────────

def smooth(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(float(np.mean(values[lo:hi])))
    return np.array(out)


# ── plotting ──────────────────────────────────────────────────────────────────

def make_figure(seeds, baseline_base, selfplay_base, output_file):
    bl_ti, bl_train, bl_vi, bl_val = collect(baseline_base, seeds)
    sp_ti, sp_train, sp_vi, sp_val = collect(selfplay_base, seeds)

    bl_tm, bl_tlo, bl_thi = ci95(bl_train)
    sp_tm, sp_tlo, sp_thi = ci95(sp_train)
    bl_vm, bl_vlo, bl_vhi = ci95(bl_val)
    sp_vm, sp_vlo, sp_vhi = ci95(sp_val)

    n = len(seeds)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        f'Baseline vs Self-Play: BabyLM BPE  (n={n} seeds)',
        fontsize=14, fontweight='bold'
    )

    # ── panel 1 · training loss (mean ± 95 % CI, smoothed) ───────────────────
    ax = axes[0]
    s_bl = smooth(bl_tm)
    s_sp = smooth(sp_tm)
    s_bllo = smooth(bl_tlo)
    s_blhi = smooth(bl_thi)
    s_splo = smooth(sp_tlo)
    s_sphi = smooth(sp_thi)

    off_bl = len(bl_ti) - len(s_bl)
    off_sp = len(sp_ti) - len(s_sp)
    ax.plot(bl_ti[off_bl:], s_bl,  color=BL_COLOR, lw=2,   label='Baseline train')
    ax.fill_between(bl_ti[off_bl:], s_bllo, s_blhi, color=BL_COLOR, alpha=0.20)
    ax.plot(sp_ti[off_sp:], s_sp,  color=SP_COLOR, lw=2, ls='--', label='Self-play train')
    ax.fill_between(sp_ti[off_sp:], s_splo, s_sphi, color=SP_COLOR, alpha=0.20)

    ax.set_xlabel('Iteration');  ax.set_ylabel('Loss')
    ax.set_title('Training Loss (mean ± 95 % CI, smoothed)')
    ax.set_ylim(bottom=3.5, top=7.0)
    ax.legend();  ax.grid(True, alpha=0.3)

    # ── panel 2 · full validation loss (mean ± 95 % CI) ─────────────────────
    ax = axes[1]
    ax.plot(bl_vi, bl_vm, color=BL_COLOR, lw=2,          label='Baseline val')
    ax.fill_between(bl_vi, bl_vlo, bl_vhi, color=BL_COLOR, alpha=0.20)
    ax.plot(sp_vi, sp_vm, color=SP_COLOR, lw=2, ls='--', label='Self-play val')
    ax.fill_between(sp_vi, sp_vlo, sp_vhi, color=SP_COLOR, alpha=0.20)

    ax.set_xlabel('Iteration');  ax.set_ylabel('Loss')
    ax.set_title('Validation Loss (mean ± 95 % CI)')
    ax.legend();  ax.grid(True, alpha=0.3)

    # ── panel 3 · zoomed final 25 %, with annotations ────────────────────────
    ax = axes[2]
    all_iters = list(bl_vi) + list(sp_vi)
    zoom_start = max(all_iters) * 0.75

    def zoom_band(vi, m, lo, hi):
        triples = [(i, a, b, c) for i, a, b, c in zip(vi, m, lo, hi) if i >= zoom_start]
        if not triples:
            return [], np.array([]), np.array([]), np.array([])
        zi  = [t[0] for t in triples]
        zm  = np.array([t[1] for t in triples])
        zlo = np.array([t[2] for t in triples])
        zhi = np.array([t[3] for t in triples])
        return zi, zm, zlo, zhi

    bzi, bzm, bzlo, bzhi = zoom_band(bl_vi, bl_vm, bl_vlo, bl_vhi)
    szi, szm, szlo, szhi = zoom_band(sp_vi, sp_vm, sp_vlo, sp_vhi)

    if bzi:
        ax.plot(bzi, bzm, color=BL_COLOR, lw=2, marker='o', ms=3, label='Baseline val')
        ax.fill_between(bzi, bzlo, bzhi, color=BL_COLOR, alpha=0.25)
        ax.annotate(f"{bzm[-1]:.4f}", xy=(bzi[-1], bzm[-1]),
                    xytext=(8, 4), textcoords='offset points',
                    color=BL_COLOR, fontsize=9, fontweight='bold')
    if szi:
        ax.plot(szi, szm, color=SP_COLOR, lw=2, marker='s', ms=3, ls='--', label='Self-play val')
        ax.fill_between(szi, szlo, szhi, color=SP_COLOR, alpha=0.25)
        ax.annotate(f"{szm[-1]:.4f}", xy=(szi[-1], szm[-1]),
                    xytext=(8, -12), textcoords='offset points',
                    color=SP_COLOR, fontsize=9, fontweight='bold')

    ax.set_xlabel('Iteration');  ax.set_ylabel('Loss')
    ax.set_title('Validation Loss — final 25 % (mean ± 95 % CI)')
    ax.legend();  ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}  (n={n} seeds)")

    if len(bl_vm) and len(sp_vm):
        bl_f, sp_f = bl_vm[-1], sp_vm[-1]
        print(f"  Baseline  final val loss (mean): {bl_f:.4f}")
        print(f"  Self-play final val loss (mean): {sp_f:.4f}")
        print(f"  Self-play advantage:             {bl_f - sp_f:+.5f}")


if __name__ == '__main__':
    make_figure(SEEDS, BASELINE_BASE, SELFPLAY_BASE, OUTPUT_FILE)
