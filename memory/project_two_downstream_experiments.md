---
name: project-two-downstream-experiments
description: CBT-NE slot-filling and BabyLM continuation diversity experiments added to the self-play pretraining paper
metadata:
  type: project
---

Two new downstream experiments added to test whether the rare-token accuracy effect from self-play pretraining (+0.14pp, p<1e-4) transfers.

**Experiment 1 — CBT-NE slot filling**: 2×2 design (BL/SP pretrain × BL/SP SFT), 30 seeds = 120 runs. Uses answer-only loss mask. Pre-registered hypotheses H_A through H_D; Bonferroni α=0.0125 across 4 tests.

**Experiment 2 — BabyLM continuation diversity**: Same 2×2 design. Metrics: distinct-1/2/3, rep-rate, TTR, mean-len-to-repeat. Bonferroni α=0.003125 across 16 tests (4 hyp × 4 metrics).

**How to apply:** When discussing downstream results or if user asks about the paper's new experiments, these are the two additions. Results will come from cluster runs, not local.

**Key files created:**
- sft_datasets/ — dataset registry (blimp, babylm_continuation, cbt_ne)
- train_sft.py, train_selfplay_sft.py — refactored to be dataset-agnostic
- data/cbt_ne/prepare.py — downloads cam-cst/cbt NE split from HuggingFace
- eval_cbt.py, eval_generation.py — evaluation scripts
- run_cbt_experiment.py, run_continuation_experiment.py — cluster orchestration
- analyze_cbt.py, analyze_continuation.py — LaTeX table generators
- config/finetune_{cbt,selfplay_cbt,continuation,selfplay_continuation}.py

**Why:** BLiMP null result explained by mechanism (discriminative tokens are high-freq function words). CBT-NE tests rare-by-construction tokens; continuation tests repetition pathology.
