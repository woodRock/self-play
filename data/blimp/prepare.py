#!/usr/bin/env python3
"""
Prepare BLiMP (Benchmark of Linguistic Minimal Pairs) for SFT.

Downloads all 67 paradigms from nyu-mll/blimp on HuggingFace.
Produces:
  train.bin         — tokenised grammatical sentences (90% split)
  val.bin           — tokenised grammatical sentences (10% split)
  eval_pairs.jsonl  — all (good, bad) token-id pairs for BLiMP evaluation
  meta.pkl          — vocab_size
"""
import os
import json
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset, get_dataset_config_names

out_dir = os.path.dirname(os.path.abspath(__file__))
enc     = tiktoken.get_encoding("gpt2")
EOT     = enc.eot_token

paradigms = get_dataset_config_names("nyu-mll/blimp")
print(f"Found {len(paradigms)} BLiMP paradigms.")

good_tokens_all = []
eval_pairs      = []

for i, paradigm in enumerate(paradigms):
    print(f"  [{i+1:2d}/{len(paradigms)}] {paradigm}", flush=True)
    ds = load_dataset("nyu-mll/blimp", paradigm, split="train")
    for ex in ds:
        good_ids = enc.encode_ordinary(ex["sentence_good"])
        bad_ids  = enc.encode_ordinary(ex["sentence_bad"])
        good_tokens_all.extend(good_ids + [EOT])
        eval_pairs.append({
            "uid":              ex["UID"],
            "field":            ex["field"],
            "linguistics_term": ex.get("linguistics_term", ""),
            "sentence_good":    ex["sentence_good"],
            "sentence_bad":     ex["sentence_bad"],
            "good_tokens":      good_ids,
            "bad_tokens":       bad_ids,
        })

print(f"\nTotal grammatical tokens : {len(good_tokens_all):,}")
print(f"Total pairs              : {len(eval_pairs):,}")

# 90/10 train/val split
n     = len(good_tokens_all)
split = int(n * 0.9)
np.array(good_tokens_all[:split], dtype=np.uint16).tofile(os.path.join(out_dir, 'train.bin'))
np.array(good_tokens_all[split:], dtype=np.uint16).tofile(os.path.join(out_dir, 'val.bin'))
print(f"train.bin : {split:,} tokens")
print(f"val.bin   : {n - split:,} tokens")

with open(os.path.join(out_dir, 'eval_pairs.jsonl'), 'w') as f:
    for pair in eval_pairs:
        f.write(json.dumps(pair) + '\n')
print(f"eval_pairs.jsonl : {len(eval_pairs):,} pairs")

with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': 50304}, f)
print("Done.")
