"""
Prepare the BabyLM dataset for character-level language modeling.

Downloads the BabyLM 10M corpus from HuggingFace (nilq/babylm-10M) and produces
train.bin / val.bin / meta.pkl in the same format as shakespeare_char,
so the same training scripts can be used unchanged.

BabyLM is a ~10M word corpus of child-directed and child-produced language
(CHILDES, children's books, simple Wikipedia, subtitles, etc.).  It is a much
richer pretraining signal than Tiny Shakespeare.

Usage:
    python data/babylm/prepare.py

Requires: pip install datasets
"""

import os
import pickle
import numpy as np
from datasets import load_dataset

print("Loading BabyLM 10M from HuggingFace (nilq/babylm-10M)...")
ds = load_dataset('nilq/babylm-10M')

print(f"Available splits: {list(ds.keys())}")

PRINTABLE_ASCII = set(chr(i) for i in range(32, 127)) | {'\n', '\t'}

def extract_text(dataset_split):
    # Filter to printable ASCII to keep vocab manageable (~95 chars vs 2500+)
    return '\n'.join(
        ''.join(c for c in ex['text'] if c in PRINTABLE_ASCII)
        for ex in dataset_split if ex.get('text')
    )

train_text = extract_text(ds['train'])
val_split  = 'validation' if 'validation' in ds else 'test'
# Cap val at ~10% of train length so evaluation stays fast
val_raw    = extract_text(ds[val_split])
val_text   = val_raw[:max(len(train_text) // 10, 100_000)]

print(f"train text length: {len(train_text):,} characters")
print(f"val   text length: {len(val_text):,} characters")

# Build character-level vocabulary (same approach as shakespeare_char).
chars = sorted(set(train_text + val_text))
vocab_size = len(chars)
print(f"vocab size: {vocab_size}")
print(f"all unique characters: {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s if c in stoi]

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids   = np.array(encode(val_text),   dtype=np.uint16)
print(f"train tokens: {len(train_ids):,}")
print(f"val   tokens: {len(val_ids):,}")

out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, 'train.bin'))
val_ids.tofile(  os.path.join(out_dir, 'val.bin'))

meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print(f"Saved train.bin, val.bin, meta.pkl to {out_dir}")
