"""
Prepare the BabyLM dataset for BPE language modeling.

Downloads BabyLM 10M from HuggingFace (nilq/babylm-10M), tokenises with the
GPT-2 BPE encoder (tiktoken), and writes train.bin / val.bin in the same
uint16 format used by openwebtext — so train.py works unchanged.

Each document is encoded and followed by the end-of-text token (50256) to
mark document boundaries.  A minimal meta.pkl is written containing only the
vocab_size so train.py picks it up automatically.

Usage:
    python data/babylm/prepare.py

Requires: pip install datasets tiktoken
"""

import os
import pickle
import numpy as np
import tiktoken
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")

print("Loading BabyLM 10M from HuggingFace (nilq/babylm-10M)...")
ds = load_dataset("nilq/babylm-10M")
print(f"Available splits: {list(ds.keys())}")


def tokenise(dataset_split):
    all_ids = []
    for ex in dataset_split:
        text = ex.get("text", "")
        if text:
            ids = enc.encode_ordinary(text)
            ids.append(enc.eot_token)   # document boundary
            all_ids.extend(ids)
    return np.array(all_ids, dtype=np.uint16)


print("Tokenising train split...")
train_ids = tokenise(ds["train"])

val_split = "validation" if "validation" in ds else "test"
print(f"Tokenising {val_split} split (capping at 10% of train)...")
val_ids_full = tokenise(ds[val_split])
val_ids = val_ids_full[: max(len(train_ids) // 10, 100_000)]

print(f"train tokens : {len(train_ids):,}")
print(f"val   tokens : {len(val_ids):,}")
print(f"vocab size   : {enc.max_token_value + 1}  (GPT-2 BPE)")

out_dir = os.path.dirname(__file__)
train_ids.tofile(os.path.join(out_dir, "train.bin"))
val_ids.tofile(os.path.join(out_dir, "val.bin"))

# minimal meta — no stoi/itos, train.py defaults to 50304 if vocab_size absent,
# but being explicit avoids any ambiguity
meta = {"vocab_size": 50304}
with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
    pickle.dump(meta, f)

print(f"Saved train.bin, val.bin, meta.pkl to {out_dir}")
