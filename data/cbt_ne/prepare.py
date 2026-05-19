"""
Prepare CBT-NE (Children's Book Test, Named Entities split) for SFT.

Downloads cam-cst/cbt (config "NE") from HuggingFace.
Produces train.jsonl, val.jsonl, test.jsonl, and meta.pkl.

Each JSONL line (train/val):
    {"prefix_ids": [...], "answer_ids": [...], "suffix_ids": [...]}

Test lines also include the answer word and candidate list:
    {"prefix_ids": [...], "answer_ids": [...], "suffix_ids": [...],
     "answer": "word", "candidates": ["w1", ..., "w10"]}

The full token sequence is prefix + answer + suffix; the SFT loss mask has
1 only at positions corresponding to answer tokens in the target.

Usage:
    python data/cbt_ne/prepare.py
"""

import os
import re
import json
import pickle
import tiktoken
from datasets import load_dataset

out_dir = os.path.dirname(os.path.abspath(__file__))
enc     = tiktoken.get_encoding("gpt2")
EOT     = enc.eot_token

# CBT uses XXXXX (five X's) as the blank marker; guard against variations.
BLANK_RE = re.compile(r'X{4,}|_XXXX_')


def find_blank(sentence):
    """Return (prefix_str, suffix_str) split at the blank, or None if not found."""
    m = BLANK_RE.search(sentence)
    if m is None:
        return None
    return sentence[:m.start()], sentence[m.end():]


def tokenize_example(ex, include_candidates=False):
    """
    Build (prefix_ids, answer_ids, suffix_ids) from a CBT-NE example.

    Handles both field-name variants seen in HuggingFace versions of CBT:
      - 'sentences' (list of 20 context strings) + 'question' (21st with blank)
      - 'sentences' (list of up to 21, last one may contain blank)
    """
    sentences = ex.get('sentences', [])
    question  = ex.get('question', '')
    answer    = ex.get('answer', '')
    # 'options' or 'candidates' depending on HF version
    candidates = ex.get('options', ex.get('candidates', []))

    # Normalise: separate context from question sentence
    if not question and sentences:
        # Some versions embed the question as the last sentence
        question  = sentences[-1]
        sentences = sentences[:-1]

    result = find_blank(question)
    if result is None:
        return None  # malformed example

    prefix_str, suffix_str = result

    # Build context text from up to 20 sentences
    context_text = ' '.join(str(s) for s in sentences[:20])
    if context_text:
        context_text = context_text + ' '

    full_prefix = context_text + prefix_str
    prefix_ids  = enc.encode_ordinary(full_prefix)
    answer_ids  = enc.encode_ordinary(answer)
    suffix_ids  = enc.encode_ordinary(suffix_str) + [EOT]

    out = {
        'prefix_ids': prefix_ids,
        'answer_ids': answer_ids,
        'suffix_ids': suffix_ids,
    }
    if include_candidates:
        out['answer']     = answer
        out['candidates'] = [str(c) for c in candidates]
    return out


def process_split(hf_split, split_name, include_candidates=False):
    examples = []
    skipped  = 0
    for ex in hf_split:
        result = tokenize_example(ex, include_candidates=include_candidates)
        if result is None:
            skipped += 1
            continue
        examples.append(result)
    print(f"  {split_name}: {len(examples)} examples ({skipped} skipped)")
    return examples


print("Downloading cam-cst/cbt (NE split) from HuggingFace…")
ds = load_dataset("cam-cst/cbt", "NE")
print(f"Splits: {list(ds.keys())}")

for hf_key, out_key, inc_cand in [
    ('train',      'train', False),
    ('validation', 'val',   False),
    ('test',       'test',  True),
]:
    split = ds.get(hf_key)
    if split is None:
        print(f"  WARNING: split {hf_key!r} not found, skipping.")
        continue
    examples = process_split(split, hf_key, include_candidates=inc_cand)
    path = os.path.join(out_dir, f'{out_key}.jsonl')
    with open(path, 'w') as f:
        for e in examples:
            f.write(json.dumps(e) + '\n')
    print(f"  wrote {path}")

meta = {'vocab_size': 50304}
with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("Done — meta.pkl written.")
