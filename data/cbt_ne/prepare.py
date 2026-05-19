"""
Prepare CBT-NE (Children's Book Test, Named Entities split) for SFT.

Uses the HuggingFace Datasets Server REST API (pure HTTP + JSON) instead of
the `datasets` library to completely avoid pyarrow parquet decoding.  This
fixes SIGILL crashes on CPUs that lack AVX2/AVX-512 (common on GPU clusters).

Downloads ~108 K training examples in batches of 100; expect ~3-8 minutes
on a typical cluster with internet access.

Produces:  train.jsonl, val.jsonl, test.jsonl, meta.pkl

Usage:
    python data/cbt_ne/prepare.py
    python data/cbt_ne/prepare.py --hf_token YOUR_TOKEN   # higher rate limits
"""

import os
import re
import sys
import json
import time
import pickle
import argparse

import requests
import tiktoken

out_dir = os.path.dirname(os.path.abspath(__file__))
enc     = tiktoken.get_encoding("gpt2")
EOT     = enc.eot_token

BLANK_RE  = re.compile(r'X{4,}')
API_BASE  = "https://datasets-server.huggingface.co/rows"
DATASET   = "cam-cst/cbt"
CONFIG    = "NE"
BATCH     = 100   # max rows per API request


# ---------------------------------------------------------------------------
# HTTP fetch with retry
# ---------------------------------------------------------------------------

def _get(session, params, token=None):
    headers = {}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    for attempt in range(6):
        try:
            r = session.get(API_BASE, params=params, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == 5:
                raise RuntimeError(f"API request failed after 6 attempts: {exc}") from exc
            wait = 2 ** attempt
            print(f"    Retrying in {wait}s ({exc})…", flush=True)
            time.sleep(wait)


def fetch_split(split_name, token=None):
    """Fetch all rows of a split from the HF Datasets Server as plain dicts."""
    session = requests.Session()
    rows    = []
    offset  = 0

    # First request tells us the total row count
    data = _get(session, {
        'dataset': DATASET, 'config': CONFIG,
        'split': split_name, 'offset': 0, 'length': 1,
    }, token=token)
    total = data.get('num_rows_total', '?')
    print(f"  {split_name}: {total} rows total", flush=True)

    while True:
        data = _get(session, {
            'dataset': DATASET, 'config': CONFIG,
            'split': split_name,
            'offset': offset, 'length': BATCH,
        }, token=token)
        batch = [item['row'] for item in data['rows']]
        rows.extend(batch)

        if len(batch) < BATCH:
            break   # final page
        offset += BATCH
        if offset % 5000 == 0:
            print(f"    {split_name}: {offset}/{total} fetched…", flush=True)
        time.sleep(0.05)   # be polite; HF rate-limits anonymous callers

    print(f"    done — {len(rows)} rows fetched", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def find_blank(sentence):
    m = BLANK_RE.search(sentence)
    if m is None:
        return None
    return sentence[:m.start()], sentence[m.end():]


def tokenize_row(row, include_candidates=False):
    sentences  = row.get('sentences', [])
    question   = row.get('question', '')
    answer     = row.get('answer', '')
    candidates = row.get('options', row.get('candidates', []))

    result = find_blank(question)
    if result is None:
        return None

    prefix_str, suffix_str = result

    context_text = ' '.join(str(s) for s in sentences[:20])
    if context_text:
        context_text += ' '

    prefix_ids = enc.encode_ordinary(context_text + prefix_str)
    answer_ids = enc.encode_ordinary(answer)
    suffix_ids = enc.encode_ordinary(suffix_str) + [EOT]

    out = {
        'prefix_ids': prefix_ids,
        'answer_ids': answer_ids,
        'suffix_ids': suffix_ids,
    }
    if include_candidates:
        out['answer']     = answer
        out['candidates'] = [str(c) for c in candidates]
    return out


def process_and_save(rows, out_name, include_candidates=False):
    examples = []
    skipped  = 0
    for row in rows:
        ex = tokenize_row(row, include_candidates=include_candidates)
        if ex is None:
            skipped += 1
            continue
        examples.append(ex)
    print(f"  {out_name}: {len(examples)} examples tokenised ({skipped} skipped)")
    path = os.path.join(out_dir, f'{out_name}.jsonl')
    with open(path, 'w') as f:
        for e in examples:
            f.write(json.dumps(e) + '\n')
    print(f"  wrote {path}")
    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_token', default=None,
                        help='HuggingFace token for higher rate limits (optional)')
    args = parser.parse_args()
    token = args.hf_token or os.environ.get('HF_TOKEN')

    if token:
        print("Using HuggingFace token (higher rate limits).")

    print("Fetching CBT-NE from HuggingFace Datasets Server (no pyarrow)…")
    t0 = time.time()

    for hf_split, out_name, inc_cand in [
        ('train',      'train', False),
        ('validation', 'val',   False),
        ('test',       'test',  True),
    ]:
        rows = fetch_split(hf_split, token=token)
        process_and_save(rows, out_name, include_candidates=inc_cand)

    with open(os.path.join(out_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump({'vocab_size': 50304}, f)

    print(f"\nDone in {time.time()-t0:.0f}s — meta.pkl written.")


if __name__ == '__main__':
    main()
