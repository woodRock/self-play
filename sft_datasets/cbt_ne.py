"""
CBT-NE dataset loader.

Reads train.jsonl / val.jsonl produced by data/cbt_ne/prepare.py.
Each line: {"prefix_ids": [...], "answer_ids": [...], "suffix_ids": [...]}

Loss mask: 1 at positions in the target (y) that correspond to answer tokens,
0 everywhere else.  The loss measures only slot-filling accuracy.
"""

import os
import json
import random
import torch
from . import register

PAD_ID = 50256  # GPT-2 EOT token reused as padding


def _load(path):
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _build_tensors(ex, block_size):
    """
    Returns (x, y, mask) lists of length block_size.

    Full token sequence: prefix + answer + suffix.
    We truncate from the left (removing prefix tokens) to fit in block_size+1,
    always preserving the answer and as much suffix as possible.
    Pads at the right with PAD_ID.

    mask[i] = 1 iff y[i] (= tokens[i+1]) is part of the answer.
    """
    prefix = list(ex['prefix_ids'])
    answer = list(ex['answer_ids'])
    suffix = list(ex['suffix_ids'])

    max_tokens = block_size + 1  # need +1 to produce x and y of length block_size

    # Truncate prefix from left if total overflows
    total = len(prefix) + len(answer) + len(suffix)
    if total > max_tokens:
        drop = total - max_tokens
        prefix = prefix[drop:]

    # If answer+suffix alone still overflows, trim suffix
    total = len(prefix) + len(answer) + len(suffix)
    if total > max_tokens:
        suffix = suffix[:max_tokens - len(prefix) - len(answer)]

    tokens = prefix + answer + suffix
    # Pad to max_tokens
    tokens += [PAD_ID] * (max_tokens - len(tokens))

    x = tokens[:block_size]
    y = tokens[1:block_size + 1]

    # Compute loss mask.
    # answer occupies tokens[p : p+a]; in y (= tokens[1:]), these are at
    # positions y[p-1] ... y[p+a-2], i.e. indices p-1 ... p+a-2.
    p = len(prefix)
    a = len(answer)
    mask = [0] * block_size
    start = max(0, p - 1)   # first answer index in y
    end   = min(block_size, p - 1 + a)  # exclusive upper bound
    for i in range(start, end):
        mask[i] = 1

    return x, y, mask


@register('cbt_ne')
def make_get_batch(data_dir, block_size, batch_size, device, device_type):
    examples = {
        split: _load(os.path.join(data_dir, f'{split}.jsonl'))
        for split in ('train', 'val')
    }

    def get_batch(split):
        pool  = examples[split]
        batch = random.choices(pool, k=batch_size)
        xs, ys, masks = [], [], []
        for ex in batch:
            x, y, m = _build_tensors(ex, block_size)
            xs.append(torch.tensor(x,  dtype=torch.long))
            ys.append(torch.tensor(y,  dtype=torch.long))
            masks.append(torch.tensor(m, dtype=torch.float))

        X    = torch.stack(xs)
        Y    = torch.stack(ys)
        mask = torch.stack(masks)

        if device_type == 'cuda':
            X    = X.pin_memory().to(device, non_blocking=True)
            Y    = Y.pin_memory().to(device, non_blocking=True)
            mask = mask.pin_memory().to(device, non_blocking=True)
        else:
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
        return X, Y, mask

    return get_batch
