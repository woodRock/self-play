"""BabyLM continuation loader — identical pipeline to BLiMP, all-ones loss mask."""

import os
import numpy as np
import torch
from . import register


@register('babylm_continuation')
def make_get_batch(data_dir, block_size, batch_size, device, device_type):
    def get_batch(split):
        fname = 'train.bin' if split == 'train' else 'val.bin'
        data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        mask = torch.ones_like(y, dtype=torch.float)
        if device_type == 'cuda':
            x    = x.pin_memory().to(device, non_blocking=True)
            y    = y.pin_memory().to(device, non_blocking=True)
            mask = mask.pin_memory().to(device, non_blocking=True)
        else:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
        return x, y, mask
    return get_batch
