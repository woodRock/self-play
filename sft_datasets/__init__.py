"""
Dataset registry for SFT training.

Maps sft_dataset name → factory function:
    factory(data_dir, block_size, batch_size, device, device_type) -> get_batch

where get_batch(split) -> (X, Y, mask) tensors.
mask is a float {0,1} tensor with 1 at positions that contribute to the loss.
"""

_REGISTRY: dict = {}


def register(name: str):
    def decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_batch_factory(name: str):
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown sft_dataset: {name!r}. Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[name]


# Import loaders to trigger registration
from . import blimp               # noqa: F401, E402
from . import babylm_continuation # noqa: F401, E402
from . import cbt_ne              # noqa: F401, E402
