# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
FLA utility functions.

Porting notes:
  - Removed torch dependency, replaced with paddle
  - Removed dependency on sglang/transformers
  - Retained core logic of input_guard and tensor_cache decorators
  - is_gather_supported checks whether tl.gather is available
"""

import functools
import logging
import os
from functools import lru_cache
from typing import Any, Callable

import paddle
import triton
import triton.language as tl

logger = logging.getLogger(__name__)


# ============================================================
# Environment flags
# ============================================================

COMPILER_MODE = os.getenv("FLA_COMPILER_MODE") == "1"


@lru_cache(maxsize=None)
def get_available_device() -> str:
    try:
        return triton.runtime.driver.active.get_current_target().backend
    except BaseException:
        return "cuda"


@lru_cache(maxsize=None)
def get_multiprocessor_count(device_idx: int = 0) -> int:
    try:
        return triton.runtime.driver.active.utils.get_device_properties(device_idx)["multiprocessor_count"]
    except BaseException:
        return -1


# tl.gather availability check (Triton >= 3.2.0)
is_gather_supported: bool = hasattr(tl, "gather")


# ============================================================
# input_guard decorator: ensure all Tensors are contiguous
# ============================================================


def input_guard(fn: Callable) -> Callable:
    """
    Ensure all input Tensors are contiguous and run on the correct CUDA device.
    Ported from SGLang, removed torch dependency, replaced with paddle.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # make all Tensors contiguous
        contiguous_args = tuple(arg.contiguous() if isinstance(arg, paddle.Tensor) else arg for arg in args)
        contiguous_kwargs = {k: (v.contiguous() if isinstance(v, paddle.Tensor) else v) for k, v in kwargs.items()}
        return fn(*contiguous_args, **contiguous_kwargs)

    return wrapper


contiguous = input_guard


# ============================================================
# tensor_cache decorator: cache results of the last N calls
# ============================================================


def tensor_cache(fn: Callable) -> Callable:
    """
    Cache results of the last cache_size calls (matched by object identity).
    Suitable for idempotent functions such as shape computations.
    """
    cache_entries: list = []
    cache_size = 4

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        nonlocal cache_entries
        for i, entry in enumerate(cache_entries):
            last_args, last_kwargs, last_result = entry
            if len(args) == len(last_args) and len(kwargs) == len(last_kwargs):
                if all(a is b for a, b in zip(args, last_args)) and all(
                    k in last_kwargs and v is last_kwargs[k] for k, v in kwargs.items()
                ):
                    # LRU: move to end
                    cache_entries = cache_entries[:i] + cache_entries[i + 1 :] + [(args, kwargs, last_result)]
                    return last_result

        result = fn(*args, **kwargs)
        if len(cache_entries) >= cache_size:
            cache_entries = cache_entries[1:]
        cache_entries.append((args, kwargs, result))
        return result

    return wrapper
