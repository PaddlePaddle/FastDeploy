# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/utils/op.py
# Original: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang (MIT License)
# Adapted for FastDeploy (PaddlePaddle) by PaddlePaddle Authors, 2025.
"""
FLA base Triton operation helpers.

Porting notes:
  - Triton kernel code is unchanged (pure GPU instructions, independent of torch/paddle)
  - Only removed dependency on sglang, replaced with local utils
"""

import os

import triton
import triton.language as tl

try:
    import triton.language.extra.libdevice as tldevice

    _HAS_LIBDEVICE = True
except ImportError:
    _HAS_LIBDEVICE = False

from fastdeploy.model_executor.ops.triton_ops.fla.utils import is_gather_supported

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1" and _HAS_LIBDEVICE:
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


@triton.jit
def safe_exp(x):
    return exp(tl.where(x <= 0, x, float("-inf")))


if not is_gather_supported:

    @triton.jit
    def gather(src, index, axis, _builder=None):
        """
        Fallback: placeholder when tl.gather is unavailable.
        """
        return None

else:
    gather = tl.gather
