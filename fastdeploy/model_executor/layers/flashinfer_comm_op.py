# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Python glue layer for the flashinfer trtllm_allreduce_residual_rmsnorm op.

Loads trtllm_comm.so via tvm_ffi.load_module, retrieves the
trtllm_allreduce_fusion function from the TVM module, then wraps the kernel
call with @register_custom_python_op so that Paddle SOT treats it as a
single opaque op node (no tracing into the function body).
"""

from typing import Tuple

import paddle

from fastdeploy.utils import get_logger, register_custom_python_op

logger = get_logger("flashinfer", "flashinfer.log")

# ---------------------------------------------------------------------------
# Lazy init: load trtllm_comm.so via tvm_ffi.load_module()
# ---------------------------------------------------------------------------

_initialized = False
_trtllm_fn = None  # tvm_ffi.core.Function for trtllm_allreduce_fusion


def _ensure_trtllm_so_loaded():
    global _initialized, _trtllm_fn
    if _initialized:
        return True

    try:
        import paddle as _paddle
        import tvm_ffi

        with _paddle.use_compat_guard(enable=True, scope={"flashinfer"}):
            from flashinfer.jit import env as jit_env
            from flashinfer.jit.comm import gen_trtllm_comm_module

            so_path = jit_env.FLASHINFER_JIT_DIR / "trtllm_comm" / "trtllm_comm.so"
            if not so_path.exists():
                from flashinfer.jit.core import build_jit_specs

                build_jit_specs([gen_trtllm_comm_module()])

        mod = tvm_ffi.load_module(str(so_path))
        _trtllm_fn = mod["trtllm_allreduce_fusion"]

        _initialized = True
        logger.info("flashinfer_comm_op: trtllm_comm.so loaded via tvm_ffi.")
    except Exception as e:
        logger.warning(f"flashinfer_comm_op: init failed: {e}")
        return False

    return True


# ---------------------------------------------------------------------------
# infer_meta for register_custom_python_op
# ---------------------------------------------------------------------------


def _trtllm_ar_rmsnorm_infer_meta(
    input_tensor: "paddle.static.MetaTensor",
    residual: "paddle.static.MetaTensor",
    weight: "paddle.static.MetaTensor",
    workspace_ptrs: "paddle.static.MetaTensor",
    world_size: int,
    world_rank: int,
    use_oneshot: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    rms_eps: float,
) -> Tuple["paddle.static.MetaTensor", "paddle.static.MetaTensor"]:
    norm_out = paddle.static.MetaTensor(shape=input_tensor.shape, dtype=input_tensor.dtype)
    residual_out = paddle.static.MetaTensor(shape=residual.shape, dtype=residual.dtype)
    return norm_out, residual_out


# ---------------------------------------------------------------------------
# @register_custom_python_op wrapper
#
# SOT sees this as a single opaque Paddle op and calls _infer_meta for shape
# inference; the function body is never traced.
# ---------------------------------------------------------------------------

# AllReduceFusionPattern::kARResidualRMSNorm == 1
_kARResidualRMSNorm = 1


@register_custom_python_op(
    name="trtllm_allreduce_residual_rmsnorm",
    infer_meta=_trtllm_ar_rmsnorm_infer_meta,
    input_names=["input_tensor", "residual", "weight", "workspace_ptrs"],
    output_names=["norm_out", "residual_out"],
    inplace_map={},
)
def trtllm_allreduce_residual_rmsnorm_op(
    input_tensor: paddle.Tensor,
    residual: paddle.Tensor,
    weight: paddle.Tensor,
    workspace_ptrs: paddle.Tensor,
    world_size: int,
    world_rank: int,
    use_oneshot: bool,
    trigger_completion_at_end: bool,
    fp32_acc: bool,
    rms_eps: float,
) -> Tuple[paddle.Tensor, paddle.Tensor]:
    """Fused allreduce + residual add + RMSNorm via FlashInfer trtllm C++ kernel.

    This function is opaque to Paddle SOT — _trtllm_fn is a tvm_ffi.core.Function
    (C++ kernel), called directly without any Python-level tensor operations that
    SOT would trace.
    """
    if not _ensure_trtllm_so_loaded():
        return None, None

    token_num = input_tensor.shape[0]
    hidden_dim = input_tensor.shape[1]

    norm_out = paddle.empty_like(input_tensor)
    residual_out = paddle.empty_like(residual)

    if token_num == 0:
        return norm_out, residual_out

    _trtllm_fn(
        input_tensor,
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptrs,
        True,  # launch_with_pdl
        use_oneshot,
        trigger_completion_at_end,
        fp32_acc,
        _kARResidualRMSNorm,
        None,  # allreduce_out
        residual,  # residual_in
        residual_out,  # residual_out
        norm_out,  # norm_out
        None,  # quant_out
        None,  # scale_out
        weight,  # rms_gamma
        rms_eps,
        None,  # scale_factor
        None,  # layout_code
    )
    return norm_out, residual_out
