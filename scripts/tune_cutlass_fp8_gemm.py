# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""UT for cutlass_fp8_fp8_half_gemm_fused"""
import paddle

from fastdeploy.utils import llm_logger as logger


def tune_cutlass_fp8_fp8_half_gemm_fused(
    ns: list,
    ks: list,
    m_min: int = 32,
    m_max: int = 32768,
):
    """
    Tune fp8 gemm.
    """
    assert len(ns) == len(ks), "The length of `ns` must be equal to that of `ks`"
    try:
        from fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_gemm_fused
    except ImportError:
        logger.warning(
            "From fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_gemm_fused failed, \
            fp8 is only support cuda arch 89+."
        )
        return
    paddle.seed(2003)
    for m in range(m_min, m_max + 32, 32):
        if m > m_max:
            break
        for idx in range(len(ns)):
            n = ns[idx]
            k = ks[idx]
            A = paddle.rand(shape=[m, k], dtype="bfloat16").astype("float8_e4m3fn")
            B = paddle.rand(shape=[n, k], dtype="bfloat16").astype("float8_e4m3fn")
            cutlass_fp8_fp8_half_gemm_fused(
                A,
                B,
                bias=None,
                transpose_x=False,
                transpose_y=True,
                output_dtype="bfloat16",
                scale=0.5,
                activation_type="identity",
            )
            paddle.device.cuda.empty_cache()


def tune_cutlass_fp8_fp8_fp8_dual_gemm_fused(
    ns: list,
    ks: list,
    m_min: int = 32,
    m_max: int = 32768,
):
    """
    Tune fp8 dual-gemm.
    """
    assert len(ns) == len(ks), "The length of `ns` must be equal to that of `ks`"
    try:
        from fastdeploy.model_executor.ops.gpu import (
            cutlass_fp8_fp8_fp8_dual_gemm_fused,
        )
    except ImportError:
        logger.warning(
            "From fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_fp8_dual_gemm_fused failed, \
            fp8 is only support cuda arch 89+."
        )
        return
    paddle.seed(2003)
    for m in range(m_min, m_max + 32, 32):
        if m > m_max:
            break
        for idx in range(len(ns)):
            n = ns[idx]
            k = ks[idx]
            A = paddle.rand(shape=[m, k], dtype="bfloat16").astype("float8_e4m3fn")
            B0 = paddle.rand(shape=[n, k], dtype="bfloat16").astype("float8_e4m3fn")
            B1 = paddle.rand(shape=[n, k], dtype="bfloat16").astype("float8_e4m3fn")
            cutlass_fp8_fp8_fp8_dual_gemm_fused(
                A,
                B0,
                B1,
                bias0=None,
                bias1=None,
                transpose_x=False,
                transpose_y=True,
                scale0=0.1,
                scale1=0.1,
                scale_out=0.5,
                activation_type="swiglu",
            )
            paddle.device.cuda.empty_cache()


def tune_per_channel_fp8_gemm_fused(
    ns: list,
    ks: list,
    m_min: int = 32,
    m_max: int = 32768,
):
    """
    Tune per-channel quant gemm.
    """
    assert len(ns) == len(ks), "The length of `ns` must be equal to that of `ks`"
    try:
        from fastdeploy.model_executor.ops.gpu import (
            per_channel_fp8_fp8_half_gemm_fused,
        )
    except ImportError:
        logger.warning(
            "From fastdeploy.model_executor.ops.gpu import per_channel_fp8_fp8_half_gemm_fused failed, \
            fp8 is only support cuda arch 89+."
        )
        return
    paddle.seed(2003)
    for m in range(m_min, m_max + 32, 32):
        if m > m_max:
            break
        for idx in range(len(ns)):
            n = ns[idx]
            k = ks[idx]
            A = paddle.rand(shape=[m, k], dtype="bfloat16").astype("float8_e4m3fn")
            B = paddle.rand(shape=[n, k], dtype="bfloat16").astype("float8_e4m3fn")
            scalar_scale = paddle.full([1], 0.168, dtype="float32")
            channel_scale = paddle.rand(shape=[n], dtype="float32")

            per_channel_fp8_fp8_half_gemm_fused(
                A,
                B,
                bias=None,
                scalar_scale=scalar_scale,
                channel_scale=channel_scale,
                transpose_x=False,
                transpose_y=True,
                output_dtype="bfloat16",
            )
            paddle.device.cuda.empty_cache()


def tune_blockwise_fp8_gemm_fused(
    ns: list,
    ks: list,
    m_min: int = 32,
    m_max: int = 32768,
):
    """
    Tune per-channel quant gemm.
    """
    assert len(ns) == len(ks), "The length of `ns` must be equal to that of `ks`"
    try:
        from fastdeploy.model_executor.ops.gpu import (
            cutlass_fp8_fp8_half_block_gemm_fused,
        )
    except ImportError:
        logger.warning(
            "From fastdeploy.model_executor.ops.gpu import cutlass_fp8_fp8_half_block_gemm_fused failed, \
            fp8 is only support cuda arch 90+."
        )
        return
    paddle.seed(2003)
    for m in range(m_min, m_max + 32, 32):
        if m > m_max:
            break
        for idx in range(len(ns)):
            n = ns[idx]
            k = ks[idx]
            scale_n = (n + 128 - 1) // 128
            scale_k = (k + 128 - 1) // 128
            A = paddle.rand(shape=[m, k], dtype="bfloat16").astype("float8_e4m3fn")
            B = paddle.rand(shape=[n, k], dtype="bfloat16").astype("float8_e4m3fn")
            a_scale = paddle.randn([scale_k, m], dtype="float32")
            b_scale = paddle.randn([scale_n, scale_k], dtype="float32")

            cutlass_fp8_fp8_half_block_gemm_fused(
                A,
                B,
                x_sacle=a_scale,
                y_sacle=b_scale,
                bias=None,
                transpose_x=False,
                transpose_y=True,
                output_dtype="bfloat16",
            )
            paddle.device.cuda.empty_cache()
