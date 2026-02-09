"""
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
"""

import paddle
from paddleformers.utils.log import logger

from fastdeploy.platforms import current_platform

from ..utils import get_sm_version


def load_deep_gemm():
    """
    Load DeepGemm module according to FastDeploy env switch.

    Returns:
        Imported deep_gemm module object.
    """

    if current_platform.is_cuda():
        if get_sm_version() == 100:
            # SM100 should use PFCC DeepGemm
            paddle.compat.enable_torch_proxy(scope={"deep_gemm"})
            try:
                from paddlefleet.ops import deep_gemm

                logger.info("Detected sm100, use PaddleFleet DeepGEMM")
            except:
                import deep_gemm

                logger.info("Detected sm100, use PFCC DeepGEMM")
        else:
            logger.info("use FastDeploy DeepGEMM")
            from fastdeploy.model_executor.ops.gpu import deep_gemm
    else:
        deep_gemm = None
    return deep_gemm


deep_gemm = load_deep_gemm()


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: paddle.Tensor,
):
    """Convert FP32 tensor to TMA-aligned packed UE8M0 format tensor"""

    from deep_gemm.utils import align, get_tma_aligned_size

    # Input validation: must be FP32 type 2D or 3D tensor
    assert x.dtype == paddle.float and x.dim() in (2, 3)

    # Step 1: Convert FP32 to UE8M0 format uint8 tensor
    # Extract FP32 exponent part through bit shift operation, convert to unsigned 8-bit integer
    ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

    # Step 2: Create padding and pack tensor
    # Get the last two dimensions of the input tensor
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    # If it's a 2D tensor, add batch dimension for unified processing
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    # Calculate TMA-aligned dimensions (aligned to 4-byte boundary)
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    # Create padded tensor with alignment and fill with valid data
    padded = paddle.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    # Pack uint8 data into int32 (pack 4 uint8 into 1 int32)
    padded = padded.view(-1).view(dtype=paddle.int).view(b, aligned_mn, aligned_k // 4)

    # Step 3: Transpose tensor to meet TMA memory access pattern requirements
    # Transpose tensor dimensions for TMA to efficiently access in MN-major order
    transposed = paddle.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int).mT
    transposed[:, :, :] = padded
    # Extract original non-padded part
    aligned_x = transposed[:, :mn, :]
    # If input was 2D tensor, remove batch dimension
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def transform_scale_ue8m0(sf, mn, weight_block_size=None):
    get_mn_major_tma_aligned_packed_ue8m0_tensor = _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
    if weight_block_size:
        assert weight_block_size == [128, 128]
        sf = sf.index_select(-2, paddle.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


def quant_weight_ue8m0(weight_dequant, weight_block_size):
    assert weight_block_size == [128, 128]
    assert weight_dequant.dtype == paddle.bfloat16, f"{weight_dequant.dtype=} {weight_dequant.shape=}"

    *batch_dims, n, k = weight_dequant.shape

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = deep_gemm.utils.math.per_block_cast_to_fp8(weight_dequant_flat, use_ue8m0=True)

    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s
