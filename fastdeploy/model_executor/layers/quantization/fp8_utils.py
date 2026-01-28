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

from fastdeploy.platforms import current_platform

from ..utils import get_sm_version

if current_platform.is_cuda():
    if get_sm_version() == 100:
        # SM100 should use PFCC DeepGemm
        paddle.compat.enable_torch_proxy(scope={"deep_gemm"})
        import deep_gemm
    else:
        from fastdeploy.model_executor.ops.gpu import deep_gemm
else:
    deep_gemm = None


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: paddle.Tensor,
):
    """将FP32张量转换为TMA对齐的packed UE8M0格式张量"""

    from deep_gemm.utils import align, get_tma_aligned_size

    # 输入验证：必须是FP32类型的2D或3D张量
    assert x.dtype == paddle.float and x.dim() in (2, 3)

    # 第一步：将FP32转换为UE8M0格式的uint8张量
    # 通过位移操作提取FP32的指数部分，转换为无符号8位整数
    ue8m0_tensor = (x.view(paddle.int) >> 23).to(paddle.uint8)

    # 第二步：创建padding并打包张量
    # 获取输入张量的最后两个维度尺寸
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    # 如果是2D张量，添加batch维度以便统一处理
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    # 计算TMA对齐的尺寸（对齐到4字节边界）
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    # 创建对齐后的padded张量，并填充有效数据
    padded = paddle.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=paddle.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    # 将uint8数据打包成int32（每4个uint8打包成1个int32）
    padded = padded.view(-1).view(dtype=paddle.int).view(b, aligned_mn, aligned_k // 4)

    # 第三步：转置张量以满足TMA的内存访问模式要求
    # 转置张量维度以便TMA能够以MN主序高效访问
    transposed = paddle.zeros((b, aligned_k // 4, aligned_mn), device=x.device, dtype=paddle.int).mT
    transposed[:, :, :] = padded
    # 截取原始非padding部分
    aligned_x = transposed[:, :mn, :]
    # 如果输入是2D张量，移除batch维度
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
