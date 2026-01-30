# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from typing import Optional

import numpy as np
import paddle

from fastdeploy.platforms import current_platform


def get_sm_version():
    prop = paddle.device.cuda.get_device_properties()
    cc = prop.major * 10 + prop.minor
    return cc


_ENABLE_MACHETE = False
if current_platform.is_cuda() and get_sm_version() == 90:
    try:
        from fastdeploy.model_executor.ops.gpu import machete_mm, machete_prepack_B

        _ENABLE_MACHETE = True
    except Exception:
        pass

MACHETE_PREPACKED_BLOCK_SHAPE = [64, 128]


def check_machete_supports_shape(in_features, out_featrues):
    if in_features and in_features % MACHETE_PREPACKED_BLOCK_SHAPE[0] != 0:
        return False
    if out_featrues and out_featrues % MACHETE_PREPACKED_BLOCK_SHAPE[1] != 0:
        return False
    return True


def query_machete_supported_group_size(in_features):
    if in_features % 128 == 0:
        return 128
    elif in_features % 64 == 0:
        return 64
    return -1


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def pack_rows(
    q_w: paddle.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == [size_k, size_n]

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.place
    q_w_np = q_w.numpy().astype(np.uint32)

    q_res = np.zeros((size_k // pack_factor, size_n), dtype=np.uint32)

    for i in range(pack_factor):
        q_res |= q_w_np[i::pack_factor, :] << num_bits * i

    q_res = paddle.to_tensor(q_res.astype(np.int32), place=orig_device)
    return q_res


def quantize_weights(
    w: paddle.Tensor,
    group_size: Optional[int],
    quant_type: str = "uint4b8",
):
    """
    Quantize weights in PaddlePaddle, similar to PyTorch implementation.

    Args:
        w: Input weight tensor (must be float type).
        quant_type: Target quantization type (e.g., `uint4`, `uint4b8`).
        group_size: Group size for quantization. If `-1`, use channel-wise quantization.
        zero_points: Whether to compute zero points (only for unsigned quant types).
        ref_zero_points_after_scales: If True, apply zero points after scales in dequantization.

    Returns:
        w_ref: Dequantized reference weights.
        w_q: Quantized weights.
        w_s: Scales (None if `group_size` is None).
    """
    assert paddle.is_floating_point(w), "w must be float type"
    assert quant_type in ["uint4b8", "uint8b128"], "only support quant_type = uint4b8, uint8b128"

    orig_device = w.place
    size_k, size_n = w.shape

    if group_size == -1:
        group_size = size_k

    # Reshape to [group_size, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape([-1, group_size, size_n])
        w = w.transpose([1, 0, 2])
        w = w.reshape([group_size, -1])

    # Compute scale for each group
    max_val = paddle.max(w, axis=0, keepdim=True)
    min_val = paddle.min(w, axis=0, keepdim=True)

    if quant_type == "uint4b8":
        max_q_val = float(7.0)
        min_q_val = float(-8.0)
    else:
        max_q_val = float(127.0)
        min_q_val = float(-128.0)

    w_s = paddle.ones([1], dtype=paddle.float32)  # unscaled case

    if group_size is not None:
        # Avoid division by zero
        max_scale = paddle.maximum(
            paddle.abs(max_val / (max_q_val if max_q_val != 0 else float("inf"))),
            paddle.abs(min_val / (min_q_val if min_q_val != 0 else float("inf"))),
        )
        w_s = max_scale

    # Quantize
    w_q = paddle.round(w / w_s).astype(paddle.int32)
    w_q = paddle.clip(w_q, min_q_val, max_q_val)

    # if hasattr(quant_type, 'bias'):  # Custom quantization bias (if applicable)
    # w_q += quant_type.bias
    if quant_type == "uint4b8":
        w_q += 8
    else:
        w_q += 128

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w_tensor):
            w_tensor = w_tensor.reshape([group_size, -1, size_n])
            w_tensor = w_tensor.transpose([1, 0, 2])
            w_tensor = w_tensor.reshape([size_k, size_n]).contiguous()
            return w_tensor

        w_q = reshape_w(w_q)
        w_s = w_s.reshape([-1, size_n]).contiguous()

    # Move tensors back to original device
    w_q = w_q.to(orig_device)
    if w_s is not None:
        w_s = w_s.to(orig_device)

    return w_q, w_s


def machete_quantize_and_pack(
    w: paddle.Tensor,
    atype: paddle.dtype,
    quant_type: str = "uint4b8",
    scale_type: str = "",
    group_size: int = -1,
):
    w_q, w_s = quantize_weights(w, group_size, quant_type=quant_type)
    num_bits = 4 if quant_type == "uint4b8" else 8
    w_q = pack_rows(w_q, num_bits, *w_q.shape)
    w_q_col = w_q.transpose([1, 0]).contiguous()  # convert to col major
    w_q_prepack = machete_prepack_B(
        w_q_col,
        atype,
        quant_type,
        scale_type,
    )
    return w_q_prepack, w_s


def machete_wint_mm(
    x: paddle.Tensor,
    w_prepack: paddle.Tensor,
    w_g_s: paddle.Tensor,
    w_g_zp: Optional[paddle.Tensor] = None,
    w_ch_s: Optional[paddle.Tensor] = None,
    w_tok_s: Optional[paddle.Tensor] = None,
    weight_dtype: str = "uint4b8",
    group_size: int = -1,
    out_dtype: str = "",
    scheduler: str = "",
):
    out = machete_mm(
        x,
        w_prepack,
        w_g_s,  # group scales
        w_g_zp,  # group zeros
        w_ch_s,  # per-channel scale
        w_tok_s,  # per-token scale
        weight_dtype,  # weight_dtype
        out_dtype,  # out_dtype
        group_size,  # group_size
        scheduler,  # scheduler
    )
    return out
