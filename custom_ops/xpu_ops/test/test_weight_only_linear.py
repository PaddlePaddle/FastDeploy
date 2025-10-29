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

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import (
    weight_only_linear_xpu as weight_only_linear,
)

np.random.seed(2025)


def np_clip_and_round(x, abs_max=127):
    return np.clip(np.around(x), -abs_max, abs_max).astype("int8")


def np_quant_weight_int4(weight_np):
    assert weight_np.dtype == np.float32  # k,n
    weight = weight_np
    # weight = np.transpose(weight_np, [1, 0])  # n,k
    max_value = np.max(np.abs(weight), axis=1).reshape(-1, 1)  # k => k,1
    quanted_weight = np_clip_and_round(weight / max_value * 7.0, 7)  # n,k
    quanted_weight = (quanted_weight[:, 1::2] & 0xF) << 4 | (quanted_weight[:, ::2] & 0xF)  # pack int4, [n,k//2]
    weight_scales = (max_value).astype(weight_np.dtype).reshape(-1)
    return quanted_weight, weight_scales.astype(np.float32)


def np_quant_weight(weight_np, algo="weight_only_int8"):
    assert weight_np.dtype == np.float32

    if algo == "weight_only_int4":
        return np_quant_weight_int4(weight_np)

    weight = weight_np
    # weight = np.transpose(weight_np, [1, 0])
    max_value = np.max(np.abs(weight), axis=1).reshape(-1, 1)
    quanted_weight = np_clip_and_round(weight / max_value * 127.0)
    weight_scales = (max_value).astype(weight_np.dtype).reshape(-1)
    return quanted_weight, weight_scales.astype(np.float32)


def int8_to_bin_np(value):
    value_np = np.int8(value)
    return np.binary_repr(value_np, width=8)


def int8_to_bin(value):
    if not -128 <= value <= 127:
        raise ValueError("int8 值必须在 -128 到 127 之间")
    return format(value & 0xFF, "08b")  # '08b' 表示 8 位二进制，高位补零


def weight_dequant_wint8(w_int, w_scale):
    w_shape = w_int.shape
    # print(f"w_shape={w_shape}")
    w_scale_new_shape = list(w_shape)
    w_scale_new_shape[-1] = 1
    w_scale_new = w_scale.reshape(w_scale_new_shape)
    w_fp32 = w_int.astype("float32") / 127.0 * w_scale_new
    return w_fp32


def weight_dequant_wint4(w_int, w_scale):
    w_shape = w_int.shape
    w_scale_new_shape = list(w_shape)
    w_scale_new_shape[-1] = 1
    # w_scale_new_shape[-2] = w_scale_new_shape[-2] * 2
    w_scale_new = w_scale.reshape(w_scale_new_shape)
    w_new_shape = list(w_shape)
    w_new_shape[-1] = w_new_shape[-1] * 2
    w_int8 = np.zeros(w_new_shape, dtype=np.int8)
    w_int8[:, :, ::2] = w_int & 0xF
    w_int8[:, :, 1::2] = (w_int >> 4) & 0xF
    w_int8 = np.where(w_int8 >= 8, w_int8 - 16, w_int8)
    w_fp32 = w_int8.astype("float32") / 7.0 * w_scale_new
    return w_fp32


def weight_dequant(w_int, w_scale, algo="weight_only_int8"):
    if algo == "weight_only_int8":
        return weight_dequant_wint8(w_int, w_scale)
    elif algo == "weight_only_int4":
        return weight_dequant_wint4(w_int, w_scale)
    else:
        return None, None


def batch_matmul(x, qw, wscale, algo, bias=None):
    w_fp32 = weight_dequant(qw, wscale, algo)
    # print(f"w_dequant={w_fp32}")
    # print(f"x.shape={x.shape}, w.shape={w_fp32.shape}")
    w_trans = np.transpose(w_fp32, [1, 0])
    y = np.matmul(x, w_trans)
    if bias is not None:
        y = y + bias
    return y


# 1) preparation
m, n, k = 64, 128, 256
algo = "weight_only_int8"
weight_dtype = "int8"
# m, n, k = 12, 14336, 8192

x_np = (np.random.random((m, k)).astype(np.float32) - 0.5) * 10
w_np = (np.random.random((n, k)).astype(np.float32) - 0.5) * 10
qw_np, wscale_np = np_quant_weight(w_np, algo)
# print(f"x_np={x_np}")
# print(f"w_np={w_np}")
# 2) np calculation
out_np = batch_matmul(x_np, qw_np, wscale_np, algo)

# 3) xpu calculation
x_pd = paddle.to_tensor(x_np).astype("bfloat16")
qw_pd = paddle.to_tensor(qw_np)
wscale_pd = paddle.to_tensor(wscale_np).astype("float32")
out_pd = weight_only_linear(x_pd, qw_pd, wscale_pd, None, weight_dtype, -1, -1)
print(f"out_pd:\n{out_pd}")
print(f"out_np:\n{out_np}")

# comparison
print(f"out_pd, mean={out_pd.mean()}, std={out_pd.std()}")
print(f"out_np, mean={out_np.mean()}, std={out_np.std()}")
sum_diff = np.sum(np.abs(out_pd.astype("float32").numpy() - out_np.astype("float32")))
print(f"sum_diff: {sum_diff}")
print(f"avg_diff: {sum_diff / (m * n)}")
