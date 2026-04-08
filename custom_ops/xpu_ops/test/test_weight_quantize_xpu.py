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

from fastdeploy.model_executor.ops.xpu import weight_quantize_xpu

np.random.seed(2025)


def np_clip_and_round(x, abs_max=127):
    return np.clip(np.around(x), -abs_max, abs_max).astype("int8")


def np_quant_weight_int4(weight_np):
    assert weight_np.dtype == np.float32  # k,n
    weight = np.transpose(weight_np, [1, 0])  # n,k
    max_value = np.max(np.abs(weight), axis=1).reshape(-1, 1)  # k => k,1
    quanted_weight = np_clip_and_round(weight / max_value * 7.0, 7)  # n,k
    quanted_weight = (quanted_weight[:, 1::2] & 0xF) << 4 | (quanted_weight[:, ::2] & 0xF)  # pack int4, [n,k//2]
    weight_scales = (max_value).astype(weight_np.dtype).reshape(-1)
    return quanted_weight, weight_scales.astype(np.float32)


def np_quant_weight(weight_np, algo="weight_only_int8"):
    assert weight_np.dtype == np.float32

    if algo == "weight_only_int4":
        return np_quant_weight_int4(weight_np)

    weight = np.transpose(weight_np, [1, 0])
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


# 1) preparation
k, n = 128, 256
algo = "weight_only_int8"
k, n = 8192, 57344

w_np = (np.random.random((k, n)).astype(np.float32) - 0.5) * 10

# 2) np calculation
qw_np, wscale_np = np_quant_weight(w_np, algo)

# 3) xpu calculation
dtype = "float32"
x_pd = paddle.to_tensor(w_np, dtype=dtype)
qw_pd, wscale_pd = weight_quantize_xpu(x_pd, algo, -1, -1)
qw_pd_trans = paddle.transpose(qw_pd, [1, 0])
# print("w_np:\n{}".format(w_np))
# print("qw_np:\n{}".format(qw_np))
# print("qw_pd:\n{}".format(qw_pd_trans))
# print("wscale_pd:\n{}".format(wscale_pd))
# print("wscale_np:\n{}".format(wscale_np))

# comparison
print(f"wscale_pd, mean={wscale_pd.mean()}, std={wscale_pd.std()}")
print(f"wscale_np, mean={wscale_np.mean()}, std={wscale_np.std()}")
print(f"qw_np, mean={qw_np.astype(np.float32).mean()}, std={qw_np.astype(np.float32).std()}")
print(f"qw_pd_trans, mean={qw_pd_trans.astype('float32').mean()}, std={qw_pd_trans.astype('float32').std()}")
sum_diff = np.sum(np.abs(qw_pd_trans.astype("float32").numpy() - qw_np.astype("float32")))
print(f"sum_diff: {sum_diff}")
