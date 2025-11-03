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

from fastdeploy.model_executor.ops.xpu import moe_expert_ffn

np.random.seed(2025)

token_num = 7
expert_num = 64
hidden_dim = 8192
ffn_inter_dim = 7168
ffn_outer_dim = ffn_inter_dim // 2
num_max_dispatch_tokens_per_rank = 128
num_rank = 8
expert_num_per_rank = expert_num // num_rank
used_in_ep_low_latency = True
hadamard_blocksize = 512

ffn_in = (np.random.random([token_num, hidden_dim]) - 0.5).astype("float32")
token_num_lod = np.full([expert_num_per_rank + 1], 0, "int32")
token_num_lod[-1] = token_num
token_num_lod[1:-1] = np.random.randint(0, token_num, [expert_num_per_rank - 1])
token_num_lod = np.sort(token_num_lod)
token_num_per_expert = token_num_lod[1:] - token_num_lod[:-1]
ffn1_w = (np.random.random([expert_num_per_rank, ffn_inter_dim, hidden_dim]) - 0.5).astype("float32")
ffn2_w = (np.random.random([expert_num_per_rank, hidden_dim, ffn_outer_dim]) - 0.5).astype("float32")
ffn2_shift = (np.random.random([1, ffn_outer_dim]) - 0.5).astype("float32")
ffn2_smooth = (np.random.random([1, ffn_outer_dim]) - 0.5).astype("float32")

if used_in_ep_low_latency:
    ffn_in_tmp = ffn_in
    ffn_in = np.zeros(
        [
            expert_num_per_rank,
            num_max_dispatch_tokens_per_rank * num_rank,
            hidden_dim,
        ],
        "float32",
    )
    for i in range(expert_num_per_rank):
        ffn_in[i][: token_num_per_expert[i]] = ffn_in_tmp[token_num_lod[i] : token_num_lod[i + 1]]
    token_num_info = token_num_per_expert
else:
    token_num_info = token_num_lod

print(f"ffn_in: {ffn_in}")
print(f"token_num_lod: {token_num_lod}")
print(f"token_num_per_expert: {token_num_per_expert}")
print(f"ffn1_w: {ffn1_w}")
print(f"ffn2_w: {ffn2_w}")


def clip_and_round(x, quant_max_bound=127):
    return np.clip(np.around(x), -quant_max_bound, quant_max_bound).astype("int8")


def weight_quant_wint8(w_fp32):
    w_max = np.max(np.abs(w_fp32), axis=-1, keepdims=True)
    w_int8 = clip_and_round(w_fp32 / w_max * 127.0)
    return w_int8, w_max.reshape([-1])


def weight_quant_wint4(w_fp32):
    w_max = np.max(np.abs(w_fp32), axis=-1, keepdims=True)
    w_int4 = clip_and_round(w_fp32 / w_max * 7.0, 7)
    w_int4 = (w_int4[:, :, 1::2] & 0xF) << 4 | (w_int4[:, :, ::2] & 0xF)  # pack int4
    return w_int4, w_max.reshape([-1])


def weight_quant(w_fp32, algo="weight_only_int8"):
    if algo == "weight_only_int8":
        return weight_quant_wint8(w_fp32)
    elif algo == "weight_only_int4":
        return weight_quant_wint4(w_fp32)
    else:
        return None, None


quant_method = "weight_only_int4"
print(f"quant_method={quant_method}, used_in_ep_low_latency={used_in_ep_low_latency}")
ffn1_quant_w, ffn1_w_scale = weight_quant(ffn1_w, quant_method)
ffn2_quant_w, ffn2_w_scale = weight_quant(ffn2_w, quant_method)
print(f"ffn1_w {ffn1_w.shape}: {ffn1_w}")
print(f"ffn2_w {ffn2_w.shape}: {ffn2_w}")
print(f"ffn1_quant_w {ffn1_quant_w.shape}: {ffn1_quant_w}")
print(f"ffn1_w_scale {ffn1_w_scale.shape}: {ffn1_w_scale}")
print(f"ffn2_quant_w {ffn2_quant_w.shape}: {ffn2_quant_w}")
print(f"ffn2_w_scale {ffn2_w_scale.shape}: {ffn2_w_scale}")


def weight_dequant_wint8(w_int, w_scale):
    w_shape = w_int.shape
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


def fwt(a):
    """
    快速 Walsh-Hadamard 变换（正向变换）
    :param a: 输入列表，长度必须是2的幂
    :return: 变换后的列表
    """
    n = len(a)
    # 检查输入长度是否为2的幂
    if n == 0 or n & (n - 1) != 0:
        raise ValueError("输入长度必须是2的幂")

    # 复制输入以避免修改原始数据
    a = a.copy()
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h <<= 1  # 等同于 h *= 2
    return a


def hadamard(_x, block_size):
    x = np.copy(_x).reshape((-1, _x.shape[-1]))
    if block_size == -1:
        return x
    m = 1
    n = x.shape[-1]
    for i in range(len(x.shape) - 1):
        m = m * x.shape[i]
    for i in range(m):
        for j in range(0, n, block_size):
            subx = x[i][j : j + block_size]
            x[i][j : j + block_size] = fwt(subx)
    return x.reshape(_x.shape)


# print(f"ffn1_w {ffn1_w.shape}: {ffn1_w}")
# ffn1_quant_w8, ffn1_w8_scale = weight_quant(ffn1_w, "weight_only_int8")
# ffn1_quant_w4, ffn1_w4_scale = weight_quant(ffn1_w, "weight_only_int4")
# print(f"ffn1_quant_w8 {ffn1_quant_w8.shape}: {ffn1_quant_w8}")
# print(f"ffn1_w8_scale {ffn1_w8_scale.shape}: {ffn1_w8_scale}")
# print(f"ffn1_quant_w4 {ffn1_quant_w4.shape}: {ffn1_quant_w4}")
# print(f"ffn1_w4_scale {ffn1_w4_scale.shape}: {ffn1_w4_scale}")

# ffn1_w8_dq = weight_dequant(ffn1_quant_w8, ffn1_w8_scale, "weight_only_int8")
# ffn1_w4_dq = weight_dequant(ffn1_quant_w4, ffn1_w4_scale, "weight_only_int4")
# print(f"ffn1_w8_dq {ffn1_w8_dq.shape}: {ffn1_w8_dq}")
# print(f"ffn1_w4_dq {ffn1_w4_dq.shape}: {ffn1_w4_dq}")


def batch_matmul(x, token_num_info, w, w_scale, algo):
    w_fp32 = weight_dequant(w, w_scale, algo)
    print(f"x {x.shape}, w {w_fp32.shape}")
    out_hidden_dim = w_fp32.shape[1]
    if not used_in_ep_low_latency:
        y = np.zeros([x.shape[0], out_hidden_dim], "float32")
        token_num_lod = token_num_info
        for i in range(expert_num_per_rank):
            start_i = token_num_lod[i]
            end_i = token_num_lod[i + 1]
            subx = x[start_i:end_i]
            subw = w_fp32[i : i + 1].transpose([0, 2, 1])
            y[start_i:end_i] = np.matmul(subx, subw)
    else:
        y = np.zeros(
            [
                expert_num_per_rank,
                num_max_dispatch_tokens_per_rank,
                out_hidden_dim,
            ],
            "float32",
        )
        token_num_per_expert = token_num_info
        for i in range(expert_num_per_rank):
            subx = x[i][: token_num_per_expert[i]]
            subw = w_fp32[i : i + 1].transpose([0, 2, 1])
            y[i][: token_num_per_expert[i]] = np.matmul(subx, subw)
    return y


def swiglu(x):
    new_shape = list(x.shape)
    new_shape[-1] //= 2
    x1 = np.copy(x[..., : new_shape[-1]])
    x2 = np.copy(x[..., new_shape[-1] :])
    y = x1 * 1.0 / (1.0 + np.exp(-x1)) * x2
    return y


ref_ffn1_out = batch_matmul(ffn_in, token_num_info, ffn1_quant_w, ffn1_w_scale, quant_method)
print(f"ref_ffn1_out {ref_ffn1_out.shape}: {ref_ffn1_out}")
ref_swiglu_out = swiglu(ref_ffn1_out)
print(f"ref_swiglu_out {ref_swiglu_out.shape}: {ref_swiglu_out}")
ref_swiglu_out = (ref_swiglu_out + ffn2_shift) * ffn2_smooth
ref_hadamard_out = hadamard(ref_swiglu_out, hadamard_blocksize)
ref_ffn2_out = batch_matmul(
    ref_hadamard_out,
    token_num_info,
    ffn2_quant_w,
    ffn2_w_scale,
    quant_method,
)

ffn_in_tensor = paddle.to_tensor(ffn_in).astype("bfloat16")
token_num_info_tensor = paddle.to_tensor(token_num_info)
ffn1_quant_w_tensor = paddle.to_tensor(ffn1_quant_w)
ffn2_quant_w_tensor = paddle.to_tensor(ffn2_quant_w)
ffn1_w_scale_tensor = paddle.to_tensor(ffn1_w_scale)
ffn2_w_scale_tensor = paddle.to_tensor(ffn2_w_scale)
ffn2_shift_tensor = paddle.to_tensor(ffn2_shift).astype("bfloat16")
ffn2_smooth_tensor = paddle.to_tensor(ffn2_smooth).astype("bfloat16")

ffn2_out = moe_expert_ffn(
    ffn_in_tensor,
    token_num_info_tensor,
    ffn1_quant_w_tensor,
    ffn2_quant_w_tensor,
    None,  # ffn1_bias
    None,  # ffn2_bias
    None,  # ffn1_act_scale
    None,  # ffn2_act_scale
    ffn1_w_scale_tensor,
    ffn2_w_scale_tensor,
    ffn2_shift_tensor,
    ffn2_smooth_tensor,
    quant_method,
    hadamard_blocksize,
    token_num,
)
ffn2_out = ffn2_out.astype("float32").numpy()
print(f"ffn2_out: {ffn2_out}")
print(f"ref_ffn2_out: {ref_ffn2_out}")

if not used_in_ep_low_latency:
    diff = np.sum(np.abs(ffn2_out - ref_ffn2_out)) / np.sum(np.abs(ffn2_out))
    print(f"diff: {diff}")
    assert diff < 0.01, f"diff: {diff}\nffn2_out:\n{ffn2_out}\nref_ffn2_out:\n{ref_ffn2_out}\n"
else:
    diff_all = 0
    for i in range(expert_num_per_rank):
        token_num_this_expert = token_num_per_expert[i]
        if token_num_this_expert == 0:
            continue
        tmp_ffn2_out = ffn2_out[i][:token_num_this_expert]
        tmp_ref_ffn2_out = ref_ffn2_out[i][:token_num_this_expert]
        diff = np.sum(np.abs(tmp_ffn2_out - tmp_ref_ffn2_out)) / np.sum(np.abs(tmp_ffn2_out))
        print(f"diff: {diff}")
        print(f"{i}, tmp_ffn2_out: {tmp_ffn2_out}")
        print(f"{i}, tmp_ref_ffn2_out: {tmp_ref_ffn2_out}")
        diff_all += diff
    diff_avg = diff_all / expert_num_per_rank
    print(f"diff_avg: {diff_avg}")
    assert diff_avg < 0.03, f"diff_avg: {diff_avg}\nffn2_out:\n{ffn2_out}\nref_ffn2_out:\n{ref_ffn2_out}\n"
