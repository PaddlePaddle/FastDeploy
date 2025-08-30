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

# tests/test_speculate_update_v3.py
import paddle

from fastdeploy.model_executor.ops.xpu import speculate_update_v3


# ---------------- NumPy 参考实现 ----------------
def speculate_update_v3_np(
    seq_lens_encoder,
    seq_lens_decoder,
    not_need_stop,
    draft_tokens,
    actual_draft_token_nums,
    accept_tokens,
    accept_num,
    stop_flags,
    seq_lens_this_time,
    is_block_step,
    stop_nums,
):
    """
    完全复现 CPU / CUDA 逻辑的 NumPy 参考版本（就地修改）。
    """
    stop_sum = 0
    real_bsz = seq_lens_this_time.shape[0]
    max_bsz = stop_flags.shape[0]
    max_draft_tokens = draft_tokens.shape[1]

    for bid in range(max_bsz):
        stop_flag_now_int = 0
        inactive = bid >= real_bsz
        block_step = (not inactive) and is_block_step[bid]

        if (not block_step) and (not inactive):

            if stop_flags[bid]:
                stop_flag_now_int = 1

            # encoder 长度为 0 时直接累加 decoder
            if seq_lens_encoder[bid] == 0:
                seq_lens_decoder[bid] += accept_num[bid]

            # draft 长度自适应
            if (seq_lens_encoder[bid] == 0) and (seq_lens_this_time[bid] > 1):
                cur_len = actual_draft_token_nums[bid]
                if accept_num[bid] - 1 == cur_len:  # 全部接受
                    if cur_len + 2 <= max_draft_tokens - 1:
                        cur_len += 2
                    elif cur_len + 1 <= max_draft_tokens - 1:
                        cur_len += 1
                    else:
                        cur_len = max_draft_tokens - 1
                else:  # 有拒绝
                    cur_len = max(1, cur_len - 1)
                actual_draft_token_nums[bid] = cur_len

            # 偿还 encoder 欠账
            if seq_lens_encoder[bid] != 0:
                seq_lens_decoder[bid] += seq_lens_encoder[bid]
                seq_lens_encoder[bid] = 0

            # 写回下一轮首 token
            draft_tokens[bid, 0] = accept_tokens[bid, accept_num[bid] - 1]

            # 停止则清零 decoder
            if stop_flag_now_int:
                seq_lens_decoder[bid] = 0

        elif inactive:
            stop_flag_now_int = 1  # padding slot 视为 stop

        stop_sum += stop_flag_now_int

    # print("stop_sum: ", stop_sum)
    not_need_stop[0] = stop_sum < stop_nums[0]

    # 返回引用，仅供一致性
    return (
        seq_lens_encoder,
        seq_lens_decoder,
        not_need_stop,
        draft_tokens,
        actual_draft_token_nums,
    )


# ---------------- 生成随机输入 ----------------
def gen_inputs(
    max_bsz=512,  # 与 CUDA BlockSize 对齐
    max_draft_tokens=16,
    real_bsz=123,  # 可自调；须 ≤ max_bsz
    seed=2022,
):
    rng = np.random.default_rng(seed)

    # 基本张量
    seq_lens_encoder = rng.integers(0, 3, size=max_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(0, 20, size=max_bsz, dtype=np.int32)
    not_need_stop = rng.integers(0, 1, size=1, dtype=np.bool_)
    draft_tokens = rng.integers(0, 1000, size=(max_bsz, max_draft_tokens), dtype=np.int64)
    actual_draft_nums = rng.integers(1, max_draft_tokens, size=max_bsz, dtype=np.int32)
    accept_tokens = rng.integers(0, 1000, size=(max_bsz, max_draft_tokens), dtype=np.int64)
    accept_num = rng.integers(1, max_draft_tokens, size=max_bsz, dtype=np.int32)
    stop_flags = rng.integers(0, 2, size=max_bsz, dtype=np.bool_)
    is_block_step = rng.integers(0, 2, size=max_bsz, dtype=np.bool_)
    stop_nums = np.array([5], dtype=np.int64)  # 阈值随意

    # seq_lens_this_time 仅取 real_bsz 长度
    seq_lens_this_time = rng.integers(1, max_draft_tokens, size=real_bsz, dtype=np.int32)

    return {
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "not_need_stop": not_need_stop,
        "draft_tokens": draft_tokens,
        "actual_draft_token_nums": actual_draft_nums,
        "accept_tokens": accept_tokens,
        "accept_num": accept_num,
        "stop_flags": stop_flags,
        "seq_lens_this_time": seq_lens_this_time,
        "is_block_step": is_block_step,
        "stop_nums": stop_nums,
        # real_bsz              = real_bsz,
        # max_bsz               = max_bsz,
        # max_draft_tokens      = max_draft_tokens
    }


# ------------------- 单测主体 -------------------
inputs = gen_inputs(max_bsz=512, max_draft_tokens=32, real_bsz=201)

# ---- Paddle 端 ----
paddle_inputs = {}
for k, v in inputs.items():
    if k in ("real_bsz", "max_bsz", "max_draft_tokens"):
        paddle_inputs[k] = v  # 纯 python int
    else:
        if k == "not_need_stop":
            paddle_inputs[k] = paddle.to_tensor(v, place=paddle.CPUPlace())
        else:
            # 其余张量保持默认 place（想测 GPU 就手动加 place=paddle.CUDAPlace(0)）
            paddle_inputs[k] = paddle.to_tensor(v)

# ---- NumPy 端 ----
# 为保证初值一致，这里必须复制 Paddle 入参的 numpy 值再传给参考实现
np_inputs = {
    k: (paddle_inputs[k].numpy().copy() if isinstance(paddle_inputs[k], paddle.Tensor) else paddle_inputs[k])
    for k in paddle_inputs
}

# 调用自定义算子
# print("seq_lens_encoder_xpu_before: ", paddle_inputs["seq_lens_encoder"])
out_pd = speculate_update_v3(**paddle_inputs)
# print("seq_lens_encoder_xpu_after: ", out_pd[0])
# print("not_need_stop: ", out_pd[2])

# speculate_update_v3 返回 5 个张量（与 Outputs 对应）
(
    seq_lens_encoder_pd,
    seq_lens_decoder_pd,
    not_need_stop_pd,
    draft_tokens_pd,
    actual_draft_nums_pd,
) = out_pd

# print("seq_lens_encoder_np_before: ", np_inputs["seq_lens_encoder"])
out_np = speculate_update_v3_np(**np_inputs)
# print("seq_lens_encoder_np_after: ", out_np[0])
# print("not_need_stop: ", out_np[2])


# ---------------- 校对 ----------------
names = [
    "seq_lens_encoder",
    "seq_lens_decoder",
    "not_need_stop",
    "draft_tokens",
    "actual_draft_token_nums",
]
pd_tensors = [
    seq_lens_encoder_pd,
    seq_lens_decoder_pd,
    not_need_stop_pd,
    draft_tokens_pd,
    actual_draft_nums_pd,
]

for name, pd_val, np_val in zip(names, pd_tensors, out_np):
    pd_arr = pd_val.numpy()
    ok = np.array_equal(pd_arr, np_val)
    print(f"{name:25s} equal :", ok)

    # 也可以加 assert，配合 pytest
    # assert all(np.array_equal(p.numpy(), n) for p,n in zip(pd_tensors, out_np))
