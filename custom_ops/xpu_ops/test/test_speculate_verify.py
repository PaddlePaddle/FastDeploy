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

import random
from typing import List

import numpy as np

# tests/speculate_verify.py
import paddle

from fastdeploy.model_executor.ops.xpu import speculate_verify


def topp_sampling_kernel(candidate_ids, candidate_scores, curand_value, candidate_len, topp, tid=0):
    """
    Python 仿真版 Top-p 样本选择函数。

    参数:
    - candidate_ids: [candidate_len] int64 array,候选 token
    - candidate_scores: [candidate_len] float32 array,对应概率
    - curand_value: float,范围在 [0,1)，模拟 GPU 中的 curand_uniform
    - candidate_len: int,候选个数
    - topp: float,TopP 截断阈值
    - tid: 模拟线程 ID,仅用于调试（非必须）

    返回:
    - 采样得到的 token(int64)
    """
    rand_top_p = curand_value * topp
    sum_scores = 0.0
    for i in range(candidate_len):
        print(
            f"debug sample i:{i} scores:{candidate_scores[i]},ids:{candidate_ids[i]},curand_value{curand_value},topp{topp}, value*topp{rand_top_p}"
        )
        sum_scores += candidate_scores[i]
        sum_scores += candidate_scores[i]
        if rand_top_p <= sum_scores:
            return candidate_ids[i]
    return candidate_ids[0]  # fallback（理论上不会走到这）


# def is_in_end(id: int, end_ids: np.ndarray, length: int) -> bool:
#     """
#     判断 id 是否存在于 end_ids 前 length 个元素中。
#     """
#     for i in range(length):
#         if id == end_ids[i]:
#             return True
#     return False

# def is_in(candidates: np.ndarray, draft: int, candidate_len: int) -> bool:
#     """
#     判断 draft 是否在 candidates 的前 candidate_len 个元素中。
#     """
#     for i in range(candidate_len):
#         if draft == candidates[i]:
#             return True
#     return False


# ---------------- NumPy 参考实现 ----------------
def speculate_verify_np(
    accept_tokens,
    accept_num,
    step_idx,
    stop_flags,
    seq_lens_encoder,
    seq_lens_decoder,
    draft_tokens,
    seq_lens_this_time,
    verify_tokens,
    verify_scores,
    max_dec_len,
    end_tokens,
    is_block_step,
    output_cum_offsets,
    actual_candidate_len,
    actual_draft_token_nums,
    topp,
    max_seq_len,
    verify_window,
    enable_topp,
):
    def is_in_end(token, end_tokens, end_length):
        return token in end_tokens[:end_length]

    def is_in(candidate_list, token, length):
        return token in candidate_list[:length]

    bsz = accept_tokens.shape[0]
    real_bsz = seq_lens_this_time.shape[0]
    max_draft_tokens = draft_tokens.shape[1]
    end_length = end_tokens.shape[0]
    max_candidate_len = verify_tokens.shape[1]
    use_topk = False
    prefill_one_step_stop = False

    # random
    initial_seed = 0
    infer_seed: List[int] = [initial_seed] * bsz
    dev_curand_states: List[float] = []

    # 循环生成随机数
    for i in range(bsz):
        current_seed = infer_seed[i]  # 这里 current_seed 总是等于 initial_seed

        # 使用当前的种子创建一个独立的随机数生成器实例
        # 这对应于 C++ 的 std::mt19937_64 engine(infer_seed[i]);
        rng = random.Random(current_seed)

        # 从独立的生成器中获取一个 [0.0, 1.0) 范围内的浮点数
        # 这对应于 C++ 的 dist(engine);
        dev_curand_states.append(rng.random())
    # --- 在函数内部进行扁平化操作 ---
    # 只有那些在 C++ 中通过指针算术访问的多维数组需要扁平化
    accept_tokens_flat = accept_tokens.reshape(-1)
    draft_tokens_flat = draft_tokens.reshape(-1)
    verify_tokens_flat = verify_tokens.reshape(-1)
    verify_scores_flat = verify_scores.reshape(-1)
    print(f"DEBUG: accept_tokens_flat shape: {accept_tokens_flat.shape}")
    print(f"DEBUG: draft_tokens_flat shape: {draft_tokens_flat.shape}")
    print(f"DEBUG: verify_tokens_flat shape: {verify_tokens_flat.shape}")
    print(f"DEBUG: verify_scores_flat shape: {verify_scores_flat.shape}")
    # 其他数组 (如 accept_num, step_idx, stop_flags, end_tokens, dev_curand_states, actual_candidate_len,
    # seq_lens_encoder, seq_lens_decoder, actual_draft_token_nums, topp_values,
    # seq_lens_this_time, max_dec_len, is_block_step, output_cum_offsets)
    # 根据其 C++ 原始定义，如果本身就是一维的，则不需要额外的 reshape。
    # 这里直接使用其原始引用，或者如果其维度不确定，也可以做 flatten()。
    # 为了明确，我们假设这些参数如果不是 (N, K) 形式，就已经是 (N,) 形式。
    print()
    # 遍历批次中的每个样本
    for bid in range(real_bsz):
        # C++: const int start_token_id = bid * max_seq_len - output_cum_offsets[bid];
        start_token_id = bid * max_seq_len - output_cum_offsets[bid]
        accept_num_now = 1
        stop_flag_now_int = 0
        print(
            f"DEBUG: start_token_id: {start_token_id}, max_seq_len: {max_seq_len}, output_cum_offsets[{bid}]: {output_cum_offsets[bid]}"
        )

        # C++: if (!(is_block_step[bid] || bid >= real_bsz))
        if not (
            is_block_step[bid] or bid >= real_bsz
        ):  # bid >= real_bsz 在 Python for 循环中天然满足，但为保持一致保留
            if stop_flags[bid]:
                stop_flag_now_int = 1
            else:
                # C++: auto *verify_tokens_now = verify_tokens + start_token_id * max_candidate_len;
                # Python: verify_tokens_now 是一个指向当前批次 verify_tokens 起始的扁平视图
                # 模拟了 C++ 中指针偏移后的“基地址”
                verify_tokens_now = verify_tokens_flat[start_token_id * max_candidate_len :]  # 从基址到末尾

                # C++: auto *draft_tokens_now = draft_tokens + bid * max_draft_tokens;
                # Python: draft_tokens_now 是当前批次 draft_tokens 起始的扁平视图
                draft_tokens_now = draft_tokens_flat[bid * max_draft_tokens :]  # 从基址到末尾

                # C++: auto *actual_candidate_len_now = actual_candidate_len + start_token_id;
                # Python: actual_candidate_len_now 是当前批次 actual_candidate_len 起始的扁平视图
                actual_candidate_len_now = actual_candidate_len[start_token_id:]  # actual_candidate_len 已经是 1D

                # C++: int i = 0;
                i = 0

                # C++: for (; i < seq_lens_this_time[bid] - 1; i++)
                for loop_i in range(seq_lens_this_time[bid] - 1):  # 使用 loop_i 作为 Python 的循环变量
                    i = loop_i  # 保持 C++ 的 i 在每次迭代中更新为当前索引

                    # C++: if (seq_lens_encoder[bid] != 0)
                    if seq_lens_encoder[bid] != 0:
                        break

                    if use_topk:
                        # C++: if (verify_tokens_now[i * max_candidate_len] == draft_tokens_now[i + 1])
                        if verify_tokens_now[i * max_candidate_len] == draft_tokens_now[i + 1]:
                            step_idx[bid] += 1
                            accept_token = draft_tokens_now[i + 1]
                            # C++: accept_tokens[bid * max_draft_tokens + i] = accept_token;
                            accept_tokens_flat[bid * max_draft_tokens + i] = accept_token

                            # C++: if (is_in_end(accept_token, end_tokens, end_length) || step_idx[bid] >= max_dec_len[bid])
                            if is_in_end(accept_token, end_tokens, end_length) or step_idx[bid] >= max_dec_len[bid]:
                                stop_flags[bid] = True
                                stop_flag_now_int = 1
                                if step_idx[bid] >= max_dec_len[bid]:
                                    accept_tokens_flat[bid * max_draft_tokens + i] = end_tokens[0]
                                break
                            else:
                                accept_num_now += 1
                        else:
                            break
                    else:  # C++: else (Top P verify)
                        # C++: auto actual_candidate_len_value = actual_candidate_len_now[i] > max_candidate_len ? max_candidate_len : actual_candidate_len_now[i];
                        actual_candidate_len_value = min(actual_candidate_len_now[i], max_candidate_len)

                        # C++: if (is_in(verify_tokens_now + i * max_candidate_len, draft_tokens_now[i + 1], actual_candidate_len_value))
                        # 传入当前候选的扁平视图
                        verify_tokens_current_candidate_view = verify_tokens_now[
                            i * max_candidate_len : (i + 1) * max_candidate_len
                        ]

                        if is_in(
                            verify_tokens_current_candidate_view,
                            draft_tokens_now[i + 1],
                            actual_candidate_len_value,
                        ):
                            step_idx[bid] += 1
                            accept_token = draft_tokens_now[i + 1]
                            accept_tokens_flat[bid * max_draft_tokens + i] = accept_token

                            if is_in_end(accept_token, end_tokens, end_length) or step_idx[bid] >= max_dec_len[bid]:
                                stop_flags[bid] = True
                                stop_flag_now_int = 1
                                if step_idx[bid] >= max_dec_len[bid]:
                                    accept_tokens_flat[bid * max_draft_tokens + i] = end_tokens[0]
                                break
                            else:
                                accept_num_now += 1
                        else:
                            # TopK verify
                            ii = i  # C++ 中 ii 从 i 开始
                            # C++: if (max_candidate_len >= 2 && verify_tokens_now[ii * max_candidate_len + 1] == draft_tokens_now[ii + 1])
                            if (
                                max_candidate_len >= 2
                                and verify_tokens_now[ii * max_candidate_len + 1] == draft_tokens_now[ii + 1]
                            ):  # top-2
                                j = 0
                                ii += 1  # C++ 中 ii 从下一个位置开始检查
                                # C++: for (; j < verify_window && ii < seq_lens_this_time[bid] - 1; j++, ii++)
                                while j < verify_window and ii < seq_lens_this_time[bid] - 1:
                                    if verify_tokens_now[ii * max_candidate_len] != draft_tokens_now[ii + 1]:
                                        break
                                    j += 1
                                    ii += 1

                                # C++: if (j >= verify_window)
                                if j >= verify_window:  # accept all
                                    accept_num_now += verify_window + 1
                                    step_idx[bid] += verify_window + 1
                                    # C++: for (; i < ii; i++)
                                    for k_accepted_idx in range(i, ii):  # i 会被更新
                                        accept_token = draft_tokens_now[k_accepted_idx + 1]
                                        accept_tokens_flat[bid * max_draft_tokens + k_accepted_idx] = accept_token

                                        if (
                                            is_in_end(
                                                accept_token,
                                                end_tokens,
                                                end_length,
                                            )
                                            or step_idx[bid] >= max_dec_len[bid]
                                        ):
                                            stop_flags[bid] = True
                                            stop_flag_now_int = 1
                                            if step_idx[bid] >= max_dec_len[bid]:
                                                accept_tokens_flat[bid * max_draft_tokens + k_accepted_idx] = (
                                                    end_tokens[0]
                                                )
                                            accept_num_now -= 1
                                            step_idx[bid] -= 1
                                            break  # 跳出内层接受循环
                            break  # 跳出主验证循环 (TopK 逻辑结束，无论成功与否)
                        # else 的 break 对应 is_in(Top P 验证失败，也不是 TopK 匹配)
                        break  # 跳出主验证循环

                # 采样阶段 (Sampling Phase)
                # C++ 中 i 变量在循环结束后会保留其最终值，直接用于采样
                # Python 同样，loop_i 的最终值赋值给了 i

                if not stop_flag_now_int:
                    accept_token: int

                    # C++: const float *verify_scores_now = verify_scores + start_token_id * max_candidate_len;
                    # Python: verify_scores_now 对应 C++ 中从 start_token_id 开始的 verify_scores 视图
                    verify_scores_now = verify_scores_flat[start_token_id * max_candidate_len :]

                    step_idx[bid] += 1

                    if enable_topp:
                        # C++: auto actual_candidate_len_value = actual_candidate_len_now[i] > max_candidate_len ? max_candidate_len : actual_candidate_len_now[i];
                        actual_candidate_len_value = min(actual_candidate_len_now[i], max_candidate_len)

                        # 传入当前候选的扁平视图
                        verify_tokens_sampling_view = verify_tokens_now[
                            i * max_candidate_len : (i + 1) * max_candidate_len
                        ]
                        verify_scores_sampling_view = verify_scores_now[
                            i * max_candidate_len : (i + 1) * max_candidate_len
                        ]

                        # C++: accept_token = topp_sampling_kernel(...)
                        accept_token = topp_sampling_kernel(
                            verify_tokens_sampling_view,
                            verify_scores_sampling_view,
                            dev_curand_states[i],  # C++: dev_curand_states + i
                            actual_candidate_len_value,
                            topp[bid],  # C++: topp[bid]
                            bid,  # C++: bid
                        )
                    else:
                        accept_token = int(verify_tokens_now[i * max_candidate_len])
                    print(
                        "debug python last accept_token",
                        accept_token,
                        "prefill_one_step_stop",
                        prefill_one_step_stop,
                    )
                    # C++: accept_tokens[bid * max_draft_tokens + i] = accept_token;
                    accept_tokens_flat[bid * max_draft_tokens + i] = accept_token

                    if prefill_one_step_stop:
                        stop_flags[bid] = True

                    if is_in_end(accept_token, end_tokens, end_length) or step_idx[bid] >= max_dec_len[bid]:
                        stop_flags[bid] = True
                        stop_flag_now_int = 1
                        if step_idx[bid] >= max_dec_len[bid]:
                            accept_tokens_flat[bid * max_draft_tokens + i] = end_tokens[0]

                accept_num[bid] = accept_num_now

    return accept_tokens, accept_num, step_idx, stop_flags


# ---------------- 生成随机输入 ----------------
def gen_speculate_verify_inputs(
    real_bsz=123,
    max_draft_tokens=16,
    max_seq_len=256,
    max_candidate_len=8,
    verify_window=2,
    end_length=4,
    enable_topp=True,
    seed=2025,
):
    rng = np.random.default_rng(seed)

    # 基础输入
    seq_lens_encoder = rng.integers(0, 3, size=real_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(1, max_draft_tokens, size=real_bsz, dtype=np.int32)
    draft_tokens = rng.integers(0, 1000, size=(real_bsz, max_draft_tokens), dtype=np.int64)
    actual_draft_token_nums = rng.integers(1, max_draft_tokens + 1, size=real_bsz, dtype=np.int32)

    seq_lens_this_time = rng.integers(1, max_seq_len + 1, size=real_bsz, dtype=np.int32)
    sum_seq_this_time = int(np.sum(seq_lens_this_time))
    # print("debug param set sum_seq_this_time",sum_seq_this_time)
    # print("debug param real_bsz * max_draft_tokens < 2k",real_bsz * max_draft_tokens)
    # print("debug sum_seq_this_time * max_candidate_len < 2k",sum_seq_this_time * max_candidate_len)

    verify_tokens = rng.integers(0, 1000, size=(sum_seq_this_time, max_candidate_len), dtype=np.int64)
    verify_scores = rng.random(size=(sum_seq_this_time, max_candidate_len)).astype(np.float32)

    max_dec_len = rng.integers(16, 64, size=real_bsz, dtype=np.int64)
    end_tokens = rng.integers(1, 1000, size=end_length, dtype=np.int64)
    is_block_step = rng.integers(0, 2, size=real_bsz, dtype=bool)

    # output_cum_offsets      = np.zeros_like(seq_lens_this_time)
    # output_cum_offsets[1:]  = np.cumsum(seq_lens_this_time[:-1])
    blank_lengths = max_seq_len - seq_lens_this_time
    output_cum_offsets = np.concatenate([[0], np.cumsum(blank_lengths[:-1])])
    output_cum_offsets = output_cum_offsets.astype("int32")
    actual_candidate_len = rng.integers(1, max_candidate_len + 1, size=sum_seq_this_time, dtype=np.int32)

    topp = (
        rng.uniform(0.8, 1.0, size=real_bsz).astype(np.float32)
        if enable_topp
        else np.zeros(real_bsz, dtype=np.float32)
    )

    # 输出（占位）
    accept_tokens = np.zeros((real_bsz, max_draft_tokens), dtype=np.int64)
    accept_num = np.zeros(real_bsz, dtype=np.int32)
    step_idx = np.zeros(real_bsz, dtype=np.int64)
    stop_flags = np.zeros(real_bsz, dtype=bool)

    return {
        "accept_tokens": accept_tokens,
        "accept_num": accept_num,
        "step_idx": step_idx,
        "stop_flags": stop_flags,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "draft_tokens": draft_tokens,
        "seq_lens_this_time": seq_lens_this_time,
        "verify_tokens": verify_tokens,
        "verify_scores": verify_scores,
        "max_dec_len": max_dec_len,
        "end_tokens": end_tokens,
        "is_block_step": is_block_step,
        "output_cum_offsets": output_cum_offsets,
        "actual_candidate_len": actual_candidate_len,
        "actual_draft_token_nums": actual_draft_token_nums,
        "topp": topp,
        "max_seq_len": max_seq_len,
        "verify_window": verify_window,
        "enable_topp": enable_topp,
    }


# ------------------- 单测主体 -------------------
# # ---- Paddle 端 ----
def run_speculate_verify_test(
    real_bsz,
    max_draft_tokens,
    max_seq_len,
    max_candidate_len,
    verify_window,
    end_length,
    enable_topp,
    seed,
):
    inputs = gen_speculate_verify_inputs(
        real_bsz=real_bsz,
        max_draft_tokens=max_draft_tokens,
        max_seq_len=max_seq_len,
        max_candidate_len=max_candidate_len,
        verify_window=verify_window,
        end_length=end_length,
        enable_topp=enable_topp,
        seed=seed,
    )

    paddle_inputs = {}

    print("========= 1 xpu process==========")

    for k, v in inputs.items():
        if isinstance(v, (int, bool)):
            paddle_inputs[k] = v
            # print(f"{k:<25} type: {type(v).__name__}, value: {v}")
        else:
            # paddle_inputs[k] = paddle.to_tensor(v, place=paddle.CPUPlace())
            paddle_inputs[k] = paddle.to_tensor(v, place=paddle.XPUPlace(0))
            # print(f"{k:<25} type: Tensor, dtype: {paddle_inputs[k].dtype}, shape: {paddle_inputs[k].shape}")

    out_pd = speculate_verify(**paddle_inputs)
    (accept_tokens_pd, accept_num_pd, step_idx_pd, stop_flags_pd) = out_pd
    pd_tensors = [accept_tokens_pd, accept_num_pd, step_idx_pd, stop_flags_pd]

    print("========= 1 end==========")
    print("========= 2 python process==========")

    # np_inputs = {k: (paddle_inputs[k].numpy().copy() if isinstance(paddle_inputs[k], paddle.Tensor)
    #                     else paddle_inputs[k])
    #                 for k in paddle_inputs}

    # out_np = speculate_verify_np(**np_inputs)
    # (accept_tokens_np, accept_num_np, step_idx_np, stop_flags_np) = out_np
    # np_tensors = [accept_tokens_np, accept_num_np, step_idx_np, stop_flags_np]

    print("=========2 end =======")

    print("========= 3 (CPU)==========")
    paddle_inputs_cpu = {}

    for k, v in inputs.items():  # 重新使用原始的 inputs 字典，确保数据原始状态
        if isinstance(v, (int, bool)):
            paddle_inputs_cpu[k] = v
            # print(f"{k:<25} type: {type(v).__name__}, value: {v}")
        else:
            # 核心修改：使用 paddle.CPUPlace()
            paddle_inputs_cpu[k] = paddle.to_tensor(v, place=paddle.CPUPlace())
            # print(f"{k:<25} type: Tensor, dtype: {paddle_inputs_cpu[k].dtype}, shape: {paddle_inputs_cpu[k].shape}")

    out_cpu = speculate_verify(**paddle_inputs_cpu)
    (accept_tokens_cpu, accept_num_cpu, step_idx_cpu, stop_flags_cpu) = out_cpu

    cpu_tensors = [
        accept_tokens_cpu,
        accept_num_cpu,
        step_idx_cpu,
        stop_flags_cpu,
    ]
    print("========= 3 (CPU) end==========")

    # ---------------- 校对 ----------------
    # print("========= python/cpu vs xpu verify ==========")

    # names = ["accept_tokens", "accept_num", "step_idx", "stop_flags"]
    # for name, pd_val, np_val in zip(names, pd_tensors, np_tensors):
    #     pd_arr = pd_val.numpy()
    #     ok     = np.array_equal(pd_arr, np_val)
    #     print(f"{name:20s} equal: {ok}")
    #     if not ok:
    #         print(f"{name} mismatch!\nPaddle:\n{pd_arr}\n\nNumPy:\n{np_val}")

    print("========= cpu vs xpu verify ==========")

    names = ["accept_tokens", "accept_num", "step_idx", "stop_flags"]
    # for name, pd_val, np_val in zip(names, pd_tensors, cpu_tensors):
    #     pd_arr = pd_val.numpy()
    #     ok     = np.array_equal(pd_arr, np_val)
    #     print(f"{name:20s} equal: {ok}")
    #     if not ok:
    #         print(f"{name} mismatch!\nPaddle:\n{pd_arr}\n\nNumPy:\n{np_val}")

    for name, pd_val, np_val in zip(names, pd_tensors, cpu_tensors):
        pd_arr = pd_val.numpy()
        ok = np.array_equal(pd_arr, np_val)
        print(f"{name:20s} equal: {ok}")
        if not ok:
            print(f"{name} mismatch!")

            # 输出不同位置的索引和对应值
            print(f"{name} mismatch!\nPaddle:\n{pd_arr}\n\nNumPy:\n{np_val}")
            mismatches = np.where(pd_arr != np_val)
            for idx in zip(*mismatches):
                print(f"  idx {idx}: Paddle = {pd_arr[idx]}, NumPy = {np_val[idx]}")

            # 如果差异太多可限制输出数量
            if len(mismatches[0]) > 20:
                print("  ... (truncated)")


# -------------------------------------
# 测试用例
# -------------------------------------
test_configs = [
    {
        "real_bsz": 4,
        "max_draft_tokens": 3,
        "max_seq_len": 30,
        "max_candidate_len": 4,
        "verify_window": 2,
        "end_length": 2,
        "enable_topp": True,
        "seed": 2025,
    },
    {
        "real_bsz": 77,
        "max_draft_tokens": 10,
        "max_seq_len": 12000,
        "max_candidate_len": 8,
        "verify_window": 2,
        "end_length": 4,
        "enable_topp": True,
        "seed": 2025,
    },
    {
        "real_bsz": 1,
        "max_draft_tokens": 2,
        "max_seq_len": 10,
        "max_candidate_len": 1,
        "verify_window": 1,
        "end_length": 1,
        "enable_topp": True,
        "seed": 42,
    },
    {
        "real_bsz": 128,
        "max_draft_tokens": 7,
        "max_seq_len": 999,
        "max_candidate_len": 5,
        "verify_window": 3,
        "end_length": 3,
        "enable_topp": True,
        "seed": 422,
    },
    {
        "real_bsz": 99,
        "max_draft_tokens": 5,
        "max_seq_len": 10,
        "max_candidate_len": 3,
        "verify_window": 4,
        "end_length": 4,
        "enable_topp": True,
        "seed": 42,
    },
    {
        "real_bsz": 1,
        "max_draft_tokens": 9,
        "max_seq_len": 11,
        "max_candidate_len": 4,
        "verify_window": 2,
        "end_length": 5,
        "enable_topp": False,
        "seed": 42,
    },
    {
        "real_bsz": 33,
        "max_draft_tokens": 5,
        "max_seq_len": 10111,
        "max_candidate_len": 5,
        "verify_window": 2,
        "end_length": 6,
        "enable_topp": False,
        "seed": 42,
    },
    {
        "real_bsz": 6,
        "max_draft_tokens": 4,
        "max_seq_len": 10001,
        "max_candidate_len": 6,
        "verify_window": 2,
        "end_length": 7,
        "enable_topp": False,
        "seed": 42,
    },
    {
        "real_bsz": 7,
        "max_draft_tokens": 3,
        "max_seq_len": 777,
        "max_candidate_len": 7,
        "verify_window": 2,
        "end_length": 5,
        "enable_topp": False,
        "seed": 42,
    },
    {
        "real_bsz": 55,
        "max_draft_tokens": 5,
        "max_seq_len": 31,
        "max_candidate_len": 9,
        "verify_window": 2,
        "end_length": 3,
        "enable_topp": False,
        "seed": 42,
    },
]

for i, cfg in enumerate(test_configs):
    print(f"\n\n======== Running Test Case {i} ========")
    run_speculate_verify_test(**cfg)
