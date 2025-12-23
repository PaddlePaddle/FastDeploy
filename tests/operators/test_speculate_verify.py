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
import unittest
from typing import List

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_verify


def topp_sampling_kernel(candidate_ids, candidate_scores, curand_value, candidate_len, topp, tid=0):
    """
    Python simulation version of the Top-p sampling function.

    Parameters:
    - candidate_ids: [candidate_len] int64 array, candidate tokens
    - candidate_scores: [candidate_len] float32 array, corresponding probabilities
    - curand_value: float, in the range [0, 1), simulating the GPU's curand_uniform
    - candidate_len: int, number of candidates
    - topp: float, Top-P truncation threshold
    - tid: simulated thread ID, for debugging purposes only (optional)

    Returns:
    - The sampled token (int64)
    """
    rand_top_p = curand_value * topp
    sum_scores = 0.0
    for i in range(candidate_len):
        sum_scores += candidate_scores[i]
        sum_scores += candidate_scores[i]
        if rand_top_p <= sum_scores:
            return candidate_ids[i]
    return candidate_ids[0]


def speculate_verify_ref(
    sampled_token_ids,
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
    benchmark_mode,
    accept_all_drafts,
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

    for i in range(bsz):
        current_seed = infer_seed[i]

        # std::mt19937_64 engine(infer_seed[i]);
        rng = random.Random(current_seed)

        dev_curand_states.append(rng.random())

    # flatten
    accept_tokens_flat = accept_tokens.reshape(-1)
    draft_tokens_flat = draft_tokens.reshape(-1)
    verify_tokens_flat = verify_tokens.reshape(-1)
    verify_scores_flat = verify_scores.reshape(-1)
    for bid in range(real_bsz):
        start_token_id = bid * max_seq_len - output_cum_offsets[bid]
        accept_num_now = 1
        stop_flag_now_int = 0

        if not (is_block_step[bid] or bid >= real_bsz):  # bid >= real_bsz reserved for consistency with gpu
            if stop_flags[bid]:
                stop_flag_now_int = 1
            else:
                verify_tokens_now = verify_tokens_flat[start_token_id * max_candidate_len :]
                draft_tokens_now = draft_tokens_flat[bid * max_draft_tokens :]
                actual_candidate_len_now = actual_candidate_len[start_token_id:]
                i = 0
                for loop_i in range(seq_lens_this_time[bid] - 1):
                    i = loop_i

                    if seq_lens_encoder[bid] != 0:
                        break

                    if use_topk:
                        if verify_tokens_now[i * max_candidate_len] == draft_tokens_now[i + 1]:
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
                            break
                    else:
                        actual_candidate_len_value = min(actual_candidate_len_now[i], max_candidate_len)
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
                            ii = i  # Start from i
                            if (
                                max_candidate_len >= 2
                                and verify_tokens_now[ii * max_candidate_len + 1] == draft_tokens_now[ii + 1]
                            ):  # top-2
                                j = 0
                                ii += 1  # Start from ii next position
                                while j < verify_window and ii < seq_lens_this_time[bid] - 1:
                                    if verify_tokens_now[ii * max_candidate_len] != draft_tokens_now[ii + 1]:
                                        break
                                    j += 1
                                    ii += 1

                                if j >= verify_window:  # accept all
                                    accept_num_now += verify_window + 1
                                    step_idx[bid] += verify_window + 1
                                    for k_accepted_idx in range(i, ii):
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
                                            break
                            break  # TopK finish
                        break  # Jump main loop

                if not stop_flag_now_int:
                    accept_token: int
                    verify_scores_now = verify_scores_flat[start_token_id * max_candidate_len :]

                    step_idx[bid] += 1

                    if enable_topp:
                        actual_candidate_len_value = min(actual_candidate_len_now[i], max_candidate_len)
                        verify_tokens_sampling_view = verify_tokens_now[
                            i * max_candidate_len : (i + 1) * max_candidate_len
                        ]
                        verify_scores_sampling_view = verify_scores_now[
                            i * max_candidate_len : (i + 1) * max_candidate_len
                        ]

                        accept_token = topp_sampling_kernel(
                            verify_tokens_sampling_view,
                            verify_scores_sampling_view,
                            dev_curand_states[i],
                            actual_candidate_len_value,
                            topp[bid],
                            bid,
                        )
                    else:
                        accept_token = int(verify_tokens_now[i * max_candidate_len])

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


def gen_speculate_verify_inputs(
    real_bsz=123,
    max_draft_tokens=16,
    max_seq_len=256,
    max_candidate_len=8,
    verify_window=2,
    end_length=4,
    enable_topp=False,
    seed=2025,
):
    rng = np.random.default_rng(seed)

    seq_lens_encoder = rng.integers(0, 3, size=real_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(1, max_draft_tokens, size=real_bsz, dtype=np.int32)
    draft_tokens = rng.integers(0, 1000, size=(real_bsz, max_draft_tokens), dtype=np.int64)
    actual_draft_token_nums = rng.integers(1, max_draft_tokens + 1, size=real_bsz, dtype=np.int32)

    seq_lens_this_time = rng.integers(1, max_seq_len + 1, size=real_bsz, dtype=np.int32)
    sum_seq_this_time = int(np.sum(seq_lens_this_time))

    sampled_token_ids = rng.integers(0, 1000, size=(sum_seq_this_time, 1), dtype=np.int64)
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

    # Output(inplace)
    accept_tokens = np.zeros((real_bsz, max_draft_tokens), dtype=np.int64)
    accept_num = np.zeros(real_bsz, dtype=np.int32)
    step_idx = np.zeros(real_bsz, dtype=np.int64)
    stop_flags = np.zeros(real_bsz, dtype=bool)

    return {
        "sampled_token_ids": sampled_token_ids,
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
        "benchmark_mode": False,
        "accept_all_drafts": False,
    }


test_configs = [
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


class TestSpeculateVerify(unittest.TestCase):
    def run_speculate_verify(
        self,
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
        paddle_inputs = {k: v if isinstance(v, (int, bool)) else paddle.to_tensor(v) for k, v in inputs.items()}
        inputs_gpu = list(paddle_inputs.values())
        speculate_verify(*inputs_gpu)
        out_gpu = [inputs_gpu[1], inputs_gpu[2], inputs_gpu[3], inputs_gpu[4]]

        paddle_inputs_ref = {k: v if isinstance(v, (int, bool)) else paddle.to_tensor(v) for k, v in inputs.items()}
        out_ref = speculate_verify_ref(**paddle_inputs_ref)

        names = ["accept_tokens", "accept_num", "step_idx", "stop_flags"]
        for _, pd_val, np_val in zip(names, out_gpu, out_ref):
            np.testing.assert_allclose(pd_val.numpy(), np_val.numpy())

    def test_speculate_verify(self):
        for config in test_configs:
            self.run_speculate_verify(**config)


if __name__ == "__main__":
    unittest.main()
