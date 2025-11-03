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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_update


def speculate_update_np(
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
    mask_rollback,
):
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
                mask_rollback[bid] = 0

            elif seq_lens_encoder[bid] == 0:
                seq_lens_decoder[bid] += accept_num[bid]
                mask_rollback[bid] = seq_lens_this_time[bid] - accept_num[bid]
            else:
                mask_rollback[bid] = 0

            if (seq_lens_encoder[bid] == 0) and (seq_lens_this_time[bid] > 1):
                cur_len = actual_draft_token_nums[bid]
                if accept_num[bid] - 1 == cur_len:
                    if cur_len + 2 <= max_draft_tokens - 1:
                        cur_len += 2
                    elif cur_len + 1 <= max_draft_tokens - 1:
                        cur_len += 1
                    else:
                        cur_len = max_draft_tokens - 1
                else:
                    cur_len = max(1, cur_len - 1)
                actual_draft_token_nums[bid] = cur_len

            if seq_lens_encoder[bid] != 0:
                seq_lens_decoder[bid] += seq_lens_encoder[bid]
                seq_lens_encoder[bid] = 0

            draft_tokens[bid, 0] = accept_tokens[bid, accept_num[bid] - 1]

        elif inactive:
            stop_flag_now_int = 1

        stop_sum += stop_flag_now_int
    not_need_stop[0] = stop_sum < stop_nums[0]

    return (
        seq_lens_encoder,
        seq_lens_decoder,
        not_need_stop,
        draft_tokens,
        actual_draft_token_nums,
    )


def gen_inputs(
    max_bsz=512,
    max_draft_tokens=16,
    real_bsz=123,
    seed=2022,
):
    rng = np.random.default_rng(seed)

    seq_lens_encoder = rng.integers(0, 3, size=max_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(0, 20, size=max_bsz, dtype=np.int32)
    not_need_stop = rng.integers(0, 1, size=1, dtype=np.bool_)
    draft_tokens = rng.integers(0, 1000, size=(max_bsz, max_draft_tokens), dtype=np.int64)
    actual_draft_nums = rng.integers(1, max_draft_tokens, size=max_bsz, dtype=np.int32)
    accept_tokens = rng.integers(0, 1000, size=(max_bsz, max_draft_tokens), dtype=np.int64)
    accept_num = rng.integers(1, max_draft_tokens, size=max_bsz, dtype=np.int32)
    stop_flags = rng.integers(0, 2, size=max_bsz, dtype=np.bool_)
    is_block_step = rng.integers(0, 2, size=max_bsz, dtype=np.bool_)
    stop_nums = np.array([5], dtype=np.int64)
    mask_rollback = np.zeros([max_bsz], dtype=np.int32)

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
        "mask_rollback": mask_rollback,
    }


class TestSpeculateUpdate(unittest.TestCase):
    def test_speculate_update(self):
        inputs = gen_inputs(max_bsz=512, max_draft_tokens=32, real_bsz=201)

        paddle_inputs = {}
        for k, v in inputs.items():
            paddle_inputs[k] = paddle.to_tensor(v)
        paddle_inputs["not_need_stop"] = paddle_inputs["not_need_stop"].to(device=paddle.CPUPlace())

        np_inputs = {
            k: (paddle_inputs[k].numpy().copy() if isinstance(paddle_inputs[k], paddle.Tensor) else paddle_inputs[k])
            for k in paddle_inputs
        }

        speculate_update(*(paddle_inputs.values()))
        pd_tensors = (
            paddle_inputs["seq_lens_encoder"],
            paddle_inputs["seq_lens_decoder"],
            paddle_inputs["not_need_stop"],
            paddle_inputs["draft_tokens"],
            paddle_inputs["actual_draft_token_nums"],
        )

        out_np = speculate_update_np(**np_inputs)

        names = [
            "seq_lens_encoder",
            "seq_lens_decoder",
            "not_need_stop",
            "draft_tokens",
            "actual_draft_token_nums",
        ]

        for name, pd_val, np_val in zip(names, pd_tensors, out_np):
            np.testing.assert_allclose(pd_val.numpy(), np_val)


if __name__ == "__main__":
    unittest.main()
