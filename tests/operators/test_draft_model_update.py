# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
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

from fastdeploy.model_executor.ops.gpu import draft_model_update


def is_in_end(id, end_ids, length):
    flag = False
    for i in range(length):
        if id == end_ids[i]:
            return True
    return flag


# recalculate data offset, offset_new is starting from index 0
def get_inter_next_tokens_start_offset(inter_next_tokens, max_seq_len, start_id, offset):
    offset_new = start_id + offset
    return inter_next_tokens[int(offset_new / max_seq_len)][int(offset_new % max_seq_len)]


def draft_model_update_kernel(
    inter_next_tokens,
    draft_tokens,
    pre_ids,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    output_cum_offsets,
    stop_flags,
    not_need_stop,
    max_dec_len,
    end_ids,
    base_model_draft_tokens,
    bsz,
    max_draft_token,
    pre_id_length,
    max_base_model_draft_token,
    end_ids_len,
    max_seq_len,
    substep,
    prefill_one_step_stop,
):
    stop_sum = 0
    for tid in range(bsz):
        stop_flag_now_int = 0
        draft_token_now = draft_tokens[tid]
        pre_ids_now = pre_ids[tid]
        base_model_draft_tokens_now = base_model_draft_tokens[tid]
        next_tokens_start_id = tid * max_seq_len - output_cum_offsets[tid]
        # next_tokens_start =
        seq_len_this_time = seq_lens_this_time[tid]
        seq_len_encoder = seq_lens_encoder[tid]
        seq_len_decoder = seq_lens_decoder[tid]

        # 1. update step_idx && seq_lens_dec
        if not stop_flags[tid]:  # seq_lens_decoder > 0 or seq_lens_encoder > 0
            token_this_time = -1
            # decoder step
            if seq_len_decoder > 0 and seq_len_encoder <= 0:
                seq_lens_decoder[tid] += seq_len_this_time
                token_this_time = get_inter_next_tokens_start_offset(
                    inter_next_tokens, max_seq_len, next_tokens_start_id, seq_len_this_time - 1
                )
                draft_token_now[0] = token_this_time
                base_model_draft_tokens_now[substep + 1] = token_this_time
                step_idx[tid] += seq_len_this_time
                pre_ids_now[step_idx[tid]] = token_this_time
            else:
                token_this_time = get_inter_next_tokens_start_offset(
                    inter_next_tokens, max_seq_len, next_tokens_start_id, 0
                )

                # seq_lens_decoder[tid] = seq_lens_encoder[tid]
                seq_lens_decoder[tid] = seq_len_encoder + seq_len_decoder
                seq_lens_encoder[tid] = 0
                pre_ids_now[1] = token_this_time
                step_idx[tid] += 1
                draft_token_now[0] = token_this_time
                base_model_draft_tokens_now[substep + 1] = token_this_time

            # multi_end
            if is_in_end(token_this_time, end_ids, end_ids_len) or prefill_one_step_stop:
                stop_flags[tid] = True
                stop_flag_now_int = 1
                # max_dec_len
            elif step_idx[tid] >= max_dec_len[tid]:
                stop_flags[tid] = True
                draft_token_now[seq_len_this_time - 1] = end_ids[0]
                base_model_draft_tokens_now[substep + 1] = end_ids[0]
                stop_flag_now_int = 1
        else:
            draft_token_now[0] = -1
            base_model_draft_tokens_now[substep + 1] = -1
            stop_flag_now_int = 1

        # 2. set end
        if not stop_flags[tid]:
            seq_lens_this_time[tid] = 1
        else:
            seq_lens_this_time[tid] = 0
            seq_lens_encoder[tid] = 0

        stop_sum = stop_sum + stop_flag_now_int
    not_need_stop[0] = stop_sum < bsz


def draft_model_update_ref(
    inter_next_tokens,
    draft_tokens,
    pre_ids,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    output_cum_offsets,
    stop_flags,
    not_need_stop,
    max_dec_len,
    end_ids,
    base_model_draft_tokens,
    max_seq_len,
    substep,
):
    seq_lens_this_time_shape = seq_lens_this_time.shape
    real_bsz = seq_lens_this_time_shape[0]
    end_ids_len = end_ids.shape[0]
    max_draft_token = draft_tokens.shape[1]
    pre_id_length = pre_ids.shape[1]
    max_base_model_draft_token = base_model_draft_tokens.shape[1]

    prefill_one_step_stop = False
    import os

    env = os.getenv("PREFILL_NODE_ONE_STEP_STOP")
    if env == "1":
        prefill_one_step_stop = True

    draft_model_update_kernel(
        inter_next_tokens,
        draft_tokens,
        pre_ids,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        output_cum_offsets,
        stop_flags,
        not_need_stop,
        max_dec_len,
        end_ids,
        base_model_draft_tokens,
        real_bsz,
        max_draft_token,
        pre_id_length,
        max_base_model_draft_token,
        end_ids_len,
        max_seq_len,
        substep,
        prefill_one_step_stop,
    )


class TestDraftModelUpdate(unittest.TestCase):
    def test_draft_model_update(self):
        self._run_paddle_test()

    def _run_paddle_test(self):
        np.random.seed(42)
        paddle.seed(42)

        max_bsz = 128
        max_draft_token = 3
        pre_id_length = 3
        max_seq_len = 100
        max_base_model_draft_token = 4
        substep = 2

        inter_next_tokens = paddle.randint(1, 100, shape=(max_bsz, max_seq_len), dtype="int64")
        draft_tokens = paddle.randint(1, 100, shape=(max_bsz, max_draft_token), dtype="int64")
        pre_ids = paddle.randint(1, 100, shape=(max_bsz, pre_id_length), dtype="int64")
        seq_lens_this_time = paddle.randint(1, 2, shape=(max_bsz,), dtype="int32")
        seq_lens_encoder = paddle.randint(1, 10, shape=(max_bsz,), dtype="int32")
        seq_lens_decoder = paddle.randint(1, 10, shape=(max_bsz,), dtype="int32")
        step_idx = paddle.randint(1, 10, shape=(max_bsz,), dtype="int64")
        output_cum_offsets = paddle.randint(0, 2, shape=(max_bsz,), dtype="int32")
        output_cum_offsets[0] = 0
        stop_flags = paddle.zeros([max_bsz], dtype="bool")
        not_need_stop = paddle.zeros([1], dtype="bool").to(device=paddle.CPUPlace())
        max_dec_len = paddle.randint(100, 102, shape=(max_bsz,), dtype="int64")
        end_ids = paddle.to_tensor([2], dtype="int64")
        base_model_draft_tokens = paddle.randint(1, 10, shape=(max_bsz, max_base_model_draft_token), dtype="int64")

        inputs = (
            inter_next_tokens,
            draft_tokens,
            pre_ids,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            output_cum_offsets,
            stop_flags,
            not_need_stop,
            max_dec_len,
            end_ids,
            base_model_draft_tokens,
            max_seq_len,
            substep,
        )
        # inplace modify, need to clone inputs
        inputs_clone = [x.clone() if isinstance(x, paddle.Tensor) else x for x in inputs]
        draft_model_update(*inputs)
        draft_model_update_ref(*inputs_clone)
        idx_list = (
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            9,
            12,
        )
        for i in idx_list:
            np.testing.assert_allclose(inputs[i].numpy(), inputs_clone[i].numpy())


if __name__ == "__main__":
    unittest.main()
