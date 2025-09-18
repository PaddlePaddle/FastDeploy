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

from fastdeploy.model_executor.ops.gpu import update_inputs_v1


def update_inputs_kernel_v1(
    not_need_stop,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_seq_lens_decoder,
    prompt_lens,
    topk_ids,
    input_ids,
    block_tables,
    stop_nums,
    stop_flags,
    is_block_step,
    next_tokens,
    bsz,
    max_bsz,
    input_ids_stride,
    block_num_per_seq,
    block_size,
):
    stop_flag_now = False
    stop_flag_now_int = np.zeros([max_bsz])
    for thread_idx in range(max_bsz):
        if thread_idx < bsz:
            stop_flag_now = stop_flags[thread_idx]
            stop_flag_now_int[thread_idx] = int(stop_flag_now)
        else:
            stop_flag_now_int[thread_idx] = 1

    for thread_idx in range(bsz):
        stop_flag_now = stop_flags[thread_idx]
        if stop_flag_now:
            seq_lens_this_time[thread_idx] = 0  # stop at next step
            seq_lens_decoder[thread_idx] = 0
            seq_lens_encoder[thread_idx] = 0
        else:
            if seq_lens_this_time[thread_idx] + seq_lens_decoder[thread_idx] >= prompt_lens[thread_idx]:
                # decoding
                seq_lens_decoder[thread_idx] += seq_lens_this_time[thread_idx]
                seq_lens_this_time[thread_idx] = 1
                seq_lens_encoder[thread_idx] = 0
                input_ids_now = input_ids[thread_idx]
                input_ids_now[0] = next_tokens[thread_idx]

                # to judge whether block is not enough
                block_table_now = block_tables[thread_idx]
                if (
                    seq_lens_this_time[thread_idx] != 0
                    and block_table_now[int(seq_lens_decoder[thread_idx] / block_size)] == -1
                ):
                    # should be scheduled by server
                    is_block_step[thread_idx] = True
                    seq_lens_this_time[thread_idx] = 0
                    stop_flags[thread_idx] = True
                    step_seq_lens_decoder[thread_idx] = seq_lens_decoder[thread_idx]
                    seq_lens_decoder[thread_idx] = 0
                    stop_flag_now_int[thread_idx] = 1
            else:
                stop_flags[thread_idx] = True
                seq_lens_this_time[thread_idx] = 0
                seq_lens_decoder[thread_idx] = 0
                seq_lens_encoder[thread_idx] = 0
                topk_ids[thread_idx] = -1
                stop_flag_now_int[thread_idx] = 1
    stop_sum = np.sum(stop_flag_now_int)
    not_need_stop[0] = stop_sum < stop_nums[0]


def update_inputs_v1_ref(
    stop_flags,
    not_need_stop,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_seq_lens_decoder,
    prompt_lens,
    topk_ids,
    input_ids,
    block_tables,
    stop_nums,
    next_tokens,
    is_block_step,
    block_size,
):
    max_bsz = stop_flags.shape[0]
    now_bsz = seq_lens_this_time.shape[0]
    input_ids_stride = input_ids.shape[1]
    block_num_per_seq = block_tables.shape[1]
    update_inputs_kernel_v1(
        not_need_stop,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_seq_lens_decoder,
        prompt_lens,
        topk_ids,
        input_ids,
        block_tables,
        stop_nums,
        stop_flags,
        is_block_step,
        next_tokens,
        now_bsz,
        max_bsz,
        input_ids_stride,
        block_num_per_seq,
        block_size,
    )


class TestUpdateInputsV1(unittest.TestCase):
    def test_update_inputs_v1(self):
        np.random.seed(2023)

        bs = 48
        max_bs = 64
        max_input_length = 100

        stop_flags = np.random.randint(0, 2, max_bs).astype("bool")
        not_need_stop = np.array([1], "bool")
        seq_lens_this_time = np.zeros([bs], "int32")
        seq_lens_encoder = np.zeros([max_bs], "int32")
        seq_lens_decoder = np.zeros([max_bs], "int32")
        for i in range(bs):
            if i % 2 == 0:
                seq_lens_encoder[i] = i
                seq_lens_this_time[i] = i
            else:
                seq_lens_decoder[i] = i
                seq_lens_this_time[i] = 1
        step_seq_lens_decoder = np.zeros([bs], "int32")
        prompt_lens = np.random.randint(0, 10, [max_bs], dtype="int64")
        topk_ids = np.zeros([bs], "int64")
        input_ids = np.random.randint(1, 10, [max_bs, max_input_length], "int64")
        block_tables = np.zeros([max_bs, 1], "int32")
        stop_nums = np.array([max_bs], "int64")
        next_tokens = np.random.randint(1, 10, [max_bs], "int64")
        is_block_step = np.random.randint(0, 2, [max_bs]).astype("bool")

        stop_flags = paddle.to_tensor(stop_flags)
        not_need_stop = paddle.to_tensor(not_need_stop, place=paddle.CPUPlace())
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder)
        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder)
        step_seq_lens_decoder = paddle.to_tensor(step_seq_lens_decoder)
        prompt_lens = paddle.to_tensor(prompt_lens)
        topk_ids = paddle.to_tensor(topk_ids)
        input_ids = paddle.to_tensor(input_ids)
        block_tables = paddle.to_tensor(block_tables)
        stop_nums = paddle.to_tensor(stop_nums)
        next_tokens = paddle.to_tensor(next_tokens)
        is_block_step = paddle.to_tensor(is_block_step)
        block_size = 1024

        inputs = (
            stop_flags,
            not_need_stop,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_seq_lens_decoder,
            prompt_lens,
            topk_ids,
            input_ids,
            block_tables,
            stop_nums,
            next_tokens,
            is_block_step,
            block_size,
        )
        # inplace modify, need to clone inputs
        inputs_clone = [x.clone() if isinstance(x, paddle.Tensor) else x for x in inputs]
        update_inputs_v1(*inputs)
        update_inputs_v1_ref(*inputs_clone)
        compare_indexs = [1, 2, 3, 4, 5, 8]
        for idx in compare_indexs:
            np.testing.assert_allclose(inputs[idx], inputs_clone[idx])


if __name__ == "__main__":
    unittest.main()
