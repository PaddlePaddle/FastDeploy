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

from fastdeploy.model_executor.ops.gpu import draft_model_preprocess


def process_splitwise_prefill(
    draft_tokens,
    input_ids,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    not_need_stop,
    is_block_step,
    batch_drop,
    pre_ids,
    accept_tokens,
    accept_num,
    base_model_seq_lens_this_time,
    base_model_seq_lens_encoder,
    base_model_seq_lens_decoder,
    base_model_step_idx,
    base_model_stop_flags,
    base_model_is_block_step,
    base_model_draft_tokens,
    bsz,
    num_model_step,
    base_model_draft_tokens_len,
    truncate_first_token,
    kvcache_scheduler_v1,
):
    not_stop_flag_sum = 0

    for tid in range(bsz):
        not_stop_flag = 0
        input_ids_now = input_ids[tid]
        accept_tokens_now = accept_tokens[tid]
        if seq_lens_encoder[tid] > 0:
            not_stop_flag = 1
            seq_len_encoder = seq_lens_encoder[tid]
            stop_flags[tid] = False
            base_model_first_token = accept_tokens_now[0]
            position = seq_len_encoder
            if truncate_first_token:
                input_ids_now[position - 1] = base_model_first_token
                seq_lens_this_time[tid] = seq_len_encoder
            else:
                input_ids_now[position] = base_model_first_token
                seq_lens_this_time[tid] = seq_len_encoder + 1
        else:
            stop_flags[tid] = True
            seq_lens_this_time[tid] = 0
            seq_lens_decoder[tid] = 0
            seq_lens_encoder[tid] = 0
            not_stop_flag = 0
        not_stop_flag_sum = not_stop_flag_sum + not_stop_flag
    not_need_stop[0] = not_stop_flag_sum > 0


def draft_model_preprocess_kernel(
    draft_tokens,
    input_ids,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    not_need_stop,
    is_block_step,
    batch_drop,
    pre_ids,
    accept_tokens,
    accept_num,
    base_model_seq_lens_this_time,
    base_model_seq_lens_encoder,
    base_model_seq_lens_decoder,
    base_model_step_idx,
    base_model_stop_flags,
    base_model_is_block_step,
    base_model_draft_tokens,
    bsz,
    num_model_step,
    base_model_draft_tokens_len,
    truncate_first_token,
    kvcache_scheduler_v1,
):
    not_stop_flag_sum = 0

    for tid in range(bsz):
        not_stop_flag = 0
        accept_tokens_now = accept_tokens[tid]
        draft_tokens_now = draft_tokens[tid]
        accept_num_now = accept_num[tid]
        input_ids_now = input_ids[tid]
        base_model_draft_tokens_now = base_model_draft_tokens[tid]
        base_model_seq_len_decoder = base_model_seq_lens_decoder[tid]
        base_model_seq_len_this_time = base_model_seq_lens_this_time[tid]
        pre_ids_now = pre_ids[tid]

        base_model_draft_tokens_now[1:base_model_draft_tokens_len] = -1

        if kvcache_scheduler_v1:
            if base_model_stop_flags[tid] and base_model_is_block_step[tid]:
                stop_flags[tid] = True
                is_block_step[tid] = True
                # Need to continue infer
        else:
            if base_model_stop_flags[tid] and base_model_is_block_step[tid]:
                batch_drop[tid] = True
                stop_flags[tid] = True

        if not (base_model_stop_flags[tid] or batch_drop[tid]):
            not_stop_flag = 1
            # 1. first token
            if seq_lens_encoder[tid] > 0:
                # Can be extended to first few tokens
                seq_len_encoder = seq_lens_encoder[tid]
                stop_flags[tid] = False
                base_model_first_token = accept_tokens_now[0]
                pre_ids_now[0] = base_model_first_token
                position = seq_len_encoder
                if truncate_first_token:
                    input_ids_now[position - 1] = base_model_first_token
                    seq_lens_this_time[tid] = seq_len_encoder
                else:
                    input_ids_now[position] = base_model_first_token
                    seq_lens_this_time[tid] = seq_len_encoder + 1
            else:
                if kvcache_scheduler_v1:
                    # 3. try to recover mtp infer in V1 mode
                    if not (base_model_is_block_step[tid] and is_block_step[tid]):
                        is_block_step[tid] = False

                if stop_flags[tid]:
                    stop_flags[tid] = False
                    # TODO: check
                    seq_lens_decoder[tid] = base_model_seq_len_decoder - base_model_seq_len_this_time
                    step_idx[tid] = base_model_step_idx[tid] - base_model_seq_len_this_time
                else:
                    # 2: Last base model generated token and first MTP token
                    seq_lens_decoder[tid] -= num_model_step - 1
                    step_idx[tid] -= num_model_step - 1

                for i in range(accept_num_now):
                    draft_tokens_now[i] = accept_tokens_now[i]
                    pre_id_pos = base_model_step_idx[tid] - (accept_num_now - i)
                    accept_token = accept_tokens_now[i]
                    pre_ids_now[pre_id_pos] = accept_token

                seq_lens_this_time[tid] = accept_num_now
        else:
            stop_flags[tid] = True
            seq_lens_this_time[tid] = 0
            seq_lens_decoder[tid] = 0
            seq_lens_encoder[tid] = 0
        not_stop_flag_sum = not_stop_flag_sum + not_stop_flag
    not_need_stop[0] = not_stop_flag_sum > 0


def DispatchRunner(
    draft_tokens,
    input_ids,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    not_need_stop,
    is_block_step,
    batch_drop,
    pre_ids,
    accept_tokens,
    accept_num,
    base_model_seq_lens_this_time,
    base_model_seq_lens_encoder,
    base_model_seq_lens_decoder,
    base_model_step_idx,
    base_model_stop_flags,
    base_model_is_block_step,
    base_model_draft_tokens,
    bsz,
    num_model_step,
    truncate_first_token,
    splitwise_prefill,
    kvcache_scheduler_v1,
):
    base_model_draft_tokens_len = base_model_draft_tokens.shape[1]
    if splitwise_prefill:
        process_splitwise_prefill(
            draft_tokens,
            input_ids,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            not_need_stop,
            is_block_step,
            batch_drop,
            pre_ids,
            accept_tokens,
            accept_num,
            base_model_seq_lens_this_time,
            base_model_seq_lens_encoder,
            base_model_seq_lens_decoder,
            base_model_step_idx,
            base_model_stop_flags,
            base_model_is_block_step,
            base_model_draft_tokens,
            bsz,
            num_model_step,
            base_model_draft_tokens_len,
            truncate_first_token,
            kvcache_scheduler_v1,
        )
    else:
        draft_model_preprocess_kernel(
            draft_tokens,
            input_ids,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            not_need_stop,
            is_block_step,
            batch_drop,
            pre_ids,
            accept_tokens,
            accept_num,
            base_model_seq_lens_this_time,
            base_model_seq_lens_encoder,
            base_model_seq_lens_decoder,
            base_model_step_idx,
            base_model_stop_flags,
            base_model_is_block_step,
            base_model_draft_tokens,
            bsz,
            num_model_step,
            base_model_draft_tokens_len,
            truncate_first_token,
            kvcache_scheduler_v1,
        )


def draft_model_preprocess_ref(
    draft_tokens,
    input_ids,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    not_need_stop,
    is_block_step,
    batch_drop,
    pre_ids,
    accept_tokens,
    accept_num,
    base_model_seq_lens_this_time,
    base_model_seq_lens_encoder,
    base_model_seq_lens_decoder,
    base_model_step_idx,
    base_model_stop_flags,
    base_model_is_block_step,
    base_model_draft_tokens,
    num_model_step,
    truncate_first_token,
    splitwise_prefill,
    kvcache_scheduler_v1,
):
    real_bsz = seq_lens_this_time.shape[0]

    DispatchRunner(
        draft_tokens,
        input_ids,
        stop_flags,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        step_idx,
        not_need_stop,
        is_block_step,
        batch_drop,
        pre_ids,
        accept_tokens,
        accept_num,
        base_model_seq_lens_this_time,
        base_model_seq_lens_encoder,
        base_model_seq_lens_decoder,
        base_model_step_idx,
        base_model_stop_flags,
        base_model_is_block_step,
        base_model_draft_tokens,
        real_bsz,
        num_model_step,
        truncate_first_token,
        splitwise_prefill,
        kvcache_scheduler_v1,
    )


class TestDraftModelPreprocess:
    def _run_tests(self):
        paddle.seed(2022)

        # Define parameters
        bsz = 10
        draft_tokens_len = 4
        input_ids_len = 100
        max_draft_token = 10

        truncate_first_token = True
        splitwise_prefill = False

        draft_tokens = paddle.randint(0, 100, [bsz, draft_tokens_len], dtype="int64")
        input_ids = paddle.randint(0, 100, [bsz, input_ids_len], dtype="int64")
        stop_flags = paddle.randint(0, 1, [bsz], dtype="int").cast("bool")
        seq_lens_this_time = paddle.randint(0, 100, [bsz], dtype="int32")
        seq_lens_encoder = paddle.randint(0, input_ids_len, [bsz], dtype="int32")
        seq_lens_decoder = paddle.randint(0, input_ids_len, [bsz], dtype="int32")
        step_idx = paddle.randint(0, 100, [bsz], dtype="int64")
        seq_lens_encoder_record = paddle.randint(0, 100, [bsz], dtype="int32")  # noqa: F841
        seq_lens_decoder_record = paddle.randint(0, 100, [bsz], dtype="int32")  # noqa: F841
        not_need_stop = paddle.zeros([1], dtype="bool").cpu()
        is_block_step = paddle.zeros([bsz], dtype="bool")
        batch_drop = paddle.zeros([bsz], dtype="bool")

        # Output tensors
        accept_tokens = paddle.randint(0, 100, [bsz, 100], dtype="int64")
        accept_num = paddle.randint(1, max_draft_token + 5, [bsz], dtype="int32")
        base_model_seq_lens_encoder = paddle.randint(0, 100, [bsz], dtype="int32")
        base_model_seq_lens_decoder = paddle.randint(0, 100, [bsz], dtype="int32")
        base_model_step_idx = paddle.randint(0, 100, [bsz], dtype="int64")
        base_model_stop_flags = paddle.zeros([bsz], dtype="bool")
        base_model_is_block_step = paddle.zeros([bsz], dtype="bool")
        base_model_draft_tokens = paddle.zeros([bsz, max_draft_token], dtype="int64")
        # Run the op
        pre_ids = input_ids.clone()
        base_model_seq_lens_this_time = seq_lens_this_time
        num_model_step = max_draft_token

        kvcache_scheduler_v1 = True
        inputs = (
            draft_tokens,
            input_ids,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            not_need_stop,
            is_block_step,
            batch_drop,
            pre_ids,
            accept_tokens,
            accept_num,
            base_model_seq_lens_this_time,
            base_model_seq_lens_encoder,
            base_model_seq_lens_decoder,
            base_model_step_idx,
            base_model_stop_flags,
            base_model_is_block_step,
            base_model_draft_tokens,
            num_model_step,
            truncate_first_token,
            splitwise_prefill,
            kvcache_scheduler_v1,
        )
        # inplace modify, need to clone inputs
        inputs_clone = [x.clone() if isinstance(x, paddle.Tensor) else x for x in inputs]
        draft_model_preprocess_ref(*inputs)
        draft_model_preprocess(*inputs_clone)
        return inputs, inputs_clone

    def test_draft_model_preprocess(self):
        results1, results2 = self._run_tests()
        np.testing.assert_allclose(results1[0], results2[0])  # draft_tokens
        np.testing.assert_allclose(results1[1], results2[1])  # input_ids
        np.testing.assert_allclose(results1[2], results2[2])  # stop_flags
        np.testing.assert_allclose(results1[3], results2[3])  # seq_lens_this_time
        np.testing.assert_allclose(results1[11], results2[11])  # accept_tokens
        np.testing.assert_allclose(results1[12], results2[12])  # accept_num
        np.testing.assert_allclose(results1[7], results2[7])  # not_need_stop


if __name__ == "__main__":
    unittest.main()
