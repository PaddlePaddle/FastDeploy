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


def draft_model_preprocess_ref(
    draft_tokens,
    input_ids,
    stop_flags,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_idx,
    not_need_stop,
    pre_ids,
    accept_tokens,
    accept_num,
    target_model_seq_lens_encoder,
    target_model_seq_lens_decoder,
    target_model_step_idx,
    target_model_stop_flags,
    max_dec_len,
    target_model_draft_tokens,
    num_model_step,
    is_splitwise_prefill,
):
    """Reference implementation for draft_model_preprocess_kernel.

    MTP state is "shadow state": initialized from target model each round.

    is_splitwise_prefill: on P-D prefill node, only prefill requests run MTP;
        decode requests are marked stopped.
    """
    real_bsz = seq_lens_this_time.shape[0]
    target_model_draft_tokens_len = target_model_draft_tokens.shape[1]
    not_stop_flag_sum = 0

    for tid in range(real_bsz):
        not_stop_flag = 0
        accept_tokens_now = accept_tokens[tid]
        draft_tokens_now = draft_tokens[tid]
        accept_num_now = int(accept_num[tid])
        input_ids_now = input_ids[tid]
        target_model_draft_tokens_now = target_model_draft_tokens[tid]
        pre_ids_now = pre_ids[tid]
        target_step = int(target_model_step_idx[tid])
        seq_len_encoder = int(seq_lens_encoder[tid])

        # Clear target_model_draft_tokens (keep first token)
        target_model_draft_tokens_now[1:target_model_draft_tokens_len] = -1

        # ----------------------------------------------------------------
        # Decision: Should MTP run?
        # ----------------------------------------------------------------
        should_skip = False

        # Target model stopped
        if bool(target_model_stop_flags[tid]):
            should_skip = True

        # Near end of max_dec_len in no splitwise_prefill mode
        if not should_skip and not is_splitwise_prefill and target_step + num_model_step >= int(max_dec_len[tid]):
            should_skip = True

        # ----------------------------------------------------------------
        # Execute
        # ----------------------------------------------------------------
        if should_skip:
            stop_flags[tid] = True
            seq_lens_this_time[tid] = 0
            seq_lens_decoder[tid] = 0
            seq_lens_encoder[tid] = 0
            step_idx[tid] = 0
            not_stop_flag = 0
        else:
            not_stop_flag = 1
            stop_flags[tid] = False

            if seq_len_encoder > 0:
                # prefill | chunk_prefill | prompt_cache | recover after preempted
                target_model_first_token = int(accept_tokens_now[0])
                pre_ids_now[0] = target_model_first_token
                input_ids_now[seq_len_encoder - 1] = target_model_first_token
                seq_lens_this_time[tid] = seq_len_encoder

                # Shadow state: prefill just finished
                step_idx[tid] = target_step - 1
            else:
                # Decode: shadow state from target model
                need_compute_token = accept_num_now
                seq_lens_decoder[tid] = int(target_model_seq_lens_decoder[tid]) - need_compute_token
                step_idx[tid] = target_step - need_compute_token

                # Prepare draft input tokens from accepted tokens
                for i in range(accept_num_now):
                    draft_tokens_now[i] = int(accept_tokens_now[i])
                    pre_id_pos = target_step - (accept_num_now - i)
                    pre_ids_now[pre_id_pos] = int(accept_tokens_now[i])
                seq_lens_this_time[tid] = accept_num_now

        not_stop_flag_sum += not_stop_flag

    not_need_stop[0] = not_stop_flag_sum > 0


class TestDraftModelPreprocess(unittest.TestCase):
    def _run_case(self, is_splitwise_prefill: bool):
        paddle.seed(2022)

        bsz = 10
        draft_tokens_len = 4
        input_ids_len = 100
        max_draft_token = 10

        draft_tokens = paddle.randint(0, 100, [bsz, draft_tokens_len], dtype="int64")
        input_ids = paddle.randint(0, 100, [bsz, input_ids_len], dtype="int64")
        stop_flags = paddle.zeros([bsz], dtype="bool")
        seq_lens_this_time = paddle.randint(0, 100, [bsz], dtype="int32")
        # Mix prefill (>0) and decode (0) requests
        seq_lens_encoder = paddle.randint(0, input_ids_len, [bsz], dtype="int32")
        seq_lens_decoder = paddle.randint(0, input_ids_len, [bsz], dtype="int32")
        step_idx = paddle.randint(0, 100, [bsz], dtype="int64")
        not_need_stop = paddle.zeros([1], dtype="bool")
        pre_ids = input_ids.clone()

        accept_tokens = paddle.randint(0, 100, [bsz, 100], dtype="int64")
        # accept_num should not exceed draft_tokens_len to avoid out-of-bounds
        accept_num = paddle.randint(1, draft_tokens_len + 1, [bsz], dtype="int32")
        target_model_seq_lens_encoder = seq_lens_encoder.clone()
        target_model_seq_lens_decoder = paddle.randint(max_draft_token + 1, 100, [bsz], dtype="int32")
        target_model_step_idx = paddle.randint(max_draft_token + 1, 100, [bsz], dtype="int64")
        target_model_stop_flags = paddle.zeros([bsz], dtype="bool")
        max_dec_len = paddle.full([bsz], 200, dtype="int64")  # int64 to match CUDA kernel
        target_model_draft_tokens = paddle.zeros([bsz, max_draft_token], dtype="int64")

        num_model_step = max_draft_token

        inputs = (
            draft_tokens,
            input_ids,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            not_need_stop,
            pre_ids,
            accept_tokens,
            accept_num,
            target_model_seq_lens_encoder,
            target_model_seq_lens_decoder,
            target_model_step_idx,
            target_model_stop_flags,
            max_dec_len,
            target_model_draft_tokens,
            num_model_step,
            is_splitwise_prefill,
        )

        # inplace modify, need to clone inputs
        inputs_clone = [x.clone() if isinstance(x, paddle.Tensor) else x for x in inputs]
        draft_model_preprocess_ref(*inputs)
        draft_model_preprocess(*inputs_clone)
        return inputs, inputs_clone

    def test_decode_mode(self):
        """Normal decode mode: is_splitwise_prefill=False"""
        results1, results2 = self._run_case(is_splitwise_prefill=False)
        for i in range(9):
            np.testing.assert_equal(
                results1[i].numpy(),
                results2[i].numpy(),
                err_msg=f"Mismatch at output index {i} (decode mode)",
            )

    def test_splitwise_prefill_mode(self):
        """Splitwise prefill node: is_splitwise_prefill=True"""
        results1, results2 = self._run_case(is_splitwise_prefill=True)
        for i in range(9):
            np.testing.assert_equal(
                results1[i].numpy(),
                results2[i].numpy(),
                err_msg=f"Mismatch at output index {i} (splitwise prefill mode)",
            )

    def test_max_bsz(self):
        """bsz == kBlockSize (1024) should succeed."""
        results1, results2 = self._run_case_bsz(bsz=1024, is_splitwise_prefill=False)
        for i in range(9):
            np.testing.assert_equal(
                results1[i].numpy(),
                results2[i].numpy(),
                err_msg=f"Mismatch at output index {i} (bsz=1024)",
            )

    def test_bsz_exceeds_block_size(self):
        """bsz > kBlockSize (1024) should raise."""
        with self.assertRaises(Exception):
            self._run_case_bsz(bsz=1025, is_splitwise_prefill=False)

    def _run_case_bsz(self, bsz: int, is_splitwise_prefill: bool):
        """Like _run_case but with a configurable bsz."""
        paddle.seed(2022)

        draft_tokens_len = 4
        input_ids_len = 100
        max_draft_token = 10

        draft_tokens = paddle.randint(0, 100, [bsz, draft_tokens_len], dtype="int64")
        input_ids = paddle.randint(0, 100, [bsz, input_ids_len], dtype="int64")
        stop_flags = paddle.zeros([bsz], dtype="bool")
        seq_lens_this_time = paddle.randint(0, 100, [bsz], dtype="int32")
        seq_lens_encoder = paddle.zeros([bsz], dtype="int32")  # all decode for simplicity
        seq_lens_decoder = paddle.randint(max_draft_token + 1, 100, [bsz], dtype="int32")
        step_idx = paddle.randint(0, 100, [bsz], dtype="int64")
        not_need_stop = paddle.zeros([1], dtype="bool")
        pre_ids = input_ids.clone()

        accept_tokens = paddle.randint(0, 100, [bsz, 100], dtype="int64")
        # accept_num should not exceed draft_tokens_len to avoid out-of-bounds
        accept_num = paddle.randint(1, draft_tokens_len + 1, [bsz], dtype="int32")
        target_model_seq_lens_encoder = paddle.zeros([bsz], dtype="int32")
        target_model_seq_lens_decoder = paddle.randint(max_draft_token + 1, 100, [bsz], dtype="int32")
        target_model_step_idx = paddle.randint(max_draft_token + 1, 100, [bsz], dtype="int64")
        target_model_stop_flags = paddle.zeros([bsz], dtype="bool")
        max_dec_len = paddle.full([bsz], 200, dtype="int64")  # int64 to match CUDA kernel
        target_model_draft_tokens = paddle.zeros([bsz, max_draft_token], dtype="int64")

        num_model_step = max_draft_token

        inputs = (
            draft_tokens,
            input_ids,
            stop_flags,
            seq_lens_this_time,
            seq_lens_encoder,
            seq_lens_decoder,
            step_idx,
            not_need_stop,
            pre_ids,
            accept_tokens,
            accept_num,
            target_model_seq_lens_encoder,
            target_model_seq_lens_decoder,
            target_model_step_idx,
            target_model_stop_flags,
            max_dec_len,
            target_model_draft_tokens,
            num_model_step,
            is_splitwise_prefill,
        )

        inputs_clone = [x.clone() if isinstance(x, paddle.Tensor) else x for x in inputs]
        draft_model_preprocess_ref(*inputs)
        draft_model_preprocess(*inputs_clone)
        return inputs, inputs_clone


if __name__ == "__main__":
    unittest.main()
