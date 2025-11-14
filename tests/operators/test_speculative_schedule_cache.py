"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_schedule_cache


def cpu_reference(
    draft_tokens,
    block_tables,
    stop_flags,
    prompt_lens,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    step_seq_lens_decoder,
    step_draft_tokens,
    step_seq_lens_this_time,
    accept_num,
    accept_tokens,
    is_block_step,
    not_need_stop,
    stop_nums,
    block_size,
    max_draft_tokens,
):
    """Pure-NumPy mirror of the CUDA kernel's logic (single block of 512 threads).

    Shapes are the same as inputs to the custom op. This mutates the provided
    NumPy arrays in-place, exactly like the kernel does.
    """
    real_bsz = seq_lens_this_time.shape[0]
    max_bsz = stop_flags.shape[0]
    draft_tokens_len = draft_tokens.shape[1]
    block_num_per_seq = block_tables.shape[1]

    max_next_step_tokens = 2 * max_draft_tokens + 2

    # Block-local reduction input per thread (threadIdx.x -> bid)
    stop_flag_now_int = np.zeros(512, dtype=np.int64)  # THREADBLOCK_SIZE = 512

    for bid in range(512):
        if bid < real_bsz:
            if not stop_flags[bid]:
                max_possible_block_idx = (seq_lens_decoder[bid] + max_next_step_tokens) // block_size
                if max_possible_block_idx < block_num_per_seq and block_tables[bid, max_possible_block_idx] == -1:
                    is_block_step[bid] = True
                    step_seq_lens_this_time[bid] = seq_lens_this_time[bid]
                    seq_lens_this_time[bid] = 0
                    stop_flags[bid] = True
                    step_seq_lens_decoder[bid] = seq_lens_decoder[bid]
                    seq_lens_decoder[bid] = 0
                    accept_num[bid] = 0
                    accept_tokens[bid, :] = -1
                    step_draft_tokens[bid, :draft_tokens_len] = draft_tokens[bid, :draft_tokens_len]
                    stop_flag_now_int[bid] = 1
                else:
                    stop_flag_now_int[bid] = 0
            else:
                stop_flag_now_int[bid] = 1
        elif bid < max_bsz:
            # Threads in [real_bsz, max_bsz) contribute 1 to reduction
            stop_flag_now_int[bid] = 1
        else:
            stop_flag_now_int[bid] = 0

    stop_sum = int(stop_flag_now_int.sum())
    not_need_stop[0] = stop_sum < int(stop_nums[0])


class TestSpeculateScheduleCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("Paddle is not compiled with CUDA; skipping GPU op test.")
        paddle.device.set_device("gpu")

    def setUp(self):
        # --- Construct a deterministic case that exercises all branches ---
        # real_bsz < max_bsz to test the padding logic in the CUB reduction
        self.real_bsz = 3
        self.max_bsz = 5  # only stop_flags has length max_bsz

        self.draft_tokens_len = 6
        self.accept_tokens_len = 5
        self.block_size = 4
        self.block_num_per_seq = 3
        self.max_draft_tokens = 2  # -> max_next_step_tokens = 6

        # Inputs that will trigger for bid 0, not trigger for bid 2, and bid 1 is already stopped
        # seq_lens_decoder + 6 // 4 -> indices: [1, 1, 4]. Index 4 is out of range -> no trigger on bid 2
        self.draft_tokens = paddle.to_tensor(
            np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [2, 2, 2, 2, 2, 2],
                    [3, 3, 3, 3, 3, 3],
                ],
                dtype=np.int64,
            )
        )
        self.block_tables = paddle.to_tensor(np.full((self.real_bsz, self.block_num_per_seq), -1, dtype=np.int32))
        # stop_flags length is max_bsz, others are real_bsz
        self.stop_flags = paddle.to_tensor(np.array([False, True, False, False, False], dtype=np.bool_))
        self.prompt_lens = paddle.to_tensor(np.array([1, 1, 1], dtype=np.int64))
        self.seq_lens_this_time = paddle.to_tensor(np.array([5, 6, 7], dtype=np.int32))
        self.seq_lens_encoder = paddle.to_tensor(np.array([1, 1, 1], dtype=np.int32))
        self.seq_lens_decoder = paddle.to_tensor(np.array([1, 1, 10], dtype=np.int32))

        # Will be filled by kernel for the triggering bids only
        self.step_seq_lens_decoder = paddle.zeros((self.real_bsz,), dtype="int32")
        self.step_draft_tokens = paddle.zeros((self.real_bsz, self.draft_tokens_len), dtype="int64")
        self.step_seq_lens_this_time = paddle.zeros((self.real_bsz,), dtype="int32")

        # Intentionally non-zero so we can verify in-place zeroing only where triggered
        self.accept_num = paddle.to_tensor(np.array([9, 8, 7], dtype=np.int32))
        self.accept_tokens = paddle.to_tensor(
            np.arange(self.real_bsz * self.accept_tokens_len, dtype=np.int64).reshape(
                self.real_bsz, self.accept_tokens_len
            )
        )
        self.is_block_step = paddle.zeros((self.real_bsz,), dtype=paddle.bool)

        # not_need_stop lives on CPU in the caller; the kernel copies to device internally
        self.not_need_stop = paddle.zeros((1,), dtype=paddle.bool).cpu()

        # Choose threshold so with: bid0 triggers, bid1 already stopped, padding (5-3)=2 -> stop_sum = 1+1+2 = 4
        # Set stop_nums to 5 so not_need_stop = (4 < 5) = True
        self.stop_nums = paddle.to_tensor([5], dtype=paddle.int64)

        # Keep NumPy copies for CPU reference
        self.np_draft_tokens = self.draft_tokens.numpy().copy()
        self.np_block_tables = self.block_tables.numpy().copy()
        self.np_stop_flags = self.stop_flags.numpy().copy()
        self.np_prompt_lens = self.prompt_lens.numpy().copy()
        self.np_seq_lens_this_time = self.seq_lens_this_time.numpy().copy()
        self.np_seq_lens_encoder = self.seq_lens_encoder.numpy().copy()
        self.np_seq_lens_decoder = self.seq_lens_decoder.numpy().copy()
        self.np_step_seq_lens_decoder = self.step_seq_lens_decoder.numpy().copy()
        self.np_step_draft_tokens = self.step_draft_tokens.numpy().copy()
        self.np_step_seq_lens_this_time = self.step_seq_lens_this_time.numpy().copy()
        self.np_accept_num = self.accept_num.numpy().copy()
        self.np_accept_tokens = self.accept_tokens.numpy().copy()
        self.np_is_block_step = self.is_block_step.numpy().copy()
        self.np_not_need_stop = self.not_need_stop.numpy().copy()
        self.np_stop_nums = self.stop_nums.numpy().copy()

    def test_correctness_against_cpu_reference(self):
        # Run GPU kernel (in-place)
        speculate_schedule_cache(
            self.draft_tokens,
            self.block_tables,
            self.stop_flags,
            self.prompt_lens,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.step_seq_lens_decoder,
            self.step_draft_tokens,
            self.step_seq_lens_this_time,
            self.accept_num,
            self.accept_tokens,
            self.is_block_step,
            self.not_need_stop,
            self.stop_nums,
            self.block_size,
            self.max_draft_tokens,
        )

        # Compute CPU reference (in-place on NumPy copies)
        cpu_reference(
            self.np_draft_tokens,
            self.np_block_tables,
            self.np_stop_flags,
            self.prompt_lens,
            self.np_seq_lens_this_time,
            self.np_seq_lens_encoder,
            self.np_seq_lens_decoder,
            self.np_step_seq_lens_decoder,
            self.np_step_draft_tokens,
            self.np_step_seq_lens_this_time,
            self.np_accept_num,
            self.np_accept_tokens,
            self.np_is_block_step,
            self.np_not_need_stop,
            self.np_stop_nums,
            self.block_size,
            self.max_draft_tokens,
        )

        # Compare all mutated tensors
        np.testing.assert_array_equal(self.step_draft_tokens.numpy(), self.np_step_draft_tokens)
        np.testing.assert_array_equal(self.accept_tokens.numpy(), self.np_accept_tokens)
        np.testing.assert_array_equal(self.stop_flags.numpy(), self.np_stop_flags)
        np.testing.assert_array_equal(self.is_block_step.numpy(), self.np_is_block_step)
        np.testing.assert_array_equal(self.seq_lens_this_time.numpy(), self.np_seq_lens_this_time)
        np.testing.assert_array_equal(self.seq_lens_decoder.numpy(), self.np_seq_lens_decoder)
        np.testing.assert_array_equal(self.step_seq_lens_decoder.numpy(), self.np_step_seq_lens_decoder)
        np.testing.assert_array_equal(self.step_seq_lens_this_time.numpy(), self.np_step_seq_lens_this_time)
        np.testing.assert_array_equal(self.accept_num.numpy(), self.np_accept_num)
        self.assertEqual(bool(self.not_need_stop.numpy()[0]), bool(self.np_not_need_stop[0]))

    def test_no_trigger_path(self):
        # Make block_tables at candidate index != -1 so nothing triggers
        # Candidate index for bid 0/1 is 1, set it to 7
        bt = self.block_tables.numpy()
        bt[:, 1] = 7
        self.block_tables = paddle.to_tensor(bt)

        # Reset outputs to distinctive values
        self.step_seq_lens_decoder[:] = 0
        self.step_draft_tokens[:] = 0
        self.step_seq_lens_this_time[:] = 0
        self.accept_num[:] = -123
        self.accept_tokens[:] = -777
        self.is_block_step[:] = False
        self.not_need_stop[:] = False

        # For not_need_stop: stopped_in_real = (bid1 True) = 1, padding = 2 -> stop_sum=3
        # With stop_nums=5 -> True
        speculate_schedule_cache(
            self.draft_tokens,
            self.block_tables,
            self.stop_flags,
            self.prompt_lens,
            self.seq_lens_this_time,
            self.seq_lens_encoder,
            self.seq_lens_decoder,
            self.step_seq_lens_decoder,
            self.step_draft_tokens,
            self.step_seq_lens_this_time,
            self.accept_num,
            self.accept_tokens,
            self.is_block_step,
            self.not_need_stop,
            self.stop_nums,
            self.block_size,
            self.max_draft_tokens,
        )

        # Nothing should have changed except not_need_stop
        np.testing.assert_array_equal(self.step_draft_tokens.numpy(), np.zeros_like(self.step_draft_tokens.numpy()))
        np.testing.assert_array_equal(self.is_block_step.numpy(), np.zeros_like(self.is_block_step.numpy()))
        np.testing.assert_array_equal(self.accept_tokens.numpy(), np.full_like(self.accept_tokens.numpy(), -777))
        np.testing.assert_array_equal(self.accept_num.numpy(), np.full_like(self.accept_num.numpy(), -123))
        self.assertTrue(bool(self.not_need_stop.numpy()[0]))


if __name__ == "__main__":
    unittest.main()
