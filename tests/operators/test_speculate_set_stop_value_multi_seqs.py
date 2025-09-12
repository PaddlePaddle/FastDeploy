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

from fastdeploy.model_executor.ops.gpu import speculate_set_stop_value_multi_seqs


class TestSpeculateSetStopValueMultiSeqs(unittest.TestCase):
    def run_op(
        self,
        accept_tokens,
        accept_num,
        pre_ids,
        step_idx,
        stop_flags,
        seq_lens,
        stop_seqs,
        stop_seqs_len,
        end_ids,
    ):
        accept_tokens_out = accept_tokens.clone()
        stop_flags_out = stop_flags.clone()
        speculate_set_stop_value_multi_seqs(
            accept_tokens_out,
            accept_num,
            pre_ids,
            step_idx,
            stop_flags_out,
            seq_lens,
            stop_seqs,
            stop_seqs_len,
            end_ids,
        )

        return {
            "accept_tokens": accept_tokens.numpy(),
            "accept_num": accept_num.numpy(),
            "pre_ids": pre_ids.numpy(),
            "step_idx": step_idx.numpy(),
            "stop_flags": stop_flags.numpy(),
            "output_accept_tokens": accept_tokens_out.numpy(),
            "output_stop_flags": stop_flags_out.numpy(),
        }

    def test_basic_functionality(self):
        # Test basic functionality with one sequence matching stop sequence
        accept_tokens = paddle.to_tensor(
            [
                [4, 5, 0, 0, 0],  # batch 0
                [1, 2, 3, 0, 0],  # batch 1
            ],
            dtype="int64",
        )
        accept_num = paddle.to_tensor([3, 4], dtype="int32")
        pre_ids = paddle.to_tensor(
            [
                [7, 8, 9, 3, 4, 5],  # batch 0
                [7, 8, 9, 1, 2, 3],  # batch 1
            ],
            dtype="int64",
        )

        step_idx = paddle.to_tensor([6, 6], dtype="int64")

        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        seq_lens = paddle.to_tensor([6, 6], dtype="int32")
        stop_seqs = paddle.to_tensor(
            [
                [3, 4, 5],  # batch 0
                [0, 0, 0],  # batch 1
            ],
            dtype="int64",
        )
        stop_seqs_len = paddle.to_tensor([3, 0], dtype="int32")
        end_ids = paddle.to_tensor([-1], dtype="int64")
        gpu_results = self.run_op(
            accept_tokens,
            accept_num,
            pre_ids,
            step_idx,
            stop_flags,
            seq_lens,
            stop_seqs,
            stop_seqs_len,
            end_ids,
        )

        expected_accept_tokens = np.array([[4, 5, -1, 0, 0], [1, 2, 3, 0, 0]])
        expected_stop_flags = np.array([True, False])

        np.testing.assert_array_equal(gpu_results["output_accept_tokens"], expected_accept_tokens)
        np.testing.assert_array_equal(gpu_results["output_stop_flags"], expected_stop_flags)

    def test_no_match(self):
        # Test case where no stop sequence matches
        accept_tokens = paddle.to_tensor(
            [[10, 20, 30, 0, 0], [40, 50, 60, 0, 0]],
            dtype="int64",
        )
        accept_num = paddle.to_tensor([3, 3], dtype="int32")
        pre_ids = paddle.to_tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype="int64")
        step_idx = paddle.to_tensor([8, 8], dtype="int64")
        stop_flags = paddle.to_tensor([False, False], dtype="bool")
        seq_lens = paddle.to_tensor([10, 10], dtype="int32")

        stop_seqs = paddle.to_tensor([[11, 12, 13], [14, 15, 16]], dtype="int64")
        stop_seqs_len = paddle.to_tensor([3, 3], dtype="int32")
        end_ids = paddle.to_tensor([-1], dtype="int64")

        gpu_results = self.run_op(
            accept_tokens,
            accept_num,
            pre_ids,
            step_idx,
            stop_flags,
            seq_lens,
            stop_seqs,
            stop_seqs_len,
            end_ids,
        )

        np.testing.assert_array_equal(gpu_results["output_accept_tokens"], accept_tokens.numpy())
        np.testing.assert_array_equal(gpu_results["output_stop_flags"], stop_flags.numpy())

    def test_partial_match(self):
        # Test case where only part of the sequence matches
        accept_tokens = paddle.to_tensor([[10, 20, 30, 0, 0]], dtype="int64")
        accept_num = paddle.to_tensor([3], dtype="int32")
        pre_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
        step_idx = paddle.to_tensor([8], dtype="int64")
        stop_flags = paddle.to_tensor([False], dtype="bool")
        seq_lens = paddle.to_tensor([10], dtype="int32")

        stop_seqs = paddle.to_tensor(
            [[5, 4, 99]],  # Only 5,4 matches (from pre_ids), 99 doesn't
            dtype="int64",
        )
        stop_seqs_len = paddle.to_tensor([3], dtype="int32")
        end_ids = paddle.to_tensor([-1], dtype="int64")

        gpu_results = self.run_op(
            accept_tokens,
            accept_num,
            pre_ids,
            step_idx,
            stop_flags,
            seq_lens,
            stop_seqs,
            stop_seqs_len,
            end_ids,
        )

        np.testing.assert_array_equal(gpu_results["output_accept_tokens"], accept_tokens.numpy())
        np.testing.assert_array_equal(gpu_results["output_stop_flags"], stop_flags.numpy())

    def test_already_stopped(self):
        # Test case where sequence is already stopped
        accept_tokens = paddle.to_tensor([[10, 20, 30, 0, 0]], dtype="int64")
        accept_num = paddle.to_tensor([3], dtype="int32")
        pre_ids = paddle.to_tensor([[1, 2, 3, 4, 5]], dtype="int64")
        step_idx = paddle.to_tensor([8], dtype="int64")
        stop_flags = paddle.to_tensor([True], dtype="bool")  # Already stopped
        seq_lens = paddle.to_tensor([10], dtype="int32")

        stop_seqs = paddle.to_tensor([[5, 4, 3]], dtype="int64")
        stop_seqs_len = paddle.to_tensor([3], dtype="int32")
        end_ids = paddle.to_tensor([-1], dtype="int64")

        gpu_results = self.run_op(
            accept_tokens,
            accept_num,
            pre_ids,
            step_idx,
            stop_flags,
            seq_lens,
            stop_seqs,
            stop_seqs_len,
            end_ids,
        )

        np.testing.assert_array_equal(gpu_results["output_accept_tokens"], accept_tokens.numpy())
        np.testing.assert_array_equal(gpu_results["output_stop_flags"], stop_flags.numpy())


if __name__ == "__main__":
    unittest.main()
