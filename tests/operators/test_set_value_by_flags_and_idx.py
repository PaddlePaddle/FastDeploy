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

from fastdeploy.model_executor.ops.gpu import set_value_by_flags_and_idx


def set_value_by_flags_and_idx_numpy(
    pre_ids_all, input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx, stop_flags
):
    """Numpy reference implementation"""
    result = pre_ids_all.copy()
    bs = seq_lens_this_time.shape[0]
    for i in range(bs):
        if stop_flags[i]:
            continue
        seq_len_enc = seq_lens_encoder[i]
        seq_len_dec = seq_lens_decoder[i]
        current_step_idx = step_idx[i]
        if seq_len_enc == 0 and seq_len_dec == 0:
            continue
        if current_step_idx > 0:
            if seq_len_enc > 0:
                token_idx = seq_len_enc - 1
                token_to_assign = input_ids[i, token_idx]
            else:
                token_to_assign = input_ids[i, 0]
            result[i, current_step_idx] = token_to_assign
    return result


class TestSetValueByFlagsAndIdxRandom(unittest.TestCase):
    """Random case testing"""

    def setUp(self):
        paddle.seed(42)
        np.random.seed(42)
        batch_size = 10
        max_length = 10
        max_input_length = 15

        # Generate random inputs
        self.pre_ids_all_np = np.random.randint(0, 1000, size=(batch_size, max_length), dtype="int64")
        self.input_ids_np = np.random.randint(0, 1000, size=(batch_size, max_input_length), dtype="int64")
        self.seq_lens_this_time_np = np.random.randint(0, max_input_length, size=(batch_size,), dtype="int32")
        self.seq_lens_encoder_np = np.random.randint(0, max_input_length, size=(batch_size,), dtype="int32")
        self.seq_lens_decoder_np = np.random.randint(0, max_input_length, size=(batch_size,), dtype="int32")
        self.step_idx_np = np.random.randint(0, max_length, size=(batch_size,), dtype="int64")
        self.stop_flags_np = np.random.choice([True, False], size=(batch_size,), p=[0.1, 0.9])

    def test_set_value_by_flags_and_idx(self):
        # NumPy baseline
        numpy_out = set_value_by_flags_and_idx_numpy(
            self.pre_ids_all_np,
            self.input_ids_np,
            self.seq_lens_this_time_np,
            self.seq_lens_encoder_np,
            self.seq_lens_decoder_np,
            self.step_idx_np,
            self.stop_flags_np,
        )
        # custom op
        pre_ids_all = paddle.to_tensor(self.pre_ids_all_np)
        set_value_by_flags_and_idx(
            pre_ids_all,
            paddle.to_tensor(self.input_ids_np),
            paddle.to_tensor(self.seq_lens_this_time_np),
            paddle.to_tensor(self.seq_lens_encoder_np),
            paddle.to_tensor(self.seq_lens_decoder_np),
            paddle.to_tensor(self.step_idx_np),
            paddle.to_tensor(self.stop_flags_np),
        )
        # Ensure outputs match exactly
        np.testing.assert_array_equal(numpy_out, pre_ids_all.numpy())


class TestSetValueByFlagsAndIdxCornerCases(unittest.TestCase):
    """Cover corner cases"""

    def test_encoder_update(self):
        # encoder case: seq_lens_encoder > 0, use last token
        pre_ids_all = np.zeros((1, 5), dtype="int64")
        input_ids = np.array([[11, 12, 13]], dtype="int64")
        seq_lens_this_time = np.array([3], dtype="int32")
        seq_lens_encoder = np.array([3], dtype="int32")
        seq_lens_decoder = np.array([0], dtype="int32")
        step_idx = np.array([0], dtype="int64")
        stop_flags = np.array([False], dtype="bool")

        expected = set_value_by_flags_and_idx_numpy(
            pre_ids_all, input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx, stop_flags
        )
        pre_ids_all_tensor = paddle.to_tensor(pre_ids_all)
        set_value_by_flags_and_idx(
            pre_ids_all_tensor,
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens_this_time),
            paddle.to_tensor(seq_lens_encoder),
            paddle.to_tensor(seq_lens_decoder),
            paddle.to_tensor(step_idx),
            paddle.to_tensor(stop_flags),
        )
        np.testing.assert_array_equal(expected, pre_ids_all_tensor.numpy())

    def test_decoder_update(self):
        # decoder case: seq_lens_encoder=0, use first token
        pre_ids_all = np.zeros((1, 4), dtype="int64")
        input_ids = np.array([[101, 102]], dtype="int64")
        seq_lens_this_time = np.array([2], dtype="int32")
        seq_lens_encoder = np.array([0], dtype="int32")
        seq_lens_decoder = np.array([2], dtype="int32")
        step_idx = np.array([2], dtype="int64")
        stop_flags = np.array([False], dtype="bool")

        expected = set_value_by_flags_and_idx_numpy(
            pre_ids_all, input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx, stop_flags
        )
        pre_ids_all_tensor = paddle.to_tensor(pre_ids_all)
        set_value_by_flags_and_idx(
            pre_ids_all_tensor,
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens_this_time),
            paddle.to_tensor(seq_lens_encoder),
            paddle.to_tensor(seq_lens_decoder),
            paddle.to_tensor(step_idx),
            paddle.to_tensor(stop_flags),
        )
        np.testing.assert_array_equal(expected, pre_ids_all_tensor.numpy())

    def test_stop_flag(self):
        # stop_flags=True, no update
        pre_ids_all = np.zeros((1, 3), dtype="int64")
        input_ids = np.array([[5, 6, 7]], dtype="int64")
        seq_lens_this_time = np.array([3], dtype="int32")
        seq_lens_encoder = np.array([3], dtype="int32")
        seq_lens_decoder = np.array([0], dtype="int32")
        step_idx = np.array([1], dtype="int64")
        stop_flags = np.array([True], dtype="bool")

        expected = set_value_by_flags_and_idx_numpy(
            pre_ids_all, input_ids, seq_lens_this_time, seq_lens_encoder, seq_lens_decoder, step_idx, stop_flags
        )
        pre_ids_all_tensor = paddle.to_tensor(pre_ids_all)
        set_value_by_flags_and_idx(
            pre_ids_all_tensor,
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens_this_time),
            paddle.to_tensor(seq_lens_encoder),
            paddle.to_tensor(seq_lens_decoder),
            paddle.to_tensor(step_idx),
            paddle.to_tensor(stop_flags),
        )
        np.testing.assert_array_equal(expected, pre_ids_all_tensor.numpy())

    def test_skip_when_both_len_zero(self):
        # seq_lens_encoder=0 and seq_lens_decoder=0, skip
        pre_ids_all = np.zeros((1, 3), dtype="int64")
        input_ids = np.array([[8, 9, 10]], dtype="int64")
        seq_lens_this_time = np.array([3], dtype="int32")
        seq_lens_encoder = np.array([0], dtype="int32")
        seq_lens_decoder = np.array([0], dtype="int32")
        step_idx = np.array([0], dtype="int64")
        stop_flags = np.array([False], dtype="bool")

        expected = pre_ids_all.copy()
        pre_ids_all_tensor = paddle.to_tensor(pre_ids_all)
        set_value_by_flags_and_idx(
            pre_ids_all_tensor,
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens_this_time),
            paddle.to_tensor(seq_lens_encoder),
            paddle.to_tensor(seq_lens_decoder),
            paddle.to_tensor(step_idx),
            paddle.to_tensor(stop_flags),
        )
        np.testing.assert_array_equal(expected, pre_ids_all_tensor.numpy())

    def test_step_idx_negative(self):
        # step_idx < 0, skip
        pre_ids_all = np.zeros((1, 3), dtype="int64")
        input_ids = np.array([[42, 43, 44]], dtype="int64")
        seq_lens_this_time = np.array([3], dtype="int32")
        seq_lens_encoder = np.array([2], dtype="int32")
        seq_lens_decoder = np.array([1], dtype="int32")
        step_idx = np.array([-1], dtype="int64")
        stop_flags = np.array([False], dtype="bool")

        expected = pre_ids_all.copy()
        pre_ids_all_tensor = paddle.to_tensor(pre_ids_all)
        set_value_by_flags_and_idx(
            pre_ids_all_tensor,
            paddle.to_tensor(input_ids),
            paddle.to_tensor(seq_lens_this_time),
            paddle.to_tensor(seq_lens_encoder),
            paddle.to_tensor(seq_lens_decoder),
            paddle.to_tensor(step_idx),
            paddle.to_tensor(stop_flags),
        )
        np.testing.assert_array_equal(expected, pre_ids_all_tensor.numpy())


if __name__ == "__main__":
    unittest.main()
