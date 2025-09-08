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

from fastdeploy.model_executor.ops.gpu import extract_text_token_output


class TestExtractTextTokenOutput(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _run_and_check(
        self,
        bsz,
        hidden_size,
        max_seq_len_v,
        max_seq_len_index_v,
        mm_token_num_len_v,
        seq_lens_this_time_v,
        cu_seqlens_q_v,
        hidden_states_v,
    ):

        max_seq_len = paddle.to_tensor([max_seq_len_v], dtype="int32")
        max_seq_len_index = paddle.to_tensor([max_seq_len_index_v], dtype="int32")
        mm_token_num_len = paddle.to_tensor([mm_token_num_len_v], dtype="int32")
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_v, dtype="int32")
        cu_seqlens_q = paddle.to_tensor(cu_seqlens_q_v, dtype="int32")
        hidden_states = paddle.to_tensor(hidden_states_v, dtype="float32")

        out = extract_text_token_output(
            max_seq_len, max_seq_len_index, mm_token_num_len, seq_lens_this_time, cu_seqlens_q, hidden_states
        )[0]
        out_np = out.numpy()

        expect = np.ones((bsz, hidden_size), dtype="float32")
        for i in range(bsz):
            true_bsz = cu_seqlens_q_v[i + 1] - 1
            if (max_seq_len_v == mm_token_num_len_v) and (i == max_seq_len_index_v):
                expect[i, :] = 0.0
            else:
                if seq_lens_this_time_v[i] != 0:
                    expect[i, :] = hidden_states_v[true_bsz, :]

        if out_np.ndim == 1:
            np.testing.assert_allclose(out_np, expect[0], rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_allclose(out_np, expect, rtol=1e-5, atol=1e-5)

    def test_basic_case(self):
        bsz, hidden_size = 2, 4
        max_seq_len_v = 3
        max_seq_len_index_v = 0
        mm_token_num_len_v = 2
        seq_lens_this_time_v = [2, 1]
        cu_seqlens_q_v = [0, 2, 3]
        hidden_states_v = np.arange(12).reshape(3, 4).astype("float32")

        self._run_and_check(
            bsz,
            hidden_size,
            max_seq_len_v,
            max_seq_len_index_v,
            mm_token_num_len_v,
            seq_lens_this_time_v,
            cu_seqlens_q_v,
            hidden_states_v,
        )

    def test_zero_case(self):
        bsz, hidden_size = 2, 4
        max_seq_len_v = 5
        max_seq_len_index_v = 1
        mm_token_num_len_v = 5
        seq_lens_this_time_v = [1, 1]
        cu_seqlens_q_v = [0, 1, 2]
        hidden_states_v = np.random.randn(2, hidden_size).astype("float32")

        self._run_and_check(
            bsz,
            hidden_size,
            max_seq_len_v,
            max_seq_len_index_v,
            mm_token_num_len_v,
            seq_lens_this_time_v,
            cu_seqlens_q_v,
            hidden_states_v,
        )


if __name__ == "__main__":
    unittest.main()
