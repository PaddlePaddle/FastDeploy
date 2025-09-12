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

from fastdeploy.model_executor.ops.gpu import fused_get_rotary_embedding


def numpy_rope(position_ids, head_dim, prompt_num=0, seq_len=None):
    """Numpy reference implementation"""
    batch_size, max_seq_len = position_ids.shape
    if seq_len is None:
        seq_len = max_seq_len - prompt_num

    inv_head_dim = 1.0 / float(head_dim)
    rope_embedding = np.empty((2, batch_size, 1, seq_len, head_dim), dtype=np.float32)

    for b in range(batch_size):
        for s in range(seq_len):
            pos = position_ids[b, s + prompt_num]
            for h in range(0, head_dim, 2):
                exponent_factor = -float(h) * inv_head_dim
                inv_freq = np.power(10000.0, exponent_factor)
                val = pos * inv_freq
                cos_val, sin_val = np.cos(val), np.sin(val)
                rope_embedding[0, b, 0, s, h : h + 2] = cos_val
                rope_embedding[1, b, 0, s, h : h + 2] = sin_val
    return rope_embedding


class TestFusedGetRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.batch_size = 2
        self.seq_len = 4
        self.head_dim = 8

    def _run_and_check(self, batch_size, seq_len, head_dim, prompt_num=0):
        input_ids = paddle.randint(0, 100, [batch_size, seq_len], dtype="int32")
        position_ids = paddle.arange(seq_len + 2 * prompt_num).tile([batch_size, 1]).astype("float32")

        head_dim_tensor = paddle.arange(head_dim, dtype="int32")

        out = fused_get_rotary_embedding(input_ids, position_ids, head_dim_tensor, prompt_num)
        out_np = out.numpy()
        ref = numpy_rope(position_ids.numpy(), head_dim, prompt_num, seq_len=seq_len)

        # check shape
        expect_shape = (2, batch_size, 1, seq_len, head_dim)
        self.assertEqual(tuple(out.shape), expect_shape)

        # check values
        np.testing.assert_allclose(out_np, ref, rtol=1e-5, atol=1e-6)

    def test_basic_case(self):
        self._run_and_check(self.batch_size, self.seq_len, self.head_dim)

    def test_minimal_head_dim(self):
        self._run_and_check(batch_size=1, seq_len=2, head_dim=2)

    def test_with_prompt_num(self):
        self._run_and_check(self.batch_size, self.seq_len, self.head_dim, prompt_num=3)


if __name__ == "__main__":
    unittest.main()
