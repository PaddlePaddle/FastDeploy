import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import fused_get_rotary_embedding


class TestFusedGetRotaryEmbedding(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.batch_size = 2
        self.seq_len = 4
        self.head_dim = 8
        self.prompt_num = 0

    def test_basic_case(self):
        """Basic functionality test"""
        input_ids = paddle.randint(0, 100, [self.batch_size, self.seq_len], dtype="int32")
        position_ids = paddle.arange(self.seq_len).tile([self.batch_size, 1]).astype("float32")
        head_dim_tensor = paddle.arange(self.head_dim, dtype="int32")

        out = fused_get_rotary_embedding(input_ids, position_ids, head_dim_tensor, self.prompt_num)
        expect_shape = (2, self.batch_size, 1, self.seq_len, self.head_dim)
        self.assertEqual(tuple(out.shape), expect_shape)

    def test_minimal_head_dim(self):
        """Minimal head_dim test"""
        batch_size, seq_len, head_dim = 1, 2, 2
        input_ids = paddle.randint(0, 100, [batch_size, seq_len], dtype="int32")
        position_ids = paddle.arange(seq_len).tile([batch_size, 1]).astype("float32")
        head_dim_tensor = paddle.arange(head_dim, dtype="int32")

        out = fused_get_rotary_embedding(input_ids, position_ids, head_dim_tensor, 0)
        expect_shape = (2, batch_size, 1, seq_len, head_dim)
        self.assertEqual(tuple(out.shape), expect_shape)

    def test_different_prompt_num(self):
        """Test with different prompt_num"""
        prompt_num = 3
        input_ids = paddle.randint(0, 100, [self.batch_size, self.seq_len], dtype="int32")
        position_ids = paddle.arange(self.seq_len + prompt_num).tile([self.batch_size, 1]).astype("float32")
        head_dim_tensor = paddle.arange(self.head_dim, dtype="int32")

        out = fused_get_rotary_embedding(input_ids, position_ids, head_dim_tensor, prompt_num)
        expect_shape = (2, self.batch_size, 1, self.seq_len, self.head_dim)
        self.assertEqual(tuple(out.shape), expect_shape)


if __name__ == "__main__":
    unittest.main()
