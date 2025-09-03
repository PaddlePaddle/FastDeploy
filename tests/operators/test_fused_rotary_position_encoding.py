import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import fused_rotary_position_encoding


class TestFusedRotaryPositionEncoding(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _make_cos_sin_cache(self, num_tokens: int, rot_dim: int) -> np.ndarray:
        """Generate cos/sin cache."""
        assert rot_dim % 2 == 0, "rot_dim must be even"
        half_dim = rot_dim // 2
        cos_np = np.random.rand(num_tokens, half_dim).astype("float32")
        sin_np = np.random.rand(num_tokens, half_dim).astype("float32")
        return np.concatenate([cos_np, sin_np], axis=1)

    def _run_op(
        self,
        query_np: np.ndarray,
        key_np: np.ndarray,
        position_ids_np: np.ndarray,
        cos_sin_cache_np: np.ndarray,
        head_size: int,
        is_neox: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run fused_rotary_position_encoding operator."""
        query = paddle.to_tensor(query_np, dtype="float32")
        key = paddle.to_tensor(key_np, dtype="float32")
        position_ids = paddle.to_tensor(position_ids_np, dtype="int32")
        cos_sin_cache = paddle.to_tensor(cos_sin_cache_np, dtype="float32")

        fused_rotary_position_encoding(query, key, position_ids, cos_sin_cache, head_size, is_neox)
        return query.numpy(), key.numpy()

    def test_basic_case(self):
        num_tokens, num_heads, head_size = 4, 2, 6
        num_kv_heads, rot_dim = 2, 4

        query_np = np.random.rand(num_tokens, num_heads, head_size).astype("float32")
        key_np = np.random.rand(num_tokens, num_kv_heads, head_size).astype("float32")
        position_ids_np = np.arange(num_tokens, dtype="int32")
        cos_sin_cache_np = self._make_cos_sin_cache(num_tokens, rot_dim)

        query_out, key_out = self._run_op(
            query_np, key_np, position_ids_np, cos_sin_cache_np, head_size, is_neox=False
        )

        self.assertEqual(query_out.shape, query_np.shape)
        self.assertEqual(key_out.shape, key_np.shape)
        self.assertFalse(np.allclose(query_out, query_np))
        self.assertFalse(np.allclose(key_out, key_np))

    def test_neox_mode(self):
        """Test NEox mode."""
        num_tokens, num_heads, head_size = 3, 2, 8
        num_kv_heads, rot_dim = 2, 8

        query_np = np.random.rand(num_tokens, num_heads, head_size).astype("float32")
        key_np = np.random.rand(num_tokens, num_kv_heads, head_size).astype("float32")
        position_ids_np = np.arange(num_tokens, dtype="int32")
        cos_sin_cache_np = self._make_cos_sin_cache(num_tokens, rot_dim)

        query_out, key_out = self._run_op(query_np, key_np, position_ids_np, cos_sin_cache_np, head_size, is_neox=True)

        self.assertEqual(query_out.shape, query_np.shape)
        self.assertEqual(key_out.shape, key_np.shape)
        self.assertFalse(np.allclose(query_out, query_np))
        self.assertFalse(np.allclose(key_out, key_np))

    def test_large_num_tokens(self):
        """Test with a large number of tokens."""
        num_tokens, num_heads, head_size = 10, 2, 4
        num_kv_heads, rot_dim = 2, 4

        query_np = np.random.rand(num_tokens, num_heads, head_size).astype("float32")
        key_np = np.random.rand(num_tokens, num_kv_heads, head_size).astype("float32")
        position_ids_np = np.arange(num_tokens, dtype="int32")
        cos_sin_cache_np = self._make_cos_sin_cache(num_tokens, rot_dim)

        query_out, key_out = self._run_op(
            query_np, key_np, position_ids_np, cos_sin_cache_np, head_size, is_neox=False
        )

        self.assertEqual(query_out.shape, query_np.shape)
        self.assertEqual(key_out.shape, key_np.shape)

    def test_exceed_max_tokens(self):
        """Test exceeding maximum number of tokens."""
        num_tokens, num_heads, head_size = 65537, 1, 4
        num_kv_heads, rot_dim = 1, 4

        query_np = np.random.rand(num_tokens, num_heads, head_size).astype("float32")
        key_np = np.random.rand(num_tokens, num_kv_heads, head_size).astype("float32")
        position_ids_np = np.arange(num_tokens, dtype="int32")
        cos_sin_cache_np = self._make_cos_sin_cache(num_tokens, rot_dim)

        with self.assertRaises(Exception):
            self._run_op(query_np, key_np, position_ids_np, cos_sin_cache_np, head_size, is_neox=False)


if __name__ == "__main__":
    unittest.main()
