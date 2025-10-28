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

from fastdeploy.model_executor.ops.gpu import fused_rotary_position_encoding


class TestFusedRotaryPositionEncoding(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)

    def _make_cos_sin_cache(self, max_position: int, rot_dim: int) -> np.ndarray:
        """Generate cos/sin cache."""
        assert rot_dim % 2 == 0, "rot_dim must be even"
        half_dim = rot_dim // 2
        inv_freq = 1.0 / (10000 ** (np.arange(0, half_dim).astype("float32") / half_dim))
        positions = np.arange(max_position, dtype="float32")
        freqs = np.outer(positions, inv_freq)  # [max_position, half_dim]
        cos_np = np.cos(freqs)
        sin_np = np.sin(freqs)
        return np.concatenate([cos_np, sin_np], axis=1).astype("float32")

    def _ref_rotary(self, query, key, position_ids, cos_sin_cache, head_size, is_neox):
        """Numpy reference implementation."""
        num_tokens, num_heads, _ = query.shape
        num_kv_heads = key.shape[1]
        rot_dim = cos_sin_cache.shape[1]
        embed_dim = rot_dim // 2

        query_ref = query.copy()
        key_ref = key.copy()

        for t in range(num_tokens):
            pos = position_ids[t]
            cos_ptr = cos_sin_cache[pos, :embed_dim]
            sin_ptr = cos_sin_cache[pos, embed_dim:]

            for h in range(num_heads):
                arr = query_ref[t, h]
                for i in range(embed_dim):
                    if is_neox:
                        x_idx, y_idx = i, embed_dim + i
                        cos, sin = cos_ptr[i], sin_ptr[i]
                    else:
                        x_idx, y_idx = 2 * i, 2 * i + 1
                        cos, sin = cos_ptr[i], sin_ptr[i]
                    x, y = arr[x_idx], arr[y_idx]
                    arr[x_idx] = x * cos - y * sin
                    arr[y_idx] = y * cos + x * sin

            for h in range(num_kv_heads):
                arr = key_ref[t, h]
                for i in range(embed_dim):
                    if is_neox:
                        x_idx, y_idx = i, embed_dim + i
                        cos, sin = cos_ptr[i], sin_ptr[i]
                    else:
                        x_idx, y_idx = 2 * i, 2 * i + 1
                        cos, sin = cos_ptr[i], sin_ptr[i]
                    x, y = arr[x_idx], arr[y_idx]
                    arr[x_idx] = x * cos - y * sin
                    arr[y_idx] = y * cos + x * sin

        return query_ref, key_ref

    def _run_op(
        self,
        query_np: np.ndarray,
        key_np: np.ndarray,
        position_ids_np: np.ndarray,
        cos_sin_cache_np: np.ndarray,
        head_size: int,
        is_neox: bool,
    ):
        """Run fused_rotary_position_encoding operator."""
        query = paddle.to_tensor(query_np, dtype="float32")
        key = paddle.to_tensor(key_np, dtype="float32")
        position_ids = paddle.to_tensor(position_ids_np, dtype="int32")
        cos_sin_cache = paddle.to_tensor(cos_sin_cache_np, dtype="float32")

        fused_rotary_position_encoding(query, key, position_ids, cos_sin_cache, head_size, is_neox)
        return query.numpy(), key.numpy()

    def _check_correctness(self, num_tokens, num_heads, num_kv_heads, head_size, rot_dim, is_neox):
        query_np = np.random.rand(num_tokens, num_heads, head_size).astype("float32")
        key_np = np.random.rand(num_tokens, num_kv_heads, head_size).astype("float32")
        position_ids_np = np.arange(num_tokens, dtype="int32")
        cos_sin_cache_np = self._make_cos_sin_cache(num_tokens, rot_dim)

        query_out, key_out = self._run_op(query_np, key_np, position_ids_np, cos_sin_cache_np, head_size, is_neox)
        query_ref, key_ref = self._ref_rotary(query_np, key_np, position_ids_np, cos_sin_cache_np, head_size, is_neox)

        np.testing.assert_allclose(query_out, query_ref, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(key_out, key_ref, rtol=1e-5, atol=1e-6)

    def test_basic_case(self):
        self._check_correctness(num_tokens=4, num_heads=2, num_kv_heads=2, head_size=6, rot_dim=4, is_neox=False)

    def test_neox_mode(self):
        self._check_correctness(num_tokens=3, num_heads=2, num_kv_heads=2, head_size=8, rot_dim=8, is_neox=True)

    def test_large_num_tokens(self):
        self._check_correctness(num_tokens=10, num_heads=2, num_kv_heads=2, head_size=4, rot_dim=4, is_neox=False)

    def test_exceed_max_tokens(self):
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
