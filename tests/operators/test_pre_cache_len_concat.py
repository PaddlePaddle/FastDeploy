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

from fastdeploy.model_executor.ops.gpu import pre_cache_len_concat


def ref_pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, block_size):
    """
    Reference implementation.
    """
    bsz = len(seq_lens_this_time)
    cu_seqlens_k = np.zeros(bsz + 1, dtype=np.int32)
    batch_ids = []
    tile_ids_per_batch = []
    total_tokens = 0
    gridx = 0

    for bid in range(bsz):
        cache_len = int(seq_lens_decoder[bid])
        q_len = int(seq_lens_this_time[bid])
        if q_len <= 0:
            cache_len = 0
        loop_times = (cache_len + block_size - 1) // block_size  # div_up
        for tile_id in range(loop_times):
            batch_ids.append(bid)
            tile_ids_per_batch.append(tile_id)
        gridx += loop_times
        total_tokens += cache_len + q_len
        cu_seqlens_k[bid + 1] = total_tokens

    return (
        cu_seqlens_k,
        np.array(batch_ids, dtype=np.int32),
        np.array(tile_ids_per_batch, dtype=np.int32),
        np.array([gridx], dtype=np.int32),
        np.array([total_tokens], dtype=np.int32),
    )


class TestPreCacheLenConcat(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    def test_smoke_shapes(self):
        bsz = 3
        max_dec_len, block_size = 16, 4

        seq_lens_decoder = np.array([8, 4, 2], dtype=np.int32)
        seq_lens_this_time = np.array([2, 3, 1], dtype=np.int32)

        seq_lens_decoder_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
        seq_lens_this_time_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

        outputs = pre_cache_len_concat(seq_lens_decoder_t, seq_lens_this_time_t, max_dec_len, block_size)
        cu_seqlens_k, batch_ids, tile_ids, num_blocks, kv_token_num = [out.numpy() for out in outputs]

        # Shape checks
        self.assertEqual(cu_seqlens_k.shape[0], bsz + 1)
        self.assertEqual(batch_ids.shape, tile_ids.shape)
        self.assertEqual(num_blocks.shape, (1,))
        self.assertEqual(kv_token_num.shape, (1,))

        # Basic value sanity checks
        self.assertTrue(np.all(np.diff(cu_seqlens_k) >= 0))  # monotonic
        self.assertGreaterEqual(num_blocks[0], 0)
        self.assertGreaterEqual(kv_token_num[0], 0)

    def test_strict_values_with_ref(self):
        max_dec_len, block_size = 16, 4

        seq_lens_decoder = np.array([8, 4, 2], dtype=np.int32)
        seq_lens_this_time = np.array([2, 3, 1], dtype=np.int32)

        seq_lens_decoder_t = paddle.to_tensor(seq_lens_decoder, dtype="int32")
        seq_lens_this_time_t = paddle.to_tensor(seq_lens_this_time, dtype="int32")

        outputs = pre_cache_len_concat(seq_lens_decoder_t, seq_lens_this_time_t, max_dec_len, block_size)
        cu_seqlens_k, batch_ids, tile_ids, num_blocks, kv_token_num = [out.numpy() for out in outputs]

        # Reference implementation
        ref_outputs = ref_pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, block_size)
        ref_cu, ref_batch_ids, ref_tile_ids, ref_num_blocks, ref_kv_token_num = ref_outputs

        # Compare all outputs against reference
        np.testing.assert_array_equal(cu_seqlens_k, ref_cu)
        np.testing.assert_array_equal(batch_ids[: len(ref_batch_ids)], ref_batch_ids)
        np.testing.assert_array_equal(tile_ids[: len(ref_tile_ids)], ref_tile_ids)
        self.assertEqual(num_blocks[0], ref_num_blocks[0])
        self.assertEqual(kv_token_num[0], ref_kv_token_num[0])


if __name__ == "__main__":
    unittest.main()
