import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import pre_cache_len_concat


class TestPreCacheLenConcat(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    def test_basic_functionality(self):
        batch_size = 3
        max_dec_len, block_size = 16, 4

        seq_lens_decoder_np = np.array([8, 4, 2], dtype=np.int32)
        seq_lens_this_time_np = np.array([2, 3, 1], dtype=np.int32)

        seq_lens_decoder = paddle.to_tensor(seq_lens_decoder_np, dtype="int32")
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_np, dtype="int32")

        outputs = pre_cache_len_concat(seq_lens_decoder, seq_lens_this_time, max_dec_len, block_size)
        cu_seqlens_k, batch_ids, tile_ids, num_blocks, kv_token_num = [out.numpy() for out in outputs]

        # Shape checks
        self.assertEqual(cu_seqlens_k.shape[0], batch_size + 1)
        self.assertEqual(batch_ids.shape, tile_ids.shape)
        self.assertEqual(num_blocks.shape, (1,))
        self.assertEqual(kv_token_num.shape, (1,))

        # Basic value checks
        self.assertTrue(np.all(np.diff(cu_seqlens_k) >= 0))
        self.assertGreaterEqual(num_blocks[0], 0)
        self.assertGreaterEqual(kv_token_num[0], 0)

        # # kv_token_num equals cu_seqlens_k[-1]
        self.assertEqual(kv_token_num[0], cu_seqlens_k[-1])


if __name__ == "__main__":
    unittest.main()
