# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import unittest

import paddle

from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
    flash_attn_func,
)


class TestFlashAttnFunc(unittest.TestCase):
    def setUp(self):
        """
        Set up the testing environment before each test..
        """
        paddle.set_device("gpu")
        paddle.set_default_dtype("bfloat16")
        prop = paddle.device.cuda.get_device_properties()
        self.sm_version = prop.major * 10 + prop.minor

    def test_fa3(self):
        if self.sm_version < 89 or self.sm_version >= 100:
            self.skipTest("Flash Attention V3 requires SM89+ but less than SM100.")
        head_dim = 128
        num_heads = 12
        kv_num_heads = 4
        seq_len = 1024
        batch_size = 4
        token_num = batch_size * seq_len
        q = paddle.rand((token_num, num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        k = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        v = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        cu_seqlens_q = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        cu_seqlens_k = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        max_seqlen_q = seq_len
        max_seqlen_k = seq_len
        attn_mask_q = None
        paddle.set_flags({"FLAGS_flash_attn_version": 3})
        flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_mask_q=attn_mask_q,
            causal=True,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            version=3,
        )

    def test_fa3_with_mask(self):
        if self.sm_version < 89 or self.sm_version >= 100:
            self.skipTest("Flash Attention V3 requires SM89+ but less than SM100.")
        head_dim = 128
        num_heads = 12
        kv_num_heads = 4
        seq_len = 1024
        batch_size = 4
        token_num = batch_size * seq_len
        q = paddle.rand((token_num, num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        k = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        v = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        cu_seqlens_q = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        cu_seqlens_k = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        max_seqlen_q = seq_len
        max_seqlen_k = seq_len

        attn_mask_q = paddle.zeros([1, 1, token_num, 4], dtype=paddle.int32)
        for bid in range(batch_size):
            attn_mask_q[:, :, seq_len * bid : seq_len * (bid + 1), :2] = seq_len * (bid + 1)
        for kv_token_id in range(token_num):
            attn_mask_q[:, :, kv_token_id, 3] = kv_token_id
        paddle.set_flags({"FLAGS_flash_attn_version": 3})
        flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_mask_q=attn_mask_q,
            causal=True,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            version=3,
        )

    def test_fa2(self):
        head_dim = 128
        num_heads = 12
        kv_num_heads = 4
        seq_len = 1024
        batch_size = 4
        token_num = batch_size * seq_len
        q = paddle.rand((token_num, num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        k = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        v = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        cu_seqlens_q = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        cu_seqlens_k = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        max_seqlen_q = seq_len
        max_seqlen_k = seq_len
        attn_mask_q = None
        paddle.set_flags({"FLAGS_flash_attn_version": 2})
        flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_mask_q=attn_mask_q,
            causal=True,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            version=2,
        )

    def test_fa2_with_mask(self):
        head_dim = 128
        num_heads = 12
        kv_num_heads = 4
        seq_len = 1024
        batch_size = 4
        token_num = batch_size * seq_len
        q = paddle.rand((token_num, num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        k = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        v = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        cu_seqlens_q = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        cu_seqlens_k = paddle.arange(0, token_num + seq_len, seq_len, dtype=paddle.int32)
        max_seqlen_q = seq_len
        max_seqlen_k = seq_len

        attn_mask_q = paddle.zeros([1, 1, token_num, 4], dtype=paddle.int32)
        for bid in range(batch_size):
            attn_mask_q[:, :, seq_len * bid : seq_len * (bid + 1), :2] = seq_len * (bid + 1)
        for kv_token_id in range(token_num):
            attn_mask_q[:, :, kv_token_id, 3] = kv_token_id
        paddle.set_flags({"FLAGS_flash_attn_version": 2})
        flash_attn_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            attn_mask_q=attn_mask_q,
            causal=True,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            version=2,
        )

    def test_fa4(self):
        if self.sm_version < 100:
            self.skipTest("Flash Attention V4 requires SM100+.")
        head_dim = 128
        num_heads = 12
        kv_num_heads = 4
        seq_len = 1024
        batch_size = 4
        token_num = batch_size * seq_len
        q = paddle.rand((token_num, num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        k = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")
        v = paddle.rand((token_num, kv_num_heads, head_dim), dtype=paddle.float32).cast("bfloat16")

        attn_mask_q = paddle.zeros([1, 1, token_num, 4], dtype=paddle.int32)
        for bid in range(batch_size):
            attn_mask_q[:, :, seq_len * bid : seq_len * (bid + 1), :2] = seq_len * (bid + 1)
        for kv_token_id in range(token_num):
            attn_mask_q[:, :, kv_token_id, 3] = kv_token_id
        flash_attn_func(
            q,
            k,
            v,
            attn_mask_q=attn_mask_q,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_dim=head_dim,
            version=4,
        )


if __name__ == "__main__":
    unittest.main()
