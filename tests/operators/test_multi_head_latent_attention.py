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

import math
import unittest

import numpy as np
import paddle

try:
    from fastdeploy.model_executor.ops.gpu import multi_head_latent_attention
except (ImportError, AttributeError):
    multi_head_latent_attention = None


def _get_sm_version():
    """Get CUDA SM version (e.g., 90 for H100, 80 for A100)."""
    try:
        if not paddle.device.is_compiled_with_cuda():
            return 0
        if paddle.device.cuda.device_count() == 0:
            return 0
        prop = paddle.device.cuda.get_device_properties()
        return prop.major * 10 + prop.minor
    except Exception:
        return 0


SM_VERSION = _get_sm_version()
HAS_CUDA = SM_VERSION > 0
OP_AVAILABLE = multi_head_latent_attention is not None


def _reference_mla_decode(
    query_np,
    kv_cache_np,
    block_tables_np,
    seq_len,
    q_num_heads,
    kv_num_heads,
    head_dim_qk,
    head_dim_v,
    block_size,
    softmax_scale,
):
    """NumPy reference for MLA decode attention with paged KV cache + GQA."""
    token_num = query_np.shape[0]
    q = query_np.reshape(token_num, q_num_heads, head_dim_qk).astype(np.float64)

    keys, values = [], []
    for pos in range(seq_len):
        blk = block_tables_np[0, pos // block_size]
        off = pos % block_size
        keys.append(kv_cache_np[blk, :, off, :head_dim_qk])
        values.append(kv_cache_np[blk, :, off, :head_dim_v])

    k = np.repeat(np.stack(keys, axis=0).astype(np.float64), q_num_heads // kv_num_heads, axis=1)
    v = np.repeat(np.stack(values, axis=0).astype(np.float64), q_num_heads // kv_num_heads, axis=1)

    out = np.zeros((token_num, q_num_heads, head_dim_v), dtype=np.float64)
    for t in range(token_num):
        for h in range(q_num_heads):
            scores = q[t : t + 1, h, :] @ k[:, h, :].T * softmax_scale
            scores_exp = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probs = scores_exp / scores_exp.sum(axis=-1, keepdims=True)
            out[t, h, :] = probs @ v[:, h, :]
    return out.reshape(token_num, q_num_heads * head_dim_v).astype(np.float32)


@unittest.skipUnless(
    HAS_CUDA and OP_AVAILABLE,
    "Requires CUDA GPU and compiled FastDeploy custom ops.",
)
class TestMultiHeadLatentAttention(unittest.TestCase):
    """Tests for multi_head_latent_attention — MLA decode correctness."""

    def setUp(self):
        paddle.set_device("gpu")
        np.random.seed(42)
        self.sm = SM_VERSION
        self.batch_size, self.token_num = 1, 1
        self.q_num_heads, self.kv_num_heads = 8, 1
        # MLA dimensions: q_head_dim = nope_size + pe_dim = 512 + 64 = 576
        self.head_dim_qk = 576  # kernel requires exactly 576
        self.head_dim_v = 512  # kv_lora_rank (nope_size), used for value/output
        self.block_size = 64
        self.max_blocks = 2
        self.max_seq_len = self.max_blocks * self.block_size
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim_qk)

    def _build_inputs(self, dtype_str="bfloat16", seq_len=5):
        """Build all input tensors for the MLA op."""
        total_blocks = self.batch_size * self.max_blocks
        q_hidden = self.q_num_heads * self.head_dim_qk

        query_fp32 = paddle.to_tensor(np.random.randn(self.token_num, q_hidden).astype(np.float32))
        kv_fp32 = paddle.to_tensor(
            np.random.randn(total_blocks, self.kv_num_heads, self.block_size, self.head_dim_qk).astype(np.float32)
        )

        if dtype_str == "float32":
            query, kv = query_fp32, kv_fp32
        else:
            query, kv = query_fp32.cast(dtype_str), kv_fp32.cast(dtype_str)

        query_ref = query.cast("float32").numpy()
        kv_ref = kv.cast("float32").numpy()
        block_tables = paddle.arange(self.max_blocks, dtype="int32").unsqueeze(0)

        compute_dtype = "bf16" if dtype_str == "bfloat16" else "fp16"
        # max_dec_len_this_time and max_len_kv are read on the HOST via
        # .data<int>()[0], so they MUST be CPU tensors.
        max_dec_len_cpu = paddle.to_tensor(np.array([self.token_num], dtype="int32")).cpu()
        max_len_kv_cpu = paddle.to_tensor(np.array([seq_len + self.token_num], dtype="int32")).cpu()
        # kv_num_blocks is passed as CPU/pinned tensor in production
        kv_num_blocks_cpu = paddle.to_tensor(np.array([1], dtype="int32")).cpu()
        args = [
            query,
            kv,
            kv,
            paddle.to_tensor([seq_len], dtype="int32"),
            paddle.to_tensor([self.token_num], dtype="int32"),
            paddle.to_tensor([0, self.token_num], dtype="int32"),
            paddle.zeros([self.token_num], dtype="int32"),
            block_tables,
            paddle.zeros([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
            kv_num_blocks_cpu,
            paddle.zeros([1], dtype="int32"),
            paddle.zeros([1], dtype="int32"),
            paddle.to_tensor([1], dtype="int32"),
            paddle.to_tensor([self.block_size], dtype="int32"),
            max_dec_len_cpu,
            max_len_kv_cpu,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            compute_dtype,
            "none",
            self.head_dim_v,
            self.max_seq_len,
            self.softmax_scale,
            0.0,
            0.0,
            0.0,
            0,
            True,
            False,
        ]
        return args, query_ref, kv_ref, block_tables.numpy()

    def _check_output(self, dtype_str, seq_len=5):
        """Run op and compare against NumPy reference."""
        if self.sm < 90:
            self.skipTest("MLA kernel requires SM >= 90 (H100+).")
        args, q_ref, kv_ref, bt_np = self._build_inputs(dtype_str=dtype_str, seq_len=seq_len)
        out = multi_head_latent_attention(*args).cast("float32").numpy()
        ref = _reference_mla_decode(
            q_ref,
            kv_ref,
            bt_np,
            seq_len,
            self.q_num_heads,
            self.kv_num_heads,
            self.head_dim_qk,
            self.head_dim_v,
            self.block_size,
            self.softmax_scale,
        )
        expected_shape = [self.token_num, self.q_num_heads * self.head_dim_v]
        self.assertEqual(list(out.shape), expected_shape)
        rtol = 5e-2 if dtype_str == "bfloat16" else 1e-2
        np.testing.assert_allclose(out, ref, rtol=rtol, atol=rtol)

    def test_decode_correctness_bf16(self):
        """BF16 single-token decode correctness against NumPy reference."""
        self._check_output("bfloat16")

    def test_decode_correctness_fp16(self):
        """FP16 single-token decode correctness against NumPy reference."""
        self._check_output("float16")


if __name__ == "__main__":
    unittest.main()
