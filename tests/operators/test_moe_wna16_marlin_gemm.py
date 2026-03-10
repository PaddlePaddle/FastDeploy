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

from fastdeploy.model_executor.ops.gpu import (
    MoeWna16MarlinGemmApi,
    gptq_marlin_repack,
    tritonmoe_preprocess_func,
)

paddle.seed(42)
np.random.seed(42)


def _get_sm_version():
    """Return GPU compute capability as a float, e.g. 8.0 for SM80."""
    if not paddle.is_compiled_with_cuda():
        return 0.0
    try:
        prop = paddle.device.cuda.get_device_properties()
        return float(f"{prop.major}.{prop.minor}")
    except Exception:
        return 0.0


def _quantize_to_uint4b8(weight_fp16):
    """Per-channel symmetric quantization to uint4b8 (zero_point=8)."""
    K, N = weight_fp16.shape
    W = weight_fp16.astype(np.float32)
    amax = np.maximum(np.abs(W).max(axis=0), 1e-10)
    scales = (amax / 7.0).astype(np.float32)
    q = np.clip(np.round(W / scales + 8.0), 0, 15).astype(np.uint8)
    return q, scales.reshape(1, N).astype(np.float16)


def _pack_gptq_int32(q_values):
    """Pack [K,N] uint8 values into GPTQ int32 layout [K//8, N]."""
    K, N = q_values.shape
    packed = np.zeros((K // 8, N), dtype=np.int32)
    for offset in range(8):
        packed |= q_values[offset::8, :].astype(np.int32) << (4 * offset)
    return packed


def _dequantize_uint4b8(q_values, scales):
    """Dequantize uint4b8 values to float32."""
    return (q_values.astype(np.float32) - 8.0) * scales.astype(np.float32)


def _build_marlin_weights(weights_list, K, N):
    """Quantize, GPTQ-pack, and Marlin-repack a list of expert weights."""
    perm = paddle.empty([0], dtype="int32")
    marlin_per_expert, all_q, all_s = [], [], []
    for w_fp16 in weights_list:
        q, s = _quantize_to_uint4b8(w_fp16)
        all_q.append(q)
        all_s.append(s)
        packed_t = paddle.to_tensor(_pack_gptq_int32(q), dtype="int32", place=paddle.CUDAPlace(0))
        marlin_per_expert.append(gptq_marlin_repack(packed_t, perm, size_k=K, size_n=N, num_bits=4))
    b_q_weight = paddle.stack(marlin_per_expert, axis=0)
    b_scales = paddle.to_tensor(np.stack(all_s, axis=0), dtype="float16", place=paddle.CUDAPlace(0))
    return b_q_weight, b_scales, all_q, all_s


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and _get_sm_version() >= 8.0,
    "Requires CUDA GPU with SM80+ (Ampere or newer)",
)
class TestMoeWna16MarlinGemm(unittest.TestCase):
    """Tests for moe_wna16_marlin_gemm — quantized MoE GEMM correctness."""

    E, K, N, BLOCK_M = 8, 256, 256, 16

    def setUp(self):
        paddle.set_device("gpu")

    def _make_inputs(self, M=16, top_k=1, seed=42):
        """Build all tensors needed by MoeWna16MarlinGemmApi."""
        np.random.seed(seed)
        a_np = (np.random.randn(M, self.K) * 0.1).astype(np.float16)
        a = paddle.to_tensor(a_np, dtype="float16", place=paddle.CUDAPlace(0))
        ws = [(np.random.randn(self.K, self.N) * 0.05).astype(np.float16) for _ in range(self.E)]
        b_q_weight, b_scales, q_vals, scales = _build_marlin_weights(ws, self.K, self.N)
        topk_ids_np = np.random.randint(0, self.E, size=(M, top_k)).astype(np.int64)
        topk_ids = paddle.to_tensor(topk_ids_np, dtype="int64", place=paddle.CUDAPlace(0))
        topk_w_np = np.random.rand(M, top_k).astype(np.float32)
        topk_weights = paddle.to_tensor(topk_w_np, dtype="float32", place=paddle.CUDAPlace(0))
        sorted_ids, expert_ids, ntokens_pp = tritonmoe_preprocess_func(topk_ids, self.E, self.BLOCK_M)
        return dict(
            a=a,
            a_np=a_np,
            b_q_weight=b_q_weight,
            b_scales=b_scales,
            topk_ids=topk_ids,
            topk_ids_np=topk_ids_np,
            topk_weights=topk_weights,
            topk_w_np=topk_w_np,
            sorted_ids=sorted_ids,
            expert_ids=expert_ids,
            ntokens_pp=ntokens_pp,
            workspace=paddle.empty([528], dtype="int32"),
            q_vals=q_vals,
            scales=scales,
        )

    def _check_output(self, M, top_k, mul_topk_weights=False):
        """Run kernel, compute NumPy reference, assert_allclose."""
        inp = self._make_inputs(M=M, top_k=top_k)
        out = MoeWna16MarlinGemmApi(
            inp["a"],
            c_or_none=None,
            b_q_weight=inp["b_q_weight"],
            b_scales=inp["b_scales"],
            global_scale_or_none=None,
            b_zeros_or_none=None,
            g_idx_or_none=None,
            perm_or_none=None,
            workspace=inp["workspace"],
            sorted_token_ids=inp["sorted_ids"],
            expert_ids=inp["expert_ids"],
            num_tokens_post_padded=inp["ntokens_pp"],
            topk_weights=inp["topk_weights"],
            moe_block_size=self.BLOCK_M,
            top_k=top_k,
            mul_topk_weights=mul_topk_weights,
            is_ep=False,
            b_q_type_str="uint4b8",
            size_m=M,
            size_n=self.N,
            size_k=self.K,
            is_k_full=True,
            use_atomic_add=True,
            use_fp32_reduce=True,
            is_zp_float=False,
        )[0]
        # NumPy reference
        a_fp32 = inp["a_np"].astype(np.float32)
        ids, w = inp["topk_ids_np"], inp["topk_w_np"]
        ref = np.zeros((M * top_k, self.N), dtype=np.float32)
        for i in range(M):
            for j in range(top_k):
                W_deq = _dequantize_uint4b8(inp["q_vals"][ids[i, j]], inp["scales"][ids[i, j]])
                row = a_fp32[i] @ W_deq
                if mul_topk_weights:
                    row *= w[i, j]
                ref[i * top_k + j] = row
        self.assertEqual(list(out.shape), [M * top_k, self.N])
        self.assertEqual(out.dtype, paddle.float16)
        np.testing.assert_allclose(out.numpy().astype(np.float32), ref, rtol=5e-2, atol=5e-2)

    def test_topk1(self):
        """top_k=1, no weight multiplication."""
        self._check_output(M=16, top_k=1, mul_topk_weights=False)

    def test_topk2_mul_weights(self):
        """top_k=2 with routing weight scaling."""
        self._check_output(M=16, top_k=2, mul_topk_weights=True)

    def test_various_sizes(self):
        """Multiple M values with top_k=1."""
        for M in (1, 8, 32):
            with self.subTest(M=M):
                self._check_output(M=M, top_k=1)


if __name__ == "__main__":
    unittest.main()
