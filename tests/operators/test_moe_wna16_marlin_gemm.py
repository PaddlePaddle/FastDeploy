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


# ---------------------------------------------------------------------------
#  Quantization helpers (uint4b8 = unsigned 4-bit, zero_point = 8)
# ---------------------------------------------------------------------------


def _quantize_to_uint4b8(weight_fp16):
    """Per-channel symmetric quantization of [K, N] FP16 weight to uint4b8.

    Returns
    -------
    q_values : ndarray [K, N] uint8  – quantized values in [0, 15]
    scales   : ndarray [1, N] float16 – per-channel scale
    """
    K, N = weight_fp16.shape
    W = weight_fp16.astype(np.float32)
    amax = np.maximum(np.abs(W).max(axis=0), 1e-10)  # [N]
    scales = (amax / 7.0).astype(np.float32)  # max positive int = 15-8 = 7
    q = np.clip(np.round(W / scales + 8.0), 0, 15).astype(np.uint8)
    return q, scales.reshape(1, N).astype(np.float16)


def _pack_gptq_int32(q_values):
    """Pack [K, N] uint8 quantised values into GPTQ int32 layout [K//8, N].

    Eight consecutive values along K are packed into one int32
    (value k%8 occupies bits [4*(k%8) : 4*(k%8)+4]).
    """
    K, N = q_values.shape
    assert K % 8 == 0
    packed = np.zeros((K // 8, N), dtype=np.int32)
    for offset in range(8):
        packed |= q_values[offset::8, :].astype(np.int32) << (4 * offset)
    return packed


def _dequantize_uint4b8(q_values, scales):
    """Dequantize [K, N] uint4b8 values to float32 using [1, N] scales."""
    return (q_values.astype(np.float32) - 8.0) * scales.astype(np.float32)


def _build_marlin_weights(weights_list, K, N):
    """Quantize, GPTQ-pack, and Marlin-repack a list of expert weights.

    Parameters
    ----------
    weights_list : list of ndarray [K, N] float16 – one per expert
    K, N         : int – weight dimensions

    Returns
    -------
    b_q_weight   : Tensor [E, K//16, N*2] int32  (Marlin layout)
    b_scales     : Tensor [E, 1, N] float16
    q_vals       : list of ndarray [K, N] uint8   (for reference dequant)
    scales       : list of ndarray [1, N] float16
    """
    perm = paddle.empty([0], dtype="int32")  # no act_order
    marlin_per_expert = []
    all_q, all_s = [], []

    for w_fp16 in weights_list:
        q, s = _quantize_to_uint4b8(w_fp16)
        all_q.append(q)
        all_s.append(s)

        packed = _pack_gptq_int32(q)
        packed_t = paddle.to_tensor(packed, dtype="int32", place=paddle.CUDAPlace(0))
        repacked = gptq_marlin_repack(packed_t, perm, size_k=K, size_n=N, num_bits=4)
        marlin_per_expert.append(repacked)

    b_q_weight = paddle.stack(marlin_per_expert, axis=0)
    b_scales = paddle.to_tensor(
        np.stack(all_s, axis=0),
        dtype="float16",
        place=paddle.CUDAPlace(0),
    )
    return b_q_weight, b_scales, all_q, all_s


# ---------------------------------------------------------------------------
#  Test class
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    paddle.is_compiled_with_cuda() and _get_sm_version() >= 8.0,
    "Requires CUDA GPU with SM80+ (Ampere or newer)",
)
class TestMoeWna16MarlinGemm(unittest.TestCase):
    """Unit tests for the moe_wna16_marlin_gemm custom CUDA operator."""

    NUM_EXPERTS = 8
    K = 128  # input dim  – must be divisible by tile_k_size (16)
    N = 64  # output dim – must be divisible by tile_n_size (64)
    BLOCK_M = 16  # MoE dispatch block size

    def setUp(self):
        paddle.set_device("gpu")

    # ----- helpers ----------------------------------------------------------

    def _make_inputs(self, M=16, top_k=1, seed=42):
        """Build all tensors needed to invoke MoeWna16MarlinGemmApi."""
        np.random.seed(seed)

        # Activations
        a_np = (np.random.randn(M, self.K) * 0.1).astype(np.float16)
        a = paddle.to_tensor(a_np, dtype="float16", place=paddle.CUDAPlace(0))

        # Expert weights (small magnitude to keep quantization error bounded)
        ws = [(np.random.randn(self.K, self.N) * 0.05).astype(np.float16) for _ in range(self.NUM_EXPERTS)]

        b_q_weight, b_scales, q_vals, scales = _build_marlin_weights(
            ws,
            self.K,
            self.N,
        )

        # Expert assignments & routing weights
        topk_ids_np = np.random.randint(
            0,
            self.NUM_EXPERTS,
            size=(M, top_k),
        ).astype(np.int64)
        topk_ids = paddle.to_tensor(
            topk_ids_np,
            dtype="int64",
            place=paddle.CUDAPlace(0),
        )

        topk_w_np = np.random.rand(M, top_k).astype(np.float32)
        topk_weights = paddle.to_tensor(
            topk_w_np,
            dtype="float32",
            place=paddle.CUDAPlace(0),
        )

        # MoE dispatch
        sorted_ids, expert_ids, ntokens_pp = tritonmoe_preprocess_func(
            topk_ids,
            self.NUM_EXPERTS,
            self.BLOCK_M,
        )

        workspace = paddle.empty([528], dtype="int32")

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
            workspace=workspace,
            q_vals=q_vals,
            scales=scales,
        )

    def _run_kernel(self, inp, M, top_k, mul_topk_weights=False):
        """Invoke MoeWna16MarlinGemmApi and return the output tensor."""
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
        )
        return out[0]

    def _reference(self, inp, M, top_k, mul_topk_weights=False):
        """NumPy reference: dequant → matmul per (token, expert) pair."""
        a_fp32 = inp["a_np"].astype(np.float32)
        ids = inp["topk_ids_np"]
        w = inp["topk_w_np"]
        ref = np.zeros((M * top_k, self.N), dtype=np.float32)
        for i in range(M):
            for j in range(top_k):
                eidx = ids[i, j]
                W_deq = _dequantize_uint4b8(inp["q_vals"][eidx], inp["scales"][eidx])
                row = a_fp32[i] @ W_deq
                if mul_topk_weights:
                    row *= w[i, j]
                ref[i * top_k + j] = row
        return ref

    # ----- A: Numerical Correctness ----------------------------------------

    def test_correctness_topk1(self):
        """top_k=1, no weight multiplication — basic GEMM correctness."""
        M, top_k = 16, 1
        inp = self._make_inputs(M=M, top_k=top_k)
        actual = self._run_kernel(inp, M, top_k, mul_topk_weights=False)
        expected = self._reference(inp, M, top_k, mul_topk_weights=False)
        np.testing.assert_allclose(
            actual.numpy().astype(np.float32),
            expected,
            rtol=5e-2,
            atol=5e-2,
        )

    def test_correctness_topk2_mul_weights(self):
        """top_k=2, mul_topk_weights=True — MoE routing with weight scaling."""
        M, top_k = 16, 2
        inp = self._make_inputs(M=M, top_k=top_k)
        actual = self._run_kernel(inp, M, top_k, mul_topk_weights=True)
        expected = self._reference(inp, M, top_k, mul_topk_weights=True)
        np.testing.assert_allclose(
            actual.numpy().astype(np.float32),
            expected,
            rtol=5e-2,
            atol=5e-2,
        )

    # ----- B: Shape Validation ---------------------------------------------

    def test_output_shape_topk1(self):
        """Output shape is [M*top_k, N] for top_k=1."""
        M, top_k = 8, 1
        inp = self._make_inputs(M=M, top_k=top_k)
        out = self._run_kernel(inp, M, top_k)
        self.assertEqual(list(out.shape), [M * top_k, self.N])

    def test_output_shape_topk2(self):
        """Output shape is [M*top_k, N] for top_k=2."""
        M, top_k = 8, 2
        inp = self._make_inputs(M=M, top_k=top_k)
        out = self._run_kernel(inp, M, top_k)
        self.assertEqual(list(out.shape), [M * top_k, self.N])

    # ----- C: Dtype ---------------------------------------------------------

    def test_output_dtype_fp16(self):
        """Output dtype matches input dtype (float16)."""
        M, top_k = 4, 1
        inp = self._make_inputs(M=M, top_k=top_k)
        out = self._run_kernel(inp, M, top_k)
        self.assertEqual(out.dtype, paddle.float16)

    # ----- D: Edge Cases ----------------------------------------------------

    def test_single_token(self):
        """Correctly handles M=1 (single token)."""
        M, top_k = 1, 1
        inp = self._make_inputs(M=M, top_k=top_k)
        out = self._run_kernel(inp, M, top_k)
        self.assertEqual(list(out.shape), [1, self.N])

    def test_zero_input(self):
        """All-zero input produces all-zero output."""
        M, top_k = 4, 1
        inp = self._make_inputs(M=M, top_k=top_k)
        inp["a"] = paddle.zeros([M, self.K], dtype="float16")
        out = self._run_kernel(inp, M, top_k)
        np.testing.assert_allclose(
            out.numpy(),
            np.zeros((M * top_k, self.N), dtype=np.float16),
            atol=1e-6,
        )

    # ----- E: Determinism ---------------------------------------------------

    def test_determinism(self):
        """Same inputs yield identical outputs across two calls."""
        M, top_k = 8, 1
        inp = self._make_inputs(M=M, top_k=top_k)
        r1 = self._run_kernel(inp, M, top_k)
        r2 = self._run_kernel(inp, M, top_k)
        np.testing.assert_array_equal(r1.numpy(), r2.numpy())


if __name__ == "__main__":
    unittest.main()
