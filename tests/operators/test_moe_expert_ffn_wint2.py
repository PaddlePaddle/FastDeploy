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

from fastdeploy.model_executor.ops.gpu import moe_expert_ffn_wint2

try:
    from fastdeploy.model_executor.ops.gpu import winx_unzip as _winx_unzip_op

    _HAS_WINX_UNZIP = True
except ImportError:
    _HAS_WINX_UNZIP = False

paddle.seed(2026)
np.random.seed(2026)


def _cutlass_rearrange(w):
    """Apply CUTLASS WINT2 weight layout rearrangement."""
    shape = w.shape
    E, Kp, N = shape
    w = w.reshape([E, Kp // 16, 16, N // 8, 8])
    w = paddle.transpose(w, perm=[0, 3, 1, 4, 2])
    return w.reshape(shape)


def _build_inputs(num_experts, hidden_size, inter_size, tokens_per_expert, dtype="bfloat16"):
    """Create correctly-shaped tensors for moe_expert_ffn_wint2."""
    gated_inter = inter_size * 2
    total_tokens = sum(tokens_per_expert)
    permute_input = paddle.randn([total_tokens, hidden_size], dtype=dtype)
    tokens_expert_prefix_sum = paddle.to_tensor(np.cumsum(tokens_per_expert).astype("int64"))

    w_up_raw = paddle.randint(0, 256, [num_experts, hidden_size // 4, gated_inter], dtype="int32").cast("uint8")
    w_down_raw = paddle.randint(0, 256, [num_experts, inter_size // 4, hidden_size], dtype="int32").cast("uint8")
    w_up = _cutlass_rearrange(w_up_raw)
    w_down = _cutlass_rearrange(w_down_raw)

    return dict(
        permute_input=permute_input,
        tokens_expert_prefix_sum=tokens_expert_prefix_sum,
        up_gate_proj_weight=w_up,
        down_proj_weight=w_down,
        up_gate_proj_bias=None,
        up_gate_proj_scale=paddle.randn([num_experts, gated_inter], dtype=dtype) * 0.01,
        down_proj_scale=paddle.randn([num_experts, hidden_size], dtype=dtype) * 0.01,
        up_gate_proj_local_scale=paddle.randint(
            0, 256, [num_experts, hidden_size // 128, gated_inter], dtype="int32"
        ).cast("uint8"),
        up_gate_proj_code_scale=paddle.randn([num_experts, gated_inter], dtype="float32") * 0.01,
        up_gate_proj_code_zp=paddle.randn([num_experts, gated_inter], dtype="float32") * 0.01,
        down_proj_local_scale=paddle.randint(
            0, 256, [num_experts, inter_size // 128, hidden_size], dtype="int32"
        ).cast("uint8"),
        down_proj_code_scale=paddle.randn([num_experts, hidden_size], dtype="float32") * 0.01,
        down_proj_code_zp=paddle.randn([num_experts, hidden_size], dtype="float32") * 0.01,
        _up_weight_raw=w_up_raw,
        _down_weight_raw=w_down_raw,
    )


def _call_op(inputs):
    """Invoke moe_expert_ffn_wint2."""
    return moe_expert_ffn_wint2(
        inputs["permute_input"],
        inputs["tokens_expert_prefix_sum"],
        inputs["up_gate_proj_weight"],
        inputs["down_proj_weight"],
        inputs["up_gate_proj_bias"],
        inputs["up_gate_proj_scale"],
        inputs["down_proj_scale"],
        inputs["up_gate_proj_local_scale"],
        inputs["up_gate_proj_code_scale"],
        inputs["up_gate_proj_code_zp"],
        inputs["down_proj_local_scale"],
        inputs["down_proj_code_scale"],
        inputs["down_proj_code_zp"],
        False,
    )


def _reference_moe_expert_ffn(inputs):
    """Decomposed reference: winx_unzip dequant -> matmul -> SwiGLU -> matmul."""
    E = inputs["_up_weight_raw"].shape[0]
    prefix = inputs["tokens_expert_prefix_sum"].numpy()
    starts = np.concatenate([[0], prefix[:-1]]).astype(int)

    dense_up_gate = _winx_unzip_op(
        inputs["_up_weight_raw"],
        inputs["up_gate_proj_local_scale"],
        inputs["up_gate_proj_code_scale"],
        inputs["up_gate_proj_code_zp"],
        inputs["up_gate_proj_scale"],
        "weight_only_int2",
    ).cast("float32")
    dense_down = _winx_unzip_op(
        inputs["_down_weight_raw"],
        inputs["down_proj_local_scale"],
        inputs["down_proj_code_scale"],
        inputs["down_proj_code_zp"],
        inputs["down_proj_scale"],
        "weight_only_int2",
    ).cast("float32")

    outputs = []
    for e in range(E):
        start, end = int(starts[e]), int(prefix[e])
        if end <= start:
            continue
        x = inputs["permute_input"][start:end].cast("float32")
        fc1 = paddle.matmul(x, dense_up_gate[e])
        inter_size = fc1.shape[-1] // 2
        act = paddle.nn.functional.silu(fc1[:, :inter_size]) * fc1[:, inter_size:]
        outputs.append(paddle.matmul(act, dense_down[e]))
    if outputs:
        return paddle.concat(outputs, axis=0).cast(inputs["permute_input"].dtype)
    return paddle.zeros_like(inputs["permute_input"][:0])


class TestMoeExpertFFNWint2(unittest.TestCase):
    """Correctness tests for the WINT2 MoE FFN op."""

    E, H, INTER = 4, 256, 128
    TOKENS = [4, 6, 2, 4]

    def setUp(self):
        paddle.set_device("gpu")

    def _check_output(self, dtype="bfloat16", tokens=None):
        """Run op, compare against decomposed reference via assert_allclose."""
        tok = tokens or self.TOKENS
        inputs = _build_inputs(self.E, self.H, self.INTER, tok, dtype=dtype)
        out = _call_op(inputs).cast("float32").numpy()
        ref = _reference_moe_expert_ffn(inputs).cast("float32").numpy()
        self.assertEqual(list(out.shape), [sum(tok), self.H])
        np.testing.assert_allclose(out, ref, rtol=5e-2, atol=5e-2)

    @unittest.skipUnless(_HAS_WINX_UNZIP, "winx_unzip needed for reference")
    def test_correctness_bf16(self):
        """Fused output matches decomposed reference (bfloat16)."""
        self._check_output(dtype="bfloat16")

    @unittest.skipUnless(_HAS_WINX_UNZIP, "winx_unzip needed for reference")
    def test_correctness_fp16(self):
        """Fused output matches decomposed reference (float16)."""
        self._check_output(dtype="float16")

    @unittest.skipUnless(_HAS_WINX_UNZIP, "winx_unzip needed for reference")
    def test_correctness_uneven_tokens(self):
        """Correctness with uneven token distribution across experts."""
        self._check_output(tokens=[1, 5, 3, 7])


if __name__ == "__main__":
    unittest.main()
