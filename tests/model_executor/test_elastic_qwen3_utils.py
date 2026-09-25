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
"""Unit tests for ``fastdeploy.model_executor.models.qwen3_elastic.utils``.

Covers the inference-only pure-paddle helpers that the elastic-attention
backend depends on:

- ``_LinearTransposed`` -- HF<->paddle weight-transpose flag
- ``AttentionRouter`` -- 3-layer MLP head router (argmax 0/1)
- ``ctx_q_pool`` -- per-sequence first-100 + last-100 mean
- ``derive_head_mask_type`` -- (retrieval_mode, toggle_type) -> {1, 0, -1}
"""

import importlib.util
import os
import unittest

import numpy as np
import paddle

# Load the target module directly from its source file. Going through
# ``fastdeploy.model_executor.models.qwen3_elastic.utils`` would trigger the
# parent ``models/__init__.py`` -> attention.ops chain, which transitively
# imports compiled custom-op symbols that may not all be present in every
# build (e.g. older fastdeploy_ops_pd_.so without ``config_for_attention``).
# These pure-paddle helpers don't need any custom op, so we side-step the
# package init entirely.
_HERE = os.path.dirname(os.path.abspath(__file__))
_UTILS_PATH = os.path.normpath(
    os.path.join(_HERE, "..", "..", "fastdeploy", "model_executor", "models", "qwen3_elastic", "utils.py")
)
_spec = importlib.util.spec_from_file_location("qwen3_elastic_utils_under_test", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)
AttentionRouter = _utils.AttentionRouter
_LinearTransposed = _utils._LinearTransposed
ctx_q_pool = _utils.ctx_q_pool
derive_head_mask_type = _utils.derive_head_mask_type

paddle.seed(0)
np.random.seed(0)


class TestLinearTransposed(unittest.TestCase):
    def test_weight_need_transpose_flag(self):
        layer = _LinearTransposed(in_features=8, out_features=16, bias=True)
        # The flag is what default_weight_loader reads to know whether to
        # transpose HF [out, in] weight into paddle [in, out] layout.
        self.assertTrue(getattr(layer.weight, "weight_need_transpose", False))
        # Paddle linear weight shape stays [in, out]
        self.assertEqual(list(layer.weight.shape), [8, 16])

    def test_forward_shape(self):
        layer = _LinearTransposed(8, 16, bias=True)
        x = paddle.randn([4, 8])
        y = layer(x)
        self.assertEqual(list(y.shape), [4, 16])


class TestAttentionRouter(unittest.TestCase):
    def setUp(self):
        paddle.seed(123)
        self.num_kv_heads = 8
        self.d_feature = 16  # tiny d_feature; we are only checking semantics
        self.router = AttentionRouter(num_kv_heads=self.num_kv_heads, d_feature=self.d_feature)

    def test_output_shape_dtype_and_range(self):
        k_pooled = paddle.randn([1, self.num_kv_heads, self.d_feature])
        z = self.router(k_pooled)
        # Spec: [B, H_kv] int32 with values in {0, 1}.
        self.assertEqual(list(z.shape), [1, self.num_kv_heads])
        self.assertEqual(z.dtype, paddle.int32)
        z_np = z.numpy()
        self.assertTrue(np.isin(z_np, [0, 1]).all())

    def test_batch_2(self):
        k_pooled = paddle.randn([2, self.num_kv_heads, self.d_feature])
        z = self.router(k_pooled)
        self.assertEqual(list(z.shape), [2, self.num_kv_heads])

    def test_argmax_matches_logits(self):
        # The router does ``argmax`` over the final 2-class logits. Manually
        # rebuild the same path and check parity.
        k_pooled = paddle.randn([1, self.num_kv_heads, self.d_feature])
        h = self.router.cls_feat_extractor(k_pooled)
        logits = self.router.cls_router_head_agnostic(h)
        ref = logits.argmax(axis=-1).astype("int32")
        out = self.router(k_pooled)
        np.testing.assert_array_equal(out.numpy(), ref.numpy())


class TestCtxQPool(unittest.TestCase):
    """ctx_q_pool == mean of first 100 + last 100 K tokens (with overlap when T<200)."""

    def _ref_pool(self, k):  # [T, H, D] -> [1, H, D]
        T = k.shape[0]
        head = k[: min(100, T)]
        tail = k[-min(100, T) :]
        cat = paddle.concat([head, tail], axis=0).astype("float32")
        return cat.mean(axis=0, keepdim=True).astype(k.dtype)

    def test_short_sequence_overlap(self):
        # T < 200 -> head and tail overlap, exactly matching HF eval path.
        T, H, D = 50, 4, 8
        k = paddle.randn([T, H, D])
        out = ctx_q_pool(k)
        ref = self._ref_pool(k)
        self.assertEqual(list(out.shape), [1, H, D])
        np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)

    def test_long_sequence_no_overlap(self):
        T, H, D = 1024, 4, 8
        k = paddle.randn([T, H, D])
        out = ctx_q_pool(k)
        ref = self._ref_pool(k)
        self.assertEqual(list(out.shape), [1, H, D])
        np.testing.assert_allclose(out.numpy(), ref.numpy(), rtol=1e-5, atol=1e-5)

    def test_varlen_two_segments(self):
        T1, T2 = 30, 300
        H, D = 4, 8
        k = paddle.randn([T1 + T2, H, D])
        cu = paddle.to_tensor([0, T1, T1 + T2], dtype="int32")
        out = ctx_q_pool(k, cu_seq_lens=cu)
        self.assertEqual(list(out.shape), [2, H, D])
        ref0 = self._ref_pool(k[:T1])
        ref1 = self._ref_pool(k[T1:])
        np.testing.assert_allclose(out[0:1].numpy(), ref0.numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(out[1:2].numpy(), ref1.numpy(), rtol=1e-5, atol=1e-5)


class TestDeriveHeadMaskType(unittest.TestCase):
    """Enumerate the 5 documented (retrieval_mode, toggle_type) pairs (§5.4)."""

    def setUp(self):
        # H_kv = 4, deliberately mixed 0/1 so each branch is meaningful.
        self.z = paddle.to_tensor([0, 1, 0, 1], dtype="int32")

    def test_full_xattn(self):
        out = derive_head_mask_type(self.z, "full", "xattn", group_size=1)
        np.testing.assert_array_equal(out.numpy(), np.array([1, 0, 1, 0], dtype=np.int32))

    def test_full_streaming(self):
        out = derive_head_mask_type(self.z, "full", "streaming", group_size=1)
        np.testing.assert_array_equal(out.numpy(), np.array([-1, 0, -1, 0], dtype=np.int32))

    def test_xattn_streaming(self):
        out = derive_head_mask_type(self.z, "xattn", "streaming", group_size=1)
        np.testing.assert_array_equal(out.numpy(), np.array([-1, 1, -1, 1], dtype=np.int32))

    def test_xattn_xattn_all_one(self):
        out = derive_head_mask_type(self.z, "xattn", "xattn", group_size=1)
        np.testing.assert_array_equal(out.numpy(), np.array([1, 1, 1, 1], dtype=np.int32))

    def test_full_full_all_zero(self):
        out = derive_head_mask_type(self.z, "full", "full", group_size=1)
        np.testing.assert_array_equal(out.numpy(), np.array([0, 0, 0, 0], dtype=np.int32))

    def test_gqa_repeat_interleave(self):
        # group_size=2 -> H_q = 2 * H_kv, each KV-head decision is duplicated.
        out = derive_head_mask_type(self.z, "full", "xattn", group_size=2)
        # base: [1, 0, 1, 0] -> repeat_interleave 2 -> [1,1,0,0,1,1,0,0]
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([1, 1, 0, 0, 1, 1, 0, 0], dtype=np.int32),
        )

    def test_unsupported_pair_raises(self):
        with self.assertRaises(NotImplementedError):
            derive_head_mask_type(self.z, "streaming", "full", group_size=1)


if __name__ == "__main__":
    unittest.main()
