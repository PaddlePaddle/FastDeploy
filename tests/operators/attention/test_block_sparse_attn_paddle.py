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
"""Tests for the paddle wrapper around ``block_sparse_attn_ops``.

The CUDA op itself ships with its own UTs in
``custom_ops/gpu_ops/block_sparse_attn``. This file targets the FastDeploy-
side paddle helpers that the wrapper applies BEFORE calling into the CUDA
binary:

- ``_replace_ones_with_count``      -- assigns unique 1-based indices to each
                                       sparse head so the kernel can read its
                                       own blockmask row.
- ``_convert_blockmask_row_reverse`` -- bool [B,H,Qb,Kb] -> int32 sorted-desc
                                       K-block index list (-1 padding) that
                                       the kernel binary-searches over.

A separate GPU smoke test asserts a tiny dense-equivalent call (no sparse
heads) returns finite, correctly-shaped output -- skipped if the standalone
``block_sparse_attn_ops`` extension is not yet built.
"""

import importlib
import importlib.util
import os
import unittest

import numpy as np
import paddle
import pytest

# File-load to avoid the ``models/__init__.py`` -> attention.ops chain that
# pulls in compiled custom-op symbols which may be missing in some builds.
_HERE = os.path.dirname(os.path.abspath(__file__))
_BSA_PATH = os.path.normpath(
    os.path.join(
        _HERE,
        "..",
        "..",
        "..",
        "fastdeploy",
        "model_executor",
        "models",
        "qwen3_elastic",
        "kernels",
        "block_sparse_attn.py",
    )
)
_spec = importlib.util.spec_from_file_location("qwen3_elastic_bsa_under_test", _BSA_PATH)
_bsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bsa)
_convert_blockmask_row_reverse = _bsa._convert_blockmask_row_reverse
_replace_ones_with_count = _bsa._replace_ones_with_count


class TestReplaceOnesWithCount(unittest.TestCase):
    def test_no_sparse_head(self):
        h = paddle.to_tensor([0, -1, 0, -1], dtype="int32")
        out, n = _replace_ones_with_count(h)
        np.testing.assert_array_equal(out.numpy(), h.numpy())
        self.assertEqual(n, 0)

    def test_mixed(self):
        # 1s sit at positions 0, 2, 5 -> they should become 1, 2, 3.
        h = paddle.to_tensor([1, 0, 1, -1, 0, 1], dtype="int32")
        out, n = _replace_ones_with_count(h)
        np.testing.assert_array_equal(
            out.numpy(),
            np.array([1, 0, 2, -1, 0, 3], dtype=np.int32),
        )
        self.assertEqual(n, 3)

    def test_all_sparse(self):
        h = paddle.to_tensor([1, 1, 1, 1], dtype="int32")
        out, n = _replace_ones_with_count(h)
        np.testing.assert_array_equal(out.numpy(), np.array([1, 2, 3, 4], dtype=np.int32))
        self.assertEqual(n, 4)


class TestConvertBlockmaskRowReverse(unittest.TestCase):
    def test_descending_indices_with_padding(self):
        # Single (B=1, H=1, Qb=1, Kb=5) row: kept blocks {0, 2, 4}
        bm = paddle.to_tensor([[[[True, False, True, False, True]]]], dtype="bool")
        out = _convert_blockmask_row_reverse(bm).numpy()[0, 0, 0]
        # Largest valid k-block first -> 4, 2, 0; padding -1 fills the rest.
        kept = sorted([i for i in out.tolist() if i != -1], reverse=True)
        self.assertEqual(kept, [4, 2, 0])
        # Total length preserved
        self.assertEqual(len(out), 5)
        # Padding only at the tail
        first_pad = next((i for i, v in enumerate(out) if v == -1), len(out))
        self.assertTrue(all(v == -1 for v in out[first_pad:]))

    def test_all_kept(self):
        bm = paddle.to_tensor([[[[True, True, True, True]]]], dtype="bool")
        out = _convert_blockmask_row_reverse(bm).numpy()[0, 0, 0]
        # All four indices present, no -1.
        self.assertEqual(sorted(out.tolist(), reverse=True), [3, 2, 1, 0])
        self.assertFalse((out == -1).any())

    def test_all_dropped(self):
        bm = paddle.to_tensor([[[[False, False, False]]]], dtype="bool")
        out = _convert_blockmask_row_reverse(bm).numpy()[0, 0, 0]
        self.assertTrue((out == -1).all())


@pytest.mark.gpu
class TestBlockSparseAttnSmoke(unittest.TestCase):
    """End-to-end smoke vs. dense scaled-dot-product reference, all heads full.

    Skipped automatically if ``block_sparse_attn_ops`` (the standalone CUDA
    extension) is not importable.
    """

    def setUp(self):
        try:
            importlib.import_module("block_sparse_attn_ops")
        except Exception:
            self.skipTest("block_sparse_attn_ops not built; skip smoke test")

    def test_full_heads_match_dense(self):
        # File-loaded copy already at module top via _bsa.
        block_sparse_attn_paddle = _bsa.block_sparse_attn_paddle

        T, H, D = 256, 4, 64
        block = 128
        Qb = (T + block - 1) // block
        Kb = Qb

        paddle.seed(0)
        q = paddle.randn([T, H, D]).astype("bfloat16")
        k = paddle.randn([T, H, D]).astype("bfloat16")
        v = paddle.randn([T, H, D]).astype("bfloat16")
        cu = paddle.to_tensor([0, T], dtype="int32")
        # head_mask_type = 0 (full) for every head -> equivalent to dense FA.
        hmt = paddle.zeros([H], dtype="int32")
        streaming_info = paddle.to_tensor([1, 16] * H, dtype="int32")
        # placeholder blockmask (no sparse heads, but the wrapper requires shape)
        blockmask = paddle.ones([1, 0, Qb, Kb], dtype="bool")

        out = block_sparse_attn_paddle(
            q,
            k,
            v,
            cu,
            cu,
            hmt,
            streaming_info,
            blockmask,
            max_seqlen_q=T,
            max_seqlen_k=T,
            is_causal=True,
            m_block_dim=block,
            n_block_dim=block,
        )
        self.assertEqual(list(out.shape), [T, H, D])
        self.assertTrue(paddle.isfinite(out).all().item())


if __name__ == "__main__":
    unittest.main()
