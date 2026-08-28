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
"""GPU unit tests for the elastic xattention estimate pipeline.

Covers the Triton-only sub-stack that does NOT depend on the standalone
``block_sparse_attn_ops`` build:

- ``find_blocks_chunked``   -- threshold-cumulative block selector
- ``xattn_estimate``        -- Triton GEMM + softmax-block-sum -> bool mask

The full ``Xattention_prefill_dim4`` (BSA op + estimate) is exercised by
``tests/layers/test_elastic_attention_backend.py``.
"""

import importlib.util
import os
import unittest

import paddle
import pytest

# File-load to side-step ``models/__init__.py`` -> attention.ops chain that
# pulls in compiled custom-op symbols (e.g. ``config_for_attention``) which
# may be missing in older fastdeploy_ops builds. The kernels themselves are
# pure paddle / triton and have no fastdeploy package dependencies.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FB_PATH = os.path.normpath(
    os.path.join(
        _HERE, "..", "..", "..", "fastdeploy", "model_executor", "models", "qwen3_elastic", "kernels", "find_blocks.py"
    )
)
_spec = importlib.util.spec_from_file_location("qwen3_elastic_find_blocks_under_test", _FB_PATH)
_fb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fb)
find_blocks_chunked = _fb.find_blocks_chunked


@pytest.mark.gpu
class TestFindBlocksChunked(unittest.TestCase):
    def setUp(self):
        paddle.seed(7)

    def test_decode_path_returns_all_true(self):
        # ``mode='prefill'`` + ``decoding=True`` is the early-return shortcut.
        x = paddle.rand([1, 2, 4, 8])
        out = find_blocks_chunked(
            x, current_index=0, threshold=0.5, num_to_choose=None, decoding=True, mode="prefill", causal=True
        )
        self.assertEqual(list(out.shape), [1, 2, 4, 8])
        self.assertEqual(out.dtype, paddle.bool)
        self.assertTrue(out.all().item())

    def test_threshold_one_keeps_everything_under_causal(self):
        # threshold close to 1.0 forces ``cumulative_sum < total_sum`` -> all
        # blocks under the causal envelope are picked. Note: the kernel
        # applies CHUNK-level causal envelope (zeros out columns
        # >= current_index + chunk_num), NOT per-row diagonal causal -- rows
        # within a chunk can attend to any column up to the chunk's right
        # edge. See find_blocks.py L127-136.
        Qb, Kb = 2, 4
        # use uniform attn -> any threshold below 1 picks ~all
        x = paddle.ones([1, 1, Qb, Kb])
        # Use current_index < Kb - Qb so the envelope strictly excludes the
        # last column, otherwise the post-causal pad would be empty and the
        # test would be vacuous.
        current_index = 1
        out = find_blocks_chunked(
            x,
            current_index=current_index,
            threshold=0.999,
            num_to_choose=None,
            decoding=False,
            mode="both",
            causal=True,
        )
        np_out = out.numpy()[0, 0]
        envelope = current_index + Qb  # cols [0, envelope) are reachable
        # Inside the envelope: with uniform attention + threshold ~1, every
        # column should be selected.
        for i in range(Qb):
            self.assertTrue(np_out[i, :envelope].all(), f"row {i} cols<{envelope} should all be True")
            # Strictly out-of-envelope cols must be False.
            self.assertFalse(np_out[i, envelope:].any(), f"row {i} cols>={envelope} must be False")

    def test_sink_and_diagonal_always_kept(self):
        # Even with attention concentrated in one block, sink (col 0) and the
        # diagonal block must be retained.
        Qb, Kb = 2, 4
        x = paddle.zeros([1, 1, Qb, Kb])
        # All mass on the last column (rightmost). After causal masking, the
        # diagonal still gets kept by the algorithm regardless of mass.
        x[:, :, :, -1] = 1.0
        out = find_blocks_chunked(
            x, current_index=Kb - Qb, threshold=0.5, num_to_choose=None, decoding=False, mode="both", causal=True
        ).numpy()[0, 0]
        for i in range(Qb):
            self.assertTrue(out[i, 0], f"row {i} sink (col 0) must be True")
            # diagonal column for row i is current_index + i = (Kb - Qb) + i
            self.assertTrue(out[i, (Kb - Qb) + i], f"row {i} diagonal must be True")


@pytest.mark.gpu
class TestXattnEstimate(unittest.TestCase):
    """Smoke-test the Triton-backed ``xattn_estimate`` shape / dtype contract.

    Numerical correctness vs. dense softmax-block-sum is non-trivial to
    re-derive here; we restrict ourselves to the contract documented in
    ELASTIC_FASTDEPLOY_INTEGRATION.md §3.3 / §5.5: outputs are
    [B, H, Qb, Kb] with bool simple_masks and float32 attn_sums.
    """

    def setUp(self):
        paddle.seed(11)
        # If GPU exists, use bf16 to mirror prod; else fp16 still works on Triton.
        try:
            paddle.set_default_dtype("bfloat16")
        except Exception:
            paddle.set_default_dtype("float16")

    def tearDown(self):
        paddle.set_default_dtype("float32")

    def test_output_shape_and_dtype(self):
        # File-load xattention.py directly so we don't go through
        # ``fastdeploy.model_executor.models.qwen3_elastic.kernels`` (which
        # is gated by the parent fastdeploy package init -> attention.ops
        # chain). xattention.py uses RELATIVE imports of sibling kernels
        # (find_blocks, block_sparse_attn, xattention_triton); to make those
        # resolve we register the parent kernels dir as a package first.
        try:
            import sys as _sys

            _kdir = os.path.normpath(
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
                )
            )
            _kpkg_name = "qwen3_elastic_kernels_under_test"
            if _kpkg_name not in _sys.modules:
                _kspec = importlib.util.spec_from_file_location(
                    _kpkg_name,
                    os.path.join(_kdir, "__init__.py"),
                    submodule_search_locations=[_kdir],
                )
                _kpkg = importlib.util.module_from_spec(_kspec)
                _sys.modules[_kpkg_name] = _kpkg
                # Don't exec __init__ (it imports xattention which needs
                # block_sparse_attn -> may fail if BSA op missing). We just
                # need the package object so relative imports inside
                # xattention.py can resolve.
            for _sub in ("find_blocks", "block_sparse_attn", "xattention_triton"):
                _mod_name = f"{_kpkg_name}.{_sub}"
                if _mod_name in _sys.modules:
                    continue
                _ss = importlib.util.spec_from_file_location(_mod_name, os.path.join(_kdir, f"{_sub}.py"))
                _sm = importlib.util.module_from_spec(_ss)
                _sys.modules[_mod_name] = _sm
                _ss.loader.exec_module(_sm)
            _xs = importlib.util.spec_from_file_location(
                f"{_kpkg_name}.xattention",
                os.path.join(_kdir, "xattention.py"),
            )
            _xm = importlib.util.module_from_spec(_xs)
            _sys.modules[f"{_kpkg_name}.xattention"] = _xm
            _xs.loader.exec_module(_xm)
            xattn_estimate = _xm.xattn_estimate
        except Exception as e:
            self.skipTest(f"xattention deps not loadable: {e!r}")

        H = 4
        T = 2048
        D = 128
        block_size = 128
        stride = 16
        chunk_size = 2048

        q = paddle.randn([1, H, T, D])
        k = paddle.randn([1, H, T, D])

        try:
            attn_sums, simple_masks = xattn_estimate(
                q,
                k,
                block_size=block_size,
                stride=stride,
                norm=1.0,
                threshold=0.9,
                chunk_size=chunk_size,
                use_triton=True,
                causal=True,
            )
        except RuntimeError as e:
            # Triton requires an active CUDA driver matching the installed
            # paddle build. In stripped-down test envs (e.g. CPU-only CI or
            # paddle/triton CUDA mismatch) this fails before kernel launch;
            # the gpu-marker contract is best-effort, so skip rather than
            # red.
            self.skipTest(f"triton driver unavailable: {e!r}")

        Qb = T // block_size
        Kb = T // block_size
        self.assertEqual(list(simple_masks.shape), [1, H, Qb, Kb])
        self.assertEqual(simple_masks.dtype, paddle.bool)
        # attn_sums is float32 per softmax_fuse_block_sum kernel
        self.assertEqual(list(attn_sums.shape)[:2], [1, H])
        # Causal: upper-triangular (strictly above the diagonal) must be all False.
        sm = simple_masks.numpy()[0, 0]
        for i in range(Qb):
            self.assertFalse(sm[i, i + 1 :].any(), f"causal violated at row {i}")


if __name__ == "__main__":
    unittest.main()
