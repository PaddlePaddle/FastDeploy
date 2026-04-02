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

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import paddle

# Stub GPU-only ops so the import chain (moe → triton_backend → fp8_utils →
# ops.gpu.deep_gemm) resolves without compiled CUDA extensions.
# Scoped via patch.dict so sys.modules is restored after the import.


class _GpuOpsStub(types.ModuleType):
    """Catchall module: returns registered sub-modules or None for unknown attrs."""

    __path__ = []  # marks as package so `import X.Y.Z` can traverse

    def __getattr__(self, name):
        fqn = f"{self.__name__}.{name}"
        sub = sys.modules.get(fqn)
        if sub is not None:
            return sub
        return None


_GPU_OPS = "fastdeploy.model_executor.ops.gpu"
_DEEP_GEMM = f"{_GPU_OPS}.deep_gemm"

_gpu_ops_stub = _GpuOpsStub(_GPU_OPS)
_deep_gemm_stub = types.ModuleType(_DEEP_GEMM)
_deep_gemm_stub.m_grouped_fp8_gemm_nt_contiguous = None
_deep_gemm_stub.m_grouped_fp8_gemm_nt_masked = None
_deep_gemm_stub.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous = None
_deep_gemm_stub.m_grouped_gemm_fp8_fp8_bf16_nt_masked = None

with patch.dict(sys.modules, {_GPU_OPS: _gpu_ops_stub, _DEEP_GEMM: _deep_gemm_stub}, clear=False):
    from fastdeploy.model_executor.layers.moe import (  # noqa: E402
        fused_moe_marlin_backend as mb,
    )


class _DummyLayer(paddle.nn.Layer):
    """Minimal FusedMoE surface for MarlinWeightOnlyMoEMethod."""

    def __init__(self, hidden=64, inter=32, experts=2):
        super().__init__()
        self.num_local_experts = self.num_experts = experts
        self.hidden_size, self.moe_intermediate_size = hidden, inter
        self.top_k = self.n_group = self.topk_group = 1
        self.topk_method = "topk"
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = paddle.zeros([experts], dtype="float32")
        self.renormalize = True
        self.fd_config = SimpleNamespace()

    def extract_moe_ffn_weights(self, sd):
        return sd["up"], sd["down"], None, None


def _make_weights(layer):
    u = [
        paddle.ones([layer.hidden_size, layer.moe_intermediate_size * 2], "float32")
        for _ in range(layer.num_local_experts)
    ]
    d = [
        paddle.ones([layer.moe_intermediate_size, layer.hidden_size], "float32")
        for _ in range(layer.num_local_experts)
    ]
    return u, d


def _init(layer):
    m = mb.MarlinWeightOnlyMoEMethod()
    m.create_weights(layer)
    return m


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestFusedMoeMarlinBackend(unittest.TestCase):
    """Tests for MarlinWeightOnlyMoEMethod."""

    def test_pure_functions(self):
        """get_scale_perms + marlin_permute_scales (both branches)."""
        perm, single = mb.get_scale_perms()
        self.assertEqual(len(perm), 64)
        self.assertEqual(len(single), 32)
        s = paddle.arange(128, dtype="float32").reshape([2, 64])
        self.assertEqual(list(mb.marlin_permute_scales(s, 16, 64, 8).shape), [2, 64])
        self.assertEqual(list(mb.marlin_permute_scales(s, 16, 64, -1).shape), [2, 64])

    def test_create_and_process(self):
        """create_weights -> process_loaded_weights end-to-end."""
        layer = _DummyLayer()
        m = _init(layer)
        self.assertTrue(hasattr(layer, "up_gate_proj_weight"))
        self.assertTrue(hasattr(layer, "down_proj_weight"))
        with patch.object(
            _gpu_ops_stub,
            "gptq_marlin_repack",
            lambda w, p, sk, sn, nb: paddle.zeros([sk // 16, sn * (nb // 2)], dtype=w.dtype),
        ):
            m.process_loaded_weights(layer, dict(zip(("up", "down"), _make_weights(layer))))

    def test_apply_topk(self):
        """apply() with default topk_method='topk'."""
        layer = _DummyLayer()
        m = _init(layer)
        gate = paddle.nn.Linear(64, 2, bias_attr=False)
        x = paddle.ones([2, 64], dtype="float32")
        with (
            patch.object(
                mb,
                "MoeWna16MarlinGemmApi",
                lambda *_a, **kw: (paddle.zeros([kw["size_m"], kw["size_n"]], "float32"),),
            ),
            patch.object(
                mb,
                "tritonmoe_preprocess_func",
                lambda ids, ne, bm: (
                    paddle.zeros([4], "int32"),
                    paddle.zeros([1], "int32"),
                    paddle.to_tensor([4], "int32"),
                ),
            ),
            patch.object(
                _gpu_ops_stub,
                "moe_topk_select",
                lambda g, b, k, *_: (
                    paddle.zeros([g.shape[0], k], "int64"),
                    paddle.ones([g.shape[0], k], "float32"),
                ),
            ),
            patch(
                "paddle.incubate.nn.functional.swiglu",
                lambda x: x[..., : x.shape[-1] // 2],
                create=True,
            ),
        ):
            out = m.apply(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
        self.assertEqual(list(out.shape), [2, 64])

    def test_apply_noaux_tc(self):
        """apply() with topk_method='noaux_tc'."""
        layer = _DummyLayer()
        layer.topk_method = "noaux_tc"
        m = _init(layer)
        gate = paddle.nn.Linear(64, 2, bias_attr=False)
        x = paddle.ones([2, 64], dtype="float32")
        with (
            patch.object(
                mb,
                "MoeWna16MarlinGemmApi",
                lambda *_a, **kw: (paddle.zeros([kw["size_m"], kw["size_n"]], "float32"),),
            ),
            patch.object(
                mb,
                "tritonmoe_preprocess_func",
                lambda ids, ne, bm: (
                    paddle.zeros([4], "int32"),
                    paddle.zeros([1], "int32"),
                    paddle.to_tensor([4], "int32"),
                ),
            ),
            patch(
                "fastdeploy.model_executor.layers.moe.moe.get_moe_scores",
                lambda g, ng, tg, k, s, b, r: (
                    g,
                    paddle.ones([g.shape[0], k], "float32"),
                    paddle.zeros([g.shape[0], k], "int64"),
                ),
            ),
            patch(
                "paddle.incubate.nn.functional.swiglu",
                lambda x: x[..., : x.shape[-1] // 2],
                create=True,
            ),
        ):
            out = m.apply(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
        self.assertEqual(list(out.shape), [2, 64])


if __name__ == "__main__":
    unittest.main()
