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

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import paddle

# ---------------------------------------------------------------------------
# Stub GPU-only ops so the import chain resolves without CUDA extensions.
# Try the real import first; only inject stubs when it is unavailable.
# After the import, explicitly remove any stale parent-package attribute
# that Python may have bound during the stub phase, preventing cross-test
# pollution (e.g. tests that access `fastdeploy.model_executor.ops.gpu` via
# attribute traversal rather than `import`).
# ---------------------------------------------------------------------------

_GPU_OPS = "fastdeploy.model_executor.ops.gpu"
_DEEP_GEMM = f"{_GPU_OPS}.deep_gemm"

_NEED_STUB = _GPU_OPS not in sys.modules


class _GpuOpsStub(types.ModuleType):
    """Catch-all module: returns registered sub-modules or ``None``."""

    __path__ = []

    def __getattr__(self, name):
        fqn = f"{self.__name__}.{name}"
        sub = sys.modules.get(fqn)
        return sub if sub is not None else None


_gpu_ops_stub = _GpuOpsStub(_GPU_OPS)
_deep_gemm_stub = types.ModuleType(_DEEP_GEMM)
_deep_gemm_stub.m_grouped_fp8_gemm_nt_contiguous = None
_deep_gemm_stub.m_grouped_fp8_gemm_nt_masked = None
_deep_gemm_stub.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous = None
_deep_gemm_stub.m_grouped_gemm_fp8_fp8_bf16_nt_masked = None

if _NEED_STUB:
    with patch.dict(sys.modules, {_GPU_OPS: _gpu_ops_stub, _DEEP_GEMM: _deep_gemm_stub}, clear=False):
        from fastdeploy.model_executor.layers.moe import fused_moe_marlin_backend as mb

    # Clean up stale attribute references that Python binds during import
    _ops_parent = sys.modules.get("fastdeploy.model_executor.ops")
    if _ops_parent is not None and getattr(_ops_parent, "gpu", None) is _gpu_ops_stub:
        try:
            delattr(_ops_parent, "gpu")
        except AttributeError:
            pass
else:
    from fastdeploy.model_executor.layers.moe import fused_moe_marlin_backend as mb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPureFunctions:
    """get_scale_perms, marlin_permute_scales, and MoE wrapper variants."""

    def test_get_scale_perms(self):
        perm, single = mb.get_scale_perms()
        assert len(perm) == 64
        assert len(single) == 32

    def test_marlin_permute_scales_group(self):
        s = paddle.arange(128, dtype="float32").reshape([2, 64])
        out = mb.marlin_permute_scales(s, 16, 64, 8)
        assert list(out.shape) == [2, 64]

    def test_marlin_permute_scales_perchannel(self):
        s = paddle.arange(128, dtype="float32").reshape([2, 64])
        out = mb.marlin_permute_scales(s, 16, 64, -1)
        assert list(out.shape) == [2, 64]

    def test_gptq_marlin_moe_repack(self):
        """Per-expert repack loop with mocked C++ op."""
        num_experts, size_k, size_n, num_bits = 2, 32, 16, 4
        b_q_weight = paddle.ones([num_experts, size_k, size_n], dtype="int32")
        perm = paddle.zeros([num_experts, size_k], dtype="int32")
        with (
            patch.dict(
                sys.modules,
                {_GPU_OPS: _gpu_ops_stub, _DEEP_GEMM: _deep_gemm_stub},
                clear=False,
            ),
            patch.object(
                _gpu_ops_stub,
                "gptq_marlin_repack",
                lambda w, p, sk, sn, nb: paddle.zeros([sk // 16, sn * (nb // 2)], dtype=w.dtype),
            ),
        ):
            out = mb.gptq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits)
        assert list(out.shape) == [num_experts, size_k // 16, size_n * (num_bits // 2)]

    def test_marlin_moe_permute_scales(self):
        """Per-expert permutation matches single-expert output."""
        num_experts, size_k, size_n, group_size = 3, 64, 64, 8
        num_groups = size_k // group_size
        s = paddle.arange(num_experts * num_groups * size_n, dtype="float32").reshape(
            [num_experts, num_groups, size_n]
        )
        out = mb.marlin_moe_permute_scales(s, size_k, size_n, group_size)
        assert list(out.shape) == [num_experts, num_groups, size_n]
        for e in range(num_experts):
            expected = mb.marlin_permute_scales(s[e], size_k, size_n, group_size)
            assert paddle.equal_all(out[e], expected).item()


class TestMarlinWeightOnlyMoEMethod:
    """create_weights, process_loaded_weights, apply."""

    def test_create_and_process(self):
        """create_weights -> process_loaded_weights with shape/dtype validation."""
        layer = _DummyLayer()
        m = _init(layer)

        # Verify create_weights set parameters with correct shape / dtype
        E, H, I = layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size
        assert list(layer.up_gate_proj_weight.shape) == [E, H // 16, I * 4]
        assert str(layer.up_gate_proj_weight.dtype).endswith("int32")
        assert list(layer.down_proj_weight.shape) == [E, I // 16, H * 2]
        assert str(layer.down_proj_weight.dtype).endswith("int32")
        assert list(layer.up_gate_proj_weight_scale.shape) == [E, 1, I * 2]
        assert list(layer.down_proj_weight_scale.shape) == [E, 1, H]

        with (
            patch.dict(
                sys.modules,
                {_GPU_OPS: _gpu_ops_stub, _DEEP_GEMM: _deep_gemm_stub},
                clear=False,
            ),
            patch.object(
                _gpu_ops_stub,
                "gptq_marlin_repack",
                lambda w, p, sk, sn, nb: paddle.zeros([sk // 16, sn * (nb // 2)], dtype=w.dtype),
            ),
        ):
            m.process_loaded_weights(layer, dict(zip(("up", "down"), _make_weights(layer))))

        # After processing: weights repacked, scales permuted — verify shapes
        # hold and scales are non-zero (not a no-op).
        assert list(layer.up_gate_proj_weight.shape) == [E, H // 16, I * 4]
        assert list(layer.down_proj_weight.shape) == [E, I // 16, H * 2]
        assert not paddle.equal_all(
            layer.up_gate_proj_weight_scale,
            paddle.zeros_like(layer.up_gate_proj_weight_scale),
        ).item()

    def test_apply_topk(self):
        """apply() with default topk_method='topk'."""
        layer = _DummyLayer()
        m = _init(layer)
        gate = paddle.nn.Linear(64, 2, bias_attr=False)
        x = paddle.ones([2, 64], dtype="float32")
        with (
            patch.dict(
                sys.modules,
                {_GPU_OPS: _gpu_ops_stub, _DEEP_GEMM: _deep_gemm_stub},
                clear=False,
            ),
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
            out = m.apply(layer, x, gate, topk_ids_hookfunc=lambda topk_ids: None)
        assert list(out.shape) == [2, 64]

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
            out = m.apply(layer, x, gate, topk_ids_hookfunc=lambda topk_ids: None)
        assert list(out.shape) == [2, 64]
