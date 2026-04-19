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
"""Unit tests for Hackathon 10th Spring No.44.
Additive unit tests for fused_moe_deepgemm_backend.py

Coverage delta over tests/layers/test_deepgemm_fused_moe.py (PR #6840):
  - Standalone helpers: infermeta, call_prefill_permute_to_masked_gemm,
    call_depermute_prefill_combine
  - Weight management: create_weights, process_weights_after_loading,
    process_loaded_weights, process_prequanted_weights
  - apply_tp with FD_USE_PHI_FP8_QUANT=False (per_token_quant input path)
  - apply_ep_prefill with num_worst_tokens > 0 (masked GEMM path)
  - shared_experts branches in apply_ep_prefill and apply_ep_decode
"""

import sys
import types
from types import SimpleNamespace

import paddle
import pytest

# ── Stub GPU-only modules ───────────────────────────────────────────────────

_SENTINEL = object()


class _GpuOpsStub(types.ModuleType):
    """GPU-only op module stub — raises NotImplementedError for unconfigured calls."""

    __path__ = []

    def __getattr__(self, name):
        fqn = f"{self.__name__}.{name}"
        sub = sys.modules.get(fqn)
        if sub is not None:
            return sub

        def _unstubbed(*args, **kwargs):
            raise NotImplementedError(f"GPU op {fqn!r} not stubbed — add monkeypatch.setattr in the test")

        _unstubbed.__name__ = name
        return _unstubbed


# Build stub objects and wire their attributes — NO sys.modules injection yet.
_ops = types.ModuleType("fastdeploy.model_executor.ops")
_ops.__path__ = []

_gpu = _GpuOpsStub("fastdeploy.model_executor.ops.gpu")
_ops.gpu = _gpu

_dg = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm")
_dg.m_grouped_fp8_gemm_nt_contiguous = None
_dg.m_grouped_fp8_gemm_nt_masked = None
_dg.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous = None
_dg.m_grouped_gemm_fp8_fp8_bf16_nt_masked = None
_gpu.deep_gemm = _dg

_triton = _GpuOpsStub("fastdeploy.model_executor.ops.triton_ops")
_ops.triton_ops = _triton

_tu = types.ModuleType("fastdeploy.model_executor.ops.triton_ops.triton_utils")
_tu.enable_compat_on_triton_kernel = lambda fn: fn
_tu.paddle_driver = None

_ep = types.ModuleType("fastdeploy.model_executor.layers.moe.ep")


class _BufferStub:
    @staticmethod
    def capture():
        return SimpleNamespace(current_stream_wait=lambda: None)


_ep.deep_ep = SimpleNamespace(Buffer=_BufferStub)

# Mapping of all stubs to inject into sys.modules during tests.
_STUB_ENTRIES = {
    "fastdeploy.model_executor.ops": _ops,
    "fastdeploy.model_executor.ops.gpu": _gpu,
    "fastdeploy.model_executor.ops.gpu.deep_gemm": _dg,
    "fastdeploy.model_executor.ops.triton_ops": _triton,
    "fastdeploy.model_executor.ops.triton_ops.triton_utils": _tu,
    "fastdeploy.model_executor.layers.moe.ep": _ep,
}

dgb = None  # populated by _install_stubs fixture


@pytest.fixture(autouse=True, scope="module")
def _install_stubs():
    """Wire GPU stubs into sys.modules; restore originals after suite."""
    global dgb  # noqa: PLW0603
    saved = {}
    for key, mod in _STUB_ENTRIES.items():
        saved[key] = sys.modules.get(key, _SENTINEL)
        sys.modules[key] = mod

    import fastdeploy.model_executor as _me

    old_ops = getattr(_me, "ops", _SENTINEL)
    _me.ops = _ops

    from fastdeploy.model_executor.layers.moe import fused_moe_deepgemm_backend as _dgb

    dgb = _dgb

    yield

    # Teardown: restore sys.modules and module attributes
    for key, orig in saved.items():
        if orig is _SENTINEL:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = orig
    if old_ops is _SENTINEL:
        if hasattr(_me, "ops"):
            delattr(_me, "ops")
    else:
        _me.ops = old_ops


class _FakeQuantConfig:
    def __init__(self, ue8m0=False):
        self.weight_block_size = [2, 2]
        self.algo = "fp8"
        self.is_checkpoint_bf16 = False
        self.deepgemm_scale_ue8m0 = ue8m0


class _StubMoELayer(paddle.nn.Layer):
    def __init__(self, experts=1, hidden=4, inter=2, phase="prefill"):
        super().__init__()
        self.num_local_experts = self.num_experts = experts
        self.hidden_size, self.moe_intermediate_size = hidden, inter
        self.ep_size, self.ep_rank = 1, 0
        self.topk_method = "noaux_tc"
        self.n_group = self.topk_group = 1
        self.top_k = 1
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = paddle.zeros([experts], dtype="float32")
        self.renormalize = True
        self.redundant_table_manger = None
        self.layer_idx = 0
        self.fd_config = SimpleNamespace(
            model_config=SimpleNamespace(
                num_max_dispatch_tokens_per_rank=2,
                model="test",
                moe_phase=SimpleNamespace(phase=phase),
            ),
            scheduler_config=SimpleNamespace(splitwise_role="prefill", max_num_batched_tokens=4),
            eplb_config=SimpleNamespace(redundant_experts_num=0),
            parallel_config=SimpleNamespace(
                ep_group=None,
                use_internode_ll_two_stage=False,
                tensor_parallel_size=1,
            ),
            load_config=SimpleNamespace(load_strategy="meta", load_choices="default_v1"),
        )
        self.weight_key_map = {
            "up_gate_proj_expert_weight_key": "up_weight_{}",
            "down_proj_expert_weight_key": "down_weight_{}",
            "up_gate_proj_expert_weight_scale_key": "up_scale_{}",
            "down_proj_expert_weight_scale_key": "down_scale_{}",
        }

    def extract_moe_ffn_weights(self, sd):
        return sd["up"], sd["down"], None, None

    def load_experts_weight(self, sd, _uk, _dk, _rearr):
        if isinstance(sd, list):
            sd = dict(sd)
        return sd["up"], sd["down"], sd["ids"], None


def _create_method(layer, qc=None):
    m = dgb.DeepGemmFusedMoeMethod(qc or _FakeQuantConfig())
    m.create_weights(layer, model_format="torch")
    return m


def _compute_scale_dims(r, c, b=2):
    return [(r + b - 1) // b, (c + b - 1) // b]


@pytest.fixture(autouse=True)
def _ensure_dist(monkeypatch):
    monkeypatch.setattr(paddle.distributed, "is_initialized", lambda: True)


# ── Tests: standalone helpers (not covered by upstream) ─────────────────────


class TestStandaloneHelpers:
    """Standalone helpers not exercised by the upstream test suite."""

    def test_infermeta(self):
        meta_in = paddle.static.MetaTensor(shape=[2, 3], dtype=paddle.float16)
        meta_w1 = paddle.static.MetaTensor(shape=[3, 4], dtype=paddle.float16)
        out = dgb.m_grouped_fp8_gemm_nt_contiguous_custom_python_op_infermeta(
            meta_in,
            meta_in,
            meta_in,
            meta_in,
            meta_in,
            meta_w1,
            meta_in,
            2,
        )
        assert out.shape == [2, 4]

    def test_permute_and_depermute(self, monkeypatch):
        monkeypatch.setattr(
            dgb,
            "prefill_permute_to_masked_gemm",
            lambda x, s, ids, ne, mt: (x, s, paddle.zeros([2, 1], "int32"), paddle.zeros([ne], "int32")),
        )
        # int32 topk_ids triggers the int64 cast branch
        px, ps, imap, tnpe = dgb.call_prefill_permute_to_masked_gemm(
            x=paddle.ones([2, 4], "float32"),
            scale=paddle.ones([2, 2], "float32"),
            topk_ids=paddle.zeros([2, 1], dtype="int32"),
            num_local_experts=1,
            max_token_num=4,
        )
        assert px.shape == [2, 4]

        monkeypatch.setattr(
            dgb,
            "depermute_prefill_combine",
            lambda x, im, tw, n: paddle.zeros([n, x.shape[-1]], "float32"),
        )
        out = dgb.call_depermute_prefill_combine(
            x=paddle.ones([1, 4, 4], "float32"),
            indice_map=paddle.zeros([2, 1], "int32"),
            topk_weights=paddle.ones([2, 1], "float32"),
            num_worst_tokens=2,
        )
        assert out.shape[0] == 2


# ── Tests: weight management (not covered by upstream) ──────────────────────


class TestWeightLifecycle:
    """Weight creation, loading, and quantization — paths not reached by upstream."""

    def test_create_weights_and_post_loading(self, monkeypatch):
        layer = _StubMoELayer()
        m = _create_method(layer)
        assert hasattr(layer, "up_gate_proj_weight")
        assert layer.up_gate_proj_weight.shape[0] == layer.num_local_experts

        monkeypatch.setattr(
            dgb.BlockWiseFP8MoEMethod,
            "process_weights_after_loading",
            lambda self, lay: None,
        )
        m.process_weights_after_loading(layer)

    def test_process_loaded_weights(self, monkeypatch):
        layer = _StubMoELayer()
        m = _create_method(layer)
        H = layer.hidden_size
        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
            lambda w, bs: (
                paddle.ones_like(w, dtype="float32"),
                paddle.ones([1, 1], "float32"),
            ),
        )
        monkeypatch.setattr(paddle.Tensor, "copy_", lambda self, src, blocking=True: None)
        monkeypatch.setattr(paddle.Tensor, "set_value", lambda self, src: None)

        sd = {
            "up": [paddle.ones([H, layer.moe_intermediate_size * 2], "float32")],
            "down": [paddle.ones([layer.moe_intermediate_size, H], "float32")],
        }
        m.process_loaded_weights(layer, sd)

    @pytest.mark.parametrize("ue8m0", [False, True])
    def test_process_prequanted_weights(self, monkeypatch, ue8m0):
        monkeypatch.setattr(dgb, "get_tensor", lambda t, _m: t)
        layer = _StubMoELayer()
        m = _create_method(layer, _FakeQuantConfig(ue8m0=ue8m0))
        up_sc = paddle.ones(
            _compute_scale_dims(layer.hidden_size, layer.moe_intermediate_size * 2),
            "float32",
        )
        dn_sc = paddle.ones(
            _compute_scale_dims(layer.moe_intermediate_size, layer.hidden_size),
            "float32",
        )
        sd = [
            ("up_scale_0", up_sc),
            ("down_scale_0", dn_sc),
            (
                "up",
                [
                    paddle.ones(
                        [layer.hidden_size, layer.moe_intermediate_size * 2],
                        "int8",
                    )
                ],
            ),
            (
                "down",
                [paddle.ones([layer.moe_intermediate_size, layer.hidden_size], "int8")],
            ),
            ("ids", [0]),
        ]
        m.process_prequanted_weights(layer, state_dict=sd, is_rearrange=False)
        assert layer.up_gate_proj_weight_scale_inv.shape[0] == layer.num_local_experts


# ── Tests: apply_tp with FD_USE_PHI_FP8_QUANT=False ────────────────────────
# Upstream test_deepgemm_fused_moe.py only runs apply_tp with the default
# FD_USE_PHI_FP8_QUANT=True; the per_token_quant input path is never hit.


class TestApplyTpPerTokenPath:
    """apply_tp with per_token_quant input — supplementary to upstream."""

    def test_per_token_quant_input_path(self, monkeypatch):
        layer = _StubMoELayer()
        m = _create_method(layer)
        H = layer.hidden_size
        gate = paddle.nn.Linear(H, layer.num_experts, bias_attr=False)
        x = paddle.ones([2, H], dtype="float32")

        monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", False)
        monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_MOE_PERMUTE", False)
        monkeypatch.setattr(
            "fastdeploy.model_executor.layers.moe.moe.get_moe_scores",
            lambda g, ng, tg, k, s, b, r, **kw: (
                g,
                paddle.ones([g.shape[0], k], "float32"),
                paddle.zeros([g.shape[0], k], "int64"),
            ),
        )
        monkeypatch.setattr(
            dgb,
            "count_tokens_per_expert_func",
            lambda ids, ne, *a: (
                paddle.zeros([ne], "int32"),
                paddle.to_tensor(0, "int32"),
            ),
        )
        monkeypatch.setattr(
            _gpu,
            "per_token_quant",
            lambda x, bs, *_: (
                paddle.zeros([x.shape[0], H], "int8"),
                paddle.ones([1, 1], "float32"),
            ),
        )
        monkeypatch.setattr(
            _gpu,
            "ep_moe_expert_dispatch_fp8",
            lambda *a, **kw: (
                paddle.zeros([2, H], "int8"),
                paddle.ones([1, 1], "float32"),
                paddle.zeros([2], "int32"),
                paddle.zeros([1], "int32"),
                paddle.zeros([1], "int32"),
                paddle.ones([2, 1], "float32"),
                paddle.zeros([2], "int32"),
                paddle.zeros([1], "int32"),
                paddle.zeros([2], "int32"),
            ),
        )
        monkeypatch.setattr(
            dgb,
            "m_grouped_fp8_gemm_nt_contiguous_custom_python_op",
            lambda pi, *_a, **_kw: paddle.zeros([pi.shape[0], H], "float32"),
        )
        monkeypatch.setattr(_gpu, "ep_moe_expert_combine", lambda ffn, *a, **kw: ffn)

        out = m.apply_tp(layer, x, gate)
        assert out.shape[-1] == H


# ── Tests: apply_ep_prefill additive paths ──────────────────────────────────
# Upstream covers: zero-token, contiguous (FD_USE_PHI_FP8_QUANT=True default),
# and prob_in_advance variants — but NOT num_worst_tokens>0 (masked GEMM)
# and NOT shared_experts.


class TestEpPrefillMaskedGemm:
    """apply_ep_prefill masked-GEMM and shared-expert paths."""

    def _make_runner(self, n, H, num_worst_tokens=0, async_finish=False):
        class Runner:
            ep_engine = SimpleNamespace(async_finish=async_finish)

            def __init__(self):
                self.num_worst_tokens = num_worst_tokens

            def moe_select(self, _layer, gate_out):
                return (
                    paddle.zeros([gate_out.shape[0], 1], "int64"),
                    paddle.ones([gate_out.shape[0], 1], "float32"),
                )

            def dispatch(self, x, topk_idx, topk_weights, **_kw):
                scale = _kw.get(
                    "x_scale_tensor",
                    paddle.ones([x.shape[0], 1], "float32"),
                )
                return (
                    (paddle.zeros([n, H], "int8"), scale),
                    topk_idx,
                    topk_weights,
                    [n],
                    None,
                    _BufferStub.capture(),
                )

            def combine(self, out, _handle, _weights, event):
                return out, event

        return Runner()

    def test_masked_gemm_path(self, monkeypatch):
        """Masked GEMM triggered when num_worst_tokens > 0."""
        layer = _StubMoELayer()
        m = _create_method(layer)
        H = layer.hidden_size
        m.ep_prefill_runner = self._make_runner(
            n=2,
            H=H,
            num_worst_tokens=2,
            async_finish=True,
        )

        monkeypatch.setattr(dgb, "let_another_thread_run", lambda: None)
        monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", False)
        monkeypatch.setattr(
            _gpu,
            "per_token_quant",
            lambda x, bs, *_: (
                paddle.zeros([x.shape[0], H], "int8"),
                paddle.ones([1, 1], "float32"),
            ),
        )
        monkeypatch.setattr(
            dgb,
            "call_prefill_permute_to_masked_gemm",
            lambda x, scale, topk_ids, num_local_experts, max_token_num: (
                x,
                scale,
                paddle.zeros([num_local_experts, max_token_num, 1], "int32"),
                paddle.zeros([num_local_experts], "int32"),
            ),
        )
        monkeypatch.setattr(dgb, "m_grouped_fp8_gemm_nt_masked", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            _gpu,
            "fused_mask_swiglu_fp8_quant",
            lambda t, tn, bs, **kw: (
                paddle.zeros_like(t),
                paddle.zeros([1], "float32"),
            ),
        )
        monkeypatch.setattr(
            dgb,
            "call_depermute_prefill_combine",
            lambda x, indice_map, topk_weights, num_worst_tokens: paddle.zeros(
                [num_worst_tokens, x.shape[-1]], "float32"
            ),
        )

        gate = paddle.nn.Linear(H, layer.num_experts, bias_attr=False)
        x = paddle.ones([2, H], dtype="float32")
        shared = paddle.nn.Linear(H, H, bias_attr=False)
        shared_calls = []
        _orig_fwd = shared.forward
        shared.forward = lambda inp: (shared_calls.append(True), _orig_fwd(inp))[1]
        out = m.apply_ep_prefill(layer, x, gate, shared_experts=shared)
        assert out.shape[-1] == H
        assert shared_calls, "shared_experts was never invoked"


# ── Tests: apply_ep_decode additive paths ───────────────────────────────────
# Upstream covers the basic masked-gemm decode path but NOT shared_experts.


class TestEpDecodeSharedExperts:
    """apply_ep_decode shared-expert branch — supplementary path."""

    def test_shared_experts_branch(self, monkeypatch):
        layer = _StubMoELayer(phase="decode")
        m = _create_method(layer)
        H = layer.hidden_size

        class _DecodeRunner:
            def moe_select(self, _layer, gate_out):
                return (
                    paddle.zeros([gate_out.shape[0], 1], "int64"),
                    paddle.ones([gate_out.shape[0], 1], "float32"),
                )

            def dispatch(self, x, _ti, _tw, **_kw):
                return (
                    (
                        paddle.empty([0, H], x.dtype),
                        paddle.empty([0, H], "float32"),
                    ),
                    paddle.zeros([layer.num_local_experts], "int32"),
                    None,
                )

            def combine(self, ffn, *_a):
                return ffn

        m.ep_decoder_runner = _DecodeRunner()

        monkeypatch.setattr(dgb, "m_grouped_fp8_gemm_nt_masked", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            _gpu,
            "fused_mask_swiglu_fp8_quant",
            lambda t, tn, bs, **kw: (
                paddle.zeros_like(t),
                paddle.zeros([1], "float32"),
            ),
        )

        shared = paddle.nn.Linear(H, H, bias_attr=False)
        shared_calls = []
        _orig_fwd = shared.forward
        shared.forward = lambda inp: (shared_calls.append(True), _orig_fwd(inp))[1]
        gate = paddle.nn.Linear(H, layer.num_experts, bias_attr=False)
        x = paddle.ones([2, H], dtype="float32")
        out = m.apply_ep_decode(layer, x, gate, shared_experts=shared)
        assert out.shape[-1] == H
        assert shared_calls, "shared_experts was never invoked"
