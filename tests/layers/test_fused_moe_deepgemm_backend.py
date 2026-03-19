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

import os
import sys
import types
from types import SimpleNamespace

import paddle
import pytest

# ── Stub GPU-only modules ───────────────────────────────────────────────────
# Stub ops.gpu so the import chain (moe → triton_backend → fp8_utils →
# ops.gpu.deep_gemm) resolves without compiled CUDA extensions.
# Must be installed BEFORE any fastdeploy import that touches ops.gpu.


class _GpuOpsStub(types.ModuleType):
    """Catchall module: returns registered sub-modules or None for unknown attrs."""

    __path__ = []  # marks as package so `import X.Y.Z` can traverse

    def __getattr__(self, name):
        # Return registered sub-modules from sys.modules so `from X import Y` works
        fqn = f"{self.__name__}.{name}"
        sub = sys.modules.get(fqn)
        if sub is not None:
            return sub
        return None


sys.modules["fastdeploy.model_executor.ops.gpu"] = _GpuOpsStub("fastdeploy.model_executor.ops.gpu")
# fp8_utils.py:52 uses `import ...ops.gpu.deep_gemm as deep_gemm`
_deep_gemm_stub = types.ModuleType("fastdeploy.model_executor.ops.gpu.deep_gemm")
# Provide dummy callables so `deep_gemm.m_grouped_*` attribute access succeeds
_deep_gemm_stub.m_grouped_fp8_gemm_nt_contiguous = None
_deep_gemm_stub.m_grouped_fp8_gemm_nt_masked = None
_deep_gemm_stub.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous = None
_deep_gemm_stub.m_grouped_gemm_fp8_fp8_bf16_nt_masked = None
sys.modules["fastdeploy.model_executor.ops.gpu.deep_gemm"] = _deep_gemm_stub
_gpu = sys.modules["fastdeploy.model_executor.ops.gpu"]

_ep_mod = types.ModuleType("fastdeploy.model_executor.layers.moe.ep")


class _BufferStub:
    @staticmethod
    def capture():
        return SimpleNamespace(current_stream_wait=lambda: None)


_ep_mod.deep_ep = SimpleNamespace(Buffer=_BufferStub)
sys.modules["fastdeploy.model_executor.layers.moe.ep"] = _ep_mod

from fastdeploy.model_executor.layers.moe import (  # noqa: E402
    fused_moe_deepgemm_backend as dgb,
)

# ── Helpers ─────────────────────────────────────────────────────────────────


class _QuantConfig:
    def __init__(self, ue8m0=False):
        self.weight_block_size = [2, 2]
        self.algo = "fp8"
        self.is_checkpoint_bf16 = False
        self.deepgemm_scale_ue8m0 = ue8m0


class _DummyLayer(paddle.nn.Layer):
    def __init__(self, experts=1, hidden=4, inter=2):
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
                moe_phase=SimpleNamespace(phase="prefill"),
            ),
            scheduler_config=SimpleNamespace(splitwise_role="prefill", max_num_batched_tokens=4),
            eplb_config=SimpleNamespace(redundant_experts_num=0),
            parallel_config=SimpleNamespace(ep_group=None, use_internode_ll_two_stage=False, tensor_parallel_size=1),
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


def _ensure_dist():
    if not paddle.distributed.is_initialized():
        os.environ.setdefault("PADDLE_TRAINER_ID", "0")
        os.environ.setdefault("PADDLE_TRAINERS_NUM", "1")
        os.environ.setdefault("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170")
        os.environ.setdefault("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170")
        paddle.distributed.init_parallel_env()


def _init(layer, qc=None):
    m = dgb.DeepGemmFusedMoeMethod(qc or _QuantConfig())
    m.create_weights(layer, model_format="torch")
    return m


def _scale_shape(r, c, b=2):
    return [(r + b - 1) // b, (c + b - 1) // b]


# ── Tests ───────────────────────────────────────────────────────────────────


def test_create_weights(monkeypatch):
    """create_weights + infermeta + process_weights_after_loading."""
    _ensure_dist()
    layer = _DummyLayer()
    m = _init(layer)
    assert hasattr(layer, "up_gate_proj_weight")
    assert layer.up_gate_proj_weight.shape[0] == layer.num_local_experts

    # infermeta — covers L57
    meta = paddle.static.MetaTensor(shape=[2, 3], dtype=paddle.float16)
    out = dgb.m_grouped_fp8_gemm_nt_contiguous_custom_python_op_infermeta(
        meta,
        meta,
        meta,
        meta,
        meta,
        paddle.static.MetaTensor(shape=[3, 4], dtype=paddle.float16),
        meta,
        2,
    )
    assert out.shape == [2, 4]

    # call_prefill_permute_to_masked_gemm — covers L76-81
    monkeypatch.setattr(
        dgb,
        "prefill_permute_to_masked_gemm",
        lambda x, s, ids, ne, mt: (x, s, paddle.zeros([2, 1], "int32"), paddle.zeros([ne], "int32")),
    )
    px_in = paddle.ones([2, 4], "float32")
    ps_in = paddle.ones([2, 2], "float32")
    ids32 = paddle.zeros([2, 1], dtype="int32")
    px, ps, imap, tnpe = dgb.call_prefill_permute_to_masked_gemm(px_in, ps_in, ids32, 1, 4)
    assert px.shape == [2, 4]

    # call_depermute_prefill_combine — covers L102-104
    monkeypatch.setattr(
        dgb,
        "depermute_prefill_combine",
        lambda x, im, tw, n: paddle.zeros([n, x.shape[-1]], "float32"),
    )
    dp_out = dgb.call_depermute_prefill_combine(
        x=paddle.ones([1, 4, 4], "float32"),
        indice_map=paddle.zeros([2, 1], "int32"),
        topk_weights=paddle.ones([2, 1], "float32"),
        num_worst_tokens=2,
    )
    assert dp_out.shape[0] == 2

    # process_weights_after_loading — covers L151
    monkeypatch.setattr(dgb.BlockWiseFP8MoEMethod, "process_weights_after_loading", lambda self, layer: None)
    m.process_weights_after_loading(layer)

    # process_loaded_weights — covers L157-180
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.utils.per_block_cast_to_fp8",
        lambda w, bs: (paddle.ones_like(w, dtype="float32"), paddle.ones([1, 1], "float32")),
    )
    # Allow copy_ / set_value to succeed without matching float8 dtype
    monkeypatch.setattr(paddle.Tensor, "copy_", lambda self, src, blocking=True: None)
    monkeypatch.setattr(paddle.Tensor, "set_value", lambda self, src: None)
    H = layer.hidden_size
    sd = {
        "up": [paddle.ones([H, layer.moe_intermediate_size * 2], "float32")],
        "down": [paddle.ones([layer.moe_intermediate_size, H], "float32")],
    }
    m.process_loaded_weights(layer, sd)


def test_process_prequanted_weights(monkeypatch):
    """process_prequanted_weights — both ue8m0 branches."""
    _ensure_dist()
    monkeypatch.setattr(dgb, "get_tensor", lambda t, _m: t)

    for ue8m0 in (False, True):
        layer = _DummyLayer()
        m = _init(layer, _QuantConfig(ue8m0=ue8m0))
        up_scale = paddle.ones(_scale_shape(layer.hidden_size, layer.moe_intermediate_size * 2), "float32")
        down_scale = paddle.ones(_scale_shape(layer.moe_intermediate_size, layer.hidden_size), "float32")
        sd = [
            ("up_scale_0", up_scale),
            ("down_scale_0", down_scale),
            ("up", [paddle.ones([layer.hidden_size, layer.moe_intermediate_size * 2], "int8")]),
            ("down", [paddle.ones([layer.moe_intermediate_size, layer.hidden_size], "int8")]),
            ("ids", [0]),
        ]
        m.process_prequanted_weights(layer, state_dict=sd, is_rearrange=False)
        assert layer.up_gate_proj_weight_scale_inv.shape[0] == layer.num_local_experts


def test_apply_tp(monkeypatch):
    """apply_tp — noaux_tc + topk paths."""
    _ensure_dist()
    layer = _DummyLayer()
    m = _init(layer)
    H = layer.hidden_size
    gate = paddle.nn.Linear(H, layer.num_experts, bias_attr=False)
    x = paddle.ones([2, H], dtype="float32")

    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", False)
    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_MOE_PERMUTE", False)
    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.moe.moe.get_moe_scores",
        lambda g, ng, tg, k, s, b, r: (
            g,
            paddle.ones([g.shape[0], k], "float32"),
            paddle.zeros([g.shape[0], k], "int64"),
        ),
    )
    monkeypatch.setattr(
        dgb,
        "count_tokens_per_expert_func",
        lambda ids, ne: (paddle.zeros([ne], "int32"), paddle.to_tensor(0, "int32")),
    )
    monkeypatch.setattr(
        _gpu,
        "per_token_quant",
        lambda x, bs, *_: (paddle.zeros([x.shape[0], H], "int8"), paddle.ones([1, 1], "float32")),
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

    out = m.apply_tp(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
    assert out.shape[-1] == H

    layer.topk_method = "topk"
    monkeypatch.setattr(
        _gpu,
        "moe_topk_select",
        lambda g, b, k, *_: (paddle.zeros([g.shape[0], k], "int64"), paddle.ones([g.shape[0], k], "float32")),
    )
    out2 = m.apply_tp(layer, x, gate)
    assert out2.shape[-1] == H

    # PHI FP8 quant path — covers L534-540
    layer.topk_method = "noaux_tc"
    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", True)
    monkeypatch.setattr(
        "paddle.incubate.nn.functional.fp8_quant_blockwise",
        lambda x, **kw: (paddle.zeros([x.shape[0], H], "int8"), paddle.ones([x.shape[0] + 1, 1], "float32")),
    )
    out3 = m.apply_tp(layer, x, gate)
    assert out3.shape[-1] == H


def test_apply_ep_prefill(monkeypatch):
    """apply_ep_prefill — with tokens and empty path."""
    _ensure_dist()
    layer = _DummyLayer()
    m = _init(layer)
    H = layer.hidden_size

    class _PrefillRunner:
        def __init__(self, n, num_worst_tokens=0):
            self._n = n
            self.ep_engine = SimpleNamespace(async_finish=True)
            self.num_worst_tokens = num_worst_tokens

        def moe_select(self, _layer, gate_out):
            return paddle.zeros([gate_out.shape[0], 1], "int64"), paddle.ones([gate_out.shape[0], 1], "float32")

        def dispatch(self, x, topk_idx, topk_weights, **_kw):
            recv_x = (paddle.zeros([self._n, H], "int8"), paddle.ones([1, 1], "float32"))
            return recv_x, topk_idx, topk_weights, [self._n], None, _BufferStub.capture()

        def combine(self, out, _handle, _weights, event):
            return out, event

    monkeypatch.setattr(dgb, "let_another_thread_run", lambda: None)
    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", False)
    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_MOE_PERMUTE", False)
    monkeypatch.setattr(
        _gpu,
        "per_token_quant",
        lambda x, bs, *_: (paddle.zeros([x.shape[0], H], "int8"), paddle.ones([1, 1], "float32")),
    )
    monkeypatch.setattr(
        dgb,
        "count_tokens_per_expert_func",
        lambda ids, ne: (paddle.zeros([ne], "int32"), paddle.to_tensor(0, "int32")),
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
    monkeypatch.setattr(dgb, "m_grouped_fp8_gemm_nt_contiguous", lambda *a, **kw: None)
    monkeypatch.setattr(
        "paddle.incubate.nn.functional.swiglu",
        lambda t, *a, **kw: paddle.zeros([t.shape[0], t.shape[1] // 2], "float32"),
    )
    monkeypatch.setattr(_gpu, "ep_moe_expert_combine", lambda ffn, *a, **kw: ffn)

    gate = paddle.nn.Linear(H, layer.num_experts, bias_attr=False)
    x = paddle.ones([2, H], dtype="float32")

    m.ep_prefill_runner = _PrefillRunner(n=2)
    out = m.apply_ep_prefill(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
    assert out.shape[-1] == H

    m.ep_prefill_runner = _PrefillRunner(n=0)
    out_empty = m.apply_ep_prefill(layer, x, gate)
    assert out_empty.shape[-1] == H

    # PHI FP8 quant path — covers L283-289 + L374-379
    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", True)
    monkeypatch.setattr(
        "paddle.incubate.nn.functional.fp8_quant_blockwise",
        lambda x, **kw: (paddle.zeros([x.shape[0], H], "int8"), paddle.ones([x.shape[0] + 1, 1], "float32")),
    )
    m.ep_prefill_runner = _PrefillRunner(n=2)
    out_phi = m.apply_ep_prefill(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
    assert out_phi.shape[-1] == H

    # num_worst_tokens > 0 branch — covers L410-482 (masked gemm path)
    monkeypatch.setattr(dgb.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", False)
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
        lambda t, tn, bs, **kw: (paddle.zeros_like(t), paddle.zeros([1], "float32")),
    )
    monkeypatch.setattr(
        dgb,
        "call_depermute_prefill_combine",
        lambda x, indice_map, topk_weights, num_worst_tokens: paddle.zeros([num_worst_tokens, x.shape[-1]], "float32"),
    )
    m.ep_prefill_runner = _PrefillRunner(n=2, num_worst_tokens=2)
    out_worst = m.apply_ep_prefill(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
    assert out_worst.shape[-1] == H


def test_apply_ep_decode(monkeypatch):
    """apply_ep_decode."""
    _ensure_dist()
    layer = _DummyLayer()
    m = _init(layer)
    H = layer.hidden_size

    class _DecodeRunner:
        def moe_select(self, _layer, gate_out):
            return paddle.zeros([gate_out.shape[0], 1], "int64"), paddle.ones([gate_out.shape[0], 1], "float32")

        def dispatch(self, x, _ti, _tw, **_kw):
            return (
                (paddle.empty([0, H], x.dtype), paddle.empty([0, H], "float32")),
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
        lambda t, tn, bs, **kw: (paddle.zeros_like(t), paddle.zeros([1], "float32")),
    )

    gate = paddle.nn.Linear(H, layer.num_experts, bias_attr=False)
    x = paddle.ones([2, H], dtype="float32")
    out = m.apply_ep_decode(layer, x, gate, topk_ids_hookfunc=lambda **_: None)
    assert out.shape[-1] == H


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
