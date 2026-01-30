"""
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
"""

import os
from types import SimpleNamespace

import paddle

from fastdeploy.model_executor.layers.moe import (
    fused_moe_deepgemm_backend as deepgemm_backend,
)
from fastdeploy.model_executor.layers.moe.ep import EPDecoderRunner, EPPrefillRunner

paddle.set_device("gpu")


class _QuantConfig:
    def __init__(self):
        self.weight_block_size = [2, 2]
        self.algo = "fp8"
        self.is_checkpoint_bf16 = False


class _DummyLayer(paddle.nn.Layer):
    def __init__(self, num_local_experts=1, hidden_size=4, moe_intermediate_size=2, topk_method="noaux_tc"):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.ep_size = 1
        self.ep_rank = 0
        self.topk_method = topk_method
        self.n_group = 1
        self.topk_group = 1
        self.top_k = 1
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = paddle.zeros([self.num_experts], dtype="float32")
        self.renormalize = True
        self.redundant_table_manger = None
        self.layer_idx = 0
        self.fd_config = SimpleNamespace(
            model_config=SimpleNamespace(
                num_max_dispatch_tokens_per_rank=2,
                model="test",
                moe_phase=SimpleNamespace(phase="prefill"),
            ),
            scheduler_config=SimpleNamespace(splitwise_role="prefill"),
            eplb_config=SimpleNamespace(redundant_experts_num=0),
            parallel_config=SimpleNamespace(ep_group=None, use_internode_ll_two_stage=False),
            load_config=SimpleNamespace(load_strategy="meta", load_choices="default_v1"),
        )
        self.weight_key_map = {
            "up_gate_proj_expert_weight_key": "up_weight_{}",
            "down_proj_expert_weight_key": "down_weight_{}",
            "up_gate_proj_expert_weight_scale_key": "up_scale_{}",
            "down_proj_expert_weight_scale_key": "down_scale_{}",
        }

    def extract_moe_ffn_weights(self, state_dict):
        return state_dict["up"], state_dict["down"], None, None

    def load_experts_weight(self, state_dict, _up_key, _down_key, _is_rearrange):
        if isinstance(state_dict, list):
            state_dict = dict(state_dict)
        return state_dict["up"], state_dict["down"], state_dict["ids"], None


def _ensure_dist_init():
    if not paddle.distributed.is_initialized():
        os.environ.setdefault("PADDLE_TRAINER_ID", "0")
        os.environ.setdefault("PADDLE_TRAINERS_NUM", "1")
        os.environ.setdefault("PADDLE_CURRENT_ENDPOINT", "127.0.0.1:6170")
        os.environ.setdefault("PADDLE_TRAINER_ENDPOINTS", "127.0.0.1:6170")
        paddle.distributed.init_parallel_env()


def _make_weights(layer):
    up = [paddle.ones([layer.hidden_size, layer.moe_intermediate_size * 2], dtype="float16")]
    down = [paddle.ones([layer.moe_intermediate_size, layer.hidden_size], dtype="float16")]
    return up, down


def _scale_shape(rows, cols, block=2):
    return [(rows + block - 1) // block, (cols + block - 1) // block]


def test_infermeta_shape():
    meta = paddle.static.MetaTensor(shape=[2, 3], dtype=paddle.float16)
    out = deepgemm_backend.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous_custom_python_op_infermeta(
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


def test_deepgemm_weights_and_apply_paths(monkeypatch):
    _ensure_dist_init()
    method = deepgemm_backend.DeepGemmFusedMoeMethod(_QuantConfig())
    layer = _DummyLayer()
    method.create_weights(layer, model_format="torch")

    up, down = _make_weights(layer)
    method.process_loaded_weights(layer, {"up": up, "down": down})
    assert layer.up_gate_proj_weight.shape[0] == layer.num_local_experts

    prequant_up = [paddle.ones([layer.hidden_size, layer.moe_intermediate_size * 2], dtype="int8")]
    prequant_down = [paddle.ones([layer.moe_intermediate_size, layer.hidden_size], dtype="int8")]
    up_scale = paddle.ones(_scale_shape(layer.hidden_size, layer.moe_intermediate_size * 2), dtype="float32")
    down_scale = paddle.ones(_scale_shape(layer.moe_intermediate_size, layer.hidden_size), dtype="float32")
    state_list = [
        ("up_scale_0", up_scale),
        ("down_scale_0", down_scale),
        ("up", prequant_up),
        ("down", prequant_down),
        ("ids", [0]),
    ]
    method.process_prequanted_weights(layer, state_dict=state_list, is_rearrange=False)
    assert layer.up_gate_proj_weight_scale_inv.shape[0] == layer.num_local_experts

    from paddle.distributed.communication import deep_ep

    orig_dispatch = deep_ep.Buffer.get_dispatch_config
    orig_combine = deep_ep.Buffer.get_combine_config

    monkeypatch.setattr(
        deep_ep.Buffer,
        "get_dispatch_config",
        staticmethod(lambda num_ranks: orig_dispatch(2) if num_ranks == 1 else orig_dispatch(num_ranks)),
    )
    monkeypatch.setattr(
        deep_ep.Buffer,
        "get_combine_config",
        staticmethod(lambda num_ranks: orig_combine(2) if num_ranks == 1 else orig_combine(num_ranks)),
    )

    method.ep_prefill_runner = EPPrefillRunner(
        top_k=layer.top_k,
        hidden_size=layer.hidden_size,
        num_experts=layer.num_experts,
        splitwise_role=layer.fd_config.scheduler_config.splitwise_role,
        num_max_dispatch_tokens_per_rank=layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
        ep_size=layer.ep_size,
        ep_rank=layer.ep_rank,
        redundant_experts_num=layer.fd_config.eplb_config.redundant_experts_num,
        ep_group=layer.fd_config.parallel_config.ep_group,
        use_internode_ll_two_stage=layer.fd_config.parallel_config.use_internode_ll_two_stage,
    )
    method.ep_decoder_runner = EPDecoderRunner(
        top_k=layer.top_k,
        hidden_size=layer.hidden_size,
        num_experts=layer.num_experts,
        splitwise_role=layer.fd_config.scheduler_config.splitwise_role,
        num_max_dispatch_tokens_per_rank=layer.fd_config.model_config.num_max_dispatch_tokens_per_rank,
        ep_size=layer.ep_size,
        ep_rank=layer.ep_rank,
        redundant_experts_num=layer.fd_config.eplb_config.redundant_experts_num,
        ep_group=layer.fd_config.parallel_config.ep_group,
        use_internode_ll_two_stage=layer.fd_config.parallel_config.use_internode_ll_two_stage,
    )

    gate = paddle.nn.Linear(layer.hidden_size, layer.num_experts, bias_attr=False)
    x = paddle.ones([2, layer.hidden_size], dtype="float16")

    out_prefill = method.apply_ep_prefill(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out_prefill.shape[-1] == layer.hidden_size

    empty = paddle.zeros([0, layer.hidden_size], dtype="float16")
    out_empty = method.apply_ep_prefill(layer, empty, gate, topk_ids_hookfunc=lambda **_k: None)
    assert list(out_empty.shape) == [0, layer.hidden_size]

    out_decode = method.apply_ep_decode(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out_decode.shape[-1] == layer.hidden_size

    out_tp = method.apply_tp(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out_tp.shape[-1] == layer.hidden_size

    layer.topk_method = "topk"
    out_tp_alt = method.apply_tp(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out_tp_alt.shape[-1] == layer.hidden_size
