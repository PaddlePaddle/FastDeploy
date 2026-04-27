"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

from __future__ import annotations

import sys
import types

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda *args, **kwargs: None

iluvatar_stub = types.ModuleType("fastdeploy.model_executor.ops.iluvatar")
iluvatar_stub.moe_expert_ffn = lambda *args, **kwargs: None
iluvatar_stub.moe_expert_dispatch = lambda *args, **kwargs: None
iluvatar_stub.moe_expert_reduce = lambda *args, **kwargs: None
iluvatar_stub.mixed_fused_paged_attention = lambda *args, **kwargs: None
iluvatar_stub.paged_attention = lambda *args, **kwargs: None
iluvatar_stub.prefill_fused_paged_attention = lambda *args, **kwargs: None
sys.modules["fastdeploy.model_executor.ops.iluvatar"] = iluvatar_stub

import fastdeploy  # noqa: E402
from fastdeploy.model_executor.layers import utils as layer_utils
from fastdeploy.model_executor.layers.moe import fused_moe_cutlass_backend as backend


def align(x, y):
    return (x + y - 1) // y * y


class DummyQuantConfig:
    def __init__(self, algo="weight_only_int8", is_quantized=False, is_checkpoint_bf16=False):
        self.algo = algo
        self.is_quantized = is_quantized
        self.is_checkpoint_bf16 = is_checkpoint_bf16

    def name(self):
        return "dummy"


class DummyFDConfig:
    def __init__(self, load_choices="default_v1"):
        self.model_config = types.SimpleNamespace(model="dummy", prefix_layer_name="prefix")
        self.load_config = types.SimpleNamespace(load_choices=load_choices)


class DummyLayer(paddle.nn.Layer):
    def __init__(
        self,
        num_experts=2,
        num_local_experts=2,
        hidden_size=2,
        moe_intermediate_size=1,
        with_bias=False,
        ep_size=1,
        topk_method="normal",
    ):
        super().__init__()
        self.fd_config = DummyFDConfig()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.with_bias = with_bias
        self.ep_size = ep_size
        self.ep_rank = 0
        self.layer_idx = 0
        self.weight_dtype = "float32"
        self.top_k = 2
        self.topk_method = topk_method
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = None
        self.is_quantized = False
        self.moe_quant_config = types.SimpleNamespace(moe_dynamic_quant=False, hadamard_block_size=128)
        self.weight_key_map = {
            "up_gate_proj_expert_weight_key": "up_gate_{}",
            "down_proj_expert_weight_key": "down_proj_{}",
            "up_gate_proj_expert_weight_scale_key": "up_gate_scale_{}",
            "down_proj_expert_weight_scale_key": "down_proj_scale_{}",
            "up_gate_proj_expert_in_scale_key": "up_gate_in_{}",
            "down_proj_expert_in_scale_key": "down_proj_in_{}",
        }
        self._up_gate_weights = None
        self._down_weights = None
        self._up_gate_bias = None
        self._down_bias = None

    def extract_moe_ffn_weights(self, state_dict):
        return self._up_gate_weights, self._down_weights, [0, 1], [0, 1]

    def extract_moe_ffn_bias(self, state_dict):
        return self._up_gate_bias, self._down_bias

    def load_experts_weight(self, state_dict, up_key, down_key, is_rearrange):
        return self._up_gate_weights, self._down_weights, [0, 1], [0, 1]


def _build_state_dict(prefix, values):
    return {prefix.format(idx): paddle.to_tensor(value) for idx, value in enumerate(values)}


class TestFusedMoeCutlassBackend:
    def test_cutlass_process_loaded_weights_with_bias(self):
        layer = DummyLayer(with_bias=True)
        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 2, 2], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 2, 2], dtype="float32")
        layer.up_gate_proj_bias = layer.create_parameter(shape=[2, 2], dtype="float32")
        layer.down_proj_bias = layer.create_parameter(shape=[2, 2], dtype="float32")

        layer._up_gate_weights = [paddle.full([2, 2], 1.0), paddle.full([2, 2], 2.0)]
        layer._down_weights = [paddle.full([2, 2], 3.0), paddle.full([2, 2], 4.0)]
        layer._up_gate_bias = [paddle.full([2], 5.0), paddle.full([2], 6.0)]
        layer._down_bias = [paddle.full([2], 7.0), paddle.full([2], 8.0)]

        method = backend.CutlassMoEMethod(None)
        method.process_loaded_weights(layer, {})

        np.testing.assert_allclose(
            layer.up_gate_proj_weight.numpy(), np.stack([np.full((2, 2), 1.0), np.full((2, 2), 2.0)])
        )
        np.testing.assert_allclose(
            layer.down_proj_weight.numpy(), np.stack([np.full((2, 2), 3.0), np.full((2, 2), 4.0)])
        )
        np.testing.assert_allclose(layer.up_gate_proj_bias.numpy(), np.stack([np.full((2,), 5.0), np.full((2,), 6.0)]))
        np.testing.assert_allclose(layer.down_proj_bias.numpy(), np.stack([np.full((2,), 7.0), np.full((2,), 8.0)]))

    def test_compute_ffn_adds_bias(self, monkeypatch):
        layer = DummyLayer(with_bias=True)
        layer.activation = "gelu"
        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 2, 2], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 2, 2], dtype="float32")
        layer.down_proj_bias = layer.create_parameter(shape=[2, 2], dtype="float32")
        layer.down_proj_bias.set_value(paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]]))

        ops_stub = types.SimpleNamespace(gpu=types.SimpleNamespace())
        monkeypatch.setattr(backend.fastdeploy.model_executor, "ops", ops_stub, raising=False)
        setattr(
            ops_stub.gpu,
            "moe_expert_ffn",
            lambda *args, **kwargs: paddle.ones([2, 2]),
        )

        method = backend.CutlassMoEMethod(None)
        permute_input = paddle.ones([2, 2])
        token_nums_per_expert = paddle.to_tensor([1, 1])
        expert_idx_per_token = paddle.to_tensor([0, 1], dtype="int64")

        out = method.compute_ffn(layer, permute_input, token_nums_per_expert, expert_idx_per_token)
        np.testing.assert_allclose(out.numpy(), np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float32))

    def test_w4a8_scale_weights_processing_ep(self):
        layer = DummyLayer(ep_size=2)
        method = backend.CutlassW4A8MoEMethod(DummyQuantConfig())
        method.create_w4a8_scale_weights(layer, layer.weight_key_map)

        state_dict = {}
        state_dict.update(_build_state_dict("up_gate_in_{}", [[2.0], [4.0]]))
        state_dict.update(_build_state_dict("down_proj_in_{}", [[1.0], [2.0]]))
        state_dict.update(_build_state_dict("up_gate_scale_{}", [[1.0, 3.0], [2.0, 4.0]]))
        state_dict.update(_build_state_dict("down_proj_scale_{}", [[5.0, 7.0], [6.0, 8.0]]))

        method.load_w4a8_scale_weights(layer, layer.weight_key_map, state_dict, [0, 1], [0, 1])

        expected_up_in_scale = np.array([0.5, 0.25], dtype=np.float32)
        expected_down_in_scale = np.array([1.0, 0.5], dtype=np.float32)
        np.testing.assert_allclose(layer.up_gate_proj_in_scale.numpy(), expected_up_in_scale)
        np.testing.assert_allclose(layer.down_proj_in_scale.numpy(), expected_down_in_scale)
        np.testing.assert_allclose(layer.up_gate_proj_in_scale_all_experts.numpy(), expected_up_in_scale)

        weight_scale = np.stack([[1.0, 3.0], [2.0, 4.0]]) / (127 * 112) / expected_up_in_scale[:, None]
        np.testing.assert_allclose(layer.up_gate_proj_weight_scale.numpy(), weight_scale.astype(np.float32))

    def test_w4a8_scale_weights_missing_key_raises(self):
        layer = DummyLayer()
        method = backend.CutlassW4A8MoEMethod(DummyQuantConfig())
        layer.weight_key_map["down_proj_expert_weight_scale_key"] = None
        method.create_w4a8_scale_weights(layer, layer.weight_key_map)

        with pytest.raises(ValueError, match="down_proj_weight_scale"):
            method.load_w4a8_scale_weights(layer, layer.weight_key_map, {}, [0], [0])

    def test_w4afp8_scale_weights_for_quantized_and_dynamic(self, monkeypatch):
        monkeypatch.setattr(backend, "w4afp8_gemm_scale_permute", lambda x: x, raising=False)

        layer = DummyLayer(ep_size=2)
        layer.is_quantized = True
        layer.moe_quant_config = types.SimpleNamespace(moe_dynamic_quant=False, hadamard_block_size=128)
        method = backend.CutlassW4AFP8MoEMethod(DummyQuantConfig(is_quantized=True))
        method.create_w4afp8_scale_weights(layer, layer.weight_key_map)

        state_dict = {}
        state_dict.update(_build_state_dict("up_gate_in_{}", [[2.0], [4.0]]))
        state_dict.update(_build_state_dict("down_proj_in_{}", [[1.0], [2.0]]))
        state_dict.update(_build_state_dict("up_gate_scale_{}", [[1.0, 3.0], [2.0, 4.0]]))
        state_dict.update(_build_state_dict("down_proj_scale_{}", [[5.0, 7.0], [6.0, 8.0]]))

        method.load_w4afp8_scale_weights(
            layer,
            layer.weight_key_map,
            state_dict,
            [0, 1],
            [0, 1],
            dynamic_scale_weight_map={},
        )

        expected_up_in_scale = np.array([0.5, 0.25], dtype=np.float32)
        np.testing.assert_allclose(layer.up_gate_proj_in_scale.numpy(), expected_up_in_scale)
        np.testing.assert_allclose(layer.up_gate_proj_in_scale_all_experts.numpy(), expected_up_in_scale)

        weight_scale = np.stack([[1.0, 3.0], [2.0, 4.0]]) / (448 * 7 * 2 ** (-9)) / expected_up_in_scale[:, None]
        np.testing.assert_allclose(layer.up_gate_proj_weight_scale.numpy(), weight_scale.astype(np.float32))

        dynamic_layer = DummyLayer()
        dynamic_layer.is_quantized = False
        dynamic_method = backend.CutlassW4AFP8MoEMethod(DummyQuantConfig(is_quantized=False))
        dynamic_method.create_w4afp8_scale_weights(dynamic_layer, dynamic_layer.weight_key_map)
        dynamic_layer.up_gate_proj_weight_scale = dynamic_layer.create_parameter(shape=[2, 2], dtype="float32")
        dynamic_layer.down_proj_weight_scale = dynamic_layer.create_parameter(shape=[2, 2], dtype="float32")
        dynamic_scales = {
            "up_gate_proj_weight_scale": [paddle.ones([2]) * 2.0, paddle.ones([2]) * 4.0],
            "down_proj_weight_scale": [paddle.ones([2]) * 3.0, paddle.ones([2]) * 5.0],
        }

        dynamic_method.load_w4afp8_scale_weights(
            dynamic_layer,
            dynamic_layer.weight_key_map,
            {},
            [0, 1],
            [0, 1],
            dynamic_scale_weight_map=dynamic_scales,
        )

        expected_dynamic = np.stack([np.full((2,), 2.0), np.full((2,), 4.0)]) / (440 * 7 * 2 ** (-9))
        np.testing.assert_allclose(
            dynamic_layer.up_gate_proj_weight_scale.numpy(), expected_dynamic.astype(np.float32)
        )

    def test_apply_ep_prefill_and_decode(self, monkeypatch):
        class DummyEvent:
            def current_stream_wait(self):
                return None

        class DummyRunner:
            def __init__(self):
                self.ep_engine = types.SimpleNamespace(async_finish=True)

            def moe_select(self, layer, gate_out):
                return paddle.to_tensor([[0, 1]]), paddle.to_tensor([[0.6, 0.4]])

            def dispatch(self, x, topk_idx, topk_weights):
                recv_x = x + 1
                recv_topk_idx = topk_idx
                recv_topk_weights = topk_weights
                recv_num_tokens_per_expert_list = [1, 0]
                return (
                    recv_x,
                    recv_topk_idx,
                    recv_topk_weights,
                    recv_num_tokens_per_expert_list,
                    object(),
                    DummyEvent(),
                )

            def combine(self, tmp_ffn_out, handle, recv_topk_weights):
                return tmp_ffn_out + 1, DummyEvent()

        class DummyDecoderRunner:
            def __init__(self):
                self.ep_engine = types.SimpleNamespace(async_finish=True)

            def moe_select(self, layer, gate_out):
                return paddle.to_tensor([[0, 1]]), paddle.to_tensor([[0.6, 0.4]])

            def dispatch(self, x, topk_idx, topk_weights, expertwise_scale=None, use_fp8=False, quant_group_size=-1):
                permute_input = paddle.ones([2, 1, 2])
                token_nums_per_expert = paddle.to_tensor([1, 0], dtype="int64")
                return permute_input, token_nums_per_expert, object()

            def combine(self, ffn_out, topk_idx, topk_weights, handle, quant_group_size=-1):
                return ffn_out + 2

        ops_stub = types.SimpleNamespace(gpu=types.SimpleNamespace())
        monkeypatch.setattr(backend.fastdeploy.model_executor, "ops", ops_stub, raising=False)
        setattr(
            ops_stub.gpu,
            "ep_moe_expert_dispatch",
            lambda *args, **kwargs: (
                paddle.ones([1, 2]),
                paddle.to_tensor([0]),
                paddle.to_tensor([1]),
                paddle.to_tensor([1]),
                paddle.to_tensor([1]),
                paddle.to_tensor([0]),
                paddle.to_tensor([0]),
                None,
            ),
        )
        setattr(
            ops_stub.gpu,
            "ep_moe_expert_combine",
            lambda *args, **kwargs: paddle.ones([1, 2]) * 3,
        )

        layer = DummyLayer(with_bias=False)
        method = backend.CutlassMoEMethod(None)
        method.ep_prefill_runner = DummyRunner()
        method.ep_decoder_runner = DummyDecoderRunner()

        monkeypatch.setattr(method, "compute_ffn", lambda *args, **kwargs: paddle.ones([1, 2]) * 2)

        x = paddle.ones([1, 2])
        gate = paddle.nn.Identity()

        out_prefill = method.apply_ep_prefill(layer, x, gate)
        np.testing.assert_allclose(out_prefill.numpy(), np.full((1, 2), 4.0))

        method.moe_quant_type = "w4a8"
        out_decode = method.apply_ep_decode(layer, x, gate)
        np.testing.assert_allclose(out_decode.numpy(), np.full((1, 2), 4.0))

    def test_apply_ep_prefill_zero_tokens(self, monkeypatch):
        class DummyEvent:
            def current_stream_wait(self):
                return None

        class DummyRunner:
            def __init__(self):
                self.ep_engine = types.SimpleNamespace(async_finish=False)

            def moe_select(self, layer, gate_out):
                return paddle.to_tensor([[0, 1]]), paddle.to_tensor([[0.6, 0.4]])

            def dispatch(self, x, topk_idx, topk_weights):
                recv_x = x * 2
                recv_num_tokens_per_expert_list = [0, 0]
                return recv_x, topk_idx, topk_weights, recv_num_tokens_per_expert_list, object(), DummyEvent()

            def combine(self, tmp_ffn_out, handle, recv_topk_weights):
                return tmp_ffn_out + 5, DummyEvent()

        method = backend.CutlassMoEMethod(None)
        method.ep_prefill_runner = DummyRunner()
        layer = DummyLayer(with_bias=False)
        x = paddle.ones([1, 2])
        gate = paddle.nn.Identity()

        out = method.apply_ep_prefill(layer, x, gate)
        np.testing.assert_allclose(out.numpy(), np.full((1, 2), 7.0))

    def test_apply_ep_decode_weight_only(self, monkeypatch):
        class DummyDecoderRunner:
            def __init__(self):
                self.ep_engine = types.SimpleNamespace(async_finish=False)

            def moe_select(self, layer, gate_out):
                return paddle.to_tensor([[0, 1]]), paddle.to_tensor([[0.6, 0.4]])

            def dispatch(self, x, topk_idx, topk_weights, expertwise_scale=None, use_fp8=False, quant_group_size=-1):
                permute_input = paddle.ones([1, 2])
                token_nums_per_expert = paddle.to_tensor([1, 0], dtype="int64")
                return permute_input, token_nums_per_expert, object()

            def combine(self, ffn_out, topk_idx, topk_weights, handle, quant_group_size=-1):
                return ffn_out + 3

        method = backend.CutlassMoEMethod(None)
        method.moe_quant_type = "weight_only_int8"
        method.ep_decoder_runner = DummyDecoderRunner()
        monkeypatch.setattr(method, "compute_ffn", lambda *args, **kwargs: paddle.ones([1, 2]) * 2)

        layer = DummyLayer(with_bias=False)
        x = paddle.ones([1, 2])
        gate = paddle.nn.Identity()
        out = method.apply_ep_decode(layer, x, gate)
        np.testing.assert_allclose(out.numpy(), np.full((1, 2), 5.0))

    def test_apply_tp_with_dispatch_and_reduce(self, monkeypatch):
        def fake_get_moe_scores(
            gate_out, n_group, topk_group, top_k, routed_scaling_factor, bias, renormalize, topk_reduce_func=None
        ):
            return gate_out, paddle.to_tensor([[0.6, 0.4]]), paddle.to_tensor([[0, 1]])

        def fake_dispatch(*args, **kwargs):
            permute_input = paddle.ones([1, 2])
            token_nums_per_expert = paddle.to_tensor([1, 0])
            permute_indices_per_token = paddle.to_tensor([0])
            topk_weights = paddle.to_tensor([[0.6, 0.4]])
            topk_idx = paddle.to_tensor([[0, 1]])
            expert_idx_per_token = paddle.to_tensor([0])
            dequant_scale = None
            max_tokens_per_expert = None
            return (
                permute_input,
                token_nums_per_expert,
                permute_indices_per_token,
                topk_weights,
                topk_idx,
                expert_idx_per_token,
                dequant_scale,
                max_tokens_per_expert,
            )

        def fake_reduce(*args, **kwargs):
            return paddle.ones([1, 2]) * 5

        monkeypatch.setattr(backend, "get_moe_scores", fake_get_moe_scores, raising=False)
        monkeypatch.setattr(backend, "moe_expert_dispatch", fake_dispatch, raising=False)
        monkeypatch.setattr(backend, "moe_expert_reduce", fake_reduce, raising=False)

        layer = DummyLayer(topk_method="noaux_tc")
        method = backend.CutlassMoEMethod(None)
        monkeypatch.setattr(method, "compute_ffn", lambda *args, **kwargs: paddle.ones([1, 2]) * 4)

        x = paddle.ones([1, 2])
        gate = paddle.nn.Identity()
        out = method.apply_tp(layer, x, gate)

        np.testing.assert_allclose(out.numpy(), np.full((1, 2), 5.0))

    def test_apply_tp_with_bias_and_w4a8(self, monkeypatch):
        dispatch_args = {}

        def fake_dispatch(*args, **kwargs):
            dispatch_args["called"] = True
            permute_input = paddle.ones([1, 2])
            token_nums_per_expert = paddle.to_tensor([1, 0])
            permute_indices_per_token = paddle.to_tensor([0])
            topk_weights = paddle.to_tensor([[0.6, 0.4]])
            topk_idx = paddle.to_tensor([[0, 1]])
            expert_idx_per_token = paddle.to_tensor([1])
            dequant_scale = paddle.ones([1])
            max_tokens_per_expert = paddle.to_tensor([1, 1])
            return (
                permute_input,
                token_nums_per_expert,
                permute_indices_per_token,
                topk_weights,
                topk_idx,
                expert_idx_per_token,
                dequant_scale,
                max_tokens_per_expert,
            )

        def fake_reduce(*args, **kwargs):
            return paddle.ones([1, 2]) * 6

        monkeypatch.setattr(backend, "moe_expert_dispatch", fake_dispatch, raising=False)
        monkeypatch.setattr(backend, "moe_expert_reduce", fake_reduce, raising=False)

        layer = DummyLayer(topk_method="default", with_bias=True)
        layer.gate_correction_bias = paddle.ones([2])
        method = backend.CutlassMoEMethod(None)
        method.moe_quant_type = "w4a8"

        def fake_compute_ffn(layer, permute_input, token_nums_per_expert, expert_idx_per_token, *args, **kwargs):
            assert expert_idx_per_token.dtype == paddle.int64
            return paddle.ones([1, 2]) * 4

        monkeypatch.setattr(method, "compute_ffn", fake_compute_ffn)

        x = paddle.ones([1, 2])
        gate = paddle.nn.Identity()
        out = method.apply_tp(layer, x, gate)
        np.testing.assert_allclose(out.numpy(), np.full((1, 2), 6.0))
        assert dispatch_args.get("called", False)

    def test_w4a8_prequanted_and_loaded_weights(self, monkeypatch):
        layer = DummyLayer(ep_size=2, hidden_size=4, moe_intermediate_size=2)
        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 2, 4], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 1, 4], dtype="float32")
        layer.up_gate_proj_weight_scale = layer.create_parameter(shape=[2, 4], dtype="float32")
        layer.down_proj_weight_scale = layer.create_parameter(shape=[2, 4], dtype="float32")
        layer.up_gate_proj_in_scale_all_experts = layer.create_parameter(shape=[2], dtype="float32")
        layer.up_gate_proj_in_scale = layer.create_parameter(shape=[2], dtype="float32")
        layer.down_proj_in_scale = layer.create_parameter(shape=[2], dtype="float32")

        layer._up_gate_weights = [paddle.ones([2, 4]), paddle.ones([2, 4]) * 2]
        layer._down_weights = [paddle.ones([1, 4]) * 3, paddle.ones([1, 4]) * 4]

        state_dict = {}
        state_dict.update(_build_state_dict("up_gate_in_{}", [[2.0], [4.0]]))
        state_dict.update(_build_state_dict("down_proj_in_{}", [[1.0], [2.0]]))
        state_dict.update(_build_state_dict("up_gate_scale_{}", [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        state_dict.update(_build_state_dict("down_proj_scale_{}", [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]))

        method = backend.CutlassW4A8MoEMethod(DummyQuantConfig())
        method.process_prequanted_weights(layer, state_dict, is_rearrange=False)
        np.testing.assert_allclose(
            layer.up_gate_proj_in_scale_all_experts.numpy(), np.array([2.0, 4.0], dtype=np.float32)
        )

        load_layer = DummyLayer(ep_size=2, hidden_size=4, moe_intermediate_size=2)
        load_layer.up_gate_proj_weight = load_layer.create_parameter(
            shape=[2, 2, 4],
            dtype="int8",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        load_layer.down_proj_weight = load_layer.create_parameter(
            shape=[2, 1, 4],
            dtype="int8",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        load_layer.up_gate_proj_weight_scale = load_layer.create_parameter(shape=[2, 4], dtype="float32")
        load_layer.down_proj_weight_scale = load_layer.create_parameter(shape=[2, 4], dtype="float32")
        load_layer.up_gate_proj_in_scale_all_experts = load_layer.create_parameter(shape=[2], dtype="float32")
        load_layer.up_gate_proj_in_scale = load_layer.create_parameter(shape=[2], dtype="float32")
        load_layer.down_proj_in_scale = load_layer.create_parameter(shape=[2], dtype="float32")
        load_layer._up_gate_weights = layer._up_gate_weights
        load_layer._down_weights = layer._down_weights

        load_state_dict = {}
        load_state_dict.update(_build_state_dict("up_gate_in_{}", [[2.0], [4.0]]))
        load_state_dict.update(_build_state_dict("down_proj_in_{}", [[1.0], [2.0]]))
        load_state_dict.update(_build_state_dict("up_gate_scale_{}", [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))
        load_state_dict.update(
            _build_state_dict("down_proj_scale_{}", [[9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]])
        )

        monkeypatch.setattr(
            backend, "weight_quantize", lambda tensor, **kwargs: (tensor.cast("int8"), paddle.ones([4]))
        )
        method.process_loaded_weights(load_layer, load_state_dict)
        assert layer.up_gate_proj_weight.shape[0] == 2

    def test_w4afp8_prequanted_weights_dynamic_and_static(self, monkeypatch):
        monkeypatch.setattr(backend, "w4afp8_gemm_weight_convert", lambda x: x, raising=False)
        layer = DummyLayer(ep_size=2)
        layer.is_quantized = True
        layer.moe_quant_config = types.SimpleNamespace(moe_dynamic_quant=False, hadamard_block_size=128)
        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 1, 2], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 1, 2], dtype="float32")
        layer.up_gate_proj_weight_scale = layer.create_parameter(shape=[2, 2], dtype="float32")
        layer.down_proj_weight_scale = layer.create_parameter(shape=[2, 2], dtype="float32")
        layer.up_gate_proj_in_scale_all_experts = layer.create_parameter(shape=[2], dtype="float32")
        layer.up_gate_proj_in_scale = layer.create_parameter(shape=[2], dtype="float32")
        layer.down_proj_in_scale = layer.create_parameter(shape=[2], dtype="float32")

        layer._up_gate_weights = [paddle.ones([1, 2]), paddle.ones([1, 2]) * 2]
        layer._down_weights = [paddle.ones([1, 2]) * 3, paddle.ones([1, 2]) * 4]

        state_dict = {}
        state_dict.update(_build_state_dict("up_gate_in_{}", [[2.0], [4.0]]))
        state_dict.update(_build_state_dict("down_proj_in_{}", [[1.0], [2.0]]))
        state_dict.update(_build_state_dict("up_gate_scale_{}", [[1.0, 2.0], [3.0, 4.0]]))
        state_dict.update(_build_state_dict("down_proj_scale_{}", [[5.0, 6.0], [7.0, 8.0]]))

        method = backend.CutlassW4AFP8MoEMethod(DummyQuantConfig(is_quantized=True))
        method.process_prequanted_weights(layer, state_dict, is_rearrange=False)

        dynamic_layer = DummyLayer()
        dynamic_layer.is_quantized = True
        dynamic_layer.moe_quant_config = types.SimpleNamespace(moe_dynamic_quant=True, hadamard_block_size=128)
        dynamic_layer.up_gate_proj_weight = dynamic_layer.create_parameter(shape=[2, 1, 2], dtype="float32")
        dynamic_layer.down_proj_weight = dynamic_layer.create_parameter(shape=[2, 1, 2], dtype="float32")
        dynamic_layer.up_gate_proj_weight_scale = dynamic_layer.create_parameter(shape=[2, 2], dtype="float32")
        dynamic_layer.down_proj_weight_scale = dynamic_layer.create_parameter(shape=[2, 2], dtype="float32")
        dynamic_layer._up_gate_weights = layer._up_gate_weights
        dynamic_layer._down_weights = layer._down_weights

        dynamic_state = {}
        dynamic_state.update(_build_state_dict("up_gate_scale_{}", [[1.0, 2.0], [3.0, 4.0]]))
        dynamic_state.update(_build_state_dict("down_proj_scale_{}", [[5.0, 6.0], [7.0, 8.0]]))
        method.process_prequanted_weights(dynamic_layer, dynamic_state, is_rearrange=False)

    def test_w4afp8_online_quantize_and_loaded_weights(self, monkeypatch):
        monkeypatch.setattr(
            backend,
            "group_wise_int4_weight_quantize",
            lambda x, group_size=128: (x.cast("int8"), paddle.ones([1])),
        )
        monkeypatch.setattr(backend, "pack", lambda x, bits=4: x)
        monkeypatch.setattr(backend, "w4afp8_gemm_weight_convert", lambda x: x, raising=False)
        monkeypatch.setattr(backend, "w4afp8_gemm_scale_permute", lambda x: x, raising=False)
        monkeypatch.setattr(backend, "free_tensor", lambda x: None)
        monkeypatch.setattr(backend, "weight_fully_copied", lambda _: True)
        monkeypatch.setattr(layer_utils, "get_orthogonal_matrix", lambda size, mode: (paddle.eye(size), 128))

        original_to = paddle.Tensor.to

        def safe_to(self, device=None, dtype=None, blocking=None):
            if device is not None and device.__class__.__name__ == "Place":
                device = "cpu"
            return original_to(self, device, dtype, blocking)

        monkeypatch.setattr(paddle.Tensor, "to", safe_to)

        layer = DummyLayer(hidden_size=2, moe_intermediate_size=2)
        layer.is_quantized = False
        layer.moe_quant_config = types.SimpleNamespace(moe_dynamic_quant=False, hadamard_block_size=128)
        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 2, 4], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 2, 2], dtype="float32")
        layer.up_gate_proj_weight.set_value(paddle.ones([2, 2, 4]))
        layer.down_proj_weight.set_value(paddle.ones([2, 2, 2]))

        method = backend.CutlassW4AFP8MoEMethod(DummyQuantConfig(is_quantized=False))
        method.model_format = "paddle"
        method.process_weights_after_loading(layer)
        method.process_weights_after_loading(layer)

        monkeypatch.setattr(backend, "rotate_model", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            backend,
            "group_wise_int4_weight_quantize",
            lambda x, group_size=128: (x.cast("int8"), paddle.ones([1])),
        )

        layer._up_gate_weights = [paddle.ones([2, 4]), paddle.ones([2, 4])]
        layer._down_weights = [paddle.ones([2, 2]), paddle.ones([2, 2])]
        method.process_loaded_weights(layer, {})

    def test_weight_only_create_and_process(self, monkeypatch):
        layer = DummyLayer(hidden_size=4, moe_intermediate_size=2)
        layer.weight_dtype = "bfloat16"
        layer.fd_config = DummyFDConfig(load_choices="default_v1")
        quant_config = DummyQuantConfig(algo="weight_only_int8", is_checkpoint_bf16=True)
        method = backend.CutlassWeightOnlyMoEMethod(quant_config)
        method.create_weights(layer, num_experts=2, hidden_size=4, moe_intermediate_size=2, model_format="paddle")

        fully_copied_calls = {"count": 0}

        def weight_fully_copied_stub(_):
            fully_copied_calls["count"] += 1
            return fully_copied_calls["count"] == 1

        monkeypatch.setattr(backend, "weight_fully_copied", weight_fully_copied_stub)
        monkeypatch.setattr(backend, "process_weight_transpose", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            backend, "weight_quantize", lambda tensor, algo=None: (tensor.cast("int8"), paddle.ones([4]))
        )

        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 4, 4], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 2, 4], dtype="float32")
        layer.up_gate_proj_weight.set_value(paddle.ones([2, 4, 4]))
        layer.down_proj_weight.set_value(paddle.ones([2, 2, 4]))

        layer._up_gate_weights = [paddle.ones([4, 4]), paddle.ones([4, 4])]
        layer._down_weights = [paddle.ones([2, 4]), paddle.ones([2, 4])]

        method.process_weights_after_loading(layer)

        load_layer = DummyLayer(hidden_size=4, moe_intermediate_size=2)
        load_layer.up_gate_proj_weight = load_layer.create_parameter(
            shape=[2, 4, 4],
            dtype="int8",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        load_layer.down_proj_weight = load_layer.create_parameter(
            shape=[2, 2, 4],
            dtype="int8",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        load_layer.up_gate_proj_weight_scale = load_layer.create_parameter(shape=[2, 4], dtype="float32")
        load_layer.down_proj_weight_scale = load_layer.create_parameter(shape=[2, 4], dtype="float32")
        load_layer._up_gate_weights = layer._up_gate_weights
        load_layer._down_weights = layer._down_weights

        method.process_loaded_weights(load_layer, {})

    def test_w4a8_create_weights_with_bias(self):
        layer = DummyLayer(with_bias=True, hidden_size=4, moe_intermediate_size=2)
        method = backend.CutlassW4A8MoEMethod(DummyQuantConfig())
        method.create_weights(layer)
        assert layer.up_gate_proj_weight.shape[0] == layer.num_local_experts
        assert layer.down_proj_bias.shape[0] == layer.num_experts

    def test_w4afp8_create_weights_branches(self):
        layer = DummyLayer(with_bias=True, hidden_size=4, moe_intermediate_size=2)
        method = backend.CutlassW4AFP8MoEMethod(DummyQuantConfig(is_quantized=False))
        method.create_weights(layer, model_format="torch")
        assert layer.up_gate_proj_weight.shape[-1] == layer.hidden_size

        quant_layer = DummyLayer(with_bias=False, hidden_size=4, moe_intermediate_size=2)
        quant_layer.is_quantized = True
        quant_method = backend.CutlassW4AFP8MoEMethod(DummyQuantConfig(is_quantized=True))
        quant_method.create_weights(quant_layer, model_format="paddle")
        assert quant_layer.up_gate_proj_weight.dtype == paddle.int8

    def test_weight_only_prequanted_and_int4_create(self):
        layer = DummyLayer(hidden_size=4, moe_intermediate_size=2)
        layer.up_gate_proj_weight = layer.create_parameter(shape=[2, 4, 4], dtype="float32")
        layer.down_proj_weight = layer.create_parameter(shape=[2, 2, 4], dtype="float32")
        layer.up_gate_proj_weight_scale = layer.create_parameter(shape=[2, 4], dtype="float32")
        layer.down_proj_weight_scale = layer.create_parameter(shape=[2, 4], dtype="float32")
        layer._up_gate_weights = [paddle.ones([4, 4]), paddle.ones([4, 4]) * 2]
        layer._down_weights = [paddle.ones([2, 4]), paddle.ones([2, 4]) * 3]

        state_dict = {}
        state_dict.update(_build_state_dict("up_gate_scale_{}", [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]))
        state_dict.update(_build_state_dict("down_proj_scale_{}", [[3.0, 3.0, 3.0, 3.0], [4.0, 4.0, 4.0, 4.0]]))

        method = backend.CutlassWeightOnlyMoEMethod(DummyQuantConfig(algo="weight_only_int8"))
        method.process_prequanted_weights(layer, state_dict, is_rearrange=False)

        int4_layer = DummyLayer(hidden_size=4, moe_intermediate_size=2)
        int4_method = backend.CutlassWeightOnlyMoEMethod(DummyQuantConfig(algo="weight_only_int4"))
        int4_method.create_weights(
            int4_layer, num_experts=2, hidden_size=4, moe_intermediate_size=2, model_format="paddle"
        )


# ---------------------------------------------------------------------------
# Real-op tests for FD_USE_PHI_MOE_PERMUTE=True (w16a16, moe_permute path)
# ---------------------------------------------------------------------------

from fastdeploy.platforms import current_platform  # noqa: E402

_CUDA_AVAILABLE = current_platform.is_cuda()
requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA required")


class RealMoELayer(paddle.nn.Layer):
    """Minimal bf16 MoE layer with real weights for moe_permute path testing."""

    def __init__(self, num_experts=4, hidden_size=64, moe_intermediate_size=32, top_k=2):
        super().__init__()
        self.fd_config = DummyFDConfig()
        self.num_experts = num_experts
        self.num_local_experts = num_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k = top_k
        self.topk_method = "noaux_tc"
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.with_bias = False
        self.ep_size = 1
        self.ep_rank = 0
        self.layer_idx = 0
        self.weight_dtype = "bfloat16"
        self.is_quantized = False
        self.activation = "swiglu"
        self.moe_quant_config = types.SimpleNamespace(moe_dynamic_quant=False, hadamard_block_size=128)
        self.gate_correction_bias = self.create_parameter(
            shape=[1, num_experts],
            dtype="float32",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        paddle.seed(0)
        self.up_gate_proj_weight = self.create_parameter(
            shape=[num_experts, hidden_size, 2 * moe_intermediate_size],
            dtype="bfloat16",
        )
        self.down_proj_weight = self.create_parameter(
            shape=[num_experts, moe_intermediate_size, hidden_size],
            dtype="bfloat16",
        )
        self.up_gate_proj_weight.set_value(
            paddle.randn([num_experts, hidden_size, 2 * moe_intermediate_size]).cast("bfloat16") * 0.01
        )
        self.down_proj_weight.set_value(
            paddle.randn([num_experts, moe_intermediate_size, hidden_size]).cast("bfloat16") * 0.01
        )


class SimpleLinearGate(paddle.nn.Layer):
    def __init__(self, hidden_size, num_experts):
        super().__init__()
        self.weight = self.create_parameter(shape=[hidden_size, num_experts], dtype="float32")

    def forward(self, x):
        return paddle.matmul(x.cast("float32"), self.weight)


class TestMoePermuteTrueRealOps:
    """Real-op tests for FD_USE_PHI_MOE_PERMUTE=True on the w16a16 path."""

    def _build(self, num_experts=4, hidden_size=64, moe_intermediate_size=32, top_k=2):
        layer = RealMoELayer(
            num_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=moe_intermediate_size,
            top_k=top_k,
        )
        gate = SimpleLinearGate(hidden_size, num_experts)
        method = backend.CutlassMoEMethod(None)
        method.moe_quant_type = "w16a16"
        return layer, gate, method

    @requires_cuda
    def test_apply_tp_moe_permute_real_ops(self, monkeypatch):
        """FD_USE_PHI_MOE_PERMUTE=True + w16a16: real moe_permute/moe_unpermute/
        count_tokens_per_expert_func/moe_expert_ffn all called end-to-end."""
        monkeypatch.setattr(backend.fastdeploy.envs, "FD_USE_PHI_MOE_PERMUTE", True)

        num_tokens, hidden_size = 8, 64
        layer, gate, method = self._build(hidden_size=hidden_size)

        paddle.seed(42)
        x = paddle.randn([num_tokens, hidden_size], dtype="bfloat16")

        # Spy: confirm moe_permute is called, moe_expert_dispatch is NOT
        permute_called = {"v": False}
        dispatch_called = {"v": False}
        original_permute = paddle.nn.functional.moe_permute

        def spy_permute(*args, **kwargs):
            permute_called["v"] = True
            return original_permute(*args, **kwargs)

        monkeypatch.setattr(paddle.nn.functional, "moe_permute", spy_permute)
        monkeypatch.setattr(
            backend,
            "moe_expert_dispatch",
            lambda *a, **kw: (_ for _ in ()).throw(AssertionError("moe_expert_dispatch must not be called")),
        )

        out = method.apply_tp(layer, x, gate)

        assert permute_called["v"], "moe_permute was not called"
        assert not dispatch_called["v"], "moe_expert_dispatch must not be called"
        assert list(out.shape) == [num_tokens, hidden_size], f"wrong output shape: {out.shape}"
        assert not paddle.isnan(out).any(), "output contains NaN"
        assert not paddle.isinf(out).any(), "output contains Inf"

    @requires_cuda
    def test_apply_ep_prefill_moe_permute_real_ops(self, monkeypatch):
        """FD_USE_PHI_MOE_PERMUTE=True + w16a16: EP prefill uses real moe_permute /
        moe_unpermute / count_tokens_per_expert_func / moe_expert_ffn end-to-end.
        The EP dispatch/combine are stubbed (no real NCCL needed).
        Use num_tokens=128 and num_experts=4 so each expert gets exactly 64 tokens
        (128 * top_k=2 / 4 experts = 64), satisfying moe_expert_ffn alignment."""
        monkeypatch.setattr(backend.fastdeploy.envs, "FD_USE_PHI_MOE_PERMUTE", True)

        # 128 tokens, top_k=2, 4 experts → 64 tokens/expert (128-aligned after padding)
        num_tokens, hidden_size = 128, 64
        layer, gate, method = self._build(num_experts=4, hidden_size=hidden_size, top_k=2)

        paddle.seed(42)
        x = paddle.randn([num_tokens, hidden_size], dtype="bfloat16")

        # Stub only the EP communication runner (dispatch/combine).
        # All on-device compute (moe_permute, moe_expert_ffn, moe_unpermute) runs for real.
        class StubEPRunner:
            ep_engine = types.SimpleNamespace(async_finish=False)

            def moe_select(self, _layer, gate_out):
                n = gate_out.shape[0]
                # Route token i to experts (i % E) and ((i+1) % E) so all experts
                # get tokens and recv_num_tokens_per_expert_list is accurate.
                E = _layer.num_local_experts
                idx0 = paddle.arange(n, dtype="int64") % E
                idx1 = (paddle.arange(n, dtype="int64") + 1) % E
                topk_ids = paddle.stack([idx0, idx1], axis=1)
                topk_weights = paddle.ones([n, _layer.top_k], dtype="float32") / _layer.top_k
                return topk_ids, topk_weights

            def dispatch(self, x, topk_idx, topk_weights, **kwargs):
                # Pass tensors through unchanged — single-rank, no real communication.
                # Compute accurate recv_num_tokens_per_expert_list from topk_idx.
                E = layer.num_local_experts
                counts = [
                    align(int((topk_idx == e).sum().item()), kwargs.get("expert_alignment", 1)) for e in range(E)
                ]
                return (
                    x,
                    topk_idx,
                    topk_weights,
                    counts,
                    object(),
                    types.SimpleNamespace(current_stream_wait=lambda: None),
                )

            def combine(self, ffn_out, handle, recv_topk_weights):
                return ffn_out, types.SimpleNamespace(current_stream_wait=lambda: None)

        method.ep_prefill_runner = StubEPRunner()

        # Spy: confirm moe_permute is called inside ep_prefill
        permute_called = {"v": False}
        original_permute = paddle.nn.functional.moe_permute

        def spy_permute(*args, **kwargs):
            permute_called["v"] = True
            return original_permute(*args, **kwargs)

        monkeypatch.setattr(paddle.nn.functional, "moe_permute", spy_permute)

        out = method.apply_ep_prefill(layer, x, gate)

        assert permute_called["v"], "moe_permute was not called in ep_prefill path"
        assert len(out.shape) == 2, f"wrong output ndim: {out.shape}"
        assert out.shape[1] == hidden_size, f"wrong hidden_size: {out.shape}"
        assert not paddle.isnan(out).any(), "output contains NaN"
        assert not paddle.isinf(out).any(), "output contains Inf"

    @requires_cuda
    def test_apply_tp_moe_permute_non_noaux_tc(self, monkeypatch):
        """FD_USE_PHI_MOE_PERMUTE=True + w16a16 + topk_method != 'noaux_tc':
        the else-branch calls moe_topk_select instead of get_moe_scores,
        then proceeds through moe_permute / moe_expert_ffn / moe_unpermute."""
        monkeypatch.setattr(backend.fastdeploy.envs, "FD_USE_PHI_MOE_PERMUTE", True)

        num_tokens, hidden_size = 8, 64
        layer, gate, method = self._build(hidden_size=hidden_size)
        # Switch to non-noaux_tc to exercise the else-branch (moe_topk_select)
        layer.topk_method = "greedy"

        paddle.seed(7)
        x = paddle.randn([num_tokens, hidden_size], dtype="bfloat16")

        # Spy on which routing function is invoked
        get_moe_scores_called = {"v": False}
        moe_topk_select_called = {"v": False}
        permute_called = {"v": False}

        original_get_moe_scores = backend.get_moe_scores
        original_moe_topk_select = fastdeploy.model_executor.ops.gpu.moe_topk_select
        original_permute = paddle.nn.functional.moe_permute

        def spy_get_moe_scores(*args, **kwargs):
            get_moe_scores_called["v"] = True
            return original_get_moe_scores(*args, **kwargs)

        def spy_moe_topk_select(*args, **kwargs):
            moe_topk_select_called["v"] = True
            return original_moe_topk_select(*args, **kwargs)

        def spy_permute(*args, **kwargs):
            permute_called["v"] = True
            return original_permute(*args, **kwargs)

        monkeypatch.setattr(backend, "get_moe_scores", spy_get_moe_scores)
        monkeypatch.setattr(fastdeploy.model_executor.ops.gpu, "moe_topk_select", spy_moe_topk_select)
        monkeypatch.setattr(paddle.nn.functional, "moe_permute", spy_permute)

        out = method.apply_tp(layer, x, gate)

        assert not get_moe_scores_called["v"], "get_moe_scores must NOT be called for non-noaux_tc"
        assert moe_topk_select_called["v"], "moe_topk_select must be called for non-noaux_tc"
        assert permute_called["v"], "moe_permute must be called"
        assert list(out.shape) == [num_tokens, hidden_size], f"wrong shape: {out.shape}"
        assert not paddle.isnan(out).any(), "output contains NaN"
        assert not paddle.isinf(out).any(), "output contains Inf"

    def test_apply_tp_with_both_latent_projs(self, monkeypatch):
        """Test apply_tp with both fc1_latent_proj and fc2_latent_proj applied."""
        fc1_called = {"count": 0}
        fc2_called = {"count": 0}

        class FC1Proj(paddle.nn.Layer):
            def forward(self, x):
                fc1_called["count"] += 1
                return x * 2

        class FC2Proj(paddle.nn.Layer):
            def forward(self, x):
                fc2_called["count"] += 1
                return x + 10

        fc1_latent_proj = FC1Proj()
        fc2_latent_proj = FC2Proj()

        def fake_get_moe_scores(
            gate_out, n_group, topk_group, top_k, routed_scaling_factor, bias, renormalize, topk_reduce_func=None
        ):
            return gate_out, paddle.to_tensor([[0.6, 0.4]]), paddle.to_tensor([[0, 1]])

        def fake_dispatch(*args, **kwargs):
            permute_input = paddle.ones([1, 2]) * 2  # fc1_latent_proj applied
            token_nums_per_expert = paddle.to_tensor([1, 0])
            permute_indices_per_token = paddle.to_tensor([0])
            topk_weights = paddle.to_tensor([[0.6, 0.4]])
            topk_idx = paddle.to_tensor([[0, 1]])
            expert_idx_per_token = paddle.to_tensor([0])
            dequant_scale = None
            max_tokens_per_expert = None
            return (
                permute_input,
                token_nums_per_expert,
                permute_indices_per_token,
                topk_weights,
                topk_idx,
                expert_idx_per_token,
                dequant_scale,
                max_tokens_per_expert,
            )

        def fake_reduce(*args, **kwargs):
            return paddle.ones([1, 2]) * 5

        monkeypatch.setattr(backend, "get_moe_scores", fake_get_moe_scores, raising=False)
        monkeypatch.setattr(backend, "moe_expert_dispatch", fake_dispatch, raising=False)
        monkeypatch.setattr(backend, "moe_expert_reduce", fake_reduce, raising=False)

        layer = DummyLayer(topk_method="noaux_tc")
        method = backend.CutlassMoEMethod(None)
        monkeypatch.setattr(method, "compute_ffn", lambda *args, **kwargs: paddle.ones([1, 2]) * 4)

        x = paddle.ones([1, 2])
        gate = paddle.nn.Identity()
        out = method.apply_tp(layer, x, gate, fc1_latent_proj=fc1_latent_proj, fc2_latent_proj=fc2_latent_proj)

        # Output should be 5 (from reduce) + 10 (from fc2_latent_proj) = 15
        np.testing.assert_allclose(out.numpy(), np.full((1, 2), 15.0))
        assert fc1_called["count"] == 1, "fc1_latent_proj should be called exactly once"
        assert fc2_called["count"] == 1, "fc2_latent_proj should be called exactly once"
