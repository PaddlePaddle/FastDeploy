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

import importlib
import sys
import types

import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda scope=None: None)
if not hasattr(paddle.nn.functional, "swiglu"):
    paddle.nn.functional.swiglu = lambda x: x

from fastdeploy.model_executor.layers.moe import fused_moe_triton_backend as backend


class DummyQuantConfig:
    def __init__(self, is_checkpoint_bf16=False, weight_block_size=(2, 2), name_value="wint8"):
        self.is_checkpoint_bf16 = is_checkpoint_bf16
        self.weight_block_size = weight_block_size
        self._name_value = name_value
        self.deepgemm_scale_ue8m0 = False

    def name(self):
        return self._name_value


class DummyQuantMethod:
    def __init__(self, quant_config):
        self.quant_config = quant_config


class DummyLoadConfig:
    def __init__(self, load_choices="default_v1"):
        self.load_choices = load_choices
        self.dynamic_load_weight = False


class DummyFDConfig:
    def __init__(self, load_choices="default_v1"):
        self.load_config = DummyLoadConfig(load_choices)
        self.model_config = types.SimpleNamespace(enable_cache=False)


class DummyGate(paddle.nn.Layer):
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, x):
        return paddle.ones([x.shape[0], self.num_experts], dtype="float32")


class DummyLayer(paddle.nn.Layer):
    def __init__(
        self,
        quant_config,
        num_local_experts=2,
        hidden_size=4,
        moe_intermediate_size=3,
        top_k=2,
        load_choices="default_v1",
        weight_dtype="float16",
    ):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k = top_k
        self.n_group = 1
        self.topk_group = 1
        self.routed_scaling_factor = 1.0
        self.renormalize = True
        self.gate_correction_bias = paddle.zeros([num_local_experts], dtype="float32")
        self.topk_method = "noaux_tc"
        self.fd_config = DummyFDConfig(load_choices)
        self.weight_dtype = weight_dtype
        self.quant_method = DummyQuantMethod(quant_config)
        self.weight_key_map = {
            "up_gate_proj_expert_weight_scale_key": "up_gate_scale_{}",
            "down_proj_expert_weight_scale_key": "down_proj_scale_{}",
            "up_gate_proj_expert_in_scale_key": "up_gate_in_scale_{}",
            "down_proj_expert_in_scale_key": "down_proj_in_scale_{}",
        }
        self._up_weights = None
        self._down_weights = None

    def extract_moe_ffn_weights(self, state_dict):
        return self._up_weights, self._down_weights, None, None


class DummyKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _runner(*args, **kwargs):
            if len(args) > 2 and isinstance(args[2], paddle.Tensor):
                args[2].set_value(paddle.zeros_like(args[2]))
            self.calls.append({"grid": grid, "args": args, "kwargs": kwargs})

        return _runner


@pytest.fixture(autouse=True)
def patch_float8(monkeypatch):
    monkeypatch.setattr(paddle, "float8_e4m3fn", paddle.float16, raising=False)
    return monkeypatch


@pytest.fixture()
def fake_ops(monkeypatch):
    def fake_moe_topk_select(gate_out, gate_correction_bias, top_k, apply_norm_weight, use_softmax):
        token_num = gate_out.shape[0]
        topk_ids = paddle.zeros([token_num, top_k], dtype="int64")
        topk_weights = paddle.ones([token_num, top_k], dtype="float32")
        return topk_ids, topk_weights

    def fake_get_moe_scores(*args, **kwargs):
        gate_out = args[0]
        token_num = gate_out.shape[0]
        top_k = args[3]
        topk_ids = paddle.zeros([token_num, top_k], dtype="int64")
        topk_weights = paddle.ones([token_num, top_k], dtype="float32")
        return gate_out, topk_weights, topk_ids

    def fake_triton_preprocess(topk_ids, num_local_experts, block_size):
        token_num = topk_ids.shape[0]
        top_k = topk_ids.shape[1]
        sorted_token_ids = paddle.arange(token_num * top_k, dtype="int32")
        expert_ids = paddle.zeros_like(sorted_token_ids)
        num_tokens_post_padded = paddle.to_tensor([token_num * top_k], dtype="int32")
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    def fake_scaled_fp8_quant(x, use_per_token_if_dynamic=True):
        x_scale = paddle.ones([x.shape[0], 1], dtype="float32")
        return x, x_scale

    def fake_hadamard_quant_fp8(x, scale, topk_ids, top_k, intermediate_size, tiled):
        return x

    def fake_fp8_quant_blockwise(x, using_pow2_scale=False, output_scale_transpose=False):
        scale = paddle.ones([x.shape[0], x.shape[1]], dtype="float32")
        return x, scale

    monkeypatch.setattr(
        backend.fastdeploy.model_executor.ops.gpu,
        "moe_topk_select",
        fake_moe_topk_select,
        raising=False,
    )
    monkeypatch.setattr(backend, "get_moe_scores", fake_get_moe_scores)
    monkeypatch.setattr(backend, "tritonmoe_preprocess_func", fake_triton_preprocess, raising=False)
    monkeypatch.setattr(
        backend.fastdeploy.model_executor.ops.gpu,
        "tritonmoe_preprocess_func",
        fake_triton_preprocess,
        raising=False,
    )
    monkeypatch.setattr(backend, "scaled_fp8_quant", fake_scaled_fp8_quant)
    monkeypatch.setattr(
        backend.fastdeploy.model_executor.ops.gpu,
        "moe_fused_hadamard_quant_fp8",
        fake_hadamard_quant_fp8,
        raising=False,
    )
    monkeypatch.setattr(paddle.incubate.nn.functional, "fp8_quant_blockwise", fake_fp8_quant_blockwise, raising=False)
    monkeypatch.setattr(paddle.incubate.nn.functional, "swiglu", lambda x: x, raising=False)
    return monkeypatch


def _make_block_scale(weight_tensor, block_size):
    return paddle.ones(
        [
            (weight_tensor.shape[0] + block_size[0] - 1) // block_size[0],
            (weight_tensor.shape[1] + block_size[1] - 1) // block_size[1],
        ],
        dtype="float32",
    )


class TestFusedMoeTritonBackend:
    def test_backend_imports_kernel_module(self, monkeypatch):
        kernel = DummyKernel()
        monkeypatch.setattr(
            backend.fastdeploy.model_executor.ops.gpu,
            "tritonmoe_preprocess_func",
            lambda *args, **kwargs: None,
            raising=False,
        )
        monkeypatch.setitem(
            sys.modules,
            "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel),
        )
        reloaded = importlib.reload(backend)
        assert hasattr(reloaded, "fused_moe_kernel_paddle")

    def test_triton_weight_only_create_and_apply(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        method = backend.TritonWeightOnlyMoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")

        layer._up_weights = [
            paddle.arange(layer.hidden_size * layer.moe_intermediate_size * 2, dtype="float32").reshape(
                [layer.hidden_size, layer.moe_intermediate_size * 2]
            )
            for _ in range(layer.num_local_experts)
        ]
        layer._down_weights = [
            paddle.arange(layer.moe_intermediate_size * layer.hidden_size, dtype="float32").reshape(
                [layer.moe_intermediate_size, layer.hidden_size]
            )
            for _ in range(layer.num_local_experts)
        ]
        method.process_loaded_weights(layer, state_dict={})

        assert paddle.any(layer.up_gate_proj_weight_scale > 0)

        kernel = DummyKernel()
        monkeypatch.setattr(backend, "fused_moe_kernel_paddle", kernel, raising=False)

        x = paddle.randn([2, layer.hidden_size], dtype="float32")
        gate = DummyGate(layer.num_local_experts)
        captured = {}

        def hook(topk_ids):
            captured["topk_ids"] = topk_ids

        _ = method.apply(layer, x, gate, topk_ids_hookfunc=hook)
        assert "topk_ids" in captured

        empty_out = method.apply(layer, paddle.zeros([0, layer.hidden_size], dtype="float32"), gate)
        assert empty_out.shape == [0, layer.hidden_size]

    def test_triton_weight_only_prequant_and_bf16_create(self, fake_ops):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=True)
        layer = DummyLayer(quant_config, weight_dtype="float32")
        method = backend.TritonWeightOnlyMoEMethod(quant_config)
        assert method.process_prequanted_weights(layer, state_dict={}) is None

        method.create_weights(layer, model_format="not_torch")
        assert list(layer.up_gate_proj_weight.shape) == [
            layer.num_local_experts,
            layer.hidden_size,
            layer.moe_intermediate_size * 2,
        ]

    def test_triton_weight_only_process_weights_after_loading_bf16(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=True)
        layer = DummyLayer(quant_config, weight_dtype="float32")
        method = backend.TritonWeightOnlyMoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")
        method.model_format = "torch"

        monkeypatch.setattr(backend, "weight_fully_copied", lambda tensor: True)
        transpose_calls = []
        monkeypatch.setattr(backend, "process_weight_transpose", lambda _layer, name: transpose_calls.append(name))
        monkeypatch.setattr(backend, "free_tensor", lambda tensor: None)

        method.process_weights_after_loading(layer)

        assert transpose_calls

    def test_triton_weight_only_process_weights_after_loading_return(self, fake_ops):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        method = backend.TritonWeightOnlyMoEMethod(quant_config)
        assert method.process_weights_after_loading(layer) is None

    def test_triton_weight_only_apply_aux_topk(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        layer.topk_method = "aux"
        method = backend.TritonWeightOnlyMoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")

        kernel = DummyKernel()
        monkeypatch.setattr(backend, "fused_moe_kernel_paddle", kernel, raising=False)

        called = {}

        def hook(topk_ids):
            called["ids"] = topk_ids

        _ = method.apply(
            layer,
            paddle.randn([1, layer.hidden_size], dtype="float32"),
            DummyGate(layer.num_local_experts),
            hook,
        )
        assert "ids" in called

    def test_wfp8afp8_method_apply_paths(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        layer.topk_method = "aux"
        method = backend.Wfp8Afp8MoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")

        kernel = DummyKernel()
        monkeypatch.setitem(
            sys.modules,
            "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel),
        )
        monkeypatch.setattr(backend, "fused_moe_kernel_paddle", kernel, raising=False)

        x = paddle.randn([1, layer.hidden_size], dtype="float32")
        gate = DummyGate(layer.num_local_experts)
        captured = {}

        def hook(topk_ids):
            captured["ids"] = topk_ids

        _ = method.apply(layer, x, gate, topk_ids_hookfunc=hook)
        assert "ids" in captured

        up_gate = [
            paddle.zeros([layer.moe_intermediate_size * 2, layer.hidden_size], dtype="float32")
            for _ in range(layer.num_local_experts)
        ]
        down_proj = [
            paddle.zeros([layer.hidden_size, layer.moe_intermediate_size], dtype="float32")
            for _ in range(layer.num_local_experts)
        ]
        method.check(layer, up_gate, down_proj)

    def test_wfp8afp8_prequant_raises(self, fake_ops):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        method = backend.Wfp8Afp8MoEMethod(quant_config)
        with pytest.raises(NotImplementedError):
            method.process_prequanted_weights(layer, state_dict={})

    def test_wfp8afp8_create_weights_bf16_branch(self, fake_ops):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=True)
        layer = DummyLayer(quant_config, weight_dtype="float32")
        method = backend.Wfp8Afp8MoEMethod(quant_config)
        method.create_weights(layer, model_format="not_torch")
        assert list(layer.down_proj_weight.shape) == [
            layer.num_local_experts,
            layer.moe_intermediate_size,
            layer.hidden_size,
        ]

    def test_wfp8afp8_process_weights_after_loading_bf16(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=True)
        layer = DummyLayer(quant_config, weight_dtype="float32")
        method = backend.Wfp8Afp8MoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")
        method.model_format = "torch"

        monkeypatch.setattr(backend, "weight_fully_copied", lambda tensor: False)
        transpose_calls = []
        monkeypatch.setattr(backend, "process_weight_transpose", lambda _layer, name: transpose_calls.append(name))
        monkeypatch.setattr(backend, "free_tensor", lambda tensor: None)

        def fake_per_token_cast_to_fp8(weight):
            return weight.cast(paddle.float16), paddle.ones([weight.shape[1], 1], dtype="float32")

        monkeypatch.setattr(
            backend.fastdeploy.model_executor.layers.utils, "per_token_cast_to_fp8", fake_per_token_cast_to_fp8
        )

        method.process_weights_after_loading(layer)
        assert transpose_calls

    def test_wfp8afp8_apply_noaux_and_empty(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        method = backend.Wfp8Afp8MoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")

        kernel = DummyKernel()
        monkeypatch.setitem(
            sys.modules,
            "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel),
        )

        _ = method.apply(
            layer, paddle.randn([1, layer.hidden_size], dtype="float32"), DummyGate(layer.num_local_experts)
        )
        empty_out = method.apply(
            layer, paddle.zeros([0, layer.hidden_size], dtype="float32"), DummyGate(layer.num_local_experts)
        )
        assert empty_out.shape == [0, layer.hidden_size]

    def test_tensorwise_prequant_and_apply(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        method = backend.TensorWiseFP8MoEMethod(quant_method=DummyQuantMethod(quant_config))
        method.create_weights(layer)

        monkeypatch.setattr(backend, "get_tensor", lambda tensor: tensor)

        state_dict = {}
        up_weight = paddle.ones([layer.hidden_size, layer.moe_intermediate_size * 2], dtype="float32")
        down_weight = paddle.ones([layer.moe_intermediate_size, layer.hidden_size], dtype="float32")
        layer._up_weights = [up_weight for _ in range(layer.num_local_experts)]
        layer._down_weights = [down_weight for _ in range(layer.num_local_experts)]
        monkeypatch.setattr(layer, "extract_moe_ffn_weights", lambda _state: (layer._up_weights, layer._down_weights))

        for idx in range(layer.num_local_experts):
            state_dict[f"up_gate_scale_{idx}"] = paddle.ones([1], dtype="float32") * (idx + 1)
            state_dict[f"down_proj_scale_{idx}"] = paddle.ones([1], dtype="float32") * (idx + 2)
            state_dict[f"up_gate_in_scale_{idx}"] = paddle.ones([1], dtype="float32") * (idx + 3)
            state_dict[f"down_proj_in_scale_{idx}"] = paddle.ones([1], dtype="float32") * (idx + 4)

        method.process_prequanted_weights(layer, state_dict)

        assert paddle.all(layer.up_gate_proj_in_scale > 0)

        kernel = DummyKernel()
        monkeypatch.setitem(
            sys.modules,
            "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel),
        )
        monkeypatch.setattr(backend, "fused_moe_kernel_paddle", kernel, raising=False)

        layer.topk_method = "aux"
        x = paddle.randn([2, layer.hidden_size], dtype="float32")
        gate = DummyGate(layer.num_local_experts)
        called = {}

        def hook(topk_ids):
            called["hooked"] = topk_ids

        _ = method.apply(layer, x, gate, topk_ids_hookfunc=hook)
        assert "hooked" in called

    def test_python_op_fused_moe_kernel_paddle(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False, weight_block_size=(2, 2))
        layer = DummyLayer(quant_config)

        kernel = DummyKernel()
        monkeypatch.setitem(
            sys.modules,
            "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel),
        )
        monkeypatch.setattr(
            paddle.static,
            "MetaTensor",
            lambda shape, dtype: types.SimpleNamespace(shape=shape, dtype=dtype),
            raising=False,
        )

        x = paddle.randn([2, layer.hidden_size], dtype="float32")
        gate = DummyGate(layer.num_local_experts)
        gate_out = gate(x)

        layer_added_weight_attrs_0 = paddle.randn(
            [layer.num_local_experts, layer.moe_intermediate_size * 2, layer.hidden_size], dtype="float32"
        )
        layer_added_weight_attrs1 = paddle.randn(
            [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size], dtype="float32"
        )
        layer_added_scale_attrs_0 = paddle.ones([layer.num_local_experts, 2, 2], dtype="float32")
        layer_added_scale_attrs1 = paddle.ones([layer.num_local_experts, 2, 2], dtype="float32")

        captured = {}

        def hook(topk_ids):
            captured["topk"] = topk_ids

        config = {
            "BLOCK_SIZE_M": 32,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }

        _ = backend.python_op_fused_moe_kernel_paddle(
            x,
            layer_added_weight_attrs_0,
            layer_added_scale_attrs_0,
            layer_added_weight_attrs1,
            layer_added_scale_attrs1,
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            layer_added_weight_attrs_0.shape[1],
            layer_added_weight_attrs1.shape[1],
            layer.num_local_experts,
            layer.moe_intermediate_size,
            layer.hidden_size,
            config,
            quant_config,
            hook,
        )

        assert "topk" in captured

        meta = backend.python_op_fused_moe_kernel_paddle_infer_meta(
            x,
            layer_added_weight_attrs_0,
            layer_added_scale_attrs_0,
            layer_added_weight_attrs1,
            layer_added_scale_attrs1,
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            layer_added_weight_attrs_0.shape[1],
            layer_added_weight_attrs1.shape[1],
            layer.num_local_experts,
            layer.moe_intermediate_size,
            layer.hidden_size,
            config,
            quant_config,
            None,
        )

        assert meta.shape == [2, layer.hidden_size]

    def test_blockwise_create_weights_and_process(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False, weight_block_size=(2, 2))
        layer = DummyLayer(quant_config)
        method = backend.BlockWiseFP8MoEMethod(quant_config)
        method.create_weights(layer, model_format="not_torch")

        transpose_calls = []
        monkeypatch.setattr(backend, "process_weight_transpose", lambda _layer, name: transpose_calls.append(name))

        method.process_weights_after_loading(layer)
        assert transpose_calls

        up_weights = [
            paddle.arange(layer.hidden_size * layer.moe_intermediate_size * 2, dtype="float32").reshape(
                [layer.hidden_size, layer.moe_intermediate_size * 2]
            )
            for _ in range(layer.num_local_experts)
        ]
        down_weights = [
            paddle.arange(layer.moe_intermediate_size * layer.hidden_size, dtype="float32").reshape(
                [layer.moe_intermediate_size, layer.hidden_size]
            )
            for _ in range(layer.num_local_experts)
        ]
        layer._up_weights = up_weights
        layer._down_weights = down_weights

        def fake_per_block_cast_to_fp8(weight, block_size):
            return weight.cast(paddle.float16), _make_block_scale(weight.transpose([1, 0]), block_size)

        monkeypatch.setattr(
            backend.fastdeploy.model_executor.layers.utils, "per_block_cast_to_fp8", fake_per_block_cast_to_fp8
        )

        method.process_loaded_weights(layer, state_dict={})

        assert paddle.any(layer.up_gate_proj_weight_scale_inv > 0)

    def test_blockwise_process_weights_after_loading_bf16(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=True, weight_block_size=(2, 2))
        layer = DummyLayer(quant_config)
        method = backend.BlockWiseFP8MoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")
        method.model_format = "torch"

        monkeypatch.setattr(backend, "weight_fully_copied", lambda tensor: False)

        def fake_per_block_cast_to_fp8(weight, block_size):
            return weight.cast(paddle.float16), _make_block_scale(weight, block_size)

        monkeypatch.setattr(
            backend.fastdeploy.model_executor.layers.utils, "per_block_cast_to_fp8", fake_per_block_cast_to_fp8
        )
        monkeypatch.setattr(backend, "free_tensor", lambda tensor: None)

        method.process_weights_after_loading(layer)

        if not hasattr(layer, "up_gate_proj_weight_scale_inv"):
            layer.up_gate_proj_weight_scale_inv = layer.create_parameter(
                shape=method.up_gate_proj_scale_shape,
                dtype="float32",
                default_initializer=paddle.nn.initializer.Constant(0),
            )

        def fake_python_op(*args, **kwargs):
            token_num = args[0].shape[0]
            hidden_size = args[12]
            return paddle.zeros([token_num, hidden_size], dtype=args[0].dtype)

        monkeypatch.setattr(backend, "python_op_fused_moe_kernel_paddle", fake_python_op)

        x = paddle.randn([2, layer.hidden_size], dtype="float32")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply(layer, x, gate)
        assert out.shape == [2, layer.hidden_size]

    def test_blockwise_check_and_apply_empty(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False, weight_block_size=(2, 2))
        layer = DummyLayer(quant_config)
        method = backend.BlockWiseFP8MoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")

        up_gate = [
            paddle.zeros([layer.hidden_size, layer.moe_intermediate_size * 2], dtype="float32")
            for _ in range(layer.num_local_experts)
        ]
        down_proj = [
            paddle.zeros([layer.moe_intermediate_size, layer.hidden_size], dtype="float32")
            for _ in range(layer.num_local_experts)
        ]
        method.check(layer, up_gate, down_proj)

        def fake_python_op(*args, **kwargs):
            token_num = args[0].shape[0]
            hidden_size = args[12]
            return paddle.zeros([token_num, hidden_size], dtype=args[0].dtype)

        monkeypatch.setattr(backend, "python_op_fused_moe_kernel_paddle", fake_python_op)

        out = method.apply(
            layer, paddle.zeros([0, layer.hidden_size], dtype="float32"), DummyGate(layer.num_local_experts)
        )
        assert out.shape == [0, layer.hidden_size]
