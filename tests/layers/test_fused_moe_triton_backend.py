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

    def test_blockwise_process_weights_ue8m0_branch(self, fake_ops, monkeypatch):
        """Test the quant_weight_ue8m0 branch in BlockWiseFP8MoEMethod.process_weights_after_loading."""
        quant_config = DummyQuantConfig(is_checkpoint_bf16=True, weight_block_size=(128, 128))
        quant_config.deepgemm_scale_ue8m0 = True
        layer = DummyLayer(quant_config, weight_dtype="bfloat16")
        method = backend.BlockWiseFP8MoEMethod(quant_config)
        method.create_weights(layer, model_format="torch")
        method.model_format = "torch"

        # Set FD_USE_PHI_FP8_QUANT to False to enter the target branch
        monkeypatch.setattr(backend.fastdeploy.envs, "FD_USE_PHI_FP8_QUANT", False)
        monkeypatch.setattr(backend, "weight_fully_copied", lambda tensor: True)

        # Mock quant_weight_ue8m0 and transform_scale_ue8m0
        quant_calls = []
        transform_calls = []

        def fake_quant_weight_ue8m0(weight_dequant, weight_block_size):
            quant_calls.append({"weight_shape": weight_dequant.shape, "block_size": weight_block_size})
            # Return fake quantized weight and scale
            n, k = weight_dequant.shape[-2], weight_dequant.shape[-1]
            out_w = paddle.zeros(weight_dequant.shape, dtype=paddle.float8_e4m3fn)
            out_s = paddle.ones([n, (k + 127) // 128], dtype="float32")
            return out_w, out_s

        def fake_transform_scale_ue8m0(sf, mn, weight_block_size=None):
            transform_calls.append({"sf_shape": sf.shape, "mn": mn, "block_size": weight_block_size})
            # Return fake transformed scale
            return paddle.ones([sf.shape[0], sf.shape[1], 1], dtype="uint8")

        monkeypatch.setattr(backend, "quant_weight_ue8m0", fake_quant_weight_ue8m0)
        monkeypatch.setattr(backend, "transform_scale_ue8m0", fake_transform_scale_ue8m0)
        monkeypatch.setattr(backend, "free_tensor", lambda tensor: None)
        monkeypatch.setattr(backend, "process_weight_transpose", lambda _layer, name: None)

        # Create unquantized weights for the layer
        num_experts = layer.num_local_experts
        hidden_size = layer.hidden_size
        moe_intermediate_size = layer.moe_intermediate_size

        # Create weight attributes that the method expects
        layer.up_gate_proj_weight = layer.create_parameter(
            shape=[num_experts, moe_intermediate_size * 2, hidden_size],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        layer.down_proj_weight = layer.create_parameter(
            shape=[num_experts, hidden_size, moe_intermediate_size],
            dtype="bfloat16",
            default_initializer=paddle.nn.initializer.Constant(0),
        )

        method.process_weights_after_loading(layer)

        # Verify the quant_weight_ue8m0 branch was executed
        assert len(quant_calls) > 0, "quant_weight_ue8m0 should have been called"
        assert len(transform_calls) > 0, "transform_scale_ue8m0 should have been called"


class DummyBF16Kernel:
    """
    Simulates fused_moe_kernel_bf16[grid](...).
    Writes zeros into the output tensor (3rd positional argument).
    """

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def _runner(*args, **kwargs):
            # output tensor is the 3rd positional argument (index 2)
            if len(args) > 2 and isinstance(args[2], paddle.Tensor):
                args[2].set_value(paddle.zeros_like(args[2]))
            self.calls.append({"grid": grid, "kwargs": kwargs})

        return _runner


class TestTritonBF16MoEMethod:
    """Unit tests for TritonBF16MoEMethod.

    Pattern mirrors TestFusedMoeTritonBackend:
    - DummyLayer / DummyGate / DummyFDConfig (reused from module top)
    - fake_ops fixture patches routing + preprocess ops
    - DummyBF16Kernel patches fused_moe_kernel_bf16
    - No real GPU kernels are executed; output shapes / attributes are verified
    """

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_layer(self, num_experts=2, hidden_size=8, intermediate_size=4, top_k=2):
        layer = DummyLayer(
            quant_config=None,
            num_local_experts=num_experts,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            top_k=top_k,
            weight_dtype="bfloat16",
        )
        return layer

    def _patch_bf16_kernel(self, monkeypatch):
        kernel = DummyBF16Kernel()
        monkeypatch.setattr(backend, "fused_moe_kernel_bf16", kernel, raising=False)
        return kernel

    # ------------------------------------------------------------------
    # __init__ / basic construction
    # ------------------------------------------------------------------

    def test_init_sets_weight_attrs(self):
        """TritonBF16MoEMethod.__init__ must expose the two weight attr names."""
        method = backend.TritonBF16MoEMethod()
        assert "up_gate_proj_weight" in method.added_weight_attrs
        assert "down_proj_weight" in method.added_weight_attrs

    def test_init_none_quant_config(self):
        method = backend.TritonBF16MoEMethod(quant_config=None)
        assert method.quant_config is None

    # ------------------------------------------------------------------
    # create_weights
    # ------------------------------------------------------------------

    def test_create_weights_registers_parameters(self):
        """After create_weights the layer should have up_gate_proj_weight and down_proj_weight."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer()
        method.create_weights(layer, model_format="torch")
        assert hasattr(layer, "up_gate_proj_weight")
        assert hasattr(layer, "down_proj_weight")

    def test_create_weights_shapes(self):
        """Weight tensors must have the correct [E, K, N] / [E, N, K] layout."""
        E, H, N = 3, 8, 4
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(num_experts=E, hidden_size=H, intermediate_size=N)
        method.create_weights(layer, model_format="torch")
        # up_gate: [E, hidden_size, intermediate*2]
        assert list(layer.up_gate_proj_weight.shape) == [E, H, N * 2]
        # down: [E, intermediate, hidden_size]
        assert list(layer.down_proj_weight.shape) == [E, N, H]

    # ------------------------------------------------------------------
    # process_loaded_weights
    # ------------------------------------------------------------------

    def test_process_loaded_weights_stacks_experts(self):
        """process_loaded_weights must stack per-expert tensors into the stacked param."""
        E, H, N = 2, 8, 4
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(num_experts=E, hidden_size=H, intermediate_size=N)
        method.create_weights(layer, model_format="torch")

        # Provide per-expert tensors via extract_moe_ffn_weights
        up_weights = [
            paddle.ones([H, N * 2], dtype="bfloat16") * (i + 1)
            for i in range(E)
        ]
        down_weights = [
            paddle.ones([N, H], dtype="bfloat16") * (i + 1)
            for i in range(E)
        ]
        layer._up_weights = up_weights
        layer._down_weights = down_weights

        method.process_loaded_weights(layer, state_dict={})

        # After stacking, shape should be [E, ...]
        assert list(layer.up_gate_proj_weight.shape) == [E, H, N * 2]
        assert list(layer.down_proj_weight.shape) == [E, N, H]

    # ------------------------------------------------------------------
    # process_prequanted_weights
    # ------------------------------------------------------------------

    def test_process_prequanted_weights_is_noop(self):
        """process_prequanted_weights should return None (no-op for BF16)."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer()
        result = method.process_prequanted_weights(layer, state_dict={})
        assert result is None

    # ------------------------------------------------------------------
    # _get_default_config — tile heuristic
    # ------------------------------------------------------------------

    def test_get_default_config_decode(self):
        """M<=32 decode path → 16x64x64."""
        method = backend.TritonBF16MoEMethod()
        cfg = method._get_default_config(M=4, N=128, K=128)
        assert cfg["BLOCK_SIZE_M"] == 16
        assert cfg["BLOCK_SIZE_N"] == 64
        assert cfg["BLOCK_SIZE_K"] == 64

    def test_get_default_config_mid(self):
        """32 < M <= 512 mid path → 32x128x64."""
        method = backend.TritonBF16MoEMethod()
        cfg = method._get_default_config(M=128, N=256, K=128)
        assert cfg["BLOCK_SIZE_M"] == 32
        assert cfg["BLOCK_SIZE_N"] == 128
        assert cfg["BLOCK_SIZE_K"] == 64

    def test_get_default_config_prefill(self):
        """M > 512 prefill path → 128x128x64."""
        method = backend.TritonBF16MoEMethod()
        cfg = method._get_default_config(M=1024, N=256, K=128)
        assert cfg["BLOCK_SIZE_M"] == 128
        assert cfg["BLOCK_SIZE_N"] == 128
        assert cfg["BLOCK_SIZE_K"] == 64

    def test_get_default_config_boundary_32(self):
        """M==32 is decode (<=32)."""
        method = backend.TritonBF16MoEMethod()
        cfg = method._get_default_config(M=32, N=64, K=64)
        assert cfg["BLOCK_SIZE_M"] == 16

    def test_get_default_config_boundary_512(self):
        """M==512 is mid (<=512)."""
        method = backend.TritonBF16MoEMethod()
        cfg = method._get_default_config(M=512, N=64, K=64)
        assert cfg["BLOCK_SIZE_M"] == 32

    def test_get_default_config_has_group_size_m(self):
        """All configs must include GROUP_SIZE_M key."""
        method = backend.TritonBF16MoEMethod()
        for M in (1, 64, 1024):
            cfg = method._get_default_config(M=M, N=64, K=64)
            assert "GROUP_SIZE_M" in cfg

    # ------------------------------------------------------------------
    # apply — empty-batch fast path
    # ------------------------------------------------------------------

    def test_apply_empty_batch_returns_zero_tensor(self, fake_ops, monkeypatch):
        """apply() with 0 tokens must return a zero tensor of shape [0, hidden_size]."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(hidden_size=8)
        method.create_weights(layer, model_format="torch")
        self._patch_bf16_kernel(monkeypatch)

        x = paddle.zeros([0, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply(layer, x, gate)

        assert list(out.shape) == [0, layer.hidden_size]

    # ------------------------------------------------------------------
    # apply — normal forward (noaux_tc routing path)
    # ------------------------------------------------------------------

    def test_apply_noaux_tc_output_shape(self, fake_ops, monkeypatch):
        """apply() noaux_tc path: output shape must be [token_num, hidden_size]."""
        T, H = 4, 8
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(hidden_size=H)
        method.create_weights(layer, model_format="torch")
        self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([T, H], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply(layer, x, gate)

        assert list(out.shape) == [T, H]

    def test_apply_noaux_tc_topk_hook_called(self, fake_ops, monkeypatch):
        """topk_ids_hookfunc must be called with topk_ids kwarg during apply()."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(hidden_size=8)
        method.create_weights(layer, model_format="torch")
        self._patch_bf16_kernel(monkeypatch)

        captured = {}

        def hook(topk_ids):
            captured["topk_ids"] = topk_ids

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        method.apply(layer, x, DummyGate(layer.num_local_experts), topk_ids_hookfunc=hook)

        assert "topk_ids" in captured

    def test_apply_noaux_tc_kernel_called_twice(self, fake_ops, monkeypatch):
        """fused_moe_kernel_bf16 must be launched twice (GEMM1 + GEMM2) per forward pass."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(hidden_size=8)
        method.create_weights(layer, model_format="torch")
        kernel = self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        method.apply(layer, x, DummyGate(layer.num_local_experts))

        assert len(kernel.calls) == 2, (
            f"Expected 2 kernel launches (GEMM1 + GEMM2), got {len(kernel.calls)}"
        )

    # ------------------------------------------------------------------
    # apply — non-noaux routing path (moe_topk_select)
    # ------------------------------------------------------------------

    def test_apply_aux_routing_path(self, fake_ops, monkeypatch):
        """When topk_method != 'noaux_tc', the moe_topk_select path is used."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(hidden_size=8)
        layer.topk_method = "aux"
        method.create_weights(layer, model_format="torch")
        self._patch_bf16_kernel(monkeypatch)

        captured = {}

        def hook(topk_ids):
            captured["ids"] = topk_ids

        x = paddle.randn([3, layer.hidden_size], dtype="bfloat16")
        out = method.apply(layer, x, DummyGate(layer.num_local_experts), topk_ids_hookfunc=hook)

        assert list(out.shape) == [3, layer.hidden_size]
        assert "ids" in captured

    # ------------------------------------------------------------------
    # apply_tp delegates to apply
    # ------------------------------------------------------------------

    def test_apply_tp_delegates_to_apply(self, fake_ops, monkeypatch):
        """apply_tp() must produce the same output shape as apply()."""
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer(hidden_size=8)
        method.create_weights(layer, model_format="torch")
        self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply_tp(layer, x, gate)

        assert list(out.shape) == [2, layer.hidden_size]

    # ------------------------------------------------------------------
    # EP methods raise NotImplementedError
    # ------------------------------------------------------------------

    def test_apply_ep_prefill_raises(self):
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer()
        with pytest.raises(NotImplementedError):
            method.apply_ep_prefill(layer, None, None)

    def test_apply_ep_decode_raises(self):
        method = backend.TritonBF16MoEMethod()
        layer = self._make_layer()
        with pytest.raises(NotImplementedError):
            method.apply_ep_decode(layer, None, None)
