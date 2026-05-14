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

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "enable_compat"):
    paddle.enable_compat = lambda scope=None: None
if not hasattr(paddle.nn.functional, "swiglu"):
    paddle.nn.functional.swiglu = lambda x: x

from fastdeploy.model_executor.layers.moe import fused_moe_triton_backend as backend


class DummyQuantConfig:
    def __init__(self, is_checkpoint_bf16=False, weight_block_size=(2, 2), name_value="wint8"):
        self.is_checkpoint_bf16 = is_checkpoint_bf16
        self.weight_block_size = weight_block_size
        self._name_value = name_value
        self.deepgemm_scale_ue8m0 = False
        self.moe_blockwise_gemm_scale_ue8m0 = False

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
        self.scheduler_config = types.SimpleNamespace(
            enable_moe_scores_elementwise_fuse=False,
            splitwise_role="mixed",
            max_num_seqs=8,
            max_num_batched_tokens=256,
        )


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
        self.routed_scaling_factor_learnable = False
        self.renormalize = True
        self.gate_correction_bias = paddle.zeros([num_local_experts], dtype="float32")
        self.topk_method = "noaux_tc"
        self.with_bias = False
        self.ep_size = 1
        self.activation = "swiglu"
        self.moe_quant_config = types.SimpleNamespace()
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
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel, fused_moe_kernel_bf16=kernel),
        )
        reloaded = importlib.reload(backend)
        assert hasattr(reloaded, "fused_moe_kernel_paddle")
        # Restore the real module: reload() permanently rebinds module-level names
        # (e.g. fused_moe_kernel_bf16) to the fake, and monkeypatch cannot undo that.
        # A second reload after monkeypatch restores sys.modules fixes the binding.
        monkeypatch.undo()
        importlib.reload(backend)

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
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel, fused_moe_kernel_bf16=kernel),
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
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel, fused_moe_kernel_bf16=kernel),
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
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel, fused_moe_kernel_bf16=kernel),
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
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel, fused_moe_kernel_bf16=kernel),
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
        quant_config.moe_blockwise_gemm_scale_ue8m0 = True
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

    def test_triton_weight_only_apply_noaux_tc_with_fd_enable_rl(self, fake_ops, monkeypatch):
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False)
        layer = DummyLayer(quant_config)
        layer.topk_method = "noaux_tc"
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

        kernel = DummyKernel()
        monkeypatch.setattr(backend, "fused_moe_kernel_paddle", kernel, raising=False)

        # Set FD_ENABLE_RL=True to trigger use_fused = False at line 313
        # This should trigger gate_out.cast('float32') at line 315
        monkeypatch.setattr(backend.fastdeploy.envs, "FD_ENABLE_RL", True)

        x = paddle.randn([1, layer.hidden_size], dtype="float32")
        gate = DummyGate(layer.num_local_experts)

        captured = {}

        def hook(topk_ids):
            captured["topk_ids"] = topk_ids

        _ = method.apply(layer, x, gate, topk_ids_hookfunc=hook)
        assert "topk_ids" in captured

    def test_python_op_learnable_scaling(self, fake_ops, monkeypatch):
        """routed_scaling_factor_learnable=True: per_expert_scale applied to topk_weights inside python_op."""
        quant_config = DummyQuantConfig(is_checkpoint_bf16=False, weight_block_size=(2, 2))
        layer = DummyLayer(quant_config)
        layer.routed_scaling_factor_learnable = True
        layer.per_expert_scale = paddle.ones([layer.num_local_experts], dtype="float32")

        kernel = DummyKernel()
        monkeypatch.setitem(
            sys.modules,
            "fastdeploy.model_executor.layers.moe.triton_moe_kernels",
            types.SimpleNamespace(fused_moe_kernel_paddle=kernel, fused_moe_kernel_bf16=kernel),
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

        up_weight = paddle.randn(
            [layer.num_local_experts, layer.moe_intermediate_size * 2, layer.hidden_size], dtype="float32"
        )
        down_weight = paddle.randn(
            [layer.num_local_experts, layer.hidden_size, layer.moe_intermediate_size], dtype="float32"
        )
        up_scale = paddle.ones([layer.num_local_experts, 2, 2], dtype="float32")
        down_scale = paddle.ones([layer.num_local_experts, 2, 2], dtype="float32")

        captured = {}

        def hook(topk_ids):
            captured["topk"] = topk_ids

        config = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1}

        _ = backend.python_op_fused_moe_kernel_paddle(
            x,
            up_weight,
            up_scale,
            down_weight,
            down_scale,
            gate_out,
            layer.gate_correction_bias,
            layer.top_k,
            up_weight.shape[1],
            down_weight.shape[1],
            layer.num_local_experts,
            layer.moe_intermediate_size,
            layer.hidden_size,
            config,
            quant_config,
            hook,
        )

        assert "topk" in captured


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


class DummyTL:
    """Minimal stub for triton.language so tests don't need a real Triton install."""

    bfloat16 = "bfloat16"
    float16 = "float16"


class TestTritonMoEMethod:
    """Unit tests for TritonMoEMethod.

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

    def _create_weights(self, method, layer):
        """Call create_weights with the mandatory kwargs that the real MoE layer supplies.

        TritonMoEMethod targets the CUDA non-torch weight layout:
          up_gate_proj_weight: [E, hidden_size, inter*2]  (K-major)
          down_proj_weight:    [E, inter, hidden_size]    (K-major)
        Therefore we must NOT pass model_format="torch"; any non-"torch" value
        (or omitting the key) lets UnquantizedFusedMoEMethod take the CUDA branch.
        """
        method.create_weights(
            layer,
            model_format="default",
            num_experts=layer.num_local_experts,
            hidden_size=layer.hidden_size,
            moe_intermediate_size=layer.moe_intermediate_size,
        )

    def _patch_bf16_kernel(self, monkeypatch):
        kernel = DummyBF16Kernel()
        monkeypatch.setattr(backend, "fused_moe_kernel_bf16", kernel, raising=False)
        # Patch tl so that `compute_type=tl.bfloat16` inside apply() does not
        # raise NameError when triton is not installed in the test environment.
        monkeypatch.setattr(backend, "tl", DummyTL(), raising=False)
        return kernel

    # ------------------------------------------------------------------
    # __init__ / basic construction
    # ------------------------------------------------------------------

    def test_init_sets_weight_attrs(self):
        """TritonMoEMethod.__init__ must expose the two weight attr names."""
        method = backend.TritonMoEMethod()
        assert "up_gate_proj_weight" in method.added_weight_attrs
        assert "down_proj_weight" in method.added_weight_attrs

    def test_init_none_quant_config(self):
        method = backend.TritonMoEMethod(quant_config=None)
        assert method.quant_config is None

    # ------------------------------------------------------------------
    # create_weights
    # ------------------------------------------------------------------

    def test_create_weights_registers_parameters(self):
        """After create_weights the layer should have up_gate_proj_weight and down_proj_weight."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer()
        self._create_weights(method, layer)
        assert hasattr(layer, "up_gate_proj_weight")
        assert hasattr(layer, "down_proj_weight")

    def test_create_weights_shapes(self):
        """Weight tensors must have the correct [E, K, N] / [E, N, K] layout."""
        E, H, N = 3, 8, 4
        method = backend.TritonMoEMethod()
        layer = self._make_layer(num_experts=E, hidden_size=H, intermediate_size=N)
        self._create_weights(method, layer)
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
        method = backend.TritonMoEMethod()
        layer = self._make_layer(num_experts=E, hidden_size=H, intermediate_size=N)
        self._create_weights(method, layer)

        # Provide per-expert tensors via extract_moe_ffn_weights
        up_weights = [paddle.ones([H, N * 2], dtype="bfloat16") * (i + 1) for i in range(E)]
        down_weights = [paddle.ones([N, H], dtype="bfloat16") * (i + 1) for i in range(E)]
        layer._up_weights = up_weights
        layer._down_weights = down_weights

        method.process_loaded_weights(layer, state_dict={})

        # After stacking, shape should be [E, ...]
        assert list(layer.up_gate_proj_weight.shape) == [E, H, N * 2]
        assert list(layer.down_proj_weight.shape) == [E, N, H]
        # Verify each expert's data is correctly stacked (expert i has value i+1)
        for i in range(E):
            expected_up = float(i + 1)
            expected_down = float(i + 1)
            actual_up = float(layer.up_gate_proj_weight[i].cast("float32").mean())
            actual_down = float(layer.down_proj_weight[i].cast("float32").mean())
            assert (
                abs(actual_up - expected_up) < 1e-3
            ), f"Expert {i} up_gate weight mean={actual_up}, expected {expected_up}"
            assert (
                abs(actual_down - expected_down) < 1e-3
            ), f"Expert {i} down_proj weight mean={actual_down}, expected {expected_down}"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # _get_default_config — tile heuristic
    # ------------------------------------------------------------------

    def test_get_default_config_decode(self):
        """M<=32 decode path → 16x64x64."""
        method = backend.TritonMoEMethod()
        cfg = method._get_default_config(M=4, E=8)
        assert cfg["BLOCK_SIZE_M"] == 16
        assert cfg["BLOCK_SIZE_N"] == 64
        assert cfg["BLOCK_SIZE_K"] == 64

    def test_get_default_config_mid(self):
        """96 < M <= 512 mid path → 64x128x64."""
        method = backend.TritonMoEMethod()
        cfg = method._get_default_config(M=128, E=8)
        assert cfg["BLOCK_SIZE_M"] == 64
        assert cfg["BLOCK_SIZE_N"] == 128
        assert cfg["BLOCK_SIZE_K"] == 64

    def test_get_default_config_prefill(self):
        """M > 512 prefill path → 128x128x64."""
        method = backend.TritonMoEMethod()
        cfg = method._get_default_config(M=1024, E=8)
        assert cfg["BLOCK_SIZE_M"] == 128
        assert cfg["BLOCK_SIZE_N"] == 128
        assert cfg["BLOCK_SIZE_K"] == 64

    def test_get_default_config_boundary_32(self):
        """M==32 is decode (<=32)."""
        method = backend.TritonMoEMethod()
        cfg = method._get_default_config(M=32, E=8)
        assert cfg["BLOCK_SIZE_M"] == 16

    def test_get_default_config_boundary_96(self):
        """M==96 is small-mid (32 < M <= 96) → BLOCK_SIZE_M=32."""
        method = backend.TritonMoEMethod()
        cfg = method._get_default_config(M=96, E=8)
        assert cfg["BLOCK_SIZE_M"] == 32

    def test_get_default_config_boundary_512(self):
        """M==512 is mid (<=512) → BLOCK_SIZE_M=64."""
        method = backend.TritonMoEMethod()
        cfg = method._get_default_config(M=512, E=8)
        assert cfg["BLOCK_SIZE_M"] == 64

    def test_get_default_config_has_group_size_m(self):
        """All configs must include GROUP_SIZE_M key."""
        method = backend.TritonMoEMethod()
        for M in (1, 64, 1024):
            cfg = method._get_default_config(M=M, E=8)
            assert "GROUP_SIZE_M" in cfg

    def test_get_default_config_block_n_boundary(self):
        """M<=64 → BLOCK_SIZE_N=64; M>64 → BLOCK_SIZE_N=128."""
        method = backend.TritonMoEMethod()
        cfg64 = method._get_default_config(M=64, E=8)
        assert cfg64["BLOCK_SIZE_N"] == 64
        cfg65 = method._get_default_config(M=65, E=8)
        assert cfg65["BLOCK_SIZE_N"] == 128

    def test_get_default_config_group_m_16(self):
        """tokens_per_expert > 128 → GROUP_SIZE_M=16."""
        method = backend.TritonMoEMethod()
        # M=1024, E=1 → tokens_per_expert=1024 > 128 → group_m=16
        cfg = method._get_default_config(M=1024, E=1)
        assert cfg["GROUP_SIZE_M"] == 16

    def test_get_default_config_group_m_1(self):
        """tokens_per_expert <= 128 → GROUP_SIZE_M=1."""
        method = backend.TritonMoEMethod()
        # M=128, E=8 → tokens_per_expert=16 <= 128 → group_m=1
        cfg = method._get_default_config(M=128, E=8)
        assert cfg["GROUP_SIZE_M"] == 1

    def test_get_default_config_num_warps(self):
        """M<=128 → num_warps=4; M>128 → num_warps=8."""
        method = backend.TritonMoEMethod()
        cfg128 = method._get_default_config(M=128, E=8)
        assert cfg128["num_warps"] == 4
        cfg256 = method._get_default_config(M=256, E=8)
        assert cfg256["num_warps"] == 8

    def test_get_default_config_num_stages(self):
        """M<=32 → num_stages=4; M>32 → num_stages=3."""
        method = backend.TritonMoEMethod()
        cfg32 = method._get_default_config(M=32, E=8)
        assert cfg32["num_stages"] == 4
        cfg33 = method._get_default_config(M=33, E=8)
        assert cfg33["num_stages"] == 3

    # ------------------------------------------------------------------
    # apply — empty-batch fast path
    # ------------------------------------------------------------------

    def test_apply_empty_batch_returns_zero_tensor(self, fake_ops, monkeypatch):
        """apply() with 0 tokens must return a zero tensor of shape [0, hidden_size]."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
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
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=H)
        self._create_weights(method, layer)
        self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([T, H], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply(layer, x, gate)

        assert list(out.shape) == [T, H]

    def test_apply_noaux_tc_topk_hook_called(self, fake_ops, monkeypatch):
        """topk_ids_hookfunc must be called with topk_ids kwarg during apply()."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
        self._patch_bf16_kernel(monkeypatch)

        captured = {}

        def hook(topk_ids):
            captured["topk_ids"] = topk_ids

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        method.apply(layer, x, DummyGate(layer.num_local_experts), topk_ids_hookfunc=hook)

        assert "topk_ids" in captured

    def test_apply_noaux_tc_kernel_called_twice(self, fake_ops, monkeypatch):
        """fused_moe_kernel_bf16 must be launched twice (GEMM1 + GEMM2) per forward pass."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
        kernel = self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        method.apply(layer, x, DummyGate(layer.num_local_experts))

        assert len(kernel.calls) == 2, f"Expected 2 kernel launches (GEMM1 + GEMM2), got {len(kernel.calls)}"

    # ------------------------------------------------------------------
    # apply — non-noaux routing path (moe_topk_select)
    # ------------------------------------------------------------------

    def test_apply_aux_routing_path(self, fake_ops, monkeypatch):
        """When topk_method != 'noaux_tc', the moe_topk_select path is used."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        layer.topk_method = "aux"
        self._create_weights(method, layer)
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
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
        self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply_tp(layer, x, gate)

        assert list(out.shape) == [2, layer.hidden_size]

    # ------------------------------------------------------------------
    # EP methods raise NotImplementedError
    # ------------------------------------------------------------------

    def test_apply_ep_prefill_raises(self):
        method = backend.TritonMoEMethod()
        layer = self._make_layer()
        with pytest.raises(NotImplementedError):
            method.apply_ep_prefill(layer, None, None)

    def test_apply_ep_decode_raises(self):
        method = backend.TritonMoEMethod()
        layer = self._make_layer()
        with pytest.raises(NotImplementedError):
            method.apply_ep_decode(layer, None, None)

    # ------------------------------------------------------------------
    # apply — kernel argument verification
    # ------------------------------------------------------------------

    def test_apply_kernel_even_ks_true(self, fake_ops, monkeypatch):
        """When hidden_size is divisible by BLOCK_SIZE_K, even_Ks=True in GEMM1."""
        method = backend.TritonMoEMethod()
        # hidden_size=64, BLOCK_SIZE_K=64 → even_Ks=True for GEMM1
        layer = self._make_layer(hidden_size=64, intermediate_size=32)
        self._create_weights(method, layer)
        kernel = self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        method.apply(layer, x, gate)

        assert len(kernel.calls) == 2
        assert kernel.calls[0]["kwargs"]["even_Ks"] is True

    def test_apply_kernel_even_ks_false(self, fake_ops, monkeypatch):
        """When hidden_size is NOT divisible by BLOCK_SIZE_K, even_Ks=False in GEMM1."""
        method = backend.TritonMoEMethod()
        # hidden_size=8, BLOCK_SIZE_K=64 → even_Ks=False for GEMM1
        layer = self._make_layer(hidden_size=8, intermediate_size=4)
        self._create_weights(method, layer)
        kernel = self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        method.apply(layer, x, gate)

        assert len(kernel.calls) == 2
        assert kernel.calls[0]["kwargs"]["even_Ks"] is False

    def test_apply_gemm2_top_k_always_1(self, fake_ops, monkeypatch):
        """GEMM2 must always be called with top_k=1 (flat token-expert pairs)."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8, top_k=4)
        self._create_weights(method, layer)
        kernel = self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        method.apply(layer, x, gate)

        assert len(kernel.calls) == 2
        assert kernel.calls[0]["kwargs"]["top_k"] == layer.top_k
        assert kernel.calls[1]["kwargs"]["top_k"] == 1

    def test_apply_gemm1_no_mul_weight_gemm2_mul_weight(self, fake_ops, monkeypatch):
        """GEMM1 has MUL_ROUTED_WEIGHT=False, GEMM2 has MUL_ROUTED_WEIGHT=True."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
        kernel = self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        method.apply(layer, x, gate)

        assert kernel.calls[0]["kwargs"]["MUL_ROUTED_WEIGHT"] is False
        assert kernel.calls[1]["kwargs"]["MUL_ROUTED_WEIGHT"] is True

    def test_apply_large_batch_config(self, fake_ops, monkeypatch):
        """Large token count picks larger tile config (BLOCK_SIZE_M=128, num_warps=8)."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
        kernel = self._patch_bf16_kernel(monkeypatch)

        # 1024 tokens → prefill config: BLOCK_SIZE_M=128
        x = paddle.randn([1024, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        method.apply(layer, x, gate)

        assert len(kernel.calls) == 2
        assert kernel.calls[0]["kwargs"]["BLOCK_SIZE_M"] == 128
        assert kernel.calls[0]["kwargs"]["num_warps"] == 8

    def test_apply_single_token_output_shape(self, fake_ops, monkeypatch):
        """Single token decode scenario."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(num_experts=128, hidden_size=16, intermediate_size=8, top_k=8)
        self._create_weights(method, layer)
        self._patch_bf16_kernel(monkeypatch)

        x = paddle.randn([1, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply(layer, x, gate)

        assert list(out.shape) == [1, layer.hidden_size]

    def test_get_moe_method_triton_branch(self, monkeypatch):
        """get_moe_method() returns TritonMoEMethod when FD_MOE_BACKEND='triton' and is_cuda()."""
        from fastdeploy.model_executor.layers.moe import moe as moe_module

        monkeypatch.setattr(moe_module, "current_platform", types.SimpleNamespace(is_cuda=lambda: True))
        monkeypatch.setattr(moe_module.envs, "FD_MOE_BACKEND", "triton")
        result = moe_module.get_moe_method()
        assert isinstance(result, backend.TritonMoEMethod)

    def test_apply_use_fused_false(self, fake_ops, monkeypatch):
        """FD_ENABLE_RL=True triggers use_fused=False branch (gate_out.cast('float32'))."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        self._create_weights(method, layer)
        self._patch_bf16_kernel(monkeypatch)

        monkeypatch.setattr(backend.fastdeploy.envs, "FD_ENABLE_RL", True)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        out = method.apply(layer, x, gate)
        assert list(out.shape) == [2, layer.hidden_size]

    def test_apply_tp_with_topk_reduce_func(self, fake_ops, monkeypatch):
        """topk_reduce_func attribute is passed through to get_moe_scores."""
        method = backend.TritonMoEMethod()
        layer = self._make_layer(hidden_size=8)
        layer.topk_reduce_func = lambda x: x
        self._create_weights(method, layer)
        self._patch_bf16_kernel(monkeypatch)

        scores_kwargs = {}

        def tracking_get_moe_scores(*args, **kwargs):
            scores_kwargs.update(kwargs)
            gate_out = args[0]
            token_num = gate_out.shape[0]
            top_k = args[3]
            topk_ids = paddle.zeros([token_num, top_k], dtype="int64")
            topk_weights = paddle.ones([token_num, top_k], dtype="float32")
            return gate_out, topk_weights, topk_ids

        monkeypatch.setattr(backend, "get_moe_scores", tracking_get_moe_scores)

        x = paddle.randn([2, layer.hidden_size], dtype="bfloat16")
        gate = DummyGate(layer.num_local_experts)
        method.apply(layer, x, gate)

        assert "topk_reduce_func" in scores_kwargs


# ===========================================================================
# Precision tests: TritonMoEMethod vs. CutlassMoEMethod (BF16)
# ===========================================================================


def _make_precision_layer_pair(num_experts, hidden_size, intermediate_size, top_k):
    """
    Build a DummyLayer with random BF16 weights and a TritonMoEMethod.

    Weight layout (CUDA non-torch): [E, H, 2N] for up_gate_proj, [E, N, H] for down_proj.
    Returns (layer, None, triton_method) for compatibility with existing test signatures.
    """
    layer = DummyLayer(
        quant_config=None,
        num_local_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        top_k=top_k,
        weight_dtype="bfloat16",
    )

    triton_method = backend.TritonMoEMethod()

    # Create weight parameters (CUDA non-torch layout)
    triton_method.create_weights(
        layer,
        model_format="default",
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
    )

    # Fill with Xavier-like random BF16 weights to produce meaningful output magnitudes.
    # W1: [E, H, 2N] — scale by 1/sqrt(H) so GEMM1 output ~O(1)
    # W2: [E, N, H] — scale by 1/sqrt(N) so GEMM2 output ~O(1)
    paddle.seed(42)
    w1_scale = 1.0 / (hidden_size**0.5)
    w2_scale = 1.0 / (intermediate_size**0.5)
    layer.up_gate_proj_weight.set_value((paddle.randn(layer.up_gate_proj_weight.shape) * w1_scale).cast("bfloat16"))
    layer.down_proj_weight.set_value((paddle.randn(layer.down_proj_weight.shape) * w2_scale).cast("bfloat16"))
    return layer, None, triton_method


def _uniform_gate(layer):
    """Gate that outputs uniform logits so every expert gets equal probability."""

    class _Gate(paddle.nn.Layer):
        def __init__(self, num_experts):
            super().__init__()
            self.num_experts = num_experts

        def forward(self, x):
            return paddle.ones([x.shape[0], self.num_experts], dtype="float32")

    return _Gate(layer.num_local_experts)


# Shapes to exercise: (token_num, hidden_size, intermediate_size, num_experts, top_k)
# Small/medium sizes to keep test runtime reasonable.
_PRECISION_SHAPES = [
    pytest.param(1, 64, 32, 8, 2, id="decode_T1_H64"),
    pytest.param(16, 64, 32, 8, 2, id="decode_T16_H64"),
    pytest.param(64, 128, 64, 8, 2, id="mid_T64_H128"),
    pytest.param(128, 128, 64, 8, 2, id="mid_T128_H128_E8"),
    pytest.param(256, 256, 128, 8, 4, id="prefill_T256_H256"),
]


@pytest.mark.skipif(not paddle.is_compiled_with_cuda(), reason="requires CUDA")
# @pytest.mark.skipif(not _triton_ops_available(), reason="triton MoE ops not available (custom ops not compiled)")
class TestTritonMoEPrecision:
    """
    Precision tests: Triton BF16 path vs. Cutlass BF16 path.

    Both paths are activated in production via the FD_MOE_BACKEND env var
    (triton vs cutlass). This test verifies they produce numerically equivalent
    results on the same shared BF16 weights and identical inputs.

    All tests run real GPU kernels (no mocking).
    Tolerance: atol=1e-2, rtol=1e-2  (both kernels use BF16 arithmetic with
    fp32 accumulation; differences come from tile ordering / rounding).
    """

    # Tolerance for comparing two independent BF16 GEMM implementations.
    # BF16 has ~7-bit mantissa (eps ~0.008). After GEMM1 + SwiGLU + GEMM2,
    # rounding differences accumulate. Use np.allclose style:
    #   |triton - cutlass| <= ATOL + RTOL * |cutlass|
    ATOL = 1e-3
    RTOL = 1e-3

    @pytest.mark.parametrize("T,H,N,E,K", _PRECISION_SHAPES)
    def test_triton_vs_cutlass(self, T, H, N, E, K):
        """Triton BF16 MoE output must agree with CUTLASS BF16 MoE output.

        Both paths use the same weight layout, routing logic, and BF16 arithmetic.
        Differences should only come from tile ordering / rounding in GEMM.
        """
        from fastdeploy.model_executor.layers.moe.fused_moe_cutlass_backend import (
            CutlassMoEMethod,
        )

        layer, _, triton_method = _make_precision_layer_pair(E, H, N, K)

        # CUTLASS method shares the same weights (already created by _make_precision_layer_pair)
        cutlass_method = CutlassMoEMethod(None)

        paddle.seed(0)
        x = (paddle.randn([T, H]) * 0.1).cast("bfloat16")

        # Use a deterministic non-uniform gate to ensure consistent routing
        # across multiple calls of noaux_tc (avoids tie-breaking ambiguity)
        class _DeterministicGate(paddle.nn.Layer):
            def __init__(self, num_experts, T):
                super().__init__()
                self.num_experts = num_experts
                paddle.seed(123)
                self._scores = paddle.randn([T, num_experts], dtype="float32") * 2.0

            def forward(self, x):
                return self._scores[: x.shape[0]]

        gate = _DeterministicGate(E, T)

        # --- Run Triton path ---
        triton_out = triton_method.apply(layer, x, gate).cast("float32").numpy()

        # --- Run CUTLASS path ---
        cutlass_out = cutlass_method.apply(layer, x, gate).cast("float32").numpy()

        # np.allclose style: |a - b| <= atol + rtol * |b|
        tol = self.ATOL + self.RTOL * np.abs(cutlass_out)
        violations = np.abs(triton_out - cutlass_out) > tol
        num_violations = int(violations.sum())
        total_elements = triton_out.size

        assert num_violations == 0, (
            f"[T={T},H={H},N={N},E={E},K={K}] "
            f"{num_violations}/{total_elements} elements exceed tolerance "
            f"(atol={self.ATOL}, rtol={self.RTOL}). "
            f"Max abs diff: {float(np.abs(triton_out - cutlass_out).max()):.2e}, "
            f"max |cutlass|: {float(np.abs(cutlass_out).max()):.2e}"
        )

    @pytest.mark.parametrize("T,H,N,E,K", _PRECISION_SHAPES)
    def test_triton_output_shape(self, T, H, N, E, K):
        """Output shape must always be [T, H] regardless of batch size."""
        layer, _, triton_method = _make_precision_layer_pair(E, H, N, K)
        x = (paddle.randn([T, H]) * 0.1).cast("bfloat16")
        gate = _uniform_gate(layer)
        out = triton_method.apply(layer, x, gate)
        assert list(out.shape) == [T, H], f"Expected [{T}, {H}], got {list(out.shape)}"

    @pytest.mark.parametrize("T,H,N,E,K", _PRECISION_SHAPES)
    def test_triton_output_dtype_is_bfloat16(self, T, H, N, E, K):
        """Output dtype must match input dtype (bfloat16)."""
        layer, _, triton_method = _make_precision_layer_pair(E, H, N, K)
        x = (paddle.randn([T, H]) * 0.1).cast("bfloat16")
        gate = _uniform_gate(layer)
        out = triton_method.apply(layer, x, gate)
        assert out.dtype == paddle.bfloat16, f"Expected bfloat16, got {out.dtype}"

    def test_zero_input_gives_zero_output(self):
        """All-zero input must produce all-zero output."""
        T, H, N, E, K = 8, 64, 32, 8, 2
        layer, _, triton_method = _make_precision_layer_pair(E, H, N, K)
        x = paddle.zeros([T, H], dtype="bfloat16")
        gate = _uniform_gate(layer)

        out = triton_method.apply(layer, x, gate).cast("float32").numpy()
        np.testing.assert_allclose(
            out,
            np.zeros_like(out),
            atol=1e-6,
            err_msg="triton: zero input should produce zero output",
        )
