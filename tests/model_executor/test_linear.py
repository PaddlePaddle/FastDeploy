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

from types import SimpleNamespace

import numpy as np
import paddle
import pytest

if not hasattr(paddle, "compat"):
    paddle.compat = SimpleNamespace()
if not hasattr(paddle.compat, "enable_torch_proxy"):
    paddle.compat.enable_torch_proxy = lambda *args, **kwargs: None

import fastdeploy.distributed.communication as communication

if not hasattr(communication, "decode_alltoall_transpose"):
    communication.decode_alltoall_transpose = lambda input_, out=None: input_
if not hasattr(communication, "tensor_model_parallel_all_reduce"):
    communication.tensor_model_parallel_all_reduce = lambda input_, group=None: input_

from fastdeploy.model_executor.layers.linear import (
    ColumnParallelLinear,
    KVBatchLinear,
    LinearBase,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from fastdeploy.platforms import current_platform


def make_fd_config(
    *,
    model_format="paddle",
    tensor_parallel_size=1,
    tensor_parallel_rank=0,
    splitwise_role="prefill",
    use_sequence_parallel_moe=False,
    load_choices="default_v0",
):
    model_config = SimpleNamespace(
        is_quantized=False,
        hidden_size=8,
        model_format=model_format,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=2,
        moe_layer_start_index=0,
        num_hidden_layers=1,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        tensor_parallel_rank=tensor_parallel_rank,
        expert_parallel_size=1,
        tp_group=None,
        use_sequence_parallel_moe=use_sequence_parallel_moe,
    )
    scheduler_config = SimpleNamespace(splitwise_role=splitwise_role, max_num_seqs=1)
    load_config = SimpleNamespace(dynamic_load_weight=False, load_choices=load_choices)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        load_config=load_config,
        quant_config=None,
    )


@pytest.fixture(autouse=True)
def _stub_platform(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(current_platform, "is_iluvatar", lambda: False)
    monkeypatch.setattr(current_platform, "is_gcu", lambda: False)
    monkeypatch.setattr(current_platform, "is_dcu", lambda: False)
    monkeypatch.setattr(current_platform, "is_maca", lambda: False)
    monkeypatch.setattr(current_platform, "is_intel_hpu", lambda: False)


def test_unquantized_method_and_linearbase_loading():
    fd_config = make_fd_config(model_format="torch")

    class DummyLayer(paddle.nn.Layer):
        def __init__(self, fd_cfg):
            super().__init__()
            self.fd_config = fd_cfg
            self.weight_shape = [2, 3]
            self.weight_dtype = "float32"
            self.with_bias = True
            self.bias = self.create_parameter(shape=[3], dtype=self.weight_dtype, is_bias=True)

    layer = DummyLayer(fd_config)
    method = UnquantizedLinearMethod()
    method.create_weights(layer, output_dim=True, model_format="torch")

    assert list(layer.weight.shape) == [3, 2]
    assert layer.weight.output_dim is False

    weight_value = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
    layer.weight.set_value(weight_value)
    method.process_weights_after_loading(layer)
    assert list(layer.weight.shape) == [2, 3]

    method.process_loaded_weights(layer, paddle.to_tensor([[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]], dtype="float64"))
    np.testing.assert_allclose(layer.weight.numpy(), [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]])

    layer.bias.set_value(paddle.to_tensor([1.0, 1.0, 1.0], dtype="float32"))
    x = paddle.to_tensor([[1.0, 0.0]], dtype="float32")
    out = method.apply(layer, x)
    np.testing.assert_allclose(out.numpy(), [[3.0, 5.0, 7.0]])


def test_linearbase_not_implemented_on_cpu(monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(current_platform, "is_iluvatar", lambda: False)
    monkeypatch.setattr(current_platform, "is_gcu", lambda: False)
    monkeypatch.setattr(current_platform, "is_dcu", lambda: False)
    monkeypatch.setattr(current_platform, "is_maca", lambda: False)
    monkeypatch.setattr(current_platform, "is_intel_hpu", lambda: False)

    fd_config = make_fd_config(model_format="paddle")
    with pytest.raises(NotImplementedError):
        LinearBase(fd_config=fd_config, input_size=2, output_size=2)


def test_linearbase_quantized_weight_keys_and_prequant_loader():
    class DummyQuantMethod:
        def __init__(self):
            self.called = False

        def create_weights(self, layer, **extra_weight_attrs):
            UnquantizedLinearMethod().create_weights(layer, **extra_weight_attrs)

        def process_prequanted_weights(self, layer, state_dict):
            self.called = True

    class DummyQuantConfig:
        dense_quant_type = "int8"

        def __init__(self, method):
            self._method = method

        def name(self):
            return "w8a8"

        def get_quant_method(self, _layer):
            return self._method

    quant_method = DummyQuantMethod()
    fd_config = make_fd_config(model_format="paddle")
    fd_config.model_config.is_quantized = True
    fd_config.quant_config = DummyQuantConfig(quant_method)

    layer = ReplicatedLinear(
        fd_config=fd_config,
        prefix="quant.linear",
        input_size=2,
        output_size=2,
        skip_quant=False,
    )
    assert layer.weight_key.endswith(".quant_weight")
    layer.load_state_dict({})
    assert quant_method.called is True


def test_replicated_linear_qkv_mqa_load_state_dict():
    fd_config = make_fd_config(model_format="paddle")
    layer = ReplicatedLinear(
        fd_config=fd_config,
        prefix="layer.qkv_a_proj_with_mqa",
        input_size=2,
        output_size=3,
        with_bias=True,
        skip_quant=True,
    )
    state_dict = {
        "layer.q_a_proj.weight": np.ones((2, 1), dtype="float32"),
        "layer.kv_a_proj_with_mqa.weight": np.full((2, 2), 2.0, dtype="float32"),
        "layer.qkv_a_proj_with_mqa.bias": np.array([0.5, 1.5, -1.0], dtype="float32"),
    }

    layer.load_state_dict(state_dict)
    np.testing.assert_allclose(layer.weight.numpy(), [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
    np.testing.assert_allclose(layer.bias.numpy(), [0.5, 1.5, -1.0])


def test_merged_replicated_linear_weight_loader_shards():
    fd_config = make_fd_config(model_format="paddle")
    layer = MergedReplicatedLinear(
        fd_config=fd_config,
        prefix="mlp",
        input_size=2,
        output_sizes=[2, 2],
        skip_quant=True,
    )
    full_weight = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype="float32")
    layer.weight_loader(layer.weight, full_weight)
    np.testing.assert_allclose(layer.weight.numpy(), full_weight.numpy())

    gate_weight = paddle.to_tensor([[9.0, 10.0], [5.0, 6.0]], dtype="float32")
    layer.weight_loader(layer.weight, gate_weight, loaded_shard_id="gate")
    np.testing.assert_allclose(layer.weight.numpy()[:, :2], gate_weight.numpy())

    up_weight = paddle.to_tensor([[1.5, 2.5], [3.5, 4.5]], dtype="float16")
    layer.weight_loader(layer.weight, up_weight, loaded_shard_id="up")
    assert layer.weight.numpy()[:, 2:].dtype == np.float32


def test_merged_column_parallel_linear_load_state_dict_and_weight_loader(monkeypatch):
    fd_config = make_fd_config(model_format="paddle", tensor_parallel_size=1)
    layer = MergedColumnParallelLinear(
        fd_config=fd_config,
        prefix="mlp.up_gate_proj",
        input_size=2,
        output_size=4,
        with_bias=False,
        skip_quant=True,
    )
    state_dict = {
        "mlp.gate_proj.weight": np.full((2, 2), 1.0, dtype="float32"),
        "mlp.up_proj.weight": np.full((2, 2), 2.0, dtype="float32"),
    }
    layer.load_state_dict(state_dict)
    expected_weight = np.concatenate([np.full((2, 2), 1.0), np.full((2, 2), 2.0)], axis=-1)
    np.testing.assert_allclose(layer.weight.numpy(), expected_weight)

    fd_config_tp = make_fd_config(model_format="paddle", tensor_parallel_size=1)
    layer_tp = MergedColumnParallelLinear(
        fd_config=fd_config_tp,
        prefix="mlp.up_gate_proj",
        input_size=2,
        output_size=4,
        skip_quant=True,
    )
    full_weight = paddle.to_tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype="float32")
    layer_tp.weight_loader(layer_tp.weight, full_weight)
    gate_weight = paddle.to_tensor([[9.0, 10.0], [7.0, 8.0]], dtype="float32")
    layer_tp.weight_loader(layer_tp.weight, gate_weight, loaded_shard_id="gate")
    np.testing.assert_allclose(layer_tp.weight.numpy()[:, :2], gate_weight.numpy())


def test_merged_column_parallel_linear_weight_loader_transpose_and_tp_shard():
    fd_config = make_fd_config(model_format="paddle", tensor_parallel_size=1)
    layer = MergedColumnParallelLinear(
        fd_config=fd_config,
        prefix="mlp.up_gate_proj",
        input_size=4,
        output_size=2,
        skip_quant=True,
    )
    layer.weight.weight_need_transpose = True
    fused_weight = paddle.arange(8, dtype="float32").reshape([2, 4])
    layer.weight_loader(layer.weight, fused_weight)
    assert layer.weight.weight_need_transpose is False
    assert list(layer.weight.shape) == [4, 2]

    fd_config_tp = make_fd_config(model_format="paddle", tensor_parallel_size=2)
    layer_tp = MergedColumnParallelLinear(
        fd_config=fd_config_tp,
        prefix="mlp.up_gate_proj",
        input_size=2,
        output_size=4,
        skip_quant=True,
    )
    shard_weight = paddle.to_tensor([[9.0, 10.0], [0.0, 0.0]], dtype="float32")
    layer_tp.weight_loader(layer_tp.weight, shard_weight, loaded_shard_id="gate")
    np.testing.assert_allclose(layer_tp.weight.numpy()[:, 0], [9.0, 0.0])


def test_column_parallel_bias_distribution():
    fd_config = make_fd_config(model_format="paddle", tensor_parallel_size=1)
    layer = ColumnParallelLinear(
        fd_config=fd_config,
        prefix="col",
        input_size=2,
        output_size=2,
        with_bias=True,
        skip_quant=True,
    )
    assert layer.bias.is_distributed is True
    assert layer.bias.split_axis == 1
    assert layer.bias.output_dim is True


def test_qkv_parallel_linear_load_weight_and_bias():
    fd_config = make_fd_config(model_format="paddle", tensor_parallel_size=2, tensor_parallel_rank=0)
    layer = QKVParallelLinear(fd_config=fd_config, prefix="attn.qkv_proj", with_bias=False)
    state_dict = {
        "attn.q_proj.weight": np.ones((8, 4), dtype="float32"),
        "attn.k_proj.weight": np.full((8, 2), 2.0, dtype="float32"),
        "attn.v_proj.weight": np.full((8, 2), 3.0, dtype="float32"),
    }
    layer.load_state_dict(state_dict)
    assert layer.weight.shape == [8, 8]


def test_qkv_parallel_linear_weight_loader_and_bias_paths():
    fd_config = make_fd_config(model_format="paddle", tensor_parallel_size=2, tensor_parallel_rank=0)
    fd_config.model_config.num_key_value_heads = 2
    layer = QKVParallelLinear(fd_config=fd_config, prefix="attn.qkv_proj", with_bias=False)
    fused_weight = paddle.arange(8 * 16, dtype="float32").reshape([8, 16])
    layer.weight_loader(layer.weight, fused_weight)
    assert layer.weight.numpy()[0, 0] == fused_weight.numpy()[0, 0]

    fd_config_single = make_fd_config(model_format="paddle", tensor_parallel_size=1, tensor_parallel_rank=0)
    fd_config_single.model_config.num_key_value_heads = 2
    layer_bias = QKVParallelLinear(fd_config=fd_config_single, prefix="attn.qkv_proj", with_bias=True)
    state_dict = {
        "attn.qkv_proj.weight": np.ones((8, 16), dtype="float32"),
        "attn.qkv_proj.bias": np.array([0.1] * 16, dtype="float32"),
    }
    layer_bias.load_state_dict(state_dict)
    np.testing.assert_allclose(layer_bias.bias.numpy()[:2], [0.1, 0.1])

    state_dict_weight = {"attn.qkv_proj.weight": np.full((8, 16), 2.0, dtype="float32")}
    layer_single = QKVParallelLinear(fd_config=fd_config_single, prefix="attn.qkv_proj", with_bias=False)
    layer_single.load_weight(state_dict_weight)
    np.testing.assert_allclose(layer_single.weight.numpy()[0, 0], 2.0)


def test_row_parallel_linear_all2all_and_forward(monkeypatch):
    fd_config_decode = make_fd_config(
        model_format="paddle",
        tensor_parallel_size=2,
        splitwise_role="decode",
        use_sequence_parallel_moe=True,
    )
    layer_decode = RowParallelLinear(
        fd_config=fd_config_decode,
        prefix="row",
        input_size=4,
        output_size=2,
        skip_quant=True,
        layer_id=0,
    )
    x = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]], dtype="float32")

    def _fake_decode_alltoall_transpose(src, dst):
        dst[:] = paddle.concat([src[:1], src[:1]], axis=-1)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.linear.decode_alltoall_transpose",
        _fake_decode_alltoall_transpose,
    )
    out = layer_decode.all2all_transpose(x)
    assert out.shape == [1, 4]

    fd_config_prefill = make_fd_config(
        model_format="paddle",
        tensor_parallel_size=2,
        splitwise_role="prefill",
        use_sequence_parallel_moe=True,
    )
    layer_prefill = RowParallelLinear(
        fd_config=fd_config_prefill,
        prefix="row",
        input_size=4,
        output_size=2,
        skip_quant=True,
        layer_id=0,
    )

    def _fake_alltoall(out, x_in, group=None):
        out[:] = x_in

    monkeypatch.setattr(paddle.distributed, "alltoall", _fake_alltoall)
    out_prefill = layer_prefill.all2all_transpose(x)
    assert out_prefill.shape == [1, 4]

    fd_config_reduce = make_fd_config(model_format="paddle", tensor_parallel_size=2)
    layer_reduce = RowParallelLinear(
        fd_config=fd_config_reduce,
        prefix="row",
        input_size=4,
        output_size=2,
        skip_quant=True,
    )
    layer_reduce.quant_method.apply = lambda _layer, data: data + 1.0

    def _fake_all_reduce(value, _group):
        return value + 2.0

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.linear.tensor_model_parallel_all_reduce",
        _fake_all_reduce,
    )
    reduced = layer_reduce.forward_cuda(paddle.ones([1, 2], dtype="float32"))
    np.testing.assert_allclose(reduced.numpy(), [[4.0, 4.0]])


def test_row_parallel_linear_padding_paths(monkeypatch):
    fd_config_decode = make_fd_config(
        model_format="paddle",
        tensor_parallel_size=2,
        splitwise_role="decode",
        use_sequence_parallel_moe=True,
    )
    layer_decode = RowParallelLinear(
        fd_config=fd_config_decode,
        prefix="row",
        input_size=4,
        output_size=2,
        skip_quant=True,
        layer_id=0,
    )
    x = paddle.to_tensor([[1.0, 2.0]], dtype="float32")

    def _fake_decode_alltoall_transpose(src, dst):
        dst[:] = paddle.concat([src[:1], src[:1]], axis=-1)

    monkeypatch.setattr(
        "fastdeploy.model_executor.layers.linear.decode_alltoall_transpose",
        _fake_decode_alltoall_transpose,
    )
    out = layer_decode.all2all_transpose(x)
    assert out.shape == [1, 4]

    fd_config_prefill = make_fd_config(
        model_format="paddle",
        tensor_parallel_size=2,
        splitwise_role="prefill",
        use_sequence_parallel_moe=True,
    )
    layer_prefill = RowParallelLinear(
        fd_config=fd_config_prefill,
        prefix="row",
        input_size=4,
        output_size=2,
        skip_quant=True,
        layer_id=0,
    )

    def _fake_alltoall(out, x_in, group=None):
        out[:] = paddle.concat([x_in, x_in], axis=-1)[:, : x_in.shape[1]]

    monkeypatch.setattr(paddle.distributed, "alltoall", _fake_alltoall)
    out_prefill = layer_prefill.all2all_transpose(x)
    assert out_prefill.shape == [1, 4]


def test_kv_batch_linear_paths():
    fd_config = make_fd_config(model_format="paddle", load_choices="default_v1")
    kv_proj = paddle.nn.Linear(2, 4, bias_attr=False)
    kv_proj.weight.set_value(paddle.arange(8, dtype="float32").reshape([2, 4]))
    layer = KVBatchLinear(
        fd_config=fd_config,
        kv_b_proj=kv_proj,
        prefix="kv_b_proj",
        kv_lora_rank=2,
        num_attention_heads=2,
        qk_nope_head_dim=1,
        v_head_dim=1,
    )
    layer.process_weights_after_loading()
    assert layer.k_b_proj_weight.shape == [2, 1, 2]
    assert layer.v_b_proj_weight.shape == [2, 2, 1]

    state_dict = {"kv_b_proj.weight": kv_proj.weight.numpy()}
    layer.load_state_dict(state_dict)
    assert layer.k_b_proj_weight.shape == [2, 1, 2]

    k_out = layer.forward_k_b(paddle.ones([2, 1, 1], dtype="float32"))
    v_out = layer.forward_v_b(paddle.ones([2, 1, 2], dtype="float32"))
    assert k_out.shape[-1] == 2
    assert v_out.shape[-1] == 1
    np.testing.assert_allclose(layer.forward(paddle.ones([2, 1, 1], dtype="float32"), proj_type="k"), k_out)
    np.testing.assert_allclose(layer.forward(paddle.ones([2, 1, 2], dtype="float32"), proj_type="v"), v_out)

    with pytest.raises(ValueError, match="proj_type must be 'k' or 'v'"):
        layer.forward(paddle.ones([1, 1, 1], dtype="float32"), proj_type="invalid")


def test_kv_batch_linear_dynamic_load_and_errors():
    fd_config = make_fd_config(model_format="paddle", load_choices="default_v0")
    fd_config.load_config.dynamic_load_weight = True
    kv_proj = paddle.nn.Linear(2, 4, bias_attr=False)
    layer = KVBatchLinear(
        fd_config=fd_config,
        kv_b_proj=kv_proj,
        prefix="kv_b_proj",
        kv_lora_rank=2,
        num_attention_heads=2,
        qk_nope_head_dim=1,
        v_head_dim=1,
    )
    assert layer.kv_b_proj is None
    layer.process_weights_after_loading()
    assert not hasattr(layer, "k_b_proj_weight")

    fd_config_error = make_fd_config(model_format="paddle", load_choices="default_v1")
    kv_proj_error = paddle.nn.Linear(2, 4, bias_attr=False)
    layer_error = KVBatchLinear(
        fd_config=fd_config_error,
        kv_b_proj=kv_proj_error,
        prefix="kv_b_proj",
        kv_lora_rank=2,
        num_attention_heads=2,
        qk_nope_head_dim=1,
        v_head_dim=None,
    )
    with pytest.raises(ValueError, match="v_head_dim should not be None"):
        layer_error.process_weights_after_loading()

    state_dict = {"kv_b_proj.weight": kv_proj_error.weight.numpy()}
    with pytest.raises(ValueError, match="v_head_dim should not be None"):
        layer_error.load_state_dict(state_dict)
