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

from types import SimpleNamespace

import numpy as np
import paddle
import pytest

from fastdeploy.model_executor.layers import linear as linear_module
from fastdeploy.model_executor.layers.linear import (
    KVBatchLinear,
    LinearBase,
    MergedColumnParallelLinear,
    MergedReplicatedLinear,
    QKVParallelLinear,
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
    return SimpleNamespace(
        model_config=SimpleNamespace(
            is_quantized=False,
            hidden_size=4,
            model_format=model_format,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=2,
            moe_layer_start_index=0,
            num_hidden_layers=1,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tensor_parallel_size,
            tensor_parallel_rank=tensor_parallel_rank,
            expert_parallel_size=1,
            tp_group=None,
            use_sequence_parallel_moe=use_sequence_parallel_moe,
        ),
        scheduler_config=SimpleNamespace(splitwise_role=splitwise_role, max_num_seqs=1),
        load_config=SimpleNamespace(dynamic_load_weight=False, load_choices=load_choices),
        quant_config=None,
    )


class TinyParam:
    def __init__(self, tensor, initialized=True, with_track=False):
        self._tensor = tensor if isinstance(tensor, paddle.Tensor) else paddle.to_tensor(tensor)
        self._initialized = initialized
        if with_track:
            self.tensor_track = SimpleNamespace(calls=[])
            self.tensor_track.mark = lambda start, end: self.tensor_track.calls.append((start, end))

    def _is_initialized(self):
        return self._initialized

    def initialize(self):
        self._initialized = True

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    def set_value(self, value):
        value_tensor = value if isinstance(value, paddle.Tensor) else paddle.to_tensor(value)
        if value_tensor.dtype != self._tensor.dtype:
            value_tensor = value_tensor.cast(self._tensor.dtype)
        self._tensor = value_tensor

    def copy_(self, src, blocking=True):
        self.set_value(src)

    def __getitem__(self, item):
        return TinyParam(self._tensor[item], initialized=True)


@pytest.fixture(autouse=True)
def _stub_platform(monkeypatch):
    for name, value in (
        ("is_cuda", False),
        ("is_xpu", True),
        ("is_iluvatar", False),
        ("is_gcu", False),
        ("is_dcu", False),
        ("is_maca", False),
        ("is_intel_hpu", False),
    ):
        monkeypatch.setattr(current_platform, name, lambda v=value: v)


def test_linearbase_and_unquantized_branches():
    layer = paddle.nn.Linear(2, 2, bias_attr=False)
    method = UnquantizedLinearMethod()
    method.process_loaded_weights(layer, paddle.ones([2, 2], dtype="float64"))
    np.testing.assert_allclose(layer.weight.numpy(), np.ones((2, 2), dtype="float32"))
    layer_init = LinearBase(
        fd_config=make_fd_config(), prefix="linear", input_size=2, output_size=3, with_bias=False, skip_quant=True
    )
    assert layer_init.weight_dtype == layer_init._dtype
    layer_pre = LinearBase.__new__(LinearBase)
    layer_pre.weight_key = "linear.weight"
    layer_pre.quant_method = UnquantizedLinearMethod()
    layer_pre.weight = TinyParam(paddle.zeros([2, 2], dtype="float32"))
    layer_pre.load_prequant_weight({"linear.weight": np.ones((2, 2), dtype="float32")})
    np.testing.assert_allclose(layer_pre.weight._tensor.numpy(), np.ones((2, 2), dtype="float32"))
    called = []
    layer_q = LinearBase.__new__(LinearBase)
    layer_q.quant_method = SimpleNamespace(process_prequanted_weights=lambda *_: called.append(True))
    layer_q.load_prequant_weight({})
    assert called
    layer_mqa = LinearBase.__new__(LinearBase)
    layer_mqa.weight_key = "block.qkv_a_proj_with_mqa.weight"
    layer_mqa.quant_method = UnquantizedLinearMethod()
    layer_mqa.weight = TinyParam(paddle.zeros([2, 3], dtype="float32"))
    layer_mqa.load_weight(
        {
            "block.q_a_proj.weight": np.ones((2, 1), dtype="float32"),
            "block.kv_a_proj_with_mqa.weight": np.full((2, 2), 2.0, dtype="float32"),
        }
    )
    np.testing.assert_allclose(layer_mqa.weight._tensor.numpy(), [[1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
    layer_state_q = LinearBase.__new__(LinearBase)
    layer_state_q.is_quantized = True
    layer_state_q.weight_key = "linear.weight"
    layer_state_q.with_bias = False
    layer_state_q.called = False
    layer_state_q.load_prequant_weight = lambda _sd: setattr(layer_state_q, "called", True)
    layer_state_q.load_state_dict({"linear.weight": np.zeros((1, 1), dtype="float32")})
    assert layer_state_q.called is True
    layer_bias = LinearBase.__new__(LinearBase)
    layer_bias.is_quantized = False
    layer_bias.weight_key = "linear.weight"
    layer_bias.bias_key = "linear.bias"
    layer_bias.with_bias = True
    layer_bias.quant_method = UnquantizedLinearMethod()
    layer_bias.weight = TinyParam(paddle.zeros([2, 3], dtype="float32"))
    layer_bias.bias = TinyParam(paddle.zeros([3], dtype="float32"))
    layer_bias.load_state_dict(
        {
            "linear.weight": np.ones((2, 3), dtype="float32"),
            "linear.bias": np.array([1.0, 2.0, 3.0], dtype="float32"),
        }
    )
    np.testing.assert_allclose(layer_bias.bias._tensor.numpy(), [1.0, 2.0, 3.0])


def test_merged_and_column_weight_paths():
    layer_init = MergedReplicatedLinear(
        fd_config=make_fd_config(), prefix="mlp", input_size=2, output_sizes=[2, 2], with_bias=False
    )
    assert layer_init.output_sizes == [2, 2]
    layer_merge = MergedReplicatedLinear.__new__(MergedReplicatedLinear)
    layer_merge.__dict__.update(fd_config=make_fd_config(model_format="paddle"), output_sizes=[2, 2])
    param = TinyParam(paddle.zeros([2, 4], dtype="float32"), initialized=False, with_track=True)
    loaded_weight = paddle.ones([2, 4], dtype="float16")
    layer_merge.weight_loader(param, loaded_weight, loaded_shard_id=None)
    assert param.tensor_track.calls == [(0, loaded_weight.shape[-1])]
    param_shard = TinyParam(paddle.zeros([2, 2], dtype=paddle.float8_e4m3fn), initialized=False)
    layer_merge.weight_loader(param_shard, paddle.ones([2, 2], dtype="int8"), loaded_shard_id="gate")
    assert param_shard._is_initialized() is True
    layer_mc = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    layer_mc.__dict__.update(
        fd_config=make_fd_config(model_format="paddle", tensor_parallel_size=1), tp_size=1, local_rank=0
    )
    param_fused = TinyParam(paddle.zeros([4, 2], dtype="float32"), initialized=False)
    param_fused.output_dim = True
    param_fused.weight_need_transpose = True
    layer_mc.weight_loader(param_fused, np.arange(8, dtype="float32").reshape(2, 4), loaded_shard_id=None)
    assert param_fused.weight_need_transpose is False
    layer_mc.__dict__.update(
        fd_config=make_fd_config(model_format="paddle", tensor_parallel_size=2), tp_size=2, local_rank=0
    )
    param_gate = TinyParam(paddle.zeros([4, 4], dtype=paddle.float8_e4m3fn), initialized=True)
    param_gate.output_dim = True
    param_gate.weight_need_transpose = True
    layer_mc.weight_loader(param_gate, paddle.ones([4, 4], dtype="int8"), loaded_shard_id="gate")
    layer_mc.local_rank = 1
    param_shape = TinyParam(paddle.zeros([4, 4], dtype="float32"), initialized=True)
    param_shape.output_dim = True
    param_shape.weight_need_transpose = False

    class _Wrapper:
        def __init__(self, array):
            self._array = array

        def get_shape(self):
            return self._array.shape

        @property
        def dtype(self):
            return self._array.dtype

        def __getitem__(self, item):
            return paddle.to_tensor(self._array[item])

    layer_mc.weight_loader(param_shape, _Wrapper(np.ones((4, 4), dtype="float32")), loaded_shard_id="up")
    layer_bias = MergedColumnParallelLinear(
        fd_config=make_fd_config(), prefix="mlp.up_gate_proj", input_size=4, output_size=4, with_bias=True
    )
    layer_bias.load_state_dict(
        {
            "mlp.gate_proj.weight": np.ones((4, 2), dtype="float32"),
            "mlp.up_proj.weight": np.ones((4, 2), dtype="float32"),
            "mlp.gate_proj.bias": np.ones((4,), dtype="float32"),
        }
    )
    np.testing.assert_allclose(layer_bias.bias.numpy(), np.ones((4,), dtype="float32"))


def test_qkv_paths():
    cfg_tp2 = make_fd_config(tensor_parallel_size=2)
    prefix = "attn.qkv_proj"
    layer_init = QKVParallelLinear(fd_config=cfg_tp2, prefix=prefix, with_bias=False)
    assert layer_init.num_kv_head_replicas == 2
    assert layer_init.kv_num_heads_per_rank == 1
    layer_w = QKVParallelLinear.__new__(QKVParallelLinear)
    layer_w.__dict__.update(
        num_heads=4,
        kv_num_heads=1,
        num_heads_per_rank=2,
        kv_num_heads_per_rank=1,
        num_kv_head_replicas=2,
        tp_size=2,
        local_rank=0,
    )
    param_fused = TinyParam(paddle.zeros([4, 8], dtype="float32"), initialized=True)
    param_fused.output_dim = True
    param_fused.weight_need_transpose = True
    layer_w.weight_loader(param_fused, np.arange(48, dtype="float32").reshape(12, 4), loaded_shard_id=None)
    param_shard = TinyParam(paddle.zeros([4, 8], dtype=paddle.float8_e4m3fn), initialized=False)
    param_shard.output_dim = True
    param_shard.weight_need_transpose = True
    layer_w.weight_loader(param_shard, paddle.ones([12, 4], dtype="int8"), loaded_shard_id="q")
    assert param_shard._is_initialized() is True
    layer_parts = QKVParallelLinear(fd_config=cfg_tp2, prefix=prefix, with_bias=False)
    layer_parts.load_weight(
        {
            "attn.q_proj.weight": np.ones((4, 4), dtype="float32"),
            "attn.k_proj.weight": np.ones((4, 2), dtype="float32"),
            "attn.v_proj.weight": np.ones((4, 2), dtype="float32"),
        }
    )
    layer_q = QKVParallelLinear.__new__(QKVParallelLinear)
    layer_q.__dict__.update(is_quantized=True, weight_key=f"{prefix}.weight", with_bias=False, called=False)
    layer_q.load_prequant_weight = lambda _sd: setattr(layer_q, "called", True)
    layer_q.load_state_dict({"attn.qkv_proj.weight": np.zeros((1, 1), dtype="float32")})
    assert layer_q.called is True
    layer_bias = QKVParallelLinear(fd_config=cfg_tp2, prefix=prefix, with_bias=True)
    layer_bias.load_state_dict(
        {
            "attn.q_proj.weight": np.ones((4, 4), dtype="float32"),
            "attn.k_proj.weight": np.ones((4, 2), dtype="float32"),
            "attn.v_proj.weight": np.ones((4, 2), dtype="float32"),
            "attn.q_proj.bias": np.ones((4,), dtype="float32"),
            "attn.k_proj.bias": np.ones((2,), dtype="float32"),
            "attn.v_proj.bias": np.ones((2,), dtype="float32"),
        }
    )
    np.testing.assert_allclose(layer_bias.bias.numpy(), np.ones((8,), dtype="float32"))


def test_row_parallel_paths(monkeypatch):
    layer_split = RowParallelLinear(
        fd_config=make_fd_config(tensor_parallel_size=2, splitwise_role="prefill", use_sequence_parallel_moe=True),
        prefix="row",
        input_size=4,
        output_size=4,
        with_bias=False,
        layer_id=0,
    )
    called = []
    layer_split.all2all_transpose = lambda x: (called.append(True) or x)
    layer_split.quant_method = SimpleNamespace(apply=lambda _layer, x: x)
    layer_split.forward_cuda(paddle.ones([2, 2], dtype="float32"))
    assert called
    layer_decode = RowParallelLinear(
        fd_config=make_fd_config(tensor_parallel_size=2, splitwise_role="decode"),
        prefix="row",
        input_size=4,
        output_size=4,
        with_bias=False,
        layer_id=-1,
    )
    monkeypatch.setattr(
        linear_module,
        "decode_alltoall_transpose",
        lambda x, out: out.set_value(paddle.zeros_like(out)),
    )
    out_decode = layer_decode.all2all_transpose(paddle.ones([1, 2], dtype="float32"))
    assert out_decode.shape[0] == 1
    layer_prefill = RowParallelLinear(
        fd_config=make_fd_config(tensor_parallel_size=2, splitwise_role="prefill"),
        prefix="row",
        input_size=4,
        output_size=2,
        with_bias=False,
        layer_id=-1,
    )
    monkeypatch.setattr(paddle.distributed, "alltoall", lambda out, x, group=None: out.set_value(x))
    out_prefill = layer_prefill.all2all_transpose(paddle.ones([1, 1], dtype="float32"))
    assert out_prefill.shape == [1, 2]


def test_kvbatch_paths():
    layer_v0 = KVBatchLinear(
        fd_config=make_fd_config(load_choices="default_v0"),
        kv_b_proj=paddle.nn.Linear(2, 4, bias_attr=False),
        prefix="kv_b_proj",
        kv_lora_rank=2,
        num_attention_heads=2,
        qk_nope_head_dim=1,
        v_head_dim=1,
    )
    assert layer_v0.kv_b_proj is None
    layer_v0.load_state_dict({"kv_b_proj.weight": paddle.arange(8, dtype="float32").reshape([2, 4])})
    assert layer_v0.k_b_proj_weight.shape[-1] == layer_v0.kv_lora_rank
    assert layer_v0.v_b_proj_weight.shape[-1] == layer_v0.v_head_dim
    layer_v1 = KVBatchLinear(
        fd_config=make_fd_config(model_format="torch", load_choices="default_v1"),
        kv_b_proj=paddle.nn.Linear(2, 4, bias_attr=False),
        prefix="kv_b_proj",
        kv_lora_rank=2,
        num_attention_heads=2,
        qk_nope_head_dim=1,
        v_head_dim=1,
    )
    layer_v1.weight_dtype = "float64"
    layer_v1.process_weights_after_loading()
    assert layer_v1.kv_b_proj is None
    x_k = paddle.ones([2, 1, 1], dtype="float64")
    x_v = paddle.ones([2, 1, 2], dtype="float64")
    out_k = layer_v1.forward_k_b(x_k)
    out_v = layer_v1.forward_v_b(x_v)
    assert out_k.shape[-1] == layer_v1.k_b_proj_weight.shape[-1]
    assert out_v.shape[-1] == layer_v1.v_b_proj_weight.shape[-1]
    layer_v1.forward(x_k, proj_type="k")
    layer_v1.forward(x_v, proj_type="v")
    with pytest.raises(ValueError):
        layer_v1.forward(x_k, proj_type="bad")
