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
    is_pre_sharded=False,
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
        load_config=SimpleNamespace(
            dynamic_load_weight=False,
            load_choices=load_choices,
            is_pre_sharded=is_pre_sharded,
        ),
        quant_config=None,
    )


class TinyParam:
    def __init__(self, tensor, initialized=True, with_track=False, write_back=None):
        self._tensor = tensor if isinstance(tensor, paddle.Tensor) else paddle.to_tensor(tensor)
        self._initialized = initialized
        self._write_back = write_back
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
        if self._write_back is not None:
            self._write_back(value_tensor)
        self._tensor = value_tensor

    def copy_(self, src, blocking=True):
        self.set_value(src)

    def __getitem__(self, item):
        def _write_back(value_tensor):
            self._tensor[item] = value_tensor

        return TinyParam(self._tensor[item], initialized=True, write_back=_write_back)


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
    np.testing.assert_allclose(param._tensor.numpy(), np.ones((2, 4), dtype="float32"))
    param_shard = TinyParam(paddle.zeros([2, 4], dtype="float32"), initialized=False)
    layer_merge.weight_loader(param_shard, paddle.ones([2, 2], dtype="int8"), loaded_shard_id="gate")
    assert param_shard._is_initialized() is True
    assert not np.allclose(param_shard._tensor.numpy()[..., :2], 0)
    assert np.allclose(param_shard._tensor.numpy()[..., 2:], 0)
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
    param_gate = TinyParam(paddle.zeros([2, 4], dtype="float32"), initialized=True)
    param_gate.output_dim = True
    param_gate.weight_need_transpose = True
    layer_mc.weight_loader(param_gate, paddle.ones([4, 2], dtype="int8"), loaded_shard_id="gate")
    assert not np.allclose(param_gate._tensor.numpy()[..., :2], 0)
    assert np.allclose(param_gate._tensor.numpy()[..., 2:], 0)
    layer_mc.local_rank = 1
    param_shape = TinyParam(paddle.zeros([2, 4], dtype="float32"), initialized=True)
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

    layer_mc.weight_loader(param_shape, _Wrapper(np.ones((2, 4), dtype="float32")), loaded_shard_id="up")
    assert np.allclose(param_shape._tensor.numpy()[..., :2], 0)
    assert np.allclose(param_shape._tensor.numpy()[..., 2:], 1)
    layer_merge_t = MergedReplicatedLinear.__new__(MergedReplicatedLinear)
    layer_merge_t.__dict__.update(fd_config=make_fd_config(model_format="paddle"), output_sizes=[1, 1])
    param_t = TinyParam(paddle.zeros([2, 2], dtype="float32"), initialized=False, with_track=True)
    param_t.weight_need_transpose = True
    param_up = TinyParam(paddle.zeros([2, 2], dtype="float32"), initialized=True, with_track=True)
    param_up.weight_need_transpose = True
    layer_merge_t.weight_loader(param_t, np.arange(4, dtype="float32").reshape(2, 2), loaded_shard_id=None)
    layer_merge_t.weight_loader(param_up, np.arange(2, dtype="float32").reshape(1, 2), loaded_shard_id="up")
    assert param_t.tensor_track.calls and param_up.tensor_track.calls
    assert np.allclose(param_up._tensor.numpy()[..., :1], 0)
    assert not np.allclose(param_up._tensor.numpy()[..., 1:], 0)
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


def test_column_parallel_load_state_dict_weight_key(monkeypatch):
    layer = MergedColumnParallelLinear.__new__(MergedColumnParallelLinear)
    layer.weight_key = "proj.weight"
    layer.with_bias = False
    layer.is_quantized = False
    layer.bias_key = "proj.bias"
    monkeypatch.setattr(LinearBase, "load_state_dict", lambda self, sd: None)
    state_dict = {"proj.weight": np.ones((2, 2), dtype="float32")}
    MergedColumnParallelLinear.load_state_dict(layer, state_dict)
    assert isinstance(state_dict["proj.weight"], paddle.Tensor)


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
        fd_config=cfg_tp2,
    )
    param_fused = TinyParam(paddle.zeros([4, 8], dtype="float32"), initialized=True)
    param_fused.output_dim = True
    param_fused.weight_need_transpose = True
    layer_w.weight_loader(param_fused, np.ones((12, 4), dtype="float32"), loaded_shard_id=None)
    np.testing.assert_allclose(param_fused._tensor.numpy(), np.ones((4, 8), dtype="float32"))

    param_split = TinyParam(paddle.zeros([4, 8], dtype="float32"), initialized=True)
    param_split.output_dim = True
    param_split.weight_need_transpose = True
    layer_w.weight_loader(param_split, np.ones((8, 4), dtype="float32"), loaded_shard_id="q")
    layer_w.weight_loader(param_split, np.ones((2, 4), dtype="float32"), loaded_shard_id="k")
    layer_w.weight_loader(param_split, np.full((2, 4), 2.0, dtype="float32"), loaded_shard_id="v")
    param_split_np = param_split._tensor.numpy()
    np.testing.assert_allclose(param_split_np[..., :4], np.ones((4, 4), dtype="float32"))
    np.testing.assert_allclose(param_split_np[..., 4:6], np.ones((4, 2), dtype="float32"))
    np.testing.assert_allclose(param_split_np[..., 6:8], np.full((4, 2), 2.0, dtype="float32"))
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
    layer_fused = QKVParallelLinear.__new__(QKVParallelLinear)
    layer_fused.weight_key = f"{prefix}.weight"
    called = []
    layer_fused.quant_method = SimpleNamespace(process_loaded_weights=lambda *_: called.append(True))
    layer_fused.load_weight({f"{prefix}.weight": np.ones((2, 2), dtype="float32")})
    assert called


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
    monkeypatch.setattr(current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(paddle.distributed, "alltoall", lambda out, x, group=None: out.set_value(x))
    out_decode = layer_decode.all2all_transpose(paddle.ones([1, 2], dtype="float32"))
    assert out_decode.shape[0] == 1
    out_decode_full = layer_decode.all2all_transpose(paddle.ones([2, 2], dtype="float32"))
    assert out_decode_full.shape[0] == 1
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True)
    layer_prefill = RowParallelLinear(
        fd_config=make_fd_config(tensor_parallel_size=2, splitwise_role="prefill"),
        prefix="row",
        input_size=4,
        output_size=2,
        with_bias=False,
        layer_id=-1,
    )
    out_prefill = layer_prefill.all2all_transpose(paddle.ones([1, 1], dtype="float32"))
    assert out_prefill.shape == [1, 2]
    layer_bias = RowParallelLinear(
        fd_config=make_fd_config(tensor_parallel_size=2, splitwise_role="prefill"),
        prefix="row",
        input_size=4,
        output_size=2,
        with_bias=True,
        layer_id=-1,
    )
    assert getattr(layer_bias.bias, "tp_row_bias", False) is True


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
    layer_v1.fd_config.load_config.dynamic_load_weight = True
    layer_v1.process_weights_after_loading()
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

    def _make_err_layer():
        return KVBatchLinear(
            fd_config=make_fd_config(load_choices="default_v1"),
            kv_b_proj=paddle.nn.Linear(2, 4, bias_attr=False),
            prefix="kv_b_proj",
            kv_lora_rank=2,
            num_attention_heads=2,
            qk_nope_head_dim=1,
            v_head_dim=None,
        )

    for fn in (
        lambda obj: obj.process_weights_after_loading(),
        lambda obj: obj.load_state_dict({"kv_b_proj.weight": paddle.arange(8, dtype="float32").reshape([2, 4])}),
    ):
        with pytest.raises(ValueError, match="v_head_dim"):
            fn(_make_err_layer())
