"""
# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from types import SimpleNamespace

import paddle

from fastdeploy.model_executor.layers.moe import (
    fused_moe_marlin_backend as marlin_backend,
)

paddle.set_device("gpu")


class _DummyLayer(paddle.nn.Layer):
    def __init__(self, hidden_size=64, moe_intermediate_size=32, topk_method="topk", num_local_experts=2):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k = 1
        self.n_group = 1
        self.topk_group = 1
        self.topk_method = topk_method
        self.routed_scaling_factor = 1.0
        self.gate_correction_bias = paddle.zeros([self.num_experts], dtype="float32")
        self.renormalize = True
        self.fd_config = SimpleNamespace()

    def extract_moe_ffn_weights(self, state_dict):
        return state_dict["up"], state_dict["down"], None, None


def _make_weights(layer):
    up = [
        paddle.ones([layer.hidden_size, layer.moe_intermediate_size * 2], dtype="float16")
        for _ in range(layer.num_local_experts)
    ]
    down = [
        paddle.ones([layer.moe_intermediate_size, layer.hidden_size], dtype="float16")
        for _ in range(layer.num_local_experts)
    ]
    return up, down


def test_marlin_process_and_apply_paths(monkeypatch):
    method = marlin_backend.MarlinWeightOnlyMoEMethod()
    layer = _DummyLayer()

    prev_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype("float16")
    method.create_weights(layer)
    paddle.set_default_dtype(prev_dtype)
    up, down = _make_weights(layer)
    method.process_loaded_weights(layer, {"up": up, "down": down})

    scales = paddle.arange(128, dtype="float32").reshape([2, 64])
    permuted = marlin_backend.marlin_permute_scales(scales, size_k=16, size_n=64, group_size=8)
    assert permuted.shape == [2, 64]

    gate = paddle.nn.Linear(layer.hidden_size, layer.num_experts, bias_attr=False)
    x = paddle.ones([2, layer.hidden_size], dtype="float16")
    monkeypatch.setattr(
        marlin_backend,
        "MoeWna16MarlinGemmApi",
        lambda *_args, **kwargs: (paddle.zeros([kwargs["size_m"], kwargs["size_n"]], dtype=x.dtype),),
    )
    out = method.apply(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out.shape == [2, layer.hidden_size]

    layer.topk_method = "noaux_tc"
    out_noaux = method.apply(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out_noaux.shape == [2, layer.hidden_size]
