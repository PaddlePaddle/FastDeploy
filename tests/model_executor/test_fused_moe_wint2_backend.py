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
    fused_moe_wint2_backend as wint2_backend,
)

paddle.set_device("gpu")


class _DummyLayer(paddle.nn.Layer):
    def __init__(self, hidden_size=128, moe_intermediate_size=128, num_local_experts=2):
        super().__init__()
        self.num_local_experts = num_local_experts
        self.num_experts = num_local_experts
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.top_k = 1
        self.n_group = 1
        self.topk_group = 1
        self.topk_method = "topk"
        self.gate_correction_bias = paddle.zeros([self.num_experts], dtype="float32")
        self.routed_scaling_factor = 1.0
        self.renormalize = True
        self.expert_id_offset = 0
        self.fd_config = SimpleNamespace()
        self.weight_key_map = {
            "up_gate_proj_expert_weight_key": "up_w_{}",
            "down_proj_expert_weight_key": "down_w_{}",
            "up_gate_proj_expert_weight_scale_key": "up_scale_{}",
            "down_proj_expert_weight_scale_key": "down_scale_{}",
            "up_gate_proj_expert_super_scales_key": "up_super_{}",
            "down_proj_expert_super_scales_key": "down_super_{}",
            "up_gate_proj_expert_code_scale_key": "up_code_scale_{}",
            "down_proj_expert_code_scale_key": "down_code_scale_{}",
            "up_gate_proj_expert_code_zp_key": "up_code_zp_{}",
            "down_proj_expert_code_zp_key": "down_code_zp_{}",
        }

    def load_experts_weight(self, state_dict, *_args, **_kwargs):
        return state_dict["up"], state_dict["down"], None, None


def _make_state_dict(layer):
    super_dtype = layer.up_gate_proj_super_scales.dtype if hasattr(layer, "up_gate_proj_super_scales") else "float32"
    up = [
        paddle.ones([layer.hidden_size // 4, layer.moe_intermediate_size * 2], dtype="uint8")
        for _ in range(layer.num_local_experts)
    ]
    down = [
        paddle.ones([layer.moe_intermediate_size // 4, layer.hidden_size], dtype="uint8")
        for _ in range(layer.num_local_experts)
    ]
    state = {"up": up, "down": down}
    for idx in range(layer.num_local_experts):
        state.update(
            {
                f"up_scale_{idx}": paddle.ones(
                    [layer.hidden_size // 128, layer.moe_intermediate_size * 2], dtype="uint8"
                ),
                f"down_scale_{idx}": paddle.ones(
                    [layer.moe_intermediate_size // 128, layer.hidden_size], dtype="uint8"
                ),
                f"up_super_{idx}": paddle.ones([layer.moe_intermediate_size * 2], dtype=super_dtype),
                f"down_super_{idx}": paddle.ones([layer.hidden_size], dtype=super_dtype),
                f"up_code_scale_{idx}": paddle.ones([layer.moe_intermediate_size * 2], dtype="float32"),
                f"down_code_scale_{idx}": paddle.ones([layer.hidden_size], dtype="float32"),
                f"up_code_zp_{idx}": paddle.ones([layer.moe_intermediate_size * 2], dtype="float32"),
                f"down_code_zp_{idx}": paddle.ones([layer.hidden_size], dtype="float32"),
            }
        )
    return state


def test_wint2_paths(monkeypatch):
    quant_config = SimpleNamespace(moe_quant_type="w4w2")
    layer = _DummyLayer()

    cutlass_method = wint2_backend.CutlassWint2FusedMoeMethod(quant_config)
    prev_dtype = paddle.get_default_dtype()
    paddle.set_default_dtype("float16")
    cutlass_method.create_weights(layer)
    paddle.set_default_dtype(prev_dtype)
    up, down = _make_state_dict(layer)["up"], _make_state_dict(layer)["down"]
    cutlass_method.check(layer, up, down)
    wint2_backend.Wint2MoeMethod.process_loaded_weights(cutlass_method, layer, None)
    cutlass_method.process_loaded_weights(layer, None)
    cutlass_method.process_prequanted_weights(layer, _make_state_dict(layer))

    gate = paddle.nn.Linear(layer.hidden_size, layer.num_experts, bias_attr=False)
    x = paddle.ones([2, layer.hidden_size], dtype="float16")
    monkeypatch.setattr(
        wint2_backend,
        "moe_expert_reduce",
        lambda _ffn_out, *_args, **_kwargs: paddle.zeros([x.shape[0], layer.hidden_size], dtype=x.dtype),
    )
    out = cutlass_method.apply(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out.shape == [2, layer.hidden_size]

    triton_method = wint2_backend.TritonWint2FusedMoeMethod(quant_config)
    triton_method.create_weights(layer)
    triton_method.process_prequanted_weights(layer, _make_state_dict(layer))
    out_triton = triton_method.apply(layer, x, gate, topk_ids_hookfunc=lambda **_k: None)
    assert out_triton.shape == [2, layer.hidden_size]
