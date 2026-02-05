import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.nn.functional import swiglu
from paddle.nn.quant import weight_only_linear, weight_quantize

from fastdeploy.model_executor.ops.gpu import (
    fused_expert_moe,
    moe_expert_dispatch,
    moe_expert_ffn,
    moe_expert_reduce,
)

paddle.seed(2025)
np.random.seed(2025)


class Expert(nn.Layer):
    def __init__(self, d_model, d_feedforward, dtype="bfloat16", quant_type="weight_only_int8"):
        super().__init__()

        self.dtype = dtype
        self.quant_type = quant_type
        self.fc0 = nn.Linear(d_model, d_feedforward * 2)
        self.fc1 = nn.Linear(d_feedforward, d_model)

        self.w0_quanted, self.s0 = weight_quantize(self.fc0.weight, quant_type, arch=80, group_size=-1)
        self.w1_quanted, self.s1 = weight_quantize(self.fc1.weight, quant_type, arch=80, group_size=-1)

    def load_weight(self, ffn0_gate_proj_weight, ffn0_up_proj_weight, ffn1_down_proj_weight):
        concated_gate_up_weight = np.concatenate([ffn0_gate_proj_weight, ffn0_up_proj_weight], axis=-1)
        ffn0_weight = paddle.to_tensor(concated_gate_up_weight).cast(self.dtype)
        ffn1_weight = paddle.to_tensor(ffn1_down_proj_weight).cast(self.dtype)

        self.fc0.weight.set_value(ffn0_weight)
        self.fc1.weight.set_value(ffn1_weight)

        self.w0_quanted, self.s0 = weight_quantize(ffn0_weight, algo=self.quant_type, arch=80, group_size=-1)
        self.w1_quanted, self.s1 = weight_quantize(ffn1_weight, algo=self.quant_type, arch=80, group_size=-1)

    def set_value(self, ffn0_weight, ffn1_weight):
        self.fc0.weight.set_value(ffn0_weight)
        self.fc1.weight.set_value(ffn1_weight)

        self.w0_quanted, self.s0 = weight_quantize(self.fc0.weight, self.quant_type, arch=80, group_size=-1)
        self.w1_quanted, self.s1 = weight_quantize(self.fc1.weight, self.quant_type, arch=80, group_size=-1)

    def forward(self, x):
        x = self.fc0(x)
        x = swiglu(x)
        return self.fc1(x)

    def forward_quant(self, x):
        x = weight_only_linear(x, self.w0_quanted.T, weight_scale=self.s0)
        x = swiglu(x)
        return weight_only_linear(x, self.w1_quanted.T, weight_scale=self.s1)


class FusedMoe:
    def __init__(
        self, input_shape: list, d_feedforward, num_experts, top_k, dtype, quant_type="None", rtol=1e-2, atol=1e-2
    ) -> None:
        self.batch_size, self.seq_len, self.d_model = input_shape
        self.d_feedforward = d_feedforward
        self.num_experts = num_experts
        self.top_k = top_k
        self.dtype = dtype
        self.quant_type = quant_type
        self.rtol = rtol
        self.atol = atol

        self._init_parameters()
        self._prepare_data()

    def _init_parameters(self):
        # 创建专家层
        self.experts = nn.LayerList(
            [Expert(self.d_model, self.d_feedforward, self.dtype, self.quant_type) for _ in range(self.num_experts)]
        )

        # 初始化门控权重
        self.gate = nn.Linear(self.d_model, self.num_experts)
        self.gate_weight = self.gate.weight.cast("float32")

    def _prepare_data(self):
        """准备输入数据"""
        self.x = paddle.randn([self.batch_size, self.seq_len, self.d_model], dtype=self.dtype)

        self.s0 = None
        self.s1 = None
        if self.quant_type == "weight_only_int8":
            self.w0 = paddle.stack([e.w0_quanted for e in self.experts], axis=0).transpose([0, 2, 1]).astype("int8")
            self.w1 = paddle.stack([e.w1_quanted for e in self.experts], axis=0).transpose([0, 2, 1]).astype("int8")
            self.s0 = paddle.stack([e.s0 for e in self.experts], axis=0).astype(self.dtype)
            self.s1 = paddle.stack([e.s1 for e in self.experts], axis=0).astype(self.dtype)
        else:
            self.w0 = paddle.stack([e.fc0.weight for e in self.experts], axis=0).astype(self.dtype)
            self.w1 = paddle.stack([e.fc1.weight for e in self.experts], axis=0).astype(self.dtype)

        self.b0 = (
            paddle.stack([e.fc0.bias for e in self.experts], axis=0)
            .reshape([self.num_experts, 1, -1])
            .astype(self.dtype)
        )
        self.b1 = (
            paddle.stack([e.fc1.bias for e in self.experts], axis=0)
            .reshape([self.num_experts, 1, -1])
            .astype(self.dtype)
        )

    def baseline_forward(self, hidden_states):
        """（逐个专家计算）"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # 路由计算
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        weights = F.softmax(logits, axis=-1)
        routing_weights, selected_experts = paddle.topk(weights, self.top_k, axis=-1)
        # 结果累加
        final_hidden_states = paddle.zeros_like(hidden_states)

        expert_mask = paddle.transpose(F.one_hot(selected_experts, num_classes=self.num_experts), [2, 1, 0])

        for expert_id in range(self.num_experts):
            expert_layer = self.experts[expert_id]
            idx, top_x = paddle.where(expert_mask[expert_id])

            current_state = paddle.index_select(hidden_states, top_x, axis=0).reshape([-1, hidden_dim])
            if self.quant_type == "None":
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx].view([-1, 1])
            else:
                current_hidden_states = expert_layer.forward_quant(current_state) * routing_weights[top_x, idx].view(
                    [-1, 1]
                )

            paddle.index_add_(
                x=final_hidden_states,
                index=top_x,
                axis=0,
                value=current_hidden_states.to(hidden_states.dtype),
            )
        final_hidden_states = paddle.reshape(final_hidden_states, [batch_size, seq_len, hidden_dim])
        return final_hidden_states

    def fused_forward(self, x):
        """测试融合实现"""
        return fused_expert_moe(
            x,
            self.gate_weight,
            self.w0,
            self.w1,
            self.b0,
            None if self.quant_type == "None" else self.s0,
            self.b1,
            None if self.quant_type == "None" else self.s1,
            self.quant_type,
            self.top_k,
            False,
            False,
        )

    def split_forward(self, hidden_states):
        """测试拆分实现"""
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape([-1, hidden_dim])

        # 路由计算
        logits = paddle.matmul(hidden_states.cast("float32"), self.gate_weight)
        scores = F.softmax(logits, axis=-1)
        (
            permute_input,
            token_nums_per_expert,
            permute_indices_per_token,
            top_k_weights,
            top_k_indices,
            expert_idx_per_token,
        ) = moe_expert_dispatch(hidden_states, scores, None, None, self.top_k, False, self.quant_type, True)

        expert_idx_per_token = None

        ffn_out = moe_expert_ffn(
            permute_input,
            token_nums_per_expert,
            self.w0,
            self.w1,
            self.b0,
            None if self.quant_type == "None" else self.s0,
            None if self.quant_type == "None" else self.s1,
            expert_idx_per_token,
            self.quant_type,
        )
        output = moe_expert_reduce(
            ffn_out,
            top_k_weights,
            permute_indices_per_token,
            top_k_indices,
            None,
            norm_topk_prob=False,
            routed_scaling_factor=1.0,
        )
        output = paddle.reshape(output, [batch_size, seq_len, hidden_dim])
        return output

    def test_consistency(self):
        base_out = self.baseline_forward(self.x)
        split_out = self.split_forward(self.x)
        fused_out = self.fused_forward(self.x)

        np.testing.assert_allclose(
            split_out.cast("float32").numpy().astype("float32"),
            base_out.cast("float32").numpy().astype("float32"),
            rtol=self.rtol,
            atol=self.atol,
        )
        np.testing.assert_allclose(
            base_out.cast("float32").numpy().astype("float32"),
            fused_out.cast("float32").numpy().astype("float32"),
            rtol=self.rtol,
            atol=self.atol,
        )


class TestMetaxFusedMoe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Class-level setup that runs once before all tests."""
        cls.set_config()
        paddle.set_default_dtype(cls.dtype)

    @classmethod
    def set_config(cls):
        """Set the configuration parameters for the test."""
        cls.dtype = "bfloat16"
        cls.supported_quant_type = ["weight_only_int8"]

        batch_size_list = [1]
        seq_len_list = [1, 128, 256, 512, 1024, 2048]
        d_model_list = [[7168, 128]]
        num_experts_list = [256]
        top_k_list = [8]

        cls.test_params = []
        for batch_size in batch_size_list:
            for seq_len in seq_len_list:
                for d_model in d_model_list:
                    for num_experts in num_experts_list:
                        for top_k in top_k_list:
                            if top_k >= num_experts:
                                continue
                            cls.test_params.append(
                                {
                                    "input_shape": [batch_size, seq_len, d_model[0]],
                                    "d_feedforward": d_model[1],
                                    "num_experts": num_experts,
                                    "top_k": top_k,
                                }
                            )

    def setUp(self):
        """Test-level setup that runs before each test."""
        pass

    def test_bfloat16_wint8_quant(self):
        rtol = 1e-2
        atol = 1e-2
        quant_type = "weight_only_int8"
        assert quant_type in self.supported_quant_type

        for param in self.test_params:
            fused_moe_test = FusedMoe(
                param["input_shape"],
                param["d_feedforward"],
                param["num_experts"],
                param["top_k"],
                self.dtype,
                quant_type,
                rtol,
                atol,
            )
            fused_moe_test.test_consistency()

    # def test_bfloat16_without_quant(self):
    #     quant_type = None
    #     assert quant_type in self.supported_quant_type

    #     for param in self.test_params:
    #         fused_moe_test = FusedMoe(param['input_shape'], param['d_feedforward'], param['num_experts'], param['top_k'], self.dtype, quant_type, self.rtol, self.atol)
    #         fused_moe_test.test_consistency()


if __name__ == "__main__":
    unittest.main()
