import json
import os
import shutil
import unittest

import paddle

from fastdeploy.config import (
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.block_wise_fp8 import (
    BlockWiseFP8Config,
)
from fastdeploy.worker.worker_process import init_distributed_environment


class FuseMoEWrapper(paddle.nn.Layer):
    def __init__(
        self,
        model_config: ModelConfig,
        tp_size: int = 1,
        ep_size: int = 1,
        ep_rank: int = 0,
        prefix: str = "layer0",
        use_deepgemm: bool = True,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.model_config = model_config
        self.prefix = prefix
        self.use_deepgemm = use_deepgemm
        self.fd_config = self.build_fd_config(use_deepgemm)
        self.fused_moe = self.build_fuse_moe()
        self.random_init_weights(dtype=paddle.bfloat16)

    def random_init_weights(self, dtype):
        paddle.seed(1024)
        # self.fused_moe.init_moe_weights()
        up_gate_proj_weight_shape = [
            self.fused_moe.num_local_experts,
            self.model_config.hidden_size,
            self.model_config.moe_intermediate_size * 2,
        ]
        down_proj_weight_shape = [
            self.fused_moe.num_local_experts,
            self.model_config.moe_intermediate_size,
            self.model_config.hidden_size,
        ]
        self.fused_moe.up_gate_proj_weight = self.fused_moe.create_parameter(
            shape=up_gate_proj_weight_shape,
            dtype=paddle.bfloat16,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        self.fused_moe.down_proj_weight = self.fused_moe.create_parameter(
            shape=down_proj_weight_shape,
            dtype=paddle.bfloat16,
            default_initializer=paddle.nn.initializer.Constant(0),
        )
        self.fused_moe.up_gate_proj_weight.set_value(
            paddle.rand(self.fused_moe.up_gate_proj_weight.shape, dtype=dtype) / 100
        )
        self.fused_moe.down_proj_weight.set_value(
            paddle.rand(self.fused_moe.down_proj_weight.shape, dtype=dtype) / 100
        )
        if self.fd_config.quant_config:
            state_dict = self.build_state_dict(self.fused_moe.up_gate_proj_weight, self.fused_moe.down_proj_weight)
            self.fused_moe.quant_method.create_weights(self.fused_moe, state_dict)

    def build_fd_config(self, use_deepgemm) -> FDConfig:
        return FDConfig(
            model_config=self.model_config,
            parallel_config=ParallelConfig(
                {
                    "tensor_parallel_size": self.tp_size,
                    "expert_parallel_size": self.ep_size,
                    "expert_parallel_rank": self.ep_rank,
                    "data_parallel_size": self.ep_size,
                }
            ),
            quant_config=BlockWiseFP8Config(weight_block_size=[64, 64]) if use_deepgemm else None,
            graph_opt_config=GraphOptimizationConfig({}),
            load_config=LoadConfig({}),
        )

    def build_fuse_moe(self) -> FusedMoE:
        weight_key_map = {
            "gate_weight_key": f"{self.prefix}.gate.weight",
            "gate_correction_bias_key": f"{self.prefix}.moe_statics.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.down_proj.weight",
        }

        fused_moe = FusedMoE(
            fd_config=self.fd_config,
            moe_intermediate_size=self.fd_config.model_config.moe_intermediate_size,
            num_experts=self.fd_config.model_config.moe_num_experts,
            top_k=self.fd_config.model_config.moe_k,
            layer_idx=0,
            weight_key_map=weight_key_map,
        )
        return fused_moe

    def build_state_dict(self, up_weights: list[paddle.Tensor], down_weights: list[paddle.Tensor]):
        local_expert_ids = list(
            range(self.fused_moe.expert_id_offset, self.fused_moe.expert_id_offset + self.fused_moe.num_local_experts)
        )
        state_dict = {}
        up_gate_proj_expert_weight_key = self.fused_moe.weight_key_map.get("up_gate_proj_expert_weight_key")
        down_proj_expert_weight_key = self.fused_moe.weight_key_map.get("down_proj_expert_weight_key")
        for expert_idx in local_expert_ids:
            down_proj_expert_weight_key_name = down_proj_expert_weight_key.format(expert_idx)
            up_gate_proj_expert_weight_key_name = up_gate_proj_expert_weight_key.format(expert_idx)
            state_dict[up_gate_proj_expert_weight_key_name] = up_weights[expert_idx - self.fused_moe.expert_id_offset]
            state_dict[down_proj_expert_weight_key_name] = down_weights[expert_idx - self.fused_moe.expert_id_offset]
        return state_dict

    def forward(self, hidden_states: paddle.Tensor, gating: paddle.nn.Layer):
        return self.fused_moe.forward(hidden_states, gating)


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.architectures = ["Ernie4_5_MoeForCausalLM"]
        self.num_tokens = 64
        self.hidden_size = 8192
        self.moe_intermediate_size = 3584
        self.moe_num_experts = 64
        self.moe_k = 8
        self.hidden_act = "silu"
        self.num_attention_heads = 64
        self.model_config = self.build_model_config()

    def build_model_config(self) -> ModelConfig:
        model_name_or_path = self.build_config_json()
        return ModelConfig(
            {
                "model": model_name_or_path,
            }
        )

    def build_config_json(self) -> str:
        config_dict = {
            "architectures": self.architectures,
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "moe_num_experts": self.moe_num_experts,
            "moe_k": self.moe_k,
            "hidden_act": self.hidden_act,
            "num_attention_heads": self.num_attention_heads,
        }
        os.makedirs("tmp", exist_ok=True)
        with open("./tmp/config.json", "w") as f:
            json.dump(config_dict, f)
        self.model_name_or_path = os.path.join(os.getcwd(), "tmp")
        return self.model_name_or_path

    def clear_tmp(self):
        if os.path.exists(self.model_name_or_path):
            shutil.rmtree(self.model_name_or_path)

    def test_fused_moe(self):
        ep_size, ep_rank = init_distributed_environment()
        hidden_states = paddle.rand((self.num_tokens, self.model_config.hidden_size), dtype=paddle.bfloat16)
        gating = paddle.nn.Linear(
            in_features=self.model_config.hidden_size, out_features=self.model_config.moe_num_experts
        )
        gating.to(dtype=paddle.float32)  # it's dtype is bfloat16 default, but the forward input is float32
        gating.weight.set_value(paddle.rand(gating.weight.shape, dtype=paddle.float32))
        os.environ["FD_USE_DEEP_GEMM"] = "1"  # use deepgemm
        fused_moe = FuseMoEWrapper(self.model_config, 1, ep_size, ep_rank)
        out = fused_moe.forward(hidden_states, gating)
        if ep_rank == 0:
            self.clear_tmp()
        return out


if __name__ == "__main__":
    unittest.main()
