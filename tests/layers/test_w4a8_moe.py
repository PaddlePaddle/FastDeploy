import json
import os
import shutil
import unittest

import numpy as np
import paddle
import paddle.device.cuda.graphs as graphs
from paddle.distributed import fleet

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.w4a8 import W4A8Config
from fastdeploy.scheduler import SchedulerConfig

paddle.set_default_dtype("bfloat16")


class FuseMoEWrapper(paddle.nn.Layer):
    def __init__(
        self,
        model_config: ModelConfig,
        tp_size: int = 1,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
        prefix: str = "layer0",
        nnodes: int = 1,
    ):
        super().__init__()
        self.model_config = model_config

        self.tp_size = tp_size
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        self.prefix = prefix
        self.fd_config = FDConfig(
            model_config=self.model_config,
            parallel_config=ParallelConfig(
                {
                    "tensor_parallel_size": self.tp_size,
                    "expert_parallel_size": self.ep_size,
                    "expert_parallel_rank": self.ep_rank,
                    "data_parallel_size": self.ep_size,
                }
            ),
            quant_config=W4A8Config(is_permuted=False, hadamard_block_size=128),
            # quant_config=W4AFP8Config(weight_scale_dict=None, act_scale_dict=None, is_permuted=False, hadamard_block_size=128),
            scheduler_config=SchedulerConfig({}),
            cache_config=CacheConfig({}),
            graph_opt_config=GraphOptimizationConfig({}),
            load_config=LoadConfig({}),
            ips=",".join(["0"] * nnodes),
        )
        self.fd_config.parallel_config.tp_group = None
        self.fd_config.parallel_config.tensor_parallel_rank = tp_rank
        self.fd_config.parallel_config.expert_parallel_size = self.ep_size
        if self.ep_size > 1:
            self.fd_config.parallel_config.ep_group = fleet.get_hybrid_communicate_group().get_model_parallel_group()
            self.fd_config.scheduler_config.splitwise_role = "mixed"
            self.fd_config.model_config.moe_phase.phase = "decode"

        weight_key_map = {
            "gate_weight_key": f"{self.prefix}.gate.weight",
            "gate_correction_bias_key": f"{self.prefix}.moe_statics.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.down_proj.weight",
            "up_gate_proj_expert_weight_scale_key": f"{self.prefix}.experts.{{}}.up_gate_proj.weight_scale",
            "down_proj_expert_weight_scale_key": f"{self.prefix}.experts.{{}}.down_proj.weight_scale",
            "up_gate_proj_expert_in_scale_key": f"{self.prefix}.experts.{{}}.up_gate_proj.activation_scale",
            "down_proj_expert_in_scale_key": f"{self.prefix}.experts.{{}}.down_proj.activation_scale",
        }

        self.fused_moe = FusedMoE(
            fd_config=self.fd_config,
            moe_intermediate_size=self.fd_config.model_config.moe_intermediate_size,
            num_experts=self.fd_config.model_config.moe_num_experts,
            top_k=self.fd_config.model_config.moe_k,
            # avoiding invoke clean_low_latency_buffer in mixed ep.
            layer_idx=666,
            weight_key_map=weight_key_map,
            topk_method="noaux_tc",
            topk_group=4,
            n_group=8,
            gate_correction_bias=paddle.zeros([self.fd_config.model_config.moe_num_experts], paddle.float32),
            # gate_correction_bias = gate_correction_bias_real_data
        )
        self.pack_num = 2
        moe_layer = self.fused_moe

        up_gate_proj_weight_shape = [
            moe_layer.num_local_experts,
            moe_layer.hidden_size // self.pack_num,
            moe_layer.moe_intermediate_size * 2,
        ]
        down_proj_weight_shape = [
            moe_layer.num_local_experts,
            moe_layer.moe_intermediate_size // self.pack_num,
            moe_layer.hidden_size,
        ]
        up_gate_proj_weight_scale_shape = [
            moe_layer.num_local_experts,
            moe_layer.moe_intermediate_size * 2,
        ]
        down_proj_weight_scale_shape = [
            moe_layer.num_local_experts,
            moe_layer.hidden_size,
        ]

        up_gate_proj_weight = (paddle.randn(up_gate_proj_weight_shape, paddle.bfloat16) * 100).cast(paddle.int8)
        down_proj_weight = (paddle.randn(down_proj_weight_shape, paddle.bfloat16) * 100).cast(paddle.int8)

        up_gate_proj_weight_scale = paddle.randn(up_gate_proj_weight_scale_shape, paddle.bfloat16)
        down_proj_weight_scale = paddle.randn(down_proj_weight_scale_shape, paddle.bfloat16)

        up_gate_proj_in_scale = paddle.randn([self.fd_config.model_config.moe_num_experts, 1], paddle.float32)
        down_proj_in_scale = paddle.randn([self.fd_config.model_config.moe_num_experts, 1], paddle.float32)

        local_expert_ids = list(
            range(moe_layer.expert_id_offset, moe_layer.expert_id_offset + moe_layer.num_local_experts)
        )
        state_dict = {}
        up_gate_proj_expert_weight_key = moe_layer.weight_key_map.get("up_gate_proj_expert_weight_key")
        up_gate_proj_expert_weight_scale_key = moe_layer.weight_key_map.get("up_gate_proj_expert_weight_scale_key")
        up_gate_proj_expert_in_scale_key = moe_layer.weight_key_map.get("up_gate_proj_expert_in_scale_key")
        down_proj_expert_weight_key = moe_layer.weight_key_map.get("down_proj_expert_weight_key")
        down_proj_expert_weight_scale_key = moe_layer.weight_key_map.get("down_proj_expert_weight_scale_key")
        down_proj_expert_in_scale_key = moe_layer.weight_key_map.get("down_proj_expert_in_scale_key")

        for expert_idx in local_expert_ids:
            up_gate_proj_expert_weight_key_name = up_gate_proj_expert_weight_key.format(expert_idx)
            up_gate_proj_expert_weight_scale_key_name = up_gate_proj_expert_weight_scale_key.format(expert_idx)
            down_proj_expert_weight_key_name = down_proj_expert_weight_key.format(expert_idx)
            down_proj_expert_weight_scale_key_name = down_proj_expert_weight_scale_key.format(expert_idx)

            state_dict[up_gate_proj_expert_weight_key_name] = up_gate_proj_weight[
                expert_idx - moe_layer.expert_id_offset
            ]
            state_dict[up_gate_proj_expert_weight_scale_key_name] = up_gate_proj_weight_scale[
                expert_idx - moe_layer.expert_id_offset
            ]
            state_dict[down_proj_expert_weight_key_name] = down_proj_weight[expert_idx - moe_layer.expert_id_offset]
            state_dict[down_proj_expert_weight_scale_key_name] = down_proj_weight_scale[
                expert_idx - moe_layer.expert_id_offset
            ]

        for expert_idx in range(self.fd_config.model_config.moe_num_experts):
            up_gate_proj_expert_in_scale_key_name = up_gate_proj_expert_in_scale_key.format(expert_idx)
            down_proj_expert_in_scale_key_name = down_proj_expert_in_scale_key.format(expert_idx)
            state_dict[up_gate_proj_expert_in_scale_key_name] = up_gate_proj_in_scale[expert_idx]
            state_dict[down_proj_expert_in_scale_key_name] = down_proj_in_scale[expert_idx]

        moe_layer.load_state_dict(state_dict)


class TestW4A8FusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.architectures = ["Ernie4_5_MoeForCausalLM"]
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
                "max_model_len": 2048,
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
            "dtype": "bfloat16",
        }

        tmp_dir = f"./tmpwedfewfef{paddle.distributed.get_rank()}"
        os.makedirs(tmp_dir, exist_ok=True)
        with open(f"./{tmp_dir}/config.json", "w") as f:
            json.dump(config_dict, f)
        self.model_name_or_path = os.path.join(os.getcwd(), tmp_dir)
        return self.model_name_or_path

    def test_fused_moe(self):
        # init_distributed_environment()

        gating = paddle.nn.Linear(self.model_config.hidden_size, self.model_config.moe_num_experts)
        gating.to(dtype=paddle.float32)  # it's dtype is bfloat16 default, but the forward input is float32
        gating.weight.set_value(paddle.rand(gating.weight.shape, dtype=paddle.float32))

        # ep_size = paddle.distributed.get_world_size()
        # ep_rank = paddle.distributed.get_rank()
        ep_size = 1
        ep_rank = 0

        tp_rank = 0
        tp_size = 1

        nnodes = (ep_size + 7) // 8

        # 这行代码必须保留，否则影响均匀性！
        paddle.seed(ep_rank + 100)

        num_layers = 54
        real_weight_layers = 9
        fused_moe = [None] * real_weight_layers
        for i in range(real_weight_layers):
            fused_moe[i] = FuseMoEWrapper(self.model_config, tp_size, tp_rank, ep_size, ep_rank, nnodes=nnodes)

        moe_cuda_graphs = [None] * 100
        cache_hidden_states = [None] * 100
        test_token_nums = [10, 20, 40, 60, 80, 100, 128, 160, 192, 256]
        # test_token_nums = [1024 * i for i in [1,2,4,8,16,32]]
        for idx, num_tokens in enumerate(test_token_nums):

            cache_hidden_states[idx] = paddle.rand((num_tokens, self.model_config.hidden_size), dtype=paddle.bfloat16)

            def fake_model_run():
                for j in range(num_layers):
                    out = fused_moe[j % real_weight_layers].fused_moe(cache_hidden_states[idx], gating)

                return out

            fake_model_run()
            moe_cuda_graphs[idx] = graphs.CUDAGraph()
            moe_cuda_graphs[idx].capture_begin()

            fake_model_run()

            moe_cuda_graphs[idx].capture_end()

            num_tests = 20
            start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            for i in range(num_tests):
                start_events[i].record()

                moe_cuda_graphs[idx].replay()

                end_events[i].record()
            paddle.device.cuda.synchronize()

            times = np.array([round(s.elapsed_time(e), 1) for s, e in zip(start_events, end_events)])[1:]
            print("num_token:", num_tokens)
            print(times[-5:])
            rdma_GB = 3.0 * num_tokens * self.moe_k * self.hidden_size / (1e9)
            times_s = (times[-1] / num_layers) / (1e3)
            print(times[-1], round(rdma_GB / times_s, 1))

            tmp_layer = fused_moe[0].fused_moe
            memory_GB = (
                tmp_layer.num_local_experts
                * tmp_layer.hidden_size
                * tmp_layer.moe_intermediate_size
                * 3
                / (1e9)
                * num_layers
            )
            print(round(memory_GB / times[-1], 1), "TB/s")

        shutil.rmtree(self.model_name_or_path)


if __name__ == "__main__":
    unittest.main()
