import json
import os
import shutil
import unittest

import numpy as np
import paddle
import paddle.device.cuda.graphs as graphs
import paddle.profiler as profiler

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.moe.moe import FusedMoE
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WINT8Config,
)
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.worker.worker_process import init_distributed_environment

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
            # quant_config=BlockWiseFP8Config(weight_block_size=[64, 64]),
            quant_config=WINT8Config({}),
            scheduler_config=SchedulerConfig({}),
            cache_config=CacheConfig({}),
            graph_opt_config=GraphOptimizationConfig({}),
            load_config=LoadConfig({}),
            ips="0,0,0,0",
        )
        self.fd_config.parallel_config.tp_group = None
        self.fd_config.parallel_config.tensor_parallel_rank = tp_rank
        self.fd_config.parallel_config.expert_parallel_size = self.ep_size
        self.fd_config.parallel_config.ep_group = paddle.distributed.new_group()

        weight_key_map = {
            "gate_weight_key": f"{self.prefix}.gate.weight",
            "gate_correction_bias_key": f"{self.prefix}.moe_statics.e_score_correction_bias",
            "up_gate_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.up_gate_proj.weight",
            "down_proj_expert_weight_key": f"{self.prefix}.experts.{{}}.down_proj.weight",
        }

        self.fused_moe = FusedMoE(
            fd_config=self.fd_config,
            moe_intermediate_size=self.fd_config.model_config.moe_intermediate_size,
            num_experts=self.fd_config.model_config.moe_num_experts,
            top_k=self.fd_config.model_config.moe_k,
            layer_idx=0,
            weight_key_map=weight_key_map,
        )
        moe_layer = self.fused_moe

        paddle.seed(1024)
        up_gate_proj_weight_shape = [
            moe_layer.num_local_experts,
            moe_layer.hidden_size,
            moe_layer.moe_intermediate_size * 2,
        ]
        down_proj_weight_shape = [
            moe_layer.num_local_experts,
            moe_layer.moe_intermediate_size,
            moe_layer.hidden_size,
        ]

        up_gate_proj_weight = paddle.randn(up_gate_proj_weight_shape, paddle.bfloat16)
        down_proj_weight = paddle.randn(down_proj_weight_shape, paddle.bfloat16)

        local_expert_ids = list(
            range(moe_layer.expert_id_offset, moe_layer.expert_id_offset + moe_layer.num_local_experts)
        )
        state_dict = {}
        up_gate_proj_expert_weight_key = moe_layer.weight_key_map.get("up_gate_proj_expert_weight_key")
        down_proj_expert_weight_key = moe_layer.weight_key_map.get("down_proj_expert_weight_key")
        for expert_idx in local_expert_ids:
            down_proj_expert_weight_key_name = down_proj_expert_weight_key.format(expert_idx)
            up_gate_proj_expert_weight_key_name = up_gate_proj_expert_weight_key.format(expert_idx)
            state_dict[up_gate_proj_expert_weight_key_name] = up_gate_proj_weight[
                expert_idx - moe_layer.expert_id_offset
            ]
            state_dict[down_proj_expert_weight_key_name] = down_proj_weight[expert_idx - moe_layer.expert_id_offset]

        moe_layer.load_state_dict(state_dict)


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.architectures = ["Ernie4_5_MoeForCausalLM"]
        self.num_tokens = 96
        self.hidden_size = 7168
        self.moe_intermediate_size = 3584
        self.moe_num_experts = 48
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
        os.makedirs("tmp", exist_ok=True)
        with open("./tmp/config.json", "w") as f:
            json.dump(config_dict, f)
        self.model_name_or_path = os.path.join(os.getcwd(), "tmp")
        return self.model_name_or_path

    def clear_tmp(self):
        if os.path.exists(self.model_name_or_path):
            shutil.rmtree(self.model_name_or_path)

    def test_fused_moe(self):
        init_distributed_environment()

        hidden_states = paddle.rand((self.num_tokens, self.model_config.hidden_size), dtype=paddle.bfloat16)
        gating = paddle.nn.Linear(self.model_config.hidden_size, self.model_config.moe_num_experts)
        gating.to(dtype=paddle.float32)  # it's dtype is bfloat16 default, but the forward input is float32
        gating.weight.set_value(paddle.rand(gating.weight.shape, dtype=paddle.float32))

        # os.environ["FD_USE_DEEP_GEMM"] = "1"  # use deepgemm
        ep_size = paddle.distributed.get_world_size()
        ep_rank = paddle.distributed.get_rank()

        tp_rank = 0
        tp_size = 1

        fused_moe = FuseMoEWrapper(self.model_config, tp_size, tp_rank, ep_size, ep_rank)

        zkk_cuda_graph = graphs.CUDAGraph()
        zkk_cuda_graph.capture_begin()

        out = fused_moe.fused_moe(hidden_states, gating)
        zkk_cuda_graph.capture_end()

        p = profiler.Profiler(
            targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
            on_trace_ready=profiler.export_chrome_tracing("./profile_log"),
        )
        p.start()

        num_tests = 20

        start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
        for i in range(num_tests):
            # Record
            start_events[i].record()

            # zkk_cuda_graph.replay()
            out = fused_moe.fused_moe(hidden_states, gating)

            end_events[i].record()
        paddle.device.cuda.synchronize()

        times = np.array([s.elapsed_time(e) for s, e in zip(start_events, end_events)])[1:]

        print(times)

        p.stop()

        # for i in range(10):
        #     out = fused_moe.fused_moe(hidden_states, gating)
        #     print(out)

        if paddle.distributed.get_rank == 0:
            self.clear_tmp()
        return out


if __name__ == "__main__":
    unittest.main()
