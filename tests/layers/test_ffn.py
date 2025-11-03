import json
import os
import shutil
import unittest

import numpy as np
import paddle
import paddle.device.cuda.graphs as graphs

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
)
from fastdeploy.model_executor.layers.quantization.block_wise_fp8 import (
    BlockWiseFP8Config,
)
from fastdeploy.model_executor.models.ernie4_5_moe import Ernie4_5_MLP
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.worker.worker_process import init_distributed_environment

paddle.set_default_dtype("bfloat16")


class FFNWrapper(paddle.nn.Layer):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.model_config = model_config

        self.intermediate_size = 3584
        self.hidden_size = self.model_config.hidden_size
        self.prefix = "hahahha"
        self.fd_config = FDConfig(
            model_config=self.model_config,
            parallel_config=ParallelConfig(
                {
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "expert_parallel_rank": 0,
                    "data_parallel_size": 1,
                }
            ),
            quant_config=BlockWiseFP8Config(weight_block_size=[128, 128]),
            # quant_config = WINT8Config({}),
            scheduler_config=SchedulerConfig({}),
            cache_config=CacheConfig({}),
            graph_opt_config=GraphOptimizationConfig({}),
            load_config=LoadConfig({}),
            ips="0.0.0.0",
        )
        self.fd_config.parallel_config.tp_group = None
        self.fd_config.parallel_config.tensor_parallel_rank = 0
        self.fd_config.parallel_config.tensor_parallel_size = 1

        self.ffn = Ernie4_5_MLP(
            fd_config=self.fd_config,
            intermediate_size=self.intermediate_size,
            prefix=self.prefix,
        )

        up_gate_proj_weight_shape = [self.hidden_size, self.intermediate_size * 2]
        down_proj_weight_shape = [self.intermediate_size, self.hidden_size]

        up_gate_proj_weight = paddle.randn(up_gate_proj_weight_shape, paddle.bfloat16)
        down_proj_weight = paddle.randn(down_proj_weight_shape, paddle.bfloat16)

        state_dict = {
            f"{self.prefix}.up_gate_proj.weight": up_gate_proj_weight,
            f"{self.prefix}.down_proj.weight": down_proj_weight,
        }

        self.ffn.load_state_dict(state_dict)


class TestFusedMoE(unittest.TestCase):
    def setUp(self) -> None:
        self.architectures = ["Ernie4_5_MoeForCausalLM"]
        self.hidden_size = 7168
        self.moe_intermediate_size = 1
        self.moe_num_experts = 1
        self.moe_k = 1
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

        tmp_dir = f"./tmpefef{paddle.distributed.get_rank()}"
        os.makedirs(tmp_dir, exist_ok=True)
        with open(f"./{tmp_dir}/config.json", "w") as f:
            json.dump(config_dict, f)
        self.model_name_or_path = os.path.join(os.getcwd(), tmp_dir)
        return self.model_name_or_path

    def test_ffn(self):
        init_distributed_environment()

        ffn = FFNWrapper(self.model_config)

        # (ZKK): disable this test,
        # CI machine does not support deepgemm blockwise_fp8, compilation error.
        return

        moe_cuda_graphs = [None] * 100
        cache_hidden_states = [None] * 100
        for idx, num_tokens in enumerate([10, 20, 40, 60, 80, 100, 128, 160, 192, 256, 512, 1024, 2048, 4096]):

            cache_hidden_states[idx] = paddle.rand((num_tokens, self.model_config.hidden_size), dtype=paddle.bfloat16)

            moe_cuda_graphs[idx] = graphs.CUDAGraph()
            moe_cuda_graphs[idx].capture_begin()

            num_layers = 80
            for _ in range(num_layers):
                out = ffn.ffn(cache_hidden_states[idx])

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
            print("num_tokens:", num_tokens)
            print(times[-5:])

            flops = num_layers * 2 * num_tokens * self.model_config.hidden_size * ffn.intermediate_size * 3
            memory = num_layers * self.model_config.hidden_size * ffn.intermediate_size * 3
            # memory += (num_layers * num_tokens * ffn.intermediate_size * 2)

            print(round(flops / times[-1] / (1024**3), 1), "TFLOPS")

            print(round(memory / times[-1] / (1024**3), 1), "TB/s")

        shutil.rmtree(self.model_name_or_path)
        return out


if __name__ == "__main__":
    unittest.main()
