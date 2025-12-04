"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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
from fastdeploy.model_executor.layers.quantization.block_wise_fp8 import (
    BlockWiseFP8Config,
)
from fastdeploy.scheduler import SchedulerConfig
from fastdeploy.worker.worker_process import init_distributed_environment

paddle.set_default_dtype("bfloat16")

gate_correction_bias_real_data = paddle.to_tensor(
    [
        32.8339,
        32.8231,
        32.8151,
        32.8131,
        32.8317,
        32.8343,
        32.8356,
        32.8270,
        32.8344,
        32.8342,
        32.8126,
        32.8299,
        32.8282,
        32.8254,
        32.8320,
        32.8280,
        32.8303,
        32.8351,
        32.8364,
        32.8347,
        32.8179,
        32.8349,
        32.8322,
        32.8323,
        32.8360,
        32.8351,
        32.8059,
        32.8352,
        32.8303,
        32.8334,
        32.8283,
        32.8265,
        32.8344,
        32.8307,
        32.8271,
        32.8343,
        32.8326,
        32.8327,
        32.8349,
        32.8356,
        32.8303,
        32.8327,
        32.8310,
        32.8363,
        32.8274,
        32.8335,
        32.8350,
        32.8255,
        32.8298,
        32.8141,
        32.8218,
        32.8362,
        32.8126,
        32.7902,
        32.8314,
        32.8356,
        32.8177,
        32.8333,
        32.8352,
        32.8354,
        32.8334,
        32.8325,
        32.7971,
        32.8319,
        32.8222,
        32.8284,
        32.8288,
        32.8355,
        32.8351,
        32.8356,
        32.8338,
        32.8346,
        32.7737,
        32.8317,
        32.8357,
        32.8345,
        32.8347,
        32.8360,
        32.8289,
        32.8268,
        32.8164,
        32.8324,
        32.8363,
        32.8308,
        32.8352,
        32.8302,
        32.8345,
        32.8298,
        32.8057,
        32.8229,
        32.8355,
        32.8325,
        32.8350,
        32.8357,
        32.8315,
        32.8327,
        32.8263,
        32.8342,
        32.8165,
        32.8349,
        32.8310,
        32.8101,
        32.8101,
        32.8081,
        32.8341,
        32.8313,
        32.8331,
        32.8299,
        32.8320,
        32.7941,
        32.8277,
        32.8287,
        32.8326,
        32.8331,
        32.8360,
        32.8295,
        32.8255,
        32.8330,
        32.8279,
        32.8210,
        32.7921,
        32.8348,
        32.8271,
        32.8297,
        32.8211,
        32.8353,
        32.8339,
        32.8335,
        32.8275,
        32.8245,
        32.8287,
        32.8352,
        32.8318,
        32.8354,
        32.8110,
        32.8347,
        32.8340,
        32.8322,
        32.8341,
        32.8316,
        32.8328,
        32.8341,
        32.8354,
        32.8264,
        32.8362,
        32.8352,
        32.8293,
        32.8292,
        32.8328,
        32.8316,
        32.8329,
        32.8308,
        32.8307,
        32.8170,
        32.8345,
        32.8356,
        32.8176,
        32.8326,
        32.8288,
        32.8355,
        32.8346,
        32.8337,
        32.8049,
        32.8315,
        32.8337,
        32.8352,
        32.7991,
        32.8304,
        32.8348,
        32.8316,
        32.8358,
        32.8279,
        32.8348,
        32.8326,
        32.8215,
        32.8281,
        32.8344,
        32.8309,
        32.8355,
        32.8337,
        32.8276,
        32.8250,
        32.8340,
        32.8322,
        32.8317,
        32.8274,
        32.8363,
        32.8277,
        32.8345,
        32.8342,
        32.8343,
        32.8355,
        32.8326,
        32.8299,
        32.8322,
        32.8351,
        32.8356,
        32.7925,
        32.8362,
        32.8170,
        32.8323,
        32.8335,
        32.8339,
        32.8193,
        32.8340,
        32.8362,
        32.8323,
        32.8328,
        32.8328,
        32.8296,
        32.8297,
        32.8344,
        32.8254,
        32.8341,
        32.8345,
        32.7967,
        32.8228,
        32.8363,
        32.8356,
        32.8317,
        32.8362,
        32.8302,
        32.8356,
        32.8239,
        32.8304,
        32.8323,
        32.8335,
        32.8196,
        32.8354,
        32.6991,
        32.8350,
        32.8337,
        32.8314,
        32.8274,
        32.8232,
        32.8305,
        32.8349,
        32.8246,
        32.8343,
        32.8339,
        32.7849,
        32.8359,
        32.8353,
        32.8352,
        32.8348,
        32.8095,
        32.8301,
        32.8350,
        32.8340,
        32.8353,
        32.8343,
        32.8344,
        32.8312,
        32.8350,
        32.8327,
        32.8231,
        32.8325,
        32.8352,
        32.8352,
        32.8293,
        32.8357,
        32.8337,
        32.8335,
        32.8348,
        32.8321,
        32.8153,
        32.8352,
        32.8265,
        32.8326,
        32.8361,
        32.8357,
        32.8312,
        32.8347,
        32.8152,
        32.8340,
        32.8272,
        32.8352,
        32.8331,
        32.8324,
        32.7952,
        32.8170,
        32.8356,
        32.8360,
        32.8298,
        32.8356,
        32.8331,
        32.8317,
        32.8349,
        32.8269,
        32.8323,
        32.8354,
        32.8350,
        32.8226,
        32.8002,
        32.8205,
        32.8329,
        32.8319,
        32.8297,
        32.8282,
        32.8356,
        32.8303,
        32.8349,
        32.8337,
        32.8247,
        32.8279,
        32.8309,
        32.8225,
        32.8337,
        32.8356,
        32.8105,
        32.8353,
        32.8361,
        32.8297,
        32.8313,
        32.8313,
        32.8363,
        32.8357,
        32.8357,
        32.8363,
        32.7806,
        32.8306,
        32.8347,
        32.8248,
        32.8334,
        32.8356,
        32.8324,
        32.8327,
        32.8284,
        32.8351,
        32.8349,
        32.8351,
        32.8171,
        32.8317,
        32.8363,
        32.8346,
        32.8335,
        32.8307,
        32.7907,
        32.8229,
        32.8346,
        32.8298,
        32.8336,
        32.8313,
        32.8349,
        32.8219,
        32.8354,
        32.8337,
        32.8294,
        32.8306,
        32.8322,
        32.8290,
        32.8333,
        32.8327,
        32.8279,
        32.8283,
        32.8338,
        32.8310,
        32.8351,
        32.8171,
        32.8310,
        32.8323,
        32.8324,
        32.8215,
        32.8314,
        32.8333,
        32.8353,
        32.8184,
        32.8344,
        32.8280,
        32.8352,
        32.8361,
        32.8308,
        32.8271,
        32.8335,
        32.8236,
        32.8350,
        32.8325,
        32.8330,
        32.8228,
        32.8352,
        32.8258,
        32.8343,
        32.8338,
        32.8292,
    ],
    dtype="float32",
)


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
            quant_config=BlockWiseFP8Config(weight_block_size=[128, 128]),
            # quant_config=WINT8Config({}),
            # quant_config=WINT4Config({}),
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
        moe_layer = self.fused_moe

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
        self.hidden_size = 4096
        self.moe_intermediate_size = 2048
        self.moe_num_experts = 160
        self.moe_k = 8
        self.num_layers = 2
        self.num_attention_heads = -1
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
        init_distributed_environment()

        gating = paddle.nn.Linear(self.model_config.hidden_size, self.model_config.moe_num_experts)
        gating.to(dtype=paddle.float32)  # it's dtype is bfloat16 default, but the forward input is float32
        gating.weight.set_value(paddle.rand(gating.weight.shape, dtype=paddle.float32))

        os.environ["FD_USE_DEEP_GEMM"] = "0"
        ep_size = paddle.distributed.get_world_size()
        ep_rank = paddle.distributed.get_rank()

        tp_rank = 0
        tp_size = 1

        nnodes = (ep_size + 7) // 8

        # 这行代码必须保留，否则影响均匀性！
        paddle.seed(ep_rank + 100)

        num_layers = self.num_layers
        real_weight_layers = num_layers // 2
        fused_moe = [None] * real_weight_layers
        for i in range(real_weight_layers):
            fused_moe[i] = FuseMoEWrapper(self.model_config, tp_size, tp_rank, ep_size, ep_rank, nnodes=nnodes)

        moe_cuda_graphs = [None] * 100
        cache_hidden_states = [None] * 100
        is_decoder = fused_moe[0].fd_config.model_config.moe_phase.phase == "decode"
        test_token_nums = [4096 * i for i in [1, 2, 4, 8]]
        if is_decoder:
            test_token_nums = [10, 20, 40, 60, 80, 100, 128, 160, 192, 256]
        for idx, num_tokens in enumerate(test_token_nums):

            cache_hidden_states[idx] = paddle.rand((num_tokens, self.model_config.hidden_size), dtype=paddle.bfloat16)

            def fake_model_run():
                for j in range(num_layers):
                    out = fused_moe[j % real_weight_layers].fused_moe(cache_hidden_states[idx], gating)

                return out

            if is_decoder:
                moe_cuda_graphs[idx] = graphs.CUDAGraph()
                moe_cuda_graphs[idx].capture_begin()

            fake_model_run()

            if is_decoder:
                moe_cuda_graphs[idx].capture_end()

            num_tests = 20
            start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            for i in range(num_tests):
                start_events[i].record()

                if is_decoder:
                    moe_cuda_graphs[idx].replay()
                else:
                    fake_model_run()

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
