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
from fastdeploy.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from fastdeploy.model_executor.layers.quantization.weight_only import (
    WINT4Config,
    WINT8Config,
)
from fastdeploy.scheduler import SchedulerConfig

paddle.set_default_dtype("bfloat16")
paddle.seed(1024)


class QuantizedLinearWrapper(paddle.nn.Layer):
    def __init__(
        self,
        model_config: ModelConfig,
        tp_size: int = 1,
        prefix: str = "layer0",
        quant_type: str = "wint4",
    ):
        super().__init__()
        self.model_config = model_config

        self.tp_size = tp_size
        self.prefix = prefix
        self.fd_config = FDConfig(
            model_config=self.model_config,
            parallel_config=ParallelConfig({"tensor_parallel_size": self.tp_size}),
            quant_config=WINT8Config({}) if quant_type == "wint8" else WINT4Config({}),
            load_config=LoadConfig({}),
            graph_opt_config=GraphOptimizationConfig({}),
            scheduler_config=SchedulerConfig({}),
            cache_config=CacheConfig({}),
        )

        self.fd_config.parallel_config.tp_group = None

        self.qkv_proj = QKVParallelLinear(
            self.fd_config,
            prefix=f"{prefix}.qkv_proj",
            with_bias=False,
        )

        self.o_proj = RowParallelLinear(
            self.fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=self.fd_config.model_config.head_dim * self.fd_config.model_config.num_attention_heads,
            output_size=self.fd_config.model_config.hidden_size,
        )

        qkv_proj_weight_shape = [
            self.qkv_proj.input_size,
            self.qkv_proj.output_size,
        ]

        o_proj_weight_shape = [
            self.o_proj.input_size,
            self.o_proj.output_size,
        ]

        state_dict = {}
        state_dict[f"{prefix}.qkv_proj.weight"] = paddle.randn(qkv_proj_weight_shape, paddle.bfloat16)
        state_dict[f"{prefix}.o_proj.weight"] = paddle.randn(o_proj_weight_shape, paddle.bfloat16)
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)

        self.input_size = self.o_proj.input_size
        self.output_size = self.qkv_proj.output_size

    def forward(self, x):
        x = self.o_proj(x)
        x = self.qkv_proj(x)
        return x


class TestQuantizedLinear(unittest.TestCase):
    def setUp(self) -> None:
        self.model_name_or_path = None
        self.model_config = self.build_model_config()

    def build_model_config(self) -> ModelConfig:
        model_path = os.getenv("TEST_MODEL_PATH")
        if model_path:
            model_cofig_path = model_path
        else:
            model_cofig_path = self.build_config_json()
        return ModelConfig(
            {
                "model": model_cofig_path,
                "max_model_len": 2048,
            }
        )

    def build_config_json(self) -> str:
        config_dict = {
            "architectures": ["Ernie4_5_MoeForCausalLM"],
            "hidden_size": 8192,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_hidden_layers": 54,
            "dtype": "bfloat16",
        }

        tmp_dir = "./tmp_wint"
        os.makedirs(tmp_dir, exist_ok=True)
        with open(f"./{tmp_dir}/config.json", "w") as f:
            json.dump(config_dict, f)
        self.model_name_or_path = os.path.join(os.getcwd(), tmp_dir)
        return self.model_name_or_path

    def run_quantized_linear(self, type="qkv_proj", quant_type="wint4"):
        quantized_linear = QuantizedLinearWrapper(self.model_config, quant_type=quant_type)
        if type == "qkv_proj":
            input_size = quantized_linear.qkv_proj.input_size
            weight_size = quantized_linear.qkv_proj.output_size * quantized_linear.qkv_proj.input_size
            mm = quantized_linear.qkv_proj
            print(f"Input Size: {input_size}, Output Size: {quantized_linear.qkv_proj.output_size}")
        elif type == "o_proj":
            input_size = quantized_linear.o_proj.input_size
            weight_size = quantized_linear.o_proj.output_size * quantized_linear.o_proj.input_size
            mm = quantized_linear.o_proj
            print(f"Input Size: {input_size}, Output Size: {quantized_linear.o_proj.output_size}")
        else:
            input_size = quantized_linear.input_size
            weight_size = (
                quantized_linear.qkv_proj.output_size * quantized_linear.qkv_proj.input_size
                + quantized_linear.o_proj.output_size * quantized_linear.o_proj.input_size
            )
            mm = quantized_linear

        print(f"========Method: {type}, Quant Type: {quant_type}=========")
        print(
            "{:<15} {:<40} {:<15} {:<15} {:<15}".format(
                "Batch Size", "Last 5 Times (us)", "Last Time (us)", "TFlops", "TB/s"
            )
        )

        num_layers = self.model_config.num_hidden_layers
        real_weight_layers = self.model_config.num_hidden_layers
        linear = [None] * real_weight_layers
        for i in range(real_weight_layers):
            linear[i] = mm

        linear_cuda_graphs = [None] * 2000
        input = [None] * 2000
        # for idx, bsz in enumerate([1024 * i for i in [1,2,4,8,16,32,64]]):
        for idx, bsz in enumerate([1, 8, 16, 32, 128, 1024]):

            input[idx] = paddle.rand((bsz, input_size), dtype=paddle.bfloat16)

            def fake_model_run():
                for j in range(num_layers):
                    out = linear[j % real_weight_layers](input[idx])

                return out

            fake_model_run()

            linear_cuda_graphs[idx] = graphs.CUDAGraph()
            linear_cuda_graphs[idx].capture_begin()

            fake_model_run()

            linear_cuda_graphs[idx].capture_end()

            num_tests = 20
            start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            for i in range(num_tests):
                start_events[i].record()

                linear_cuda_graphs[idx].replay()

                end_events[i].record()
            paddle.device.synchronize()

            times = np.array([round(s.elapsed_time(e), 2) for s, e in zip(start_events, end_events)])[1:]
            times = times * 1e3 / num_layers
            times = np.array([round(time, 2) for time in times])
            last_5_times = times[-5:]
            last_time = times[-1]  # us

            flops = 2 * bsz * weight_size
            memory = weight_size
            tfloaps = round(flops / (1e12) / (last_time * 1e-6), 1)
            tbps = round(memory / (1e12) / (last_time * 1e-6), 1)
            print("{:<15} {:<40} {:<15} {:<15} {:<15}".format(bsz, str(last_5_times), last_time, tfloaps, tbps))

    def test_quantized_linear(self):
        for type in ["qkv_proj", "o_proj", "out_proj+qkv_proj"]:
            for quant_type in ["wint4", "wint8"]:
                for use_machete in ["0", "1"]:
                    os.environ["FD_USE_MACHETE"] = use_machete
                    self.run_quantized_linear(type, quant_type)
            self.run_quantized_linear(type, "block_wise_fp8")

    def tearDown(self) -> None:
        if self.model_name_or_path:
            print("Remove tmp model config file")
            shutil.rmtree(self.model_name_or_path)


class TestQuantizedLinearGroupSize64(TestQuantizedLinear):
    def setUp(self) -> None:
        self.model_name_or_path = None
        self.model_config = self.build_model_config()

    def build_model_config(self) -> ModelConfig:
        model_path = os.getenv("TEST_MODEL_PATH")
        if model_path:
            model_cofig_path = model_path
        else:
            model_cofig_path = self.build_config_json()
        return ModelConfig(
            {
                "model": model_cofig_path,
                "max_model_len": 2048,
            }
        )

    def build_config_json(self) -> str:
        config_dict = {
            "architectures": ["Ernie4_5_MoeForCausalLM"],
            "hidden_size": 2880,
            "head_dim": 64,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_hidden_layers": 24,
            "dtype": "bfloat16",
        }

        tmp_dir = "./tmp_wint"
        os.makedirs(tmp_dir, exist_ok=True)
        with open(f"./{tmp_dir}/config.json", "w") as f:
            json.dump(config_dict, f)
        self.model_name_or_path = os.path.join(os.getcwd(), tmp_dir)
        return self.model_name_or_path

    def test_quantized_linear(self):
        for type in ["qkv_proj", "o_proj"]:
            for quant_type in ["wint4", "wint8"]:
                for use_machete in ["0", "1"]:
                    os.environ["FD_USE_MACHETE"] = use_machete
                    self.run_quantized_linear(type, quant_type)


if __name__ == "__main__":
    unittest.main()
