"""
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
"""

from unittest.mock import Mock

import numpy as np
import paddle
import paddle.device.cuda.graphs as graphs

from fastdeploy.config import (
    CacheConfig,
    FDConfig,
    GraphOptimizationConfig,
    ParallelConfig,
    SchedulerConfig,
)


class FakeModelConfig:
    def __init__(self):
        self.hidden_size = 768
        self.intermediate_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.rms_norm_eps = 1e-6
        self.tie_word_embeddings = True
        self.ori_vocab_size = 32000
        self.moe_layer_start_index = 8
        self.pretrained_config = Mock()
        self.pretrained_config.prefix_name = "test"
        self.num_key_value_heads = 1
        self.head_dim = 1
        self.is_quantized = False
        self.hidden_act = "relu"
        self.vocab_size = 32000
        self.hidden_dropout_prob = 0.1
        self.initializer_range = 0.02
        self.max_position_embeddings = 512
        self.tie_word_embeddings = True
        self.model_format = "auto"
        self.enable_mm = False
        self.max_model_len = 512


def get_default_test_fd_config():
    graph_opt_config = GraphOptimizationConfig(args={})
    scheduler_config = SchedulerConfig(args={})
    scheduler_config.max_num_seqs = 1
    parallel_config = ParallelConfig(args={})
    parallel_config.data_parallel_rank = 1
    cache_config = CacheConfig({})
    model_config = FakeModelConfig()
    fd_config = FDConfig(
        graph_opt_config=graph_opt_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        model_config=model_config,
        test_mode=True,
    )
    return fd_config


class OpPerformanceTester:
    def __init__(self, op_name, op_fn, num_layers=20, weight_size=None, gate=None):
        self.op_name = op_name
        self.op_fn = op_fn
        self.num_layers = num_layers
        self.weight_size = weight_size
        self.gate = gate

    def _fake_model_run(self, x):
        for j in range(self.num_layers):
            if self.gate:
                out = self.op_fn(x, self.gate)
            else:
                out = self.op_fn(x)
        return out

    def benchmark(self, input_size, batch_sizes, dtype="bfloat16", num_warmup=1, num_tests=10):
        print(f"======== {self.op_name} Performance ========")
        print(
            "{:<15} {:<40} {:<15} {:<15} {:<15}".format(
                "Batch Size", "Last 5 Times (us)", "Last Time (us)", "TFlops", "TB/s"
            )
        )

        for idx, bsz in enumerate(batch_sizes):
            x = paddle.rand((bsz, input_size), dtype=dtype)

            self._fake_model_run(x)

            graph = graphs.CUDAGraph()
            graph.capture_begin()
            self._fake_model_run(x)
            graph.capture_end()

            start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]
            end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(num_tests)]

            for i in range(num_tests):
                start_events[i].record()
                graph.replay()
                end_events[i].record()

            paddle.device.synchronize()

            times = np.array([round(s.elapsed_time(e), 2) for s, e in zip(start_events, end_events)])[num_warmup:]
            times = times * 1e3 / self.num_layers  # us / layer
            times = np.array([round(time, 2) for time in times])
            last_5_times = times[-5:]
            last_time = times[-1]

            tfloaps = None
            tbps = None
            if self.weight_size:
                flops = 2 * bsz * self.weight_size
                memory = self.weight_size
                tfloaps = round(flops / 1e12 / (last_time * 1e-6), 1)
                tbps = round(memory / 1e12 / (last_time * 1e-6), 1)

                print("{:<15} {:<40} {:<15} {:<15} {:<15}".format(bsz, str(last_5_times), last_time, tfloaps, tbps))
            else:
                print("{:<15} {:<40} {:<15}".format(bsz, str(last_5_times), last_time))
