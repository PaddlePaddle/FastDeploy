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

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import paddle
import paddle.device.cuda.graphs as graphs

from fastdeploy.config import (
    CacheConfig,
    CommitConfig,
    DeviceConfig,
    EarlyStopConfig,
    FDConfig,
    GraphOptimizationConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
)
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.layers.attention import (
    AttentionBackend,
    get_attention_backend,
)
from fastdeploy.model_executor.layers.attention.append_attn_backend import (
    allocate_launch_related_buffer,
)
from fastdeploy.model_executor.layers.quantization.mix_quant import MixQuantConfig
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.models.ernie4_5_moe import Ernie4_5_Attention
from fastdeploy.model_executor.ops.gpu import get_padding_offset

if "nvidia graphics device" in paddle.device.cuda.get_device_name().lower():
    # (ZKK): CI machine.
    os.environ.setdefault("DG_NVCC_OVERRIDE_CPP_STANDARD", "17")


class TestAttentionPerformance(unittest.TestCase):
    def setUp(self):
        """
        Set up the testing environment before each test.
        This includes creating configurations, initializing the model,
        and preparing a random state dictionary.
        """
        print("Setting up test environment...")
        paddle.set_device("gpu")
        paddle.set_default_dtype("bfloat16")

        self.model_dir = self.create_model_config_json()
        self.fd_config = self.create_fd_config_from_model_path(self.model_dir, tensor_parallel_size=1)
        self.fd_config.parallel_config.tp_group = [0]

        # Initialize Attention Layer
        os.environ["FD_ATTENTION_BACKEND"] = "APPEND_ATTN"
        attn_cls = get_attention_backend()
        self.attn_backend = attn_cls(
            self.fd_config,
            kv_num_heads=self.fd_config.model_config.num_key_value_heads
            // self.fd_config.parallel_config.tensor_parallel_size,
            num_heads=self.fd_config.model_config.num_attention_heads
            // self.fd_config.parallel_config.tensor_parallel_size,
            head_dim=self.fd_config.model_config.head_dim,
            encoder_block_shape_q=64,
            decoder_block_shape_q=16,
        )

        num_layers = self.fd_config.model_config.num_hidden_layers
        self.attention_layer = [None] * num_layers
        for i in range(num_layers):
            self.attention_layer[i] = Ernie4_5_Attention(self.fd_config, layer_id=i, prefix="test_layer")
            state_dict = self.create_random_attention_state_dict(self.fd_config, prefix="test_layer")
            self.attention_layer[i].load_state_dict(state_dict)

        def attn_forward(forward_meta, hidden_states):
            for i in range(num_layers):
                hidden_states = self.attention_layer[i](forward_meta, hidden_states)

            return hidden_states

        self.attn_forward = attn_forward

        self.cache_quant_type_str = getattr(self.attention_layer[0].attn, "cache_quant_type_str", "none")

        print("===== Initialization Complete =====")

    def tearDown(self):
        """
        Clean up the environment after each test.
        """
        print("\nTearing down test environment...")
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
            print(f"Successfully removed temporary directory: {self.model_dir}")

    # region Helper Functions
    def create_model_config_json(self) -> str:
        """
        Creates a temporary directory and writes the model configuration to a 'config.json' file.
        """
        config_dict = {
            "architectures": ["Ernie4_5_MoeForCausalLM"],
            "dtype": "bfloat16",
            "max_position_embeddings": 131072,
            "max_model_len": 131072,
            "head_dim": 128,
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "num_hidden_layers": 57,
        }
        model_dir = tempfile.mkdtemp(prefix="tmp_model_config_")
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Successfully created config.json at: {config_path}")
        return model_dir

    def create_fd_config_from_model_path(self, model_path, tensor_parallel_size=1):
        """Creates a complete FDConfig from a model path."""
        model_args = {"model": model_path, "dtype": "bfloat16"}
        model_config = ModelConfig(model_args)
        model_config.tensor_parallel_size = tensor_parallel_size
        parallel_config = ParallelConfig({"tensor_parallel_size": tensor_parallel_size, "data_parallel_size": 1})
        cache_config = CacheConfig(
            {
                "block_size": 64,
                "model_cfg": model_config,
                "tensor_parallel_size": tensor_parallel_size,
            }
        )
        return FDConfig(
            model_config=model_config,
            cache_config=cache_config,
            parallel_config=parallel_config,
            scheduler_config=SchedulerConfig({}),
            load_config=LoadConfig({}),
            quant_config=MixQuantConfig(
                dense_quant_type="block_wise_fp8",
                moe_quant_type="block_wise_fp8",
                kv_cache_quant_type="float8_e4m3fn",
            ),
            graph_opt_config=GraphOptimizationConfig({}),
            commit_config=CommitConfig(),
            device_config=DeviceConfig({}),
            speculative_config=SpeculativeConfig({}),
            early_stop_config=EarlyStopConfig({}),
        )

    def create_random_attention_state_dict(self, fd_config: FDConfig, prefix: str) -> dict:
        """
        Creates a state_dict with random weights for the Ernie4_5_Attention layer.
        """
        hidden_size = fd_config.model_config.hidden_size
        tp_size = fd_config.parallel_config.tensor_parallel_size
        tensor_dtype = getattr(paddle, fd_config.model_config.dtype)

        q_dims = fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim
        kv_dims = fd_config.model_config.num_key_value_heads * fd_config.model_config.head_dim
        total_output_dim = q_dims + 2 * kv_dims
        qkv_proj_output_dim_tp = total_output_dim // tp_size
        qkv_weight_shape = [hidden_size, qkv_proj_output_dim_tp]

        o_proj_input_dim = fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim
        o_proj_input_dim_tp = o_proj_input_dim // tp_size
        o_proj_weight_shape = [o_proj_input_dim_tp, hidden_size]

        qkv_weight = paddle.randn(qkv_weight_shape, dtype=tensor_dtype)
        o_proj_weight = paddle.randn(o_proj_weight_shape, dtype=tensor_dtype)

        kv_num_heads_tp = fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size
        activation_scale_shape = [kv_num_heads_tp]
        activation_scale_tensor = paddle.full(shape=activation_scale_shape, fill_value=1.0, dtype=tensor_dtype)

        state_dict = {
            f"{prefix}.qkv_proj.weight": qkv_weight,
            f"{prefix}.o_proj.weight": o_proj_weight,
            f"{prefix}.cachek_matmul.activation_scale": activation_scale_tensor,
            f"{prefix}.cachev_matmul.activation_scale": activation_scale_tensor,
        }
        return state_dict

    def create_forward_meta(
        self,
        batch_size: int,
        seq_len: int,
        mode: ForwardMode,
        fd_config: FDConfig,
        attn_backend: AttentionBackend,
        cache_quant_type_str: str = "none",
    ) -> ForwardMeta:
        """
        Creates a high-fidelity ForwardMeta object.
        """
        if mode == ForwardMode.EXTEND:
            seq_lens_encoder = paddle.full([batch_size], seq_len, dtype="int32")
            seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
            seq_lens_this_time = seq_lens_encoder
        elif mode == ForwardMode.DECODE:
            seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
            seq_lens_decoder = paddle.full([batch_size], seq_len, dtype="int32")
            seq_lens_this_time = paddle.ones([batch_size], dtype="int32")
        else:
            raise ValueError(f"Unsupported ForwardMode: {mode}")

        attn_backend_buffers = allocate_launch_related_buffer(
            max_batch_size=batch_size,
            max_model_len=fd_config.model_config.max_model_len,
            encoder_block_shape_q=64,
            decoder_block_shape_q=16,
            decoder_step_token_num=fd_config.speculative_config.num_speculative_tokens + 1,
            num_heads=fd_config.model_config.num_attention_heads,
            kv_num_heads=fd_config.model_config.num_key_value_heads,
            block_size=fd_config.cache_config.block_size,
        )

        block_size = fd_config.cache_config.block_size
        max_model_len = fd_config.model_config.max_model_len
        max_blocks_per_seq = (max_model_len + block_size - 1) // block_size
        allocated_blocks_per_seq = seq_len // block_size + 1
        allocated_num_blocks = allocated_blocks_per_seq * batch_size
        head_dim = fd_config.model_config.head_dim
        kv_num_heads_tp = fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size
        num_layers = fd_config.model_config.num_hidden_layers
        cache_type = fd_config.model_config.dtype
        if cache_quant_type_str != "none":
            cache_type = "uint8"
        cache_shape = (allocated_num_blocks, kv_num_heads_tp, block_size, head_dim)
        scale_shape = (allocated_num_blocks, kv_num_heads_tp, block_size)
        caches = []
        for _ in range(num_layers):
            key_cache = paddle.randint(0, 255, shape=cache_shape, dtype="int32").cast(cache_type)
            value_cache = paddle.randint(0, 255, shape=cache_shape, dtype="int32").cast(cache_type)
            caches.extend([key_cache, value_cache])
            if cache_quant_type_str == "block_wise_fp8":
                key_cache_scale = paddle.rand(shape=scale_shape, dtype=fd_config.model_config.dtype)
                value_cache_scale = paddle.rand(shape=scale_shape, dtype=fd_config.model_config.dtype)
                caches.extend([key_cache_scale, value_cache_scale])

        block_tables = paddle.zeros(shape=(batch_size, max_blocks_per_seq), dtype="int32")
        for i in range(batch_size):
            for j in range(allocated_blocks_per_seq):
                block_tables[i, j] = i * allocated_blocks_per_seq + j

        tmp_position_ids = paddle.arange(fd_config.model_config.max_model_len).reshape((1, -1))
        rope_emb = get_rope(
            rotary_dim=fd_config.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=fd_config.model_config.rope_theta,
            model_config=fd_config.model_config,
            partial_rotary_factor=fd_config.model_config.partial_rotary_factor,
        )

        input_ids = paddle.zeros([batch_size, seq_len if mode == ForwardMode.EXTEND else 1], dtype="int64")
        token_num = paddle.sum(seq_lens_this_time)
        ids_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k = get_padding_offset(
            input_ids, token_num, seq_lens_this_time
        )

        forward_meta = ForwardMeta(
            ids_remove_padding=ids_remove_padding,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            seq_lens_this_time=seq_lens_this_time,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            batch_id_per_token=batch_id_per_token,
            block_tables=block_tables,
            caches=caches,
            rotary_embs=rope_emb,
            step_use_cudagraph=False,
            attn_backend=attn_backend,
            forward_mode=ForwardMode.MIXED,
            attn_mask=None,
            attn_mask_offsets=None,
            **attn_backend_buffers,
        )
        return forward_meta

    def test_decode_performance_with_prefill(self):
        # Test parameters
        test_steps = 100
        act_tensor_dtype = paddle.bfloat16

        # prefill_batch_size = 1
        # prefill_seq_len = 4096

        # prefill_hidden_states = paddle.randn(
        #     [prefill_batch_size * prefill_seq_len, self.fd_config.model_config.hidden_size],
        #     dtype=act_tensor_dtype,
        # )

        # forward_meta = self.create_forward_meta(
        #     batch_size=prefill_batch_size,
        #     seq_len=prefill_seq_len,
        #     mode=ForwardMode.EXTEND,
        #     fd_config=self.fd_config,
        #     attn_backend=self.attn_backend,
        #     cache_quant_type_str=self.cache_quant_type_str,
        # )

        # self.attn_backend.init_attention_metadata(forward_meta)
        # self.attn_forward(forward_meta, prefill_hidden_states)

        # paddle.device.synchronize()

        # import paddle.profiler as profiler
        # p = profiler.Profiler(
        #     targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        #     on_trace_ready=profiler.export_chrome_tracing("./profile_log"),
        # )
        # p.start()
        # p.step()

        # start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(test_steps)]
        # end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(test_steps)]
        # for i in range(test_steps):
        #     start_events[i].record()

        #     self.attn_forward(forward_meta, prefill_hidden_states)

        #     end_events[i].record()
        # paddle.device.synchronize()

        # times = np.array([round(s.elapsed_time(e), 1) for s, e in zip(start_events, end_events)])[1:]
        # print(times[-5:])
        # return

        # p.stop()

        # p = profiler.Profiler(
        #     targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        #     on_trace_ready=profiler.export_chrome_tracing("./profile_log"),
        # )

        # p.start()
        # p.step()

        for decode_batch_size in [32, 16, 8, 4, 2]:
            decode_hidden_states = paddle.randn(
                [decode_batch_size, self.fd_config.model_config.hidden_size], dtype=act_tensor_dtype
            )

            forward_meta = self.create_forward_meta(
                batch_size=decode_batch_size,
                seq_len=36 * 1024,
                mode=ForwardMode.DECODE,
                fd_config=self.fd_config,
                attn_backend=self.attn_backend,
                cache_quant_type_str=self.cache_quant_type_str,
            )

            self.attn_backend.init_attention_metadata(forward_meta)

            paddle.device.synchronize()

            # 必须要先预热一次！因为预处理被放到了第一层再做了！
            self.attn_forward(forward_meta, decode_hidden_states)

            attn_cuda_graphs = graphs.CUDAGraph()
            attn_cuda_graphs.capture_begin()

            self.attn_forward(forward_meta, decode_hidden_states)

            attn_cuda_graphs.capture_end()

            start_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(test_steps)]
            end_events = [paddle.device.cuda.Event(enable_timing=True) for _ in range(test_steps)]
            for i in range(test_steps):
                start_events[i].record()

                attn_cuda_graphs.replay()

                end_events[i].record()
            paddle.device.synchronize()

            times = np.array([round(s.elapsed_time(e), 1) for s, e in zip(start_events, end_events)])[1:]
            print(times[-5:])

            del forward_meta

        # p.stop()


if __name__ == "__main__":
    unittest.main()
