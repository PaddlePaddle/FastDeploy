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
import time
import types
import unittest

import numpy as np
import paddle
from paddle import nn

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
from fastdeploy.model_executor.layers.quantization import parse_quant_config
from fastdeploy.model_executor.layers.rotary_embedding import get_rope
from fastdeploy.model_executor.models.ernie4_5_moe import Ernie4_5_Attention
from fastdeploy.model_executor.pre_and_post_process import pre_process


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
        # Adjust config for the test
        self.fd_config.model_config.max_model_len = 2 * (9000 + 128)
        self.fd_config.model_config.num_hidden_layers = 1
        self.fd_config.parallel_config.tp_group = [0]

        # Mock quantization config
        mock_args = types.SimpleNamespace()
        mock_args.quantization = None
        # NOTE: Dense Gemm 跑block_wise_fp8请使用下面这一行. 同时设置config里量化相关选项.
        # mock_args.quantization = {"quantization": "block_wise_fp8"}
        mock_args.dynamic_load_weight = False
        quant_config = parse_quant_config(mock_args, self.fd_config.model_config, is_ernie=1, is_v1_loader=1)
        self.fd_config.quant_config = quant_config

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
        self.attention_layer = Ernie4_5_Attention(self.fd_config, layer_id=0, prefix="test_layer")
        state_dict = self.create_random_attention_state_dict(self.fd_config, prefix="test_layer")
        self.attention_layer.load_state_dict(state_dict)
        self.attention_layer.attn.cache_quant_type_str = "block_wise_fp8"
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
            "bos_token_id": 1,
            "eos_token_id": 2,
            "dtype": "bfloat16",
            "hidden_act": "silu",
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "max_position_embeddings": 131072,
            "model_type": "ernie4_5_moe",
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_hidden_layers": 5,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-05,
            "use_cache": False,
            "vocab_size": 103424,
            "rope_theta": 500000,
            "use_rmsnorm": True,
            "use_bias": False,
            "moe_num_experts": 64,
            "moe_layer_start_index": 1,
            "moe_intermediate_size": 3584,
            "moe_capacity": [64, 64, 64],
            "moe_gate": "topk",
            "moe_k": 4,
            "moe_layer_interval": 1,
            "moe_use_aux_free": True,
            "num_nextn_predict_layers": 1,
            "tie_word_embeddings": False,
            "is_quantized": False,
            # NOTE: 跑量化推理请取消注释
            # "quantization_config": {
            #     "dense_quant_type": "block_wise_fp8",
            #     "moe_quant_type": "block_wise_fp8",
            #     "kv_cache_quant_type": "float8_e4m3fn",
            #     "quantization": "mix_quant",
            # },
        }
        model_dir = tempfile.mkdtemp(prefix="tmp_model_config_")
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        print(f"Successfully created config.json at: {config_path}")
        return model_dir

    def create_random_attention_state_dict(self, fd_config: FDConfig, prefix: str) -> dict:
        """
        Creates a state_dict with random weights for the Ernie4_5_Attention layer.
        """
        print("INFO: Creating random weights for testing...")
        with paddle.no_grad():
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

            kv_num_heads_tp = (
                fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size
            )
            activation_scale_shape = [kv_num_heads_tp]
            activation_scale_tensor = paddle.full(shape=activation_scale_shape, fill_value=1.0, dtype=tensor_dtype)

            state_dict = {
                f"{prefix}.qkv_proj.weight": qkv_weight,
                f"{prefix}.o_proj.weight": o_proj_weight,
                f"{prefix}.cachek_matmul.activation_scale": activation_scale_tensor,
                f"{prefix}.cachev_matmul.activation_scale": activation_scale_tensor,
            }
        return state_dict

    def create_attn_backend_buffers(self, m_config: ModelConfig, batch_size: int, block_size: int) -> dict:
        """
        Pre-allocates metadata buffers required by the Attention backend.
        """
        encoder_block_shape_q = 64
        decoder_block_shape_q = 16
        decoder_step_token_num = 1
        num_heads = m_config.num_attention_heads
        kv_num_heads = m_config.num_key_value_heads
        group_size = np.ceil(num_heads / kv_num_heads)

        decode_max_tile_size = (
            1024 * batch_size * np.ceil((decoder_step_token_num * group_size) / decoder_block_shape_q)
        )
        encode_max_tile_size = batch_size * np.ceil((m_config.max_model_len * group_size) / encoder_block_shape_q)
        kv_max_tile_size = batch_size * np.ceil(m_config.max_model_len / block_size)

        return {
            "decoder_batch_ids": paddle.full([int(decode_max_tile_size)], 0, dtype="int32"),
            "decoder_tile_ids_per_batch": paddle.full([int(decode_max_tile_size)], 0, dtype="int32"),
            "decoder_num_blocks_cpu": paddle.full([1], 0, dtype="int32").pin_memory(),
            "decoder_num_blocks_device": paddle.full([1], 0, dtype="int32"),
            "decoder_chunk_size_device": paddle.full([1], 64, dtype="int32"),
            "max_len_tensor_cpu": paddle.full([8], 0, dtype="int32").cpu(),
            "encoder_batch_ids": paddle.full([int(encode_max_tile_size)], 0, dtype="int32"),
            "encoder_tile_ids_per_batch": paddle.full([int(encode_max_tile_size)], 0, dtype="int32"),
            "encoder_num_blocks_x_cpu": paddle.full([1], 0, dtype="int32").cpu(),
            "kv_batch_ids": paddle.full([int(kv_max_tile_size)], 0, dtype="int32"),
            "kv_tile_ids_per_batch": paddle.full([int(kv_max_tile_size)], 0, dtype="int32"),
            "kv_num_blocks_x_cpu": paddle.full([1], 0, dtype="int32").cpu(),
        }

    def create_forward_meta(
        self,
        batch_size: int,
        seq_len: int,
        mode: ForwardMode,
        fd_config: FDConfig,
        attn_backend: AttentionBackend,
        past_kv_len: int = 0,
        existing_caches: list[paddle.Tensor] | None = None,
        existing_block_tables: paddle.Tensor | None = None,
        use_dynamic_quant: bool = False,
        free_blocks_pool: list[int] | None = None,
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
            seq_lens_decoder = paddle.full([batch_size], past_kv_len, dtype="int32")
            seq_lens_this_time = paddle.ones([batch_size], dtype="int32")
        else:
            raise ValueError(f"Unsupported ForwardMode: {mode}")

        attn_backend_buffers = self.create_attn_backend_buffers(
            fd_config.model_config, batch_size, fd_config.cache_config.block_size
        )

        if existing_caches is None:
            block_size = fd_config.cache_config.block_size
            max_model_len = fd_config.model_config.max_model_len
            num_blocks_per_seq = (max_model_len + block_size - 1) // block_size
            num_blocks = num_blocks_per_seq * batch_size
            head_dim = fd_config.model_config.head_dim
            kv_num_heads_tp = (
                fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size
            )
            num_layers = fd_config.model_config.num_hidden_layers
            cache_type = fd_config.model_config.dtype
            if use_dynamic_quant:
                cache_type = "uint8"
            cache_shape = (num_blocks, kv_num_heads_tp, block_size, head_dim)
            scale_shape = (num_blocks, kv_num_heads_tp, block_size)
            caches = []
            for _ in range(num_layers):
                key_cache = paddle.randint(0, 255, shape=cache_shape, dtype="int32").cast(cache_type)
                value_cache = paddle.randint(0, 255, shape=cache_shape, dtype="int32").cast(cache_type)
                caches.extend([key_cache, value_cache])
                if use_dynamic_quant:
                    key_cache_scale = paddle.rand(shape=scale_shape, dtype=fd_config.model_config.dtype)
                    value_cache_scale = paddle.rand(shape=scale_shape, dtype=fd_config.model_config.dtype)
                    caches.extend([key_cache_scale, value_cache_scale])
        else:
            caches = existing_caches

        if existing_block_tables is None:
            block_size = fd_config.cache_config.block_size
            max_model_len = fd_config.model_config.max_model_len
            num_blocks_per_seq = (max_model_len + block_size - 1) // block_size
            if free_blocks_pool is None:
                total_blocks_for_this_run = num_blocks_per_seq * batch_size
                free_blocks_pool = list(range(total_blocks_for_this_run - 1, -1, -1))
            block_tables = paddle.zeros(shape=(batch_size, num_blocks_per_seq), dtype="int32")
            num_blocks_to_alloc = (seq_len + block_size - 1) // block_size
            for i in range(batch_size):
                for j in range(num_blocks_to_alloc):
                    if not free_blocks_pool:
                        raise RuntimeError("Out of free blocks during test setup!")
                    block_tables[i, j] = free_blocks_pool.pop()
        else:
            block_tables = existing_block_tables

        tmp_position_ids = paddle.arange(fd_config.model_config.max_model_len).reshape((1, -1))
        rope_emb = get_rope(
            rotary_dim=fd_config.model_config.head_dim,
            position_ids=tmp_position_ids,
            base=fd_config.model_config.rope_theta,
            model_config=fd_config.model_config,
            partial_rotary_factor=fd_config.model_config.partial_rotary_factor,
        )

        input_ids = paddle.zeros([batch_size, seq_len if mode == ForwardMode.EXTEND else 1], dtype="int64")
        (
            ids_remove_padding,
            batch_id_per_token,
            cu_seqlens_q,
            cu_seqlens_k,
            output_cum_offsets,
            output_padding_offset,
        ) = pre_process(input_ids, seq_lens_this_time, False, None, seq_lens_encoder, seq_lens_decoder)

        meta = ForwardMeta(
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
        return meta, free_blocks_pool

    def profile_attention_layer(
        self,
        title: str,
        model: nn.Layer,
        hidden_states: paddle.Tensor,
        forward_meta: ForwardMeta,
        warmup_steps: int,
        test_steps: int,
    ):
        print(f"\n--- {title} ---")
        print(f"Input shape: {hidden_states.shape}")

        for _ in range(warmup_steps):
            _ = model(forward_meta, hidden_states)
        paddle.device.cuda.synchronize()

        start_time = time.time()
        for _ in range(test_steps):
            _ = model(forward_meta, hidden_states)
        paddle.device.cuda.synchronize()
        end_time = time.time()

        total_time = end_time - start_time
        avg_latency_ms = (total_time / test_steps) * 1000
        print(f"Result: Average latency is {avg_latency_ms:.4f} ms over {test_steps} steps.")
        return avg_latency_ms

    def create_fd_config_from_model_path(self, model_path, tensor_parallel_size=1):
        """Creates a complete FDConfig from a model path."""
        model_args = {"model": model_path, "dtype": "bfloat16"}
        model_config = ModelConfig(model_args)
        model_config.tensor_parallel_size = tensor_parallel_size
        parallel_config = ParallelConfig({"tensor_parallel_size": tensor_parallel_size, "data_parallel_size": 1})
        cache_config = CacheConfig(
            {
                "block_size": 64,
                "gpu_memory_utilization": 0.9,
                "cache_dtype": "bfloat16",
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
            graph_opt_config=GraphOptimizationConfig({}),
            commit_config=CommitConfig(),
            device_config=DeviceConfig({}),
            speculative_config=SpeculativeConfig({}),
            early_stop_config=EarlyStopConfig({}),
            plas_attention_config=None,
            test_mode=True,
        )

    # endregion

    def test_decode_performance_with_prefill(self):
        """
        Tests decode performance after a long prefill, using a pre-allocate,
        fill, and then profile pattern.
        """
        # Test parameters
        warmup_steps = 10
        test_steps = 100
        prefill_batch_size = 1
        decode_batch_size = 10  # This can be configured as needed
        prefill_seq_len = 9000
        use_dynamic_quant = True
        act_tensor_dtype = paddle.bfloat16

        # --- Step 1: Pre-allocate KV Cache for the max batch size ---
        print(f"\n--- Step 1: Pre-allocating KV Cache for max batch size {decode_batch_size} ---")
        large_meta, free_blocks_pool = self.create_forward_meta(
            batch_size=decode_batch_size,
            seq_len=prefill_seq_len,
            mode=ForwardMode.EXTEND,
            fd_config=self.fd_config,
            attn_backend=self.attn_backend,
            use_dynamic_quant=use_dynamic_quant,
        )
        print(f"Large meta created with Block Tables shape: {large_meta.block_tables.shape}")

        # --- Step 2: Run Prefill to populate the first cache slot ---
        print(f"\n--- Step 2: Running Prefill (BS={prefill_batch_size}, SeqLen={prefill_seq_len}) ---")
        prefill_hidden_states = paddle.randn(
            [prefill_batch_size * prefill_seq_len, self.fd_config.model_config.hidden_size],
            dtype=act_tensor_dtype,
        )

        prefill_meta_view, temp_pool = self.create_forward_meta(
            batch_size=prefill_batch_size,
            seq_len=prefill_seq_len,
            mode=ForwardMode.EXTEND,
            fd_config=self.fd_config,
            attn_backend=self.attn_backend,
            existing_caches=large_meta.caches,
            existing_block_tables=large_meta.block_tables[:prefill_batch_size],
            use_dynamic_quant=use_dynamic_quant,
            free_blocks_pool=free_blocks_pool,
        )

        self.attn_backend.init_attention_metadata(prefill_meta_view)
        with paddle.no_grad():
            _ = self.attention_layer(prefill_meta_view, prefill_hidden_states)
        paddle.device.cuda.synchronize()
        print("Prefill complete.")

        # --- Step 3: Profile Decode performance on all copies ---
        print(f"\n--- Step 3: Profiling Decode (BS={decode_batch_size}) ---")
        decode_hidden_states = paddle.randn(
            [decode_batch_size * 1, self.fd_config.model_config.hidden_size], dtype=act_tensor_dtype
        )

        decode_meta, _ = self.create_forward_meta(
            batch_size=decode_batch_size,
            seq_len=1,
            mode=ForwardMode.DECODE,
            fd_config=self.fd_config,
            attn_backend=self.attn_backend,
            past_kv_len=prefill_seq_len,
            existing_caches=large_meta.caches,
            existing_block_tables=large_meta.block_tables,
            use_dynamic_quant=use_dynamic_quant,
            free_blocks_pool=temp_pool,
        )

        self.attn_backend.init_attention_metadata(decode_meta)

        self.profile_attention_layer(
            f"Decode Perf (BS={decode_batch_size} after 1x{prefill_seq_len}-token Prefill)",
            self.attention_layer,
            decode_hidden_states,
            decode_meta,
            warmup_steps,
            test_steps,
        )


if __name__ == "__main__":
    unittest.main()
