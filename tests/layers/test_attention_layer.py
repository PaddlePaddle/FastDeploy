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

from __future__ import annotations

import os
import time

import numpy as np
import paddle
from paddle import nn

from fastdeploy.config import (
    CacheConfig,
    CommitConfig,
    DecodingConfig,
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

# from fastdeploy.config import FDConfig, ModelConfig, ParallelConfig, CacheConfig, SchedulerConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.layers.attention import (
    AttentionBackend,
    get_attention_backend,
)
from fastdeploy.model_executor.layers.attention.attention import Attention
from fastdeploy.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from fastdeploy.model_executor.layers.quantization import parse_quant_config
from fastdeploy.model_executor.layers.rotary_embedding import get_rope


def get_padding_offset(bsz, max_seq_len, seq_lens_this_time):
    cum_offsets_now = paddle.cumsum(max_seq_len - seq_lens_this_time, dtype="int32")
    cum_offsets = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cum_offsets[1:] = cum_offsets_now
    token_num = paddle.sum(seq_lens_this_time)
    padding_offsets = paddle.zeros(shape=(token_num), dtype="int32")
    cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
    cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
    for i in range(bsz):
        seq_len_now = seq_lens_this_time[i]
        cum_offset = cum_offsets[i]
        for j in range(seq_len_now):
            padding_offsets[i * max_seq_len - cum_offset + j] = cum_offset
        cum_seq_len = (i + 1) * max_seq_len - cum_offsets[i + 1]
        cu_seqlens_q[i + 1] = cum_seq_len
        cu_seqlens_k[i + 1] = cum_seq_len
    return padding_offsets, cum_offsets[:-1], cu_seqlens_q, cu_seqlens_k


class RopeEmbedding:
    def __init__(self, use_neox_rotary_style=False):
        self.use_neox_rotary_style = use_neox_rotary_style
        self.base = 10000

    def get_neox_style_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, 1, D]
        emb = paddle.concat([freqs, freqs], axis=-1).reshape((bsz, max_seq_len, 1, head_dim))

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def get_rotary_position_embedding(self, position_ids, head_dim):
        bsz, max_seq_len = position_ids.shape[:2]
        rot_emb = paddle.zeros((2, bsz, max_seq_len, 1, head_dim // 2), dtype="float32")
        inv_freq = self.base ** (-paddle.arange(0, head_dim, 2, dtype="float32") / head_dim)

        # shape: [B, S, D/2]
        freqs = paddle.einsum("ij,k->ijk", position_ids.cast("float32"), inv_freq)
        # shape: [B, S, D/2]
        emb = paddle.stack([freqs], axis=-1).reshape((bsz, max_seq_len, head_dim // 2))
        # shape: [B, S, 1, D]
        emb = paddle.unsqueeze(emb, 2)

        rot_emb[0] = paddle.cos(emb)
        rot_emb[1] = paddle.sin(emb)
        return rot_emb

    def _apply_rope(self, rotary_emb, q, k, v=None, causal=False):
        # sin [sequence_length, embed_size_per_head//2]
        # cos [sequence_length, embed_size_per_head//2]
        # sin, cos = paddle.chunk(rp, 2, axis=-1)
        seq, head_dim = q.shape[2], q.shape[3]
        cos, sin = paddle.chunk(rotary_emb, 2, axis=0)
        cos = paddle.squeeze(cos, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        sin = paddle.squeeze(sin, axis=0).transpose([0, 2, 1, 3])[:, :, :seq, :]
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]

        if self.use_neox_rotary_style:
            sin_pos = sin
            cos_pos = cos
            # NeoX Stype：前后半部分分块旋转
            rotate_half_q = paddle.reshape(
                paddle.stack(
                    [
                        -q[:, :, :, q.shape[-1] // 2 :],
                        q[:, :, :, : q.shape[-1] // 2],
                    ],
                    axis=-1,
                ),
                paddle.shape(q),
            )
            rotate_half_k = paddle.reshape(
                paddle.stack(
                    [
                        -k[:, :, :, k.shape[-1] // 2 :],
                        k[:, :, :, : k.shape[-1] // 2],
                    ],
                    axis=-1,
                ),
                paddle.shape(k),
            )
        else:
            # import pdb;pdb.set_trace()
            sin_pos = paddle.reshape(paddle.stack([sin, sin], axis=-1), [1, 1, seq, head_dim])
            # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
            cos_pos = paddle.reshape(paddle.stack([cos, cos], axis=-1), [1, 1, seq, head_dim])
            # GPT Stype：奇偶位置分块旋转
            rotate_half_q = paddle.reshape(
                paddle.stack([-q[:, :, :, 1::2], q[:, :, :, 0::2]], axis=-1),
                paddle.shape(q),
            )
            rotate_half_k = paddle.reshape(
                paddle.stack([-k[:, :, :, 1::2], k[:, :, :, 0::2]], axis=-1),
                paddle.shape(k),
            )

        query = paddle.add(paddle.multiply(q, cos_pos), paddle.multiply(rotate_half_q, sin_pos))

        key = paddle.add(paddle.multiply(k, cos_pos), paddle.multiply(rotate_half_k, sin_pos))

        return paddle.cast(query, q.dtype), paddle.cast(key, k.dtype)


# ===================================================================
# 修改后的 Ernie4_5_Attention 类
# ===================================================================
class Ernie4_5_Attention(nn.Layer):
    def __init__(self, fd_config: FDConfig, layer_id: int, prefix: str) -> None:
        super().__init__()

        self.qkv_proj = QKVParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            fd_config=fd_config,
            prefix=f"{prefix}.o_proj",
            input_size=fd_config.model_config.head_dim * fd_config.model_config.num_attention_heads,
            output_size=fd_config.model_config.hidden_size,
        )
        self.attn = Attention(
            fd_config=fd_config,
            layer_id=layer_id,
            prefix=prefix,
            use_neox_rotary_style=False,
        )

        # ================== 新增: 权重初始化逻辑 (参考 FusedMoE 单测) ==================
        # 这段代码确保 Attention 层在实例化时就自动初始化权重
        print(f"INFO: Initializing weights for {prefix} inside __init__...")
        paddle.seed(1024 + layer_id)  # 使用 layer_id 保证不同层的随机种子不同

        with paddle.no_grad():
            # 从 fd_config 获取维度信息
            hidden_size = fd_config.model_config.hidden_size
            tp_size = fd_config.parallel_config.tensor_parallel_size
            tensor_dtype = paddle.to_tensor(0, dtype=fd_config.model_config.dtype).dtype

            # QKVParallelLinear (ColumnParallel) 的权重形状
            q_dims = fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim
            kv_dims = fd_config.model_config.num_key_value_heads * fd_config.model_config.head_dim
            total_output_dim = q_dims + 2 * kv_dims
            qkv_proj_output_dim_tp = total_output_dim // tp_size

            qkv_weight_shape = [hidden_size, qkv_proj_output_dim_tp]
            # qkv_bias_shape = [qkv_proj_output_dim_tp]

            # RowParallelLinear (RowParallel) 的权重形状
            o_proj_input_dim = fd_config.model_config.num_attention_heads * fd_config.model_config.head_dim
            o_proj_input_dim_tp = o_proj_input_dim // tp_size

            o_proj_weight_shape = [o_proj_input_dim_tp, hidden_size]
            # o_proj_bias_shape = [hidden_size]

            # 创建随机张量
            qkv_weight = paddle.randn(qkv_weight_shape, dtype=tensor_dtype)
            # qkv_bias = paddle.zeros(qkv_bias_shape, dtype=tensor_dtype)
            o_proj_weight = paddle.randn(o_proj_weight_shape, dtype=tensor_dtype)
            # o_proj_bias = paddle.zeros(o_proj_bias_shape, dtype=tensor_dtype)

            # 构建 state_dict 并加载
            state_dict = {
                f"{prefix}.qkv_proj.weight": qkv_weight,
                # f"{prefix}.qkv_proj.bias": qkv_bias,
                f"{prefix}.o_proj.weight": o_proj_weight,
                # f"{prefix}.o_proj.bias": o_proj_bias,
            }

            self.load_state_dict(state_dict)
        print(f"INFO: Weights for {prefix} loaded successfully.")
        # =============================================================================

    # ================== 新增: load_state_dict 方法 ==================
    def load_state_dict(self, state_dict):
        # 这个方法将权重分发给正确的子模块
        # 注意: QKVParallelLinear 和 RowParallelLinear 内部已经处理好了
        # 如何从一个大的 state_dict 中只挑选自己需要的部分
        self.qkv_proj.load_state_dict(state_dict)
        self.o_proj.load_state_dict(state_dict)
        # self.attn 可能没有需要加载的权重，但保留调用是好的实践
        # self.attn.load_state_dict(state_dict)

    # =================================================================

    def forward(
        self,
        forward_meta: ForwardMeta,
        hidden_states: paddle.Tensor,
    ):
        qkv_out = self.qkv_proj(hidden_states)

        attn_out = self.attn(
            qkv=qkv_out,
            forward_meta=forward_meta,
        )

        output = self.o_proj(attn_out)

        return output


# ===================================================================
# 步骤 3: 创建 ForwardMeta 的专业辅助函数
# ===================================================================
# NEW: 这个新函数精确地模拟了 initialize_attn_backend 的缓冲区创建逻辑
def _create_attn_backend_buffers(m_config: ModelConfig, batch_size: int, block_size: int) -> dict:
    """
    根据 GPUModelRunner.initialize_attn_backend 的逻辑，预分配 Attention 后端所需的元数据缓冲区。
    """
    # 这些是 Attention Kernel 的内部参数
    encoder_block_shape_q = 64
    decoder_block_shape_q = 16
    # 假设非 speculative decoding
    decoder_step_token_num = 1

    num_heads = m_config.num_attention_heads
    kv_num_heads = m_config.num_key_value_heads
    group_size = np.ceil(num_heads / kv_num_heads)

    # 核心计算: 确定缓冲区的最大尺寸，以应对最坏情况
    decode_max_tile_size = 1024 * batch_size * np.ceil((decoder_step_token_num * group_size) / decoder_block_shape_q)
    encode_max_tile_size = batch_size * np.ceil((m_config.max_model_len * group_size) / encoder_block_shape_q)
    kv_max_tile_size = batch_size * np.ceil(m_config.max_model_len / block_size)

    # 创建并返回包含所有缓冲区的字典
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
        "max_len_kv_cpu": paddle.full([1], 0, dtype="int32").cpu(),
    }


def create_forward_meta(
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
    Creates a high-fidelity ForwardMeta object, strictly following the logic and
    data structures of the production `initialize_kv_cache` function.
    """
    # ... (seq_lens, cu_seqlens, attn_backend_buffers calculations remain the same) ...
    if mode == ForwardMode.EXTEND:
        total_tokens = batch_size * seq_len
        seq_lens_encoder = paddle.full([batch_size], seq_len, dtype="int32")
        seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_this_time = seq_lens_encoder
    elif mode == ForwardMode.DECODE:
        total_tokens = batch_size
        seq_lens_encoder = paddle.zeros([batch_size], dtype="int32")
        seq_lens_decoder = paddle.full([batch_size], past_kv_len, dtype="int32")
        seq_lens_this_time = paddle.ones([batch_size], dtype="int32")
    else:
        raise ValueError(f"Unsupported ForwardMode: {mode}")

    cu_seqlens_q = paddle.arange(0, total_tokens + 1, seq_len if mode == ForwardMode.EXTEND else 1, dtype="int32")
    cu_seqlens_k = cu_seqlens_q

    attn_backend_buffers = _create_attn_backend_buffers(
        fd_config.model_config, batch_size, fd_config.cache_config.block_size
    )

    # --- Cache Creation Block: Replicated from `initialize_kv_cache` ---
    if existing_caches is None:
        # --- MODIFIED BLOCK START ---
        # The following logic is now aligned with your `init_tensor` reference function.
        # We calculate the required blocks for this specific test run, rather than using a global value.

        block_size = fd_config.cache_config.block_size
        max_model_len = fd_config.model_config.max_model_len

        # NEW: Calculate the number of blocks needed per sequence for the entire lifetime.
        num_blocks_per_seq = (max_model_len + block_size - 1) // block_size

        # NEW: Calculate the total number of blocks needed for the entire batch.
        # This is the `max_block_num` from your reference code.
        num_blocks = num_blocks_per_seq * batch_size

        # --- MODIFIED BLOCK END ---

        head_dim = fd_config.model_config.head_dim
        kv_num_heads_tp = fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size
        num_layers = fd_config.model_config.num_hidden_layers

        cache_type = fd_config.model_config.dtype
        if use_dynamic_quant:
            cache_type = "uint8"

        # MODIFIED: The shape now uses the dynamically calculated `num_blocks`.
        cache_shape = (num_blocks, kv_num_heads_tp, block_size, head_dim)
        scale_shape = (num_blocks, kv_num_heads_tp, block_size)

        caches = []
        for _ in range(num_layers):
            key_cache = paddle.full(shape=cache_shape, fill_value=0, dtype=cache_type)
            value_cache = paddle.full(shape=cache_shape, fill_value=0, dtype=cache_type)
            caches.append(key_cache)
            caches.append(value_cache)

            if use_dynamic_quant:
                key_cache_scale = paddle.full(shape=scale_shape, fill_value=0, dtype=fd_config.model_config.dtype)
                value_cache_scale = paddle.full(shape=scale_shape, fill_value=0, dtype=fd_config.model_config.dtype)
                caches.append(key_cache_scale)
                caches.append(value_cache_scale)
    else:
        caches = existing_caches

    if existing_block_tables is None:
        block_size = fd_config.cache_config.block_size
        max_model_len = fd_config.model_config.max_model_len

        num_blocks_per_seq = (max_model_len + block_size - 1) // block_size

        if free_blocks_pool is None:
            # MODIFIED: The pool of free blocks is now sized according to the blocks we
            # just calculated for our cache, making the test self-contained.
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

    # Rope
    # rope = RopeEmbedding()
    tmp_position_ids = paddle.arange(fd_config.model_config.max_model_len).reshape((1, -1))
    print("====RyanDebug, the fd_config.model_config.max_model_len is:", fd_config.model_config.max_model_len)
    # rope_emb = rope.get_rotary_position_embedding(tmp_position_ids, fd_config.model_config.head_dim)

    rope_emb = get_rope(
        rotary_dim=fd_config.model_config.head_dim,
        position_ids=tmp_position_ids,
        base=fd_config.model_config.rope_theta,
        model_config=fd_config.model_config,
        partial_rotary_factor=fd_config.model_config.partial_rotary_factor,
    )
    # print("===RyanDebug, the rope_emb_2 is :", rope_emb)

    # padding offset
    max_seq_len_for_call = seq_len if mode != ForwardMode.DECODE else 1
    padding_offset, cum_offset_old, cu_seqlens_q_old, cu_seqlens_k_old = get_padding_offset(
        batch_size, max_seq_len_for_call, seq_lens_this_time
    )

    meta = ForwardMeta(
        input_ids=paddle.zeros([batch_size, seq_len if mode == ForwardMode.EXTEND else 1], dtype="int64"),
        ids_remove_padding=paddle.zeros([total_tokens], dtype="int64"),
        seq_lens_encoder=seq_lens_encoder,
        seq_lens_decoder=seq_lens_decoder,
        seq_lens_this_time=seq_lens_this_time,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        block_tables=block_tables,
        caches=caches,
        rotary_embs=rope_emb,
        step_use_cudagraph=False,
        attn_backend=attn_backend,
        forward_mode=ForwardMode.MIXED,  # Append Attn 只有这一个模式!
        attn_mask=None,
        attn_mask_offsets=None,
        **attn_backend_buffers,
    )
    # Just different name
    meta.batch_id_per_token = padding_offset

    return meta, free_blocks_pool


# ===================================================================
# 步骤 4: 性能测试核心函数
# ===================================================================
def profile_attention_layer(
    title: str,
    model: nn.Layer,
    hidden_states: paddle.Tensor,
    forward_meta: ForwardMeta,
    warmup_steps: int,
    test_steps: int,
):
    print(f"\n--- {title} ---")
    print(f"Input shape: {hidden_states.shape}")

    # 预热
    for _ in range(warmup_steps):
        _ = model(forward_meta, hidden_states)
    paddle.device.cuda.synchronize()

    # 正式测试
    start_time = time.time()
    for _ in range(test_steps):
        _ = model(forward_meta, hidden_states)
    paddle.device.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / test_steps) * 1000
    print(f"Result: Average latency is {avg_latency_ms:.4f} ms over {test_steps} steps.")
    return avg_latency_ms


def create_fd_config_from_model_path(model_path, tensor_parallel_size=1):
    """从模型路径创建完整的 FDConfig，会自动读取 config.json 并初始化所有配置项。"""

    # --- 已有的配置 ---
    model_args = {
        "model": model_path,
        "dtype": "bfloat16",
        "runner": "generate",
        "convert": "none",
    }
    model_config = ModelConfig(model_args)
    # 确保内部张量并行大小与外部参数同步
    model_config.tensor_parallel_size = tensor_parallel_size

    parallel_args = {"tensor_parallel_size": tensor_parallel_size, "data_parallel_size": 1}
    parallel_config = ParallelConfig(parallel_args)

    cache_args = {
        "block_size": 64,
        "gpu_memory_utilization": 0.9,
        "cache_dtype": "bfloat16",
        "model_cfg": model_config,
        "tensor_parallel_size": tensor_parallel_size,
    }
    cache_config = CacheConfig(cache_args)

    scheduler_args = {"name": "local", "max_num_seqs": 256, "max_num_batched_tokens": 32768}
    scheduler_config = SchedulerConfig(scheduler_args)

    load_config = LoadConfig({})
    graph_opt_config = GraphOptimizationConfig({})

    # --- 新增的配置（使用合理的默认值）---

    # 对于此测试，这些配置不是核心，因此使用默认构造函数即可
    commit_config = CommitConfig()
    device_config = DeviceConfig({})
    decoding_config = DecodingConfig({})
    speculative_config = SpeculativeConfig({})
    early_stop_config = EarlyStopConfig({})

    # 可选配置，对于此测试可以为 None
    plas_attention_config = None

    # --- 组装最终的 FDConfig ---
    fd_config = FDConfig(
        model_config=model_config,
        cache_config=cache_config,
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        load_config=load_config,
        graph_opt_config=graph_opt_config,
        # 注入所有新增的配置对象
        commit_config=commit_config,
        device_config=device_config,
        decoding_config=decoding_config,
        speculative_config=speculative_config,
        early_stop_config=early_stop_config,
        plas_attention_config=plas_attention_config,
        # 传递 test_mode=True 以跳过生产环境的检查
        test_mode=True,
    )

    return fd_config


# ===================================================================
# 步骤 5: 主程序
# ===================================================================
# ===================================================================
# 步骤 5: 主程序 (修改后)
# ===================================================================
def run_prefill_and_copy_decode_test():
    """
    本函数基于一个清晰的“预分配-填充-复制”模式，实现以下测试场景：
    1. 预分配：为最终的 decode_batch_size (96) 分配完整的、空的 KV Cache 物理空间和 block_tables。
    2. 填充：仅使用 prefill_batch_size (1) 对缓存的第一个“槽位”进行前向计算，填充真实数据。
    3. 复制：将第一个槽位的数据，物理复制到其余95个槽位中。
    4. 解码：在96个拥有独立、相同数据的 KV Cache 副本上执行性能测试。

    此实现逻辑清晰，代码优雅，非常适合作为功能验证和性能分析的模板。
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="""Run performance test for the Attention layer with physical KV Cache copying.
        This script tests the decode performance for a given batch size after a long prefill."""
    )
    parser.add_argument(
        "-b", "--bs", type=int, required=True, help="The decode batch size to be tested (e.g., 10, 32, 96)."
    )
    args = parser.parse_args()

    # ===================================================================
    # 步骤 0: 全局变量设置
    # ===================================================================
    warmup_steps = 10
    test_steps = 1000

    # 分离 prefill 和 decode 的 batch size
    prefill_batch_size = 1
    decode_batch_size = args.bs
    prefill_seq_len = 9000

    use_dynamic_quant = True
    paddle.set_device("gpu")
    act_tensor_dtype = paddle.bfloat16

    config_path = "./"
    fd_config = create_fd_config_from_model_path(config_path, tensor_parallel_size=1)

    fd_config.model_config.max_model_len = 2 * (prefill_seq_len + 128)
    fd_config.model_config.num_hidden_layers = 1
    fd_config.parallel_config.tp_group = [0]
    print("====RYanDebug, the quantization_config is:", fd_config.model_config.quantization_config)

    # ... 其他配置保持不变 ...
    import types

    mock_args = types.SimpleNamespace()
    mock_args.quantization = {"quantization": "block_wise_fp8"}
    mock_args.dynamic_load_weight = False
    quant_config = parse_quant_config(
        mock_args,
        fd_config.model_config,
        is_ernie=1,
        is_v1_loader=1,
    )
    fd_config.quant_config = quant_config
    # ===================================================================

    # ... (Attention Backend 和 Layer 初始化保持不变) ...
    print("===== Initializing Attention Backend and Model Layer =====")
    os.environ["FD_ATTENTION_BACKEND"] = "APPEND_ATTN"
    attn_cls = get_attention_backend()
    attn_backend = attn_cls(
        fd_config,
        kv_num_heads=fd_config.model_config.num_key_value_heads // fd_config.parallel_config.tensor_parallel_size,
        num_heads=fd_config.model_config.num_attention_heads // fd_config.parallel_config.tensor_parallel_size,
        head_dim=fd_config.model_config.head_dim,
        encoder_block_shape_q=64,
        decoder_block_shape_q=16,
    )
    attention_layer = Ernie4_5_Attention(fd_config, layer_id=0, prefix="test_layer")
    attention_layer.attn.cache_quant_type_str = "block_wise_fp8"
    print("===== Initialization Complete =====")

    # --- 步骤 1: 预分配。为 decode_batch_size (96) 创建一个大的、空的元数据对象 ---
    print(f"\n--- Step 1: Pre-allocating KV Cache for max batch size {decode_batch_size} ---")

    # 这个 'large_meta' 对象包含了足够96个序列使用的物理缓存(caches)和地址簿(block_tables)
    large_meta, free_blocks_pool = create_forward_meta(
        batch_size=decode_batch_size,  # 关键：使用最终的 batch_size 来分配资源
        seq_len=prefill_seq_len,
        mode=ForwardMode.EXTEND,
        fd_config=fd_config,
        attn_backend=attn_backend,
        use_dynamic_quant=use_dynamic_quant,
    )
    print("Large meta object created with:")
    print(f"  - Caches: {len(large_meta.caches)} tensors")
    print(f"  - Block Tables shape: {large_meta.block_tables.shape}")

    # --- 步骤 2: 填充。仅对第一个序列执行 Prefill ---
    print(f"\n--- Step 2: Running Prefill to fill the first cache slot (BS=1, SeqLen={prefill_seq_len}) ---")

    prefill_hidden_states = paddle.randn(
        [prefill_batch_size * prefill_seq_len, fd_config.model_config.hidden_size], dtype=act_tensor_dtype
    )

    # 创建一个临时的、只包含第一个序列的 meta，但让它指向大的物理缓存
    # 这是通过复用 large_meta.caches 和 large_meta.block_tables 的第一个切片来实现的
    prefill_meta_view, temp_pool = create_forward_meta(
        batch_size=prefill_batch_size,
        seq_len=prefill_seq_len,
        mode=ForwardMode.EXTEND,
        fd_config=fd_config,
        attn_backend=attn_backend,
        existing_caches=large_meta.caches,  # 复用大的物理缓存
        existing_block_tables=large_meta.block_tables[:prefill_batch_size],  # 只使用第一行地址
        use_dynamic_quant=use_dynamic_quant,
        free_blocks_pool=free_blocks_pool,  # 传递剩余的空闲池
    )

    attn_backend.init_attention_metadata(prefill_meta_view)

    # 执行 Prefill，这会填充 large_meta.caches 中由 block_tables[0] 指向的物理块
    with paddle.no_grad():
        _ = attention_layer(prefill_meta_view, prefill_hidden_states)
    paddle.device.cuda.synchronize()
    print("Prefill complete. The first slot of the KV Cache is now populated.")

    # --- 步骤 3: 复制。将第一个槽位的数据物理复制到其余95个槽位 ---
    print(f"\n--- Step 3: Replicating KV Cache data from slot 0 to slots 1-{decode_batch_size-1} ---")

    block_size = fd_config.cache_config.block_size
    num_blocks_for_prefill = (prefill_seq_len + block_size - 1) // block_size

    with paddle.no_grad():
        # 获取源 block 地址 (只取实际用到的部分)
        source_blocks = large_meta.block_tables[0, :num_blocks_for_prefill]

        # 遍历所有层
        num_tensors_per_layer = 4 if use_dynamic_quant else 2
        for layer_idx in range(len(large_meta.caches) // num_tensors_per_layer):
            base_idx = layer_idx * num_tensors_per_layer

            # 为目标序列 (1 到 95) 复制数据
            for seq_idx in range(1, decode_batch_size):
                target_blocks = large_meta.block_tables[seq_idx, :num_blocks_for_prefill]

                # 使用高级索引进行高效的物理复制
                # 复制 Key Cache
                large_meta.caches[base_idx][target_blocks] = large_meta.caches[base_idx][source_blocks]
                # 复制 Value Cache
                large_meta.caches[base_idx + 1][target_blocks] = large_meta.caches[base_idx + 1][source_blocks]

                if use_dynamic_quant:
                    # 复制 Key Scale
                    large_meta.caches[base_idx + 2][target_blocks] = large_meta.caches[base_idx + 2][source_blocks]
                    # 复制 Value Scale
                    large_meta.caches[base_idx + 3][target_blocks] = large_meta.caches[base_idx + 3][source_blocks]

    paddle.device.cuda.synchronize()  # 确保所有复制操作完成
    print("KV Cache replication completed successfully.")

    # --- 步骤 4: 解码。在96个独立的 Cache 副本上进行性能测试 ---
    print(f"\n--- Step 4: Profiling Decode with independent cache copies (BS={decode_batch_size}) ---")

    decode_hidden_states = paddle.randn(
        [decode_batch_size * 1, fd_config.model_config.hidden_size], dtype=act_tensor_dtype
    )

    # 创建最终的 decode_meta，它使用包含了96份副本的 caches 和完整的 block_tables
    decode_meta, _ = create_forward_meta(
        batch_size=decode_batch_size,
        seq_len=1,
        mode=ForwardMode.DECODE,
        fd_config=fd_config,
        attn_backend=attn_backend,
        past_kv_len=prefill_seq_len,
        existing_caches=large_meta.caches,  # 使用被完全填充的大缓存
        existing_block_tables=large_meta.block_tables,  # 使用完整的地址簿
        use_dynamic_quant=use_dynamic_quant,
        free_blocks_pool=temp_pool,  # 使用经过Prefill消耗后的空闲池
    )

    attn_backend.init_attention_metadata(decode_meta)

    # 执行性能测试
    profile_attention_layer(
        f"Decode Perf with Copied Cache (BS={decode_batch_size} after 1x{prefill_seq_len}-token Prefill)",
        attention_layer,
        decode_hidden_states,
        decode_meta,
        warmup_steps,
        test_steps,
    )


if __name__ == "__main__":
    # 运行实现了物理复制的测试函数
    run_prefill_and_copy_decode_test()
    # run_prefill_then_decode_test()
