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

import numpy as np
import paddle
import paddle.distributed as dist
import triton
import triton.language as tl
from paddleformers.utils.log import logger

from fastdeploy.cache_manager.routing_cache_manager import RoutingHostBufferView
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)


@enable_compat_on_triton_kernel
@triton.jit
def _save_routing_kernel_v2(
    device_routing_buffer_PTR,
    TOPK_IDS_PTR,
    LAYER_IDX,
    TOKEN_NUM,
    TOP_K,
    NUM_MOE_LAYERS,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    token_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    token_mask = token_offsets < TOKEN_NUM
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < TOP_K

    load_mask = token_mask[:, None] & k_mask[None, :]
    topk_vals = tl.load(
        TOPK_IDS_PTR + token_offsets[:, None] * TOP_K + k_offsets[None, :],
        mask=load_mask,
    )

    STRIDE_TOKEN = NUM_MOE_LAYERS * TOP_K
    STRIDE_LAYER = TOP_K
    output_ptrs = (
        device_routing_buffer_PTR
        + token_offsets[:, None] * STRIDE_TOKEN
        + LAYER_IDX * STRIDE_LAYER
        + k_offsets[None, :]
    )
    tl.store(output_ptrs, topk_vals, mask=load_mask)


def save_routing_to_buffer_v2(
    device_routing_buffer: paddle.Tensor,
    topk_ids: paddle.Tensor,
    layer_idx: int,
    tp_size: int,
    ep_size: int,
    tp_group: dist.communication.group.Group,
    total_token_num: int = -1,
):
    token_num_per_rank = topk_ids.shape[0]
    if token_num_per_rank == 0:
        return
    if tp_size > 1 and ep_size > 1:
        topk_ids_all = paddle.zeros([token_num_per_rank * tp_size, topk_ids.shape[1]], dtype=topk_ids.dtype)
        paddle.distributed.all_gather(topk_ids_all, topk_ids, tp_group)
        assert (
            total_token_num >= token_num_per_rank
        ), f"[R3] total_token_num={total_token_num} < token_num_per_rank={token_num_per_rank}"
        topk_ids = topk_ids_all[:total_token_num, :]

    token_num, top_k = topk_ids.shape
    buf_max_tokens, num_moe_layers, buf_top_k = device_routing_buffer.shape

    assert (
        token_num <= buf_max_tokens
    ), f"[R3] token_num={token_num} exceeds device_routing_buffer capacity={buf_max_tokens}"
    assert (
        top_k == buf_top_k
    ), f"[R3] top_k mismatch: topk_ids.top_k={top_k} vs device_routing_buffer.top_k={buf_top_k}"
    assert 0 <= layer_idx < num_moe_layers, f"[R3] layer_idx={layer_idx} out of range [0, {num_moe_layers})"

    BLOCK_SIZE_M = 128
    BLOCK_SIZE_K = triton.next_power_of_2(top_k)
    grid = (triton.cdiv(token_num, BLOCK_SIZE_M),)
    _save_routing_kernel_v2[grid](
        device_routing_buffer,
        topk_ids,
        LAYER_IDX=layer_idx,
        TOKEN_NUM=token_num,
        TOP_K=top_k,
        NUM_MOE_LAYERS=num_moe_layers,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )


class RoutedExpertsCapturer:
    """
    Worker-side routing capture: manages GPU transient buffer and GPU→CPU scatter.
    Does NOT manage request lifecycle — that is handled by RoutingCacheManager on the Engine side.
    """

    def __init__(self, fd_config: FDConfig, total_block_num: int):
        self.fd_config = fd_config
        self.max_num_seqs = fd_config.scheduler_config.max_num_seqs

        # Read routing params from centralized config
        rrc = fd_config.routing_replay_config
        self.num_moe_layers = rrc.num_moe_layers
        self.moe_top_k = rrc.moe_top_k
        self.routing_dtype = rrc.routing_dtype
        self.tp_rank = fd_config.parallel_config.tensor_parallel_rank

        logger.info(f"[R3] RoutedExpertsCapturer config: {rrc}")

        self._init_routing_cache(dtype=self.routing_dtype, total_block_num=total_block_num)

    def _init_routing_cache(self, dtype: str, total_block_num: int):
        """Initialize GPU transient buffer, staging buffers, and CPU pinned buffers."""
        max_num_kv_tokens = total_block_num * self.fd_config.cache_config.block_size

        # Small GPU transient buffer: only current step's token routing
        # TODO(Chengyanfu): Use max_num_batched_tokens to replace get_max_chunk_tokens()
        max_num_batched_tokens = self.fd_config.get_max_chunk_tokens()
        shape = [max_num_batched_tokens, self.num_moe_layers, self.moe_top_k]

        self.device_routing_buffer = paddle.full(shape=shape, fill_value=-1, dtype=dtype)
        self.routing_staging_buf = paddle.full(shape=shape, fill_value=-1, dtype=dtype)
        self.slot_mapping_staging_buf = paddle.zeros([max_num_batched_tokens], dtype=paddle.int64)

        self.cpu_routing_buf = paddle.zeros(shape, dtype=dtype).pin_memory()
        self.cpu_slot_mapping_buf = paddle.zeros([max_num_batched_tokens], dtype=paddle.int64).pin_memory()
        self._pending_save = None  # {"num_tokens": int}

        # Lazy attach to SharedMemory routing_host_buffer (created by Engine after profiling)
        self.routing_host_view = None
        self._routing_host_view_attach_attempted = False
        self._routing_host_view_shm_name = (
            f"routing_host_buffer.{str(self.fd_config.parallel_config.local_engine_worker_queue_port)}"
        )
        self._routing_host_view_shape = (max_num_kv_tokens, self.num_moe_layers, self.moe_top_k)
        self._routing_host_view_dtype = dtype

        gpu_buffer_bytes = int(np.prod(self.device_routing_buffer.shape)) * np.dtype(dtype).itemsize
        logger.info(
            f"[R3] GPU transient routing buffer: {self.device_routing_buffer.shape} "
            f"({gpu_buffer_bytes / 1024:.1f} KB)"
        )

    def _try_attach_routing_host_view(self):
        """Lazily attach to SharedMemory routing_host_buffer on first use."""
        if self._routing_host_view_attach_attempted:
            return
        self._routing_host_view_attach_attempted = True
        try:
            self.routing_host_view = RoutingHostBufferView(
                shape=self._routing_host_view_shape,
                dtype=self._routing_host_view_dtype,
                shm_name=self._routing_host_view_shm_name,
            )
            logger.info(f"[R3] Attached to RoutingHostBuffer SharedMemory: {self._routing_host_view_shm_name}")
        except FileNotFoundError:
            logger.warning(
                f"[R3] RoutingHostBuffer SharedMemory {self._routing_host_view_shm_name} not found. "
                "Routing capture will be skipped."
            )

    def prepare_pending_save(self, num_tokens: int, slot_mapping_gpu: paddle.Tensor):
        """
        Enqueue D2D + async D2H for routing data and slot_mapping.
        Must be called before post_process_event.record().
        All ops are enqueued on the current CUDA stream; CPU returns immediately.

        1. D2D (non-blocking): device_routing_buffer → routing_staging_buf
        2. D2D (non-blocking): slot_mapping_gpu → slot_mapping_staging_buf
        3. async D2H: routing_staging_buf → cpu_routing_buf
        4. async D2H: slot_mapping_staging_buf → cpu_slot_mapping_buf
        """
        if num_tokens > 0:
            # D2D: GPU → staging
            self.routing_staging_buf.copy_(self.device_routing_buffer, False)
            self.slot_mapping_staging_buf.copy_(slot_mapping_gpu, False)
            # Async D2H: staging → CPU pinned
            self.cpu_routing_buf.copy_(self.routing_staging_buf, False)
            self.cpu_slot_mapping_buf.copy_(self.slot_mapping_staging_buf, False)
            self._pending_save = {"num_tokens": num_tokens}
        else:
            self._pending_save = None

    def flush_pending_save(self):
        """
        Pure CPU operation. Called after post_process_event.synchronize(),
        which guarantees all D2D and D2H transfers have completed.
        Scatter from CPU pinned buffers to SharedMemory.
        """
        pending = self._pending_save
        if pending is None:
            return
        self._pending_save = None

        if self.routing_host_view is None:
            if not self._routing_host_view_attach_attempted:
                self._try_attach_routing_host_view()
            if self.routing_host_view is None:
                return

        num_tokens = pending["num_tokens"]
        # NOTE(gongshaotian): Slice pinned memory tensor maybe cause problem.
        data = self.cpu_routing_buf.cpu()[:num_tokens].numpy()
        slot_np = self.cpu_slot_mapping_buf.cpu()[:num_tokens].numpy()

        self.routing_host_view.scatter(slot_np, data)

    def get_device_routing_buffer(self) -> paddle.Tensor:
        return self.device_routing_buffer

    def clear(self):
        """Clear GPU buffer and pending save state. Used during RL round cleanup."""
        self.device_routing_buffer.fill_(-1)
        self._pending_save = None


# Backward compatibility alias
RoutingReplayManager = RoutedExpertsCapturer
