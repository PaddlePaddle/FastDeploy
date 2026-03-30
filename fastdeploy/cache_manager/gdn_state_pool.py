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
GDN (Gated Delta Network) State Pool — pre-allocated GPU tensor pool
for conv and SSM states used by Qwen3.5 linear attention layers.

Design:
  - Analogous to paged KV cache block pool, but each slot stores a
    complete per-request state (conv state + SSM state).
  - All GDN layers share a single pool object, indexed by layer_idx.
  - Slot 0 is reserved as a zero-filled padding sentinel.
    PAD_SLOT_ID (-1) is mapped to slot 0 via +1 offset when building
    gdn_slot_ids in ForwardMeta, so reads return zero and writes are
    harmless.

Pool layouts:
  conv_pool: [num_gdn_layers, pool_size, conv_dim, conv_kernel_size - 1]
  ssm_pool:  [num_gdn_layers, pool_size, num_v_heads, head_k_dim, head_v_dim]

  where pool_size = max_num_seqs + 1 (slot 0 = padding sentinel).
"""

import logging
from typing import List

import paddle

logger = logging.getLogger(__name__)

PAD_SLOT_ID = -1


class GDNSlotAllocator:
    """Lightweight CPU-only slot allocator for GDN state pool.

    Used by ResourceManagerV1 on the scheduler side to manage slot IDs
    without requiring paddle/GPU access. The corresponding GPU tensors
    live in GDNStatePool on the worker side.

    Slot 0 is reserved as a padding sentinel. Valid slots: 1..max_num_seqs.
    """

    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs
        self._free_slots: List[int] = list(range(max_num_seqs, 0, -1))

    def allocate(self) -> int:
        """Allocate a single slot ID.

        Returns:
            Allocated slot ID (1-based).

        Raises:
            RuntimeError: If no free slots available.
        """
        if not self._free_slots:
            raise RuntimeError(f"GDNSlotAllocator: no free slots (max_num_seqs={self.max_num_seqs})")
        return self._free_slots.pop()

    def free(self, slot_id: int):
        """Return a slot ID to the free list.

        Args:
            slot_id: Slot ID to free (1-based). Slot 0 is silently ignored.
        """
        if slot_id > 0:
            self._free_slots.append(slot_id)

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)


class GDNStatePool:
    """Pre-allocated GPU tensor pool for GDN conv and SSM states.

    Args:
        max_num_seqs: Maximum number of concurrent sequences.
        num_gdn_layers: Number of GDN (linear_attention) layers in the model.
        conv_dim: TP-local convolution dimension (key_dim * 2 + value_dim) // tp_size.
        conv_kernel_size: Causal conv1d kernel width (e.g. 4).
        num_v_heads: TP-local number of value heads (num_v_heads // tp_size).
        head_k_dim: Per-head key dimension.
        head_v_dim: Per-head value dimension.
        conv_dtype: Data type for conv state pool (default: bfloat16).
    """

    def __init__(
        self,
        max_num_seqs: int,
        num_gdn_layers: int,
        conv_dim: int,
        conv_kernel_size: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_dtype: paddle.dtype = paddle.bfloat16,
    ):
        self.max_num_seqs = max_num_seqs
        self.num_gdn_layers = num_gdn_layers
        self.conv_dim = conv_dim
        self.conv_kernel_size = conv_kernel_size
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim

        # pool_size = max_num_seqs + 1; slot 0 is the padding sentinel
        pool_size = max_num_seqs + 1

        # Conv state pool: [num_gdn_layers, pool_size, conv_dim, conv_kernel_size - 1]
        conv_state_len = conv_kernel_size - 1
        self.conv_pool = paddle.zeros(
            [num_gdn_layers, pool_size, conv_dim, conv_state_len],
            dtype=conv_dtype,
        )

        # SSM state pool: [num_gdn_layers, pool_size, num_v_heads, head_k_dim, head_v_dim]
        # K-first layout matching FLA kernel native format.
        # float32 for numerical stability (SSM state accumulates over many steps).
        self.ssm_pool = paddle.zeros(
            [num_gdn_layers, pool_size, num_v_heads, head_k_dim, head_v_dim],
            dtype=paddle.float32,
        )

        conv_mem_mb = (num_gdn_layers * pool_size * conv_dim * conv_state_len * paddle.finfo(conv_dtype).bits // 8) / (
            1024 * 1024
        )
        ssm_mem_mb = (num_gdn_layers * pool_size * num_v_heads * head_k_dim * head_v_dim * 4) / (1024 * 1024)
        logger.info(
            f"GDNStatePool allocated: "
            f"conv_pool {list(self.conv_pool.shape)} ({conv_mem_mb:.1f} MB), "
            f"ssm_pool {list(self.ssm_pool.shape)} ({ssm_mem_mb:.1f} MB)"
        )

        # Free slot list: valid slots are 1..max_num_seqs (slot 0 is sentinel)
        self._free_slots: List[int] = list(range(max_num_seqs, 0, -1))

        logger.info(
            f"GDNStatePool: {len(self._free_slots)} free slots available " f"(slot 0 reserved as padding sentinel)"
        )

    def allocate(self, n: int = 1) -> List[int]:
        """Allocate n slot IDs from the free list.

        Args:
            n: Number of slots to allocate.

        Returns:
            List of allocated slot IDs (1-based, already offset for pool indexing).

        Raises:
            RuntimeError: If not enough free slots available.
        """
        if len(self._free_slots) < n:
            raise RuntimeError(f"GDNStatePool: cannot allocate {n} slots, " f"only {len(self._free_slots)} free")
        allocated = [self._free_slots.pop() for _ in range(n)]
        return allocated

    def free(self, slot_ids: List[int]):
        """Return slot IDs to the free list and zero-out their state.

        Args:
            slot_ids: List of slot IDs to free (1-based pool indices).
                      Slot 0 (padding sentinel) is silently ignored.
        """
        valid = [s for s in slot_ids if s > 0]
        if not valid:
            return
        self.reset_slots(valid)
        self._free_slots.extend(valid)

    @property
    def num_free_slots(self) -> int:
        """Number of currently available slots."""
        return len(self._free_slots)

    def get_layer_conv_pool(self, layer_idx: int) -> paddle.Tensor:
        """Get conv state pool for a specific GDN layer.

        Returns:
            Tensor of shape [pool_size, conv_dim, conv_kernel_size - 1]
        """
        return self.conv_pool[layer_idx]

    def get_layer_ssm_pool(self, layer_idx: int) -> paddle.Tensor:
        """Get SSM state pool for a specific GDN layer.

        Returns:
            Tensor of shape [pool_size, num_v_heads, head_k_dim, head_v_dim]
        """
        return self.ssm_pool[layer_idx]

    def reset_slots(self, slot_ids: List[int]):
        """Zero-out conv and SSM state for given slots across all layers.

        Used when requests finish and their slots are returned to the free list.

        Args:
            slot_ids: List of slot indices to reset (already +1 offset applied).
        """
        if not slot_ids:
            return
        idx = paddle.to_tensor(slot_ids, dtype=paddle.int64)
        for layer_idx in range(self.num_gdn_layers):
            self.conv_pool[layer_idx][idx] = 0
            self.ssm_pool[layer_idx][idx] = 0

    @staticmethod
    def offset_slot_ids(raw_slot_ids: paddle.Tensor) -> paddle.Tensor:
        """Apply +1 offset to raw slot IDs so PAD_SLOT_ID (-1) maps to slot 0.

        Args:
            raw_slot_ids: [batch_size] int32, may contain PAD_SLOT_ID (-1).

        Returns:
            Offset slot IDs where -1 -> 0, 0 -> 1, 1 -> 2, etc.
        """
        return raw_slot_ids + 1
