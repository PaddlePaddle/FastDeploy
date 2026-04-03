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

StagingManager: manages staging buffers for per-block storage transfers.

Wraps a StorageConnector and provides batch_set_block / batch_get_block
methods that transparently gather scattered per-layer host memory into
contiguous staging buffers (for writes) or scatter contiguous staging
data back to per-layer host memory (for reads).

The caller (CacheTransferManager) does not need to know about the
staging buffer details.
"""

import ctypes
from typing import TYPE_CHECKING, Dict, List

from paddleformers.utils.log import logger

if TYPE_CHECKING:
    from .base import StorageConnector


# Buffer kinds for key/value cache and optional FP8 scales
_CACHE_KINDS = ("key", "value")
_SCALE_KINDS = ("key_scale", "value_scale")


class StagingManager:
    """
    Manages pinned staging buffers for per-block (all-layers-packed) storage I/O.

    Staging buffers are allocated once via ``initialize()`` and reused across
    calls.  Separate read/write buffers ensure thread safety between
    concurrent ``batch_get_block`` (read from storage) and
    ``batch_set_block`` (write to storage) operations.

    Memory layout per staging buffer (for one kind, e.g. "key")::

        [block_0_layer_0 | block_0_layer_1 | ... | block_0_layer_N-1 |
         block_1_layer_0 | block_1_layer_1 | ... | block_1_layer_N-1 |
         ...
         block_B_layer_0 | ... | block_B_layer_N-1 ]

        where B = staging_batch_size, N = num_layers,
        each segment is ``per_layer_stride`` bytes.

    Args:
        connector: Underlying StorageConnector for RDMA transfers.
        staging_batch_size: Max blocks processed in one staging round.
    """

    def __init__(
        self,
        connector: "StorageConnector",
        staging_batch_size: int = 64,
    ):
        self._connector = connector
        self._staging_batch_size = staging_batch_size

        # Populated by initialize()
        self._num_layers: int = 0
        self._strides: Dict[str, int] = {}  # kind -> bytes per block per layer
        self._bufs: Dict[str, int] = {}  # "{read|write}_{kind}" -> pinned ptr
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
        self,
        num_layers: int,
        strides: Dict[str, int],
    ) -> None:
        """
        Allocate and RDMA-register staging buffers.

        Must be called after the storage connector is connected and
        host block strides are known.

        Args:
            num_layers: Number of transformer layers.
            strides: Per-layer stride in bytes for each kind.
                Required keys: ``"key"``, ``"value"``.
                Optional keys: ``"key_scale"``, ``"value_scale"`` (FP8).
        """
        if self._initialized:
            return

        from fastdeploy.cache_manager.ops import cuda_host_alloc

        self._num_layers = num_layers
        self._strides = dict(strides)

        kinds = list(strides.keys())
        total_bytes = 0
        for direction in ("read", "write"):
            for kind in kinds:
                per_block = num_layers * strides[kind]
                buf_bytes = self._staging_batch_size * per_block
                buf_name = f"{direction}_{kind}"

                ptr = cuda_host_alloc(buf_bytes)
                self._bufs[buf_name] = ptr
                total_bytes += buf_bytes

                # Register with RDMA so batch_get / batch_set can use it
                if self._connector is not None:
                    self._connector.register_buffer(ptr, buf_bytes)

        logger.info(
            f"[StagingManager] Allocated {len(kinds) * 2} staging buffers: "
            f"{total_bytes / 1024**3:.3f} GB total "
            f"({self._staging_batch_size} blocks x {num_layers} layers, "
            f"kinds={kinds})"
        )

        self._initialized = True

    def shutdown(self) -> None:
        """Free all staging buffers."""
        if not self._initialized:
            return

        from fastdeploy.cache_manager.ops import cuda_host_free

        for buf_name, ptr in self._bufs.items():
            if ptr:
                try:
                    cuda_host_free(ptr)
                except Exception as e:
                    logger.warning(f"[StagingManager] Failed to free {buf_name}: {e}")
        self._bufs.clear()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def staging_batch_size(self) -> int:
        return self._staging_batch_size

    def total_staging_bytes(self) -> int:
        """Total pinned memory used by all staging buffers (for segment budget)."""
        total = 0
        for kind, stride in self._strides.items():
            per_block = self._num_layers * stride
            # read + write
            total += 2 * self._staging_batch_size * per_block
        return total

    def compute_staging_bytes(
        self,
        num_layers: int,
        strides: Dict[str, int],
    ) -> int:
        """
        Compute staging memory needed *before* allocating (for segment budget).

        Call this before connector.connect() to include staging in
        global_segment_size.
        """
        total = 0
        for kind, stride in strides.items():
            total += 2 * self._staging_batch_size * num_layers * stride
        return total

    # ------------------------------------------------------------------
    # Gather / Scatter helpers
    # ------------------------------------------------------------------

    def _gather_block(
        self,
        direction: str,
        kind: str,
        batch_offset: int,
        cpu_block_id: int,
        host_ptrs: List[int],
    ) -> None:
        """
        Gather one block from per-layer host buffers into contiguous staging.

        Args:
            direction: "read" or "write".
            kind: "key", "value", "key_scale", or "value_scale".
            batch_offset: Index of this block within the staging batch.
            cpu_block_id: Host block ID.
            host_ptrs: Per-layer base pointers (len == num_layers).
        """
        stride = self._strides[kind]
        buf = self._bufs[f"{direction}_{kind}"]
        block_base = buf + batch_offset * (self._num_layers * stride)

        for layer_idx in range(self._num_layers):
            src = host_ptrs[layer_idx] + cpu_block_id * stride
            dst = block_base + layer_idx * stride
            ctypes.memmove(dst, src, stride)

    def _scatter_block(
        self,
        direction: str,
        kind: str,
        batch_offset: int,
        cpu_block_id: int,
        host_ptrs: List[int],
    ) -> None:
        """
        Scatter one block from contiguous staging into per-layer host buffers.

        Args:
            direction: "read" or "write".
            kind: "key", "value", "key_scale", or "value_scale".
            batch_offset: Index of this block within the staging batch.
            cpu_block_id: Host block ID.
            host_ptrs: Per-layer base pointers (len == num_layers).
        """
        stride = self._strides[kind]
        buf = self._bufs[f"{direction}_{kind}"]
        block_base = buf + batch_offset * (self._num_layers * stride)

        for layer_idx in range(self._num_layers):
            src = block_base + layer_idx * stride
            dst = host_ptrs[layer_idx] + cpu_block_id * stride
            ctypes.memmove(dst, src, stride)

    # ------------------------------------------------------------------
    # Public block-level I/O
    # ------------------------------------------------------------------

    def batch_set_block(
        self,
        keys_per_kind: Dict[str, List[str]],
        host_ptrs_per_kind: Dict[str, List[int]],
        cpu_block_ids: List[int],
    ) -> List[bool]:
        """
        Write blocks (all layers packed per key) to storage.

        For each block, gathers per-layer host data into the write staging
        buffer, then calls the connector's ``batch_set`` once per chunk.

        Args:
            keys_per_kind: ``{kind: [key_for_block_0, key_for_block_1, ...]}``
                Each kind (e.g. "key", "value") maps to a list of storage keys
                aligned with ``cpu_block_ids``.
            host_ptrs_per_kind: ``{kind: per_layer_ptrs}``
                Each kind maps to a list of per-layer base pointers.
            cpu_block_ids: Source CPU block IDs.

        Returns:
            List[bool]: True for each block where ALL kinds succeeded.
        """
        if not self._initialized:
            logger.warning("[StagingManager] batch_set_block: not initialized")
            return [False] * len(cpu_block_ids)

        num_blocks = len(cpu_block_ids)
        block_success = [True] * num_blocks
        batch_size = self._staging_batch_size
        kinds = list(keys_per_kind.keys())

        # Precompute per-kind constants (invariant across all chunks)
        per_block_bytes = {kind: self._num_layers * self._strides[kind] for kind in kinds}
        write_bufs = {kind: self._bufs[f"write_{kind}"] for kind in kinds}

        for chunk_start in range(0, num_blocks, batch_size):
            chunk_end = min(chunk_start + batch_size, num_blocks)
            chunk_size = chunk_end - chunk_start

            # Gather into write staging and build flat batch_set args in one pass
            flat_keys: List[str] = []
            flat_ptrs: List[int] = []
            flat_sizes: List[int] = []
            flat_index: List[int] = []  # maps flat idx -> block idx

            for b in range(chunk_size):
                bi = chunk_start + b
                for kind in kinds:
                    self._gather_block("write", kind, b, cpu_block_ids[bi], host_ptrs_per_kind[kind])
                    flat_keys.append(keys_per_kind[kind][bi])
                    flat_ptrs.append(write_bufs[kind] + b * per_block_bytes[kind])
                    flat_sizes.append(per_block_bytes[kind])
                    flat_index.append(bi)

            results = self._connector.batch_set(flat_keys, flat_ptrs, flat_sizes)

            for flat_idx, ok in enumerate(results):
                if not ok:
                    block_success[flat_index[flat_idx]] = False

        return block_success

    def batch_get_block(
        self,
        keys_per_kind: Dict[str, List[str]],
        host_ptrs_per_kind: Dict[str, List[int]],
        cpu_block_ids: List[int],
    ) -> List[bool]:
        """
        Read blocks (all layers packed per key) from storage.

        Calls the connector's ``batch_get`` into the read staging buffer,
        then scatters data back to per-layer host buffers for successful blocks.

        Args:
            keys_per_kind: ``{kind: [key_for_block_0, key_for_block_1, ...]}``
            host_ptrs_per_kind: ``{kind: per_layer_ptrs}``
            cpu_block_ids: Target CPU block IDs.

        Returns:
            List[bool]: True for each block where ALL kinds succeeded.
        """
        if not self._initialized:
            logger.warning("[StagingManager] batch_get_block: not initialized")
            return [False] * len(cpu_block_ids)

        num_blocks = len(cpu_block_ids)
        block_success = [True] * num_blocks
        batch_size = self._staging_batch_size
        kinds = list(keys_per_kind.keys())

        for chunk_start in range(0, num_blocks, batch_size):
            chunk_end = min(chunk_start + batch_size, num_blocks)
            chunk_size = chunk_end - chunk_start

            # Build flat batch_get args
            flat_keys: List[str] = []
            flat_ptrs: List[int] = []
            flat_sizes: List[int] = []
            flat_index: List[int] = []

            for b in range(chunk_size):
                bi = chunk_start + b
                for kind in kinds:
                    per_block_bytes = self._num_layers * self._strides[kind]
                    buf = self._bufs[f"read_{kind}"]
                    flat_keys.append(keys_per_kind[kind][bi])
                    flat_ptrs.append(buf + b * per_block_bytes)
                    flat_sizes.append(per_block_bytes)
                    flat_index.append(bi)

            results = self._connector.batch_get(flat_keys, flat_ptrs, flat_sizes)

            # Mark failures
            for flat_idx, ok in enumerate(results):
                if not ok:
                    block_success[flat_index[flat_idx]] = False

            # Scatter successful blocks from staging to per-layer host buffers
            for b in range(chunk_size):
                bi = chunk_start + b
                if not block_success[bi]:
                    continue
                for kind in kinds:
                    self._scatter_block("read", kind, b, cpu_block_ids[bi], host_ptrs_per_kind[kind])

        return block_success
