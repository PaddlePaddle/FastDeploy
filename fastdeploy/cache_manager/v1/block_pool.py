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

import threading
import traceback
from abc import ABC
from typing import Any, Dict, List, Optional

from fastdeploy.utils import get_logger

from .metadata import CacheBlockMetadata

logger = get_logger("block_pool", "cache_manager.log")


class BlockPool(ABC):
    """
    Abstract base class for block pool management.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        """
        Initialize the block pool.

        Args:
            num_blocks: Total number of blocks in the pool
            block_size: Size of each block in bytes
        """
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._lock = threading.RLock()

        # Track free and used blocks
        self._free_blocks: List[int] = list(range(num_blocks))
        self._used_blocks: set = set()

        # Block metadata
        self._metadata: Dict[int, CacheBlockMetadata] = {}

    def allocate(self, num_blocks: int) -> Optional[List[int]]:
        """
        Allocate blocks from the pool.

        Args:
            num_blocks: Number of blocks to allocate

        Returns:
            List of allocated block indices if successful, None if not enough blocks
        """
        with self._lock:
            if num_blocks == 0:
                return []

            if num_blocks > len(self._free_blocks):
                logger.warning(
                    f"BlockPool.allocate failed: not enough blocks, "
                    f"requested={num_blocks}, available={len(self._free_blocks)}"
                )
                return None

            allocated = self._free_blocks[-num_blocks:]
            del self._free_blocks[-num_blocks:]
            self._used_blocks.update(allocated)

            return allocated

    def release(self, block_indices: List[int]) -> None:
        """
        Release blocks back to the pool.

        Args:
            block_indices: List of block indices to release
        """
        with self._lock:
            for idx in block_indices:
                if idx in self._used_blocks:
                    self._used_blocks.remove(idx)
                    self._free_blocks.append(idx)
                    # Clear metadata
                    self._metadata.pop(idx, None)
                else:
                    logger.error(
                        f"BlockPool.release: block_id={idx} NOT in used_blocks! "
                        f"request_blocks={block_indices}, "
                        f"is_in_free_blocks={idx in self._free_blocks}, "
                        f"is_valid_block_id={0 <= idx < self.num_blocks}"
                    )
                    logger.error(f"BlockPool.release callstack:\n{traceback.format_exc()}")

    def get_metadata(self, block_idx: int) -> Optional[CacheBlockMetadata]:
        """
        Get metadata for a block.

        Args:
            block_idx: Block index

        Returns:
            Block metadata or None if not found
        """
        return self._metadata.get(block_idx)

    def set_metadata(
        self,
        block_idx: int,
        metadata: CacheBlockMetadata,
    ) -> None:
        """
        Set metadata for a block.

        Args:
            block_idx: Block index
            metadata: Block metadata to set
        """
        self._metadata[block_idx] = metadata

    def available_blocks(self) -> int:
        """Get number of available blocks."""
        return len(self._free_blocks)

    def used_blocks(self) -> int:
        """Get number of used blocks."""
        return len(self._used_blocks)

    def reset(self) -> None:
        """Reset the block pool."""
        with self._lock:
            self._free_blocks = list(range(self.num_blocks))
            self._used_blocks.clear()
            self._metadata.clear()

    def resize(self, new_num_blocks: int) -> bool:
        """
        Resize the block pool.

        Supports both expansion and shrinking. Shrinking will fail if
        there are more used blocks than the new size.

        Args:
            new_num_blocks: New total number of blocks

        Returns:
            True if resize was successful, False otherwise
        """
        with self._lock:
            current_used = len(self._used_blocks)

            # Cannot shrink below currently used blocks
            if new_num_blocks < current_used:
                return False

            old_num_blocks = self.num_blocks
            self.num_blocks = new_num_blocks

            if new_num_blocks > old_num_blocks:
                # Expansion: add new free blocks
                new_blocks = list(range(old_num_blocks, new_num_blocks))
                self._free_blocks.extend(new_blocks)
            elif new_num_blocks < old_num_blocks:
                # Shrinking: remove free blocks beyond new size
                blocks_to_keep = set(range(new_num_blocks))
                self._free_blocks = [b for b in self._free_blocks if b in blocks_to_keep]
                # Clean up metadata for removed blocks
                for block_id in range(new_num_blocks, old_num_blocks):
                    self._metadata.pop(block_id, None)

            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "available": len(self._free_blocks),
            "used": len(self._used_blocks),
        }


class DeviceBlockPool(BlockPool):
    """
    GPU device memory block pool.

    Manages KV cache blocks on GPU memory.
    Does not track per-device blocks - device affinity is handled elsewhere.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
    ):
        """
        Initialize the device block pool.

        Args:
            num_blocks: Total number of blocks in the pool
            block_size: Size of each block in bytes
        """
        super().__init__(num_blocks, block_size)

    def get_stats(self) -> Dict[str, Any]:
        """Get device pool statistics."""
        stats = super().get_stats()
        return stats


class HostBlockPool(BlockPool):
    """
    CPU host memory block pool.

    Manages KV cache blocks on CPU memory (pinned memory for fast GPU transfer).
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        use_pinned_memory: bool = True,
    ):
        """
        Initialize the host block pool.

        Args:
            num_blocks: Total number of blocks
            block_size: Size of each block in bytes
            use_pinned_memory: Whether to use pinned (page-locked) memory
        """
        super().__init__(num_blocks, block_size)
        self.use_pinned_memory = use_pinned_memory

    def get_stats(self) -> Dict[str, Any]:
        """Get host pool statistics."""
        stats = super().get_stats()
        stats["use_pinned_memory"] = self.use_pinned_memory
        return stats
