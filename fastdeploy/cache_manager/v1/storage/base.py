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
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class StorageScheduler(ABC):
    """
    Abstract base class for storage scheduler operations.

    Used by CacheManager (Scheduler process) to query storage
    existence without performing actual data transfer.

    Minimal interface for backend implementations:
      - ``connect`` / ``disconnect`` — lifecycle
      - ``batch_exists`` — the only query method required by CacheManager
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from fastdeploy.utils import get_logger

        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False
        self.logger = get_logger("mooncake_storage", "cache_manager.log")

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every backend
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the storage backend.

        Implementations must be idempotent: if already connected
        (``self._connected is True``) return ``True`` immediately without
        re-initialising the underlying client.

        Returns:
            True if connection was successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        pass

    @abstractmethod
    def batch_exists(self, keys: List[str]) -> List[bool]:
        """
        Batch check existence of multiple keys.

        Args:
            keys: List of storage keys to check

        Returns:
            List of booleans corresponding to each key's existence
        """
        pass

    # ------------------------------------------------------------------
    # Concrete methods
    # ------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Check if connected to storage."""
        return self._connected

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "connected": self._connected,
            "config": self.config,
        }


class StorageConnector(ABC):
    """
    Abstract base class for storage connector operations.

    Used by CacheController (Worker process) to perform actual
    data transfer operations with the storage backend.

    All get/set operations use zero-copy semantics: callers pass raw memory
    pointers (int) and sizes (int, bytes) so the backend can perform direct
    RDMA transfers without an intermediate copy.

    Minimal interface for backend implementations:
      - ``connect`` / ``disconnect`` — lifecycle
      - ``batch_get`` — prefetch from storage
      - ``batch_set`` — backup to storage
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from paddleformers.utils.log import logger

        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False
        self.logger = logger

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every backend
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the storage backend.

        Implementations must be idempotent: if already connected
        (``self._connected is True``) return ``True`` immediately without
        re-initialising the underlying client.

        Returns:
            True if connection was successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        pass

    @abstractmethod
    def batch_get(
        self,
        keys: List[str],
        dst_ptrs: List[int],
        sizes: List[int],
    ) -> List[bool]:
        """
        Batch get multiple objects from storage into pre-allocated zero-copy buffers.

        Args:
            keys: List of storage keys
            dst_ptrs: List of destination memory pointers (must be registered if RDMA)
            sizes: List of expected sizes in bytes

        Returns:
            List of booleans indicating success for each key
        """
        pass

    @abstractmethod
    def batch_set(
        self,
        keys: List[str],
        src_ptrs: List[int],
        sizes: List[int],
    ) -> List[bool]:
        """
        Batch set multiple objects into storage from zero-copy source buffers.

        Args:
            keys: List of storage keys
            src_ptrs: List of source memory pointers (must be registered if RDMA)
            sizes: List of data sizes in bytes

        Returns:
            List of booleans indicating success for each key
        """
        pass

    # ------------------------------------------------------------------
    # Concrete methods — backends may override for efficiency
    # ------------------------------------------------------------------

    def register_buffer(self, buffer_ptr: int, buffer_size: int) -> None:
        """
        Register a memory buffer with the storage backend for zero-copy transfer.

        This must be called before using the buffer pointer in get/set operations
        when the backend requires RDMA memory registration (e.g., Mooncake).
        Backends that do not need registration can leave this as a no-op.

        Args:
            buffer_ptr: Raw pointer (int) to the start of the memory region
            buffer_size: Size of the memory region in bytes

        Raises:
            RuntimeError: If registration fails
        """
        pass

    def batch_exists(self, keys: List[str]) -> List[bool]:
        """
        Batch check key existence. Backends that support it should override.
        Default returns False for all keys (conservative: assume missing).
        """
        return [False] * len(keys)

    def batch_delete(self, keys: List[str]) -> List[bool]:
        """
        Delete multiple keys. Backends can override for efficiency.
        Default returns False for all keys.
        """
        return [False] * len(keys)

    def clear(self) -> int:
        """
        Clear all data from storage. Optional — backends that support it
        should override. Default is a no-op returning 0.
        """
        return 0

    def is_connected(self) -> bool:
        """Check if connected to storage."""
        return self._connected

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "connected": self._connected,
            "config": self.config,
        }
