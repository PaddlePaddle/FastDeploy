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
    existence and metadata without performing actual data transfer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the storage scheduler.

        Args:
            config: Storage configuration
        """
        from fastdeploy.utils import get_logger

        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False
        self.logger = get_logger("mooncake_storage", "cache_manager.log")

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
    def exists(self, key: str) -> bool:
        """
        Check if a single key exists in storage.

        Args:
            key: Storage key to check

        Returns:
            True if key exists
        """
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

    @abstractmethod
    def query_prefix_count(
        self,
        k_keys: List[str],
        v_keys: List[str],
        k_scale_keys: Optional[List[str]] = None,
        v_scale_keys: Optional[List[str]] = None,
    ) -> int:
        """
        Query the number of consecutive valid KV cache blocks from the beginning.

        Checks k/v key pairs (and optionally scale key pairs) in order and
        returns the count of leading pairs where all keys exist.

        Args:
            k_keys: List of K-cache keys
            v_keys: List of V-cache keys (same length as k_keys)
            k_scale_keys: Optional list of K-scale keys (FP8 quantization)
            v_scale_keys: Optional list of V-scale keys (FP8 quantization)

        Returns:
            Number of consecutive valid blocks from the start
        """
        pass

    @abstractmethod
    def list_keys(self, prefix: str = "") -> List[str]:
        """
        List keys with a given prefix.

        Args:
            prefix: Key prefix to filter

        Returns:
            List of matching keys
        """
        pass

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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the storage connector.

        Args:
            config: Storage configuration
        """
        from paddleformers.utils.log import logger

        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False
        self.logger = logger

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

    @abstractmethod
    def get(self, key: str, dst_ptr: int, size: int) -> bool:
        """
        Get data from storage into a pre-allocated zero-copy buffer.

        Args:
            key: Storage key
            dst_ptr: Destination memory pointer (int, must be registered if RDMA)
            size: Expected size in bytes

        Returns:
            True if get was successful
        """
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
    def set(self, key: str, src_ptr: int, size: int) -> bool:
        """
        Set data in storage from a zero-copy source buffer.

        Args:
            key: Storage key
            src_ptr: Source memory pointer (int, must be registered if RDMA)
            size: Size of data in bytes

        Returns:
            True if set was successful
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

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete data from storage.

        Args:
            key: Storage key to delete

        Returns:
            True if deletion was successful
        """
        pass

    @abstractmethod
    def clear(self) -> int:
        """
        Clear all data from storage.

        Returns:
            Number of keys cleared
        """
        pass

    def is_connected(self) -> bool:
        """Check if connected to storage."""
        return self._connected

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "connected": self._connected,
            "config": self.config,
        }
