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
        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the storage backend.

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
        Check if a key exists in storage.

        Args:
            key: Storage key to check

        Returns:
            True if key exists
        """
        pass

    @abstractmethod
    def query(self, keys: List[str]) -> Dict[str, bool]:
        """
        Query multiple keys for existence.

        Args:
            keys: List of keys to query

        Returns:
            Dictionary mapping keys to existence status
        """
        pass

    @abstractmethod
    def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a key.

        Args:
            key: Storage key

        Returns:
            Metadata dictionary or None if not found
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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the storage connector.

        Args:
            config: Storage configuration
        """
        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the storage backend.

        Returns:
            True if connection was successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the storage backend."""
        pass

    @abstractmethod
    def get(self, key: str, dst_buffer: Any) -> bool:
        """
        Get data from storage.

        Args:
            key: Storage key
            dst_buffer: Destination buffer to write data

        Returns:
            True if get was successful
        """
        pass

    @abstractmethod
    def set(self, key: str, src_buffer: Any, size: int) -> bool:
        """
        Set data in storage.

        Args:
            key: Storage key
            src_buffer: Source buffer to read data from
            size: Size of data in bytes

        Returns:
            True if set was successful
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
    def clear(self, prefix: str = "") -> int:
        """
        Clear data from storage.

        Args:
            prefix: Key prefix to clear (empty for all)

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
