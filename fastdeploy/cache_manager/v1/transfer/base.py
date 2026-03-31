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
from typing import Any, Dict, Optional


class TransferConnector(ABC):
    """
    Abstract base class for transfer connector operations.

    Used by CacheController (Worker process) to perform cross-node
    and cross-process data transfer operations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the transfer connector.

        Args:
            config: Transfer configuration
        """
        self.config = config or {}
        self._lock = threading.RLock()
        self._connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the transfer backend.

        Returns:
            True if connection was successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the transfer backend."""
        pass

    @abstractmethod
    def send(
        self,
        dst_addr: str,
        src_buffer: Any,
        size: int,
        dst_offset: int = 0,
    ) -> bool:
        """
        Send data to a remote destination.

        Args:
            dst_addr: Destination address
            src_buffer: Source buffer to read data from
            size: Size of data in bytes
            dst_offset: Offset at destination

        Returns:
            True if send was successful
        """
        pass

    @abstractmethod
    def recv(
        self,
        src_addr: str,
        dst_buffer: Any,
        size: int,
        src_offset: int = 0,
    ) -> bool:
        """
        Receive data from a remote source.

        Args:
            src_addr: Source address
            dst_buffer: Destination buffer to write data
            size: Size of data in bytes
            src_offset: Offset at source

        Returns:
            True if receive was successful
        """
        pass

    @abstractmethod
    def send_async(
        self,
        dst_addr: str,
        src_buffer: Any,
        size: int,
        dst_offset: int = 0,
    ) -> Any:
        """
        Asynchronously send data to a remote destination.

        Args:
            dst_addr: Destination address
            src_buffer: Source buffer to read data from
            size: Size of data in bytes
            dst_offset: Offset at destination

        Returns:
            Handle for tracking the async operation
        """
        pass

    @abstractmethod
    def recv_async(
        self,
        src_addr: str,
        dst_buffer: Any,
        size: int,
        src_offset: int = 0,
    ) -> Any:
        """
        Asynchronously receive data from a remote source.

        Args:
            src_addr: Source address
            dst_buffer: Destination buffer to write data
            size: Size of data in bytes
            src_offset: Offset at source

        Returns:
            Handle for tracking the async operation
        """
        pass

    @abstractmethod
    def wait(self, handle: Any, timeout: float = -1) -> bool:
        """
        Wait for an async operation to complete.

        Args:
            handle: Handle from send_async or recv_async
            timeout: Timeout in seconds (-1 for infinite)

        Returns:
            True if operation completed successfully
        """
        pass

    @abstractmethod
    def register_buffer(self, buffer: Any, addr: str) -> bool:
        """
        Register a buffer for RDMA operations.

        Args:
            buffer: Buffer to register
            addr: Address to associate with buffer

        Returns:
            True if registration was successful
        """
        pass

    @abstractmethod
    def unregister_buffer(self, addr: str) -> bool:
        """
        Unregister a buffer.

        Args:
            addr: Address of buffer to unregister

        Returns:
            True if unregistration was successful
        """
        pass

    def is_connected(self) -> bool:
        """Check if connected to transfer backend."""
        return self._connected

    def get_stats(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            "connected": self._connected,
            "config": self.config,
        }
