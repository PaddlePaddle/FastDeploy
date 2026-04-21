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

import mmap
import os
from typing import Any, Dict, Optional

from ..base import TransferConnector


class IPCConnector(TransferConnector):
    """
    IPC connector for cross-process transfer on same node.

    Uses shared memory for efficient data transfer between
    processes on the same machine.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize IPC connector.

        Args:
            config: Configuration with keys:
                - shm_path: Shared memory path prefix
                - buffer_size: Default buffer size
                - max_buffers: Maximum number of buffers
        """
        super().__init__(config)
        self._shm_buffers: Dict[str, mmap.mmap] = {}
        self._shm_paths: Dict[str, str] = {}

    def connect(self) -> bool:
        """Connect to IPC backend."""
        try:
            self._connected = True
            return True
        except Exception:
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from IPC backend."""
        # Clean up shared memory
        for name, shm in self._shm_buffers.items():
            try:
                shm.close()
            except Exception:
                pass

        # Remove shared memory files
        for name, path in self._shm_paths.items():
            try:
                os.unlink(path)
            except Exception:
                pass

        self._shm_buffers.clear()
        self._shm_paths.clear()
        self._connected = False

    def send(
        self,
        dst_addr: str,
        src_buffer: Any,
        size: int,
        dst_offset: int = 0,
    ) -> bool:
        """Send data via shared memory."""
        if not self._connected:
            return False

        if dst_addr not in self._shm_buffers:
            return False

        try:
            shm = self._shm_buffers[dst_addr]
            shm.seek(dst_offset)
            shm.write(src_buffer[:size])
            return True
        except Exception:
            return False

    def recv(
        self,
        src_addr: str,
        dst_buffer: Any,
        size: int,
        src_offset: int = 0,
    ) -> bool:
        """Receive data via shared memory."""
        if not self._connected:
            return False

        if src_addr not in self._shm_buffers:
            return False

        try:
            shm = self._shm_buffers[src_addr]
            shm.seek(src_offset)
            data = shm.read(size)
            dst_buffer[:size] = data
            return True
        except Exception:
            return False

    def send_async(
        self,
        dst_addr: str,
        src_buffer: Any,
        size: int,
        dst_offset: int = 0,
    ) -> Any:
        """Asynchronously send data via shared memory."""
        # For shared memory, async is similar to sync
        success = self.send(dst_addr, src_buffer, size, dst_offset)
        return {"success": success, "addr": dst_addr}

    def recv_async(
        self,
        src_addr: str,
        dst_buffer: Any,
        size: int,
        src_offset: int = 0,
    ) -> Any:
        """Asynchronously receive data via shared memory."""
        # For shared memory, async is similar to sync
        success = self.recv(src_addr, dst_buffer, size, src_offset)
        return {"success": success, "addr": src_addr}

    def wait(self, handle: Any, timeout: float = -1) -> bool:
        """Wait for IPC operation completion."""
        if handle is None:
            return False
        return handle.get("success", False)

    def register_buffer(self, buffer: Any, addr: str) -> bool:
        """Register a shared memory buffer."""
        if not self._connected:
            return False

        try:
            # Create shared memory file
            shm_path = f"/dev/shm/kv_cache_{addr}"
            shm_fd = os.open(shm_path, os.O_CREAT | os.O_RDWR, 0o666)

            # Size the file
            buffer_size = len(buffer) if hasattr(buffer, "__len__") else self.config.get("buffer_size", 1024 * 1024)
            os.ftruncate(shm_fd, buffer_size)

            # Map the file
            shm = mmap.mmap(shm_fd, buffer_size)
            os.close(shm_fd)

            self._shm_buffers[addr] = shm
            self._shm_paths[addr] = shm_path

            return True
        except Exception:
            return False

    def unregister_buffer(self, addr: str) -> bool:
        """Unregister a shared memory buffer."""
        if addr not in self._shm_buffers:
            return False

        try:
            self._shm_buffers[addr].close()
            del self._shm_buffers[addr]

            if addr in self._shm_paths:
                os.unlink(self._shm_paths[addr])
                del self._shm_paths[addr]

            return True
        except Exception:
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get IPC connector statistics."""
        stats = super().get_stats()
        stats.update(
            {
                "registered_buffers": len(self._shm_buffers),
                "buffer_addresses": list(self._shm_buffers.keys()),
            }
        )
        return stats
