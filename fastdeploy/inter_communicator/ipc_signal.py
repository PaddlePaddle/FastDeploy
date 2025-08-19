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

import fcntl
import os
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from fastdeploy import envs
from fastdeploy.utils import shared_memory_logger


def shared_memory_exists(name: str) -> bool:
    """Check if a shared memory block with the given name exists.

    Args:
        name: The unique identifier of the shared memory block.

    Returns:
        True if the shared memory exists, False otherwise.
    """
    try:
        shm = SharedMemory(name=name, create=False)
        shm.close()
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False


class IPCSignal:
    """A shared memory wrapper for inter-process communication using numpy arrays.

    Allows creating or connecting to existing shared memory blocks and synchronizing
    numpy array data between processes.

    Attributes:
        shm: The underlying SharedMemory object.
        value: Numpy array interface to the shared memory buffer.
    """

    def __init__(
        self,
        name: str,
        array: np.ndarray,
        dtype: np.dtype,
        suffix: int = None,
        create: bool = True,
        expected_consumers: int = None,
    ) -> None:
        """Initialize or connect to a shared memory block.

        Args:
            name: Unique identifier for the shared memory block.
            array: Numpy array template defining shape and data type.
            dtype: Data type of the array (must match array.dtype).
            suffix: Suffix number that will be appended to the name.
            create: If True, creates new memory block; otherwise connects to existing.

        Raises:
            AssertionError: If create=True but memory already exists, or dtype mismatch.
        """
        shared_memory_logger.debug(f"Initializing signal, args:\n{locals()}")
        shared_memory_logger.debug(f"FD_ENABLE_IPC_AUTO_CLEAN={envs.FD_ENABLE_IPC_AUTO_CLEAN}")

        assert isinstance(array, np.ndarray), "Input must be a numpy array"
        assert dtype == array.dtype, "Specified dtype must match array dtype"

        self.is_creator = create
        self.expected_consumers = expected_consumers

        # Set a suffix for name to avoid name conflict while there are multiple engine launched
        if suffix is not None:
            name = name + f".{suffix}"
            if envs.FD_IPC_APPEND_SUFFIX:
                name = name + f".{envs.FD_IPC_APPEND_SUFFIX}"

        if create:
            assert not shared_memory_exists(name), f"ShareMemory: {name} already exists"
            shared_memory_logger.info(f"Shared memory created: {name}")
            self.shm = SharedMemory(create=True, size=array.nbytes, name=name)
            self.value: np.ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)
            self.value[:] = array  # Initialize with input array data
            if envs.FD_ENABLE_IPC_AUTO_CLEAN and expected_consumers is not None:
                self.async_unlink_if_all_consumers_ready(interval=5)  # check the status preriodically
        else:
            shared_memory_logger.info(f"Shared memory attached: {name}")
            self.shm = SharedMemory(name=name)
            self.value: np.ndarray = np.ndarray(array.shape, dtype=array.dtype, buffer=self.shm.buf)
            if envs.FD_ENABLE_IPC_AUTO_CLEAN:
                self.notify_creator_ready()

    def clear(self) -> None:
        """Release system resources and unlink the shared memory block."""
        try:
            self.shm.close()
            if shared_memory_exists(self.shm.name):
                self.shm.unlink()
            # additional control blocks for creator to clean up
            if self.is_creator:
                self.ctl.close()
                if shared_memory_exists(self.ctl.name):
                    self.ctl.unlink()
                os.remove(f"/dev/shm/{self.shm.name}.lock")
        except:
            pass

    def async_unlink_if_all_consumers_ready(self, interval=60, timeout=3600) -> None:
        """
        Periodically checks whether all consumer slots have been marked as ready.
        Once all are ready, it asynchronously unlinks the main shared memory block
        and its associated consumer flag segment, ensuring /dev/shm handles are
        cleaned up even if some processes exit unexpectedly.

        Args:
            interval: Interval (in seconds) to check the status of all consumers.
            timeout: Maximum allowed waiting time before giving up.
        """

        def unlink_if_all_consumers_ready(interval, timeout):
            start_loop_time = time.time()
            while time.time() - start_loop_time < timeout:
                ready_consumers = struct.unpack_from("<Q", self.ctl.buf, 0)[0]
                shared_memory_logger.debug(
                    f"Waiting for all consumers to attach, name: {self.shm.name}, "
                    f"current: {ready_consumers}, expected: {self.expected_consumers}"
                )
                if ready_consumers >= self.expected_consumers:
                    self.shm.unlink()
                    self.ctl.close()
                    self.ctl.unlink()
                    os.remove(self.ctl_lock)
                    shared_memory_logger.info(f"All consumers ready, creator unlinking: {self.shm.name}")
                    shared_memory_logger.info(f"Control block unlinking: {self.ctl.name}")
                    return
                time.sleep(interval)

        ctl_name = self.shm.name + ".ctl"
        assert not shared_memory_exists(ctl_name), f"ShareMemory: {ctl_name} already exists"
        self.ctl = SharedMemory(create=True, size=8, name=ctl_name)
        struct.pack_into("<Q", self.ctl.buf, 0, 0)
        shared_memory_logger.info(f"Control block created: {ctl_name}")

        # prepare lock
        self.ctl_lock = f"/dev/shm/{self.ctl.name}.lock"
        try:
            fd = os.open(self.ctl_lock, os.O_CREAT | os.O_RDWR, 0o600)
            os.close(fd)
        except Exception:
            pass

        t = threading.Thread(target=unlink_if_all_consumers_ready, args=[interval, timeout], daemon=True)
        t.start()

    def notify_creator_ready(self) -> None:
        """
        Atomically increments the shared counter in the control segment to notify
        the creator process that this consumer has finished mapping the memory.
        """
        try:
            ctl_name = self.shm.name + ".ctl"
            ctl = SharedMemory(name=ctl_name)
            with open(self.ctl_lock, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                val = struct.unpack_from("<Q", ctl.buf, 0)[0]
                struct.pack_into("<Q", ctl.buf, 0, val + 1)
                fcntl.flock(f, fcntl.LOCK_UN)
            ctl.close()
            shared_memory_logger.info(f"Notified creator: {self.shm.name}, current attached: {val + 1}")
        except:
            pass
