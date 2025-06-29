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

import numpy as np
from multiprocessing.shared_memory import SharedMemory

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

    def __init__(self,
                 name: str,
                 array: np.ndarray,
                 dtype: np.dtype,
                 suffix: int = None,
                 create: bool = True) -> None:
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
        assert isinstance(array, np.ndarray), "Input must be a numpy array"
        assert dtype == array.dtype, "Specified dtype must match array dtype"

        # Set a suffix for name to avoid name conflict while there are multiple engine launched
        if suffix is not None:
            name = name + f".{suffix}"

        if create:
            assert not shared_memory_exists(
                name), f"ShareMemory: {name} already exists"
            self.shm = SharedMemory(create=True, size=array.nbytes, name=name)
            self.value: np.ndarray = np.ndarray(array.shape,
                                                dtype=array.dtype,
                                                buffer=self.shm.buf)
            self.value[:] = array  # Initialize with input array data
        else:
            self.shm = SharedMemory(name=name)
            self.value: np.ndarray = np.ndarray(array.shape,
                                                dtype=array.dtype,
                                                buffer=self.shm.buf)

    def clear(self) -> None:
        """Release system resources and unlink the shared memory block."""
        if shared_memory_exists(self.shm.name):
            self.shm.close()
            self.shm.unlink()
