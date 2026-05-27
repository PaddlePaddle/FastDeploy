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


class RequestManager:
    """
    Unified interface for request slot management.

    Wraps the existing stop_flags / tasks_list arrays from V1's
    ResourceManager — no new data structures are created.  All slot
    operations should go through this interface rather than directly
    manipulating the underlying arrays.

    Thread-safety:
      - acquire_slot / get_available_position / available_batch: called
        only inside schedule() (single-writer), no locking needed.
      - release_slot: also safe to call from other threads (e.g.
        finish_requests) because single-element list assignment is
        GIL-atomic under CPython.
    """

    def __init__(self, stop_flags: list[bool], tasks_list: list):
        self.stop_flags = stop_flags
        self.tasks_list = tasks_list
        self.max_num_seqs = len(stop_flags)

    def acquire_slot(self, idx: int, task) -> None:
        """Occupy slot idx with task."""
        self.stop_flags[idx] = False
        self.tasks_list[idx] = task

    def release_slot(self, idx: int) -> None:
        """Release a slot. Safe to call from any thread."""
        self.stop_flags[idx] = True
        self.tasks_list[idx] = None

    def get_available_position(self) -> int:
        """Return the first available slot index, or raise RuntimeError."""
        for idx in range(self.max_num_seqs):
            if self.stop_flags[idx]:
                return idx
        raise RuntimeError("No available position for new request")

    def available_batch(self) -> int:
        """Return the number of free slots."""
        return sum(self.stop_flags)
