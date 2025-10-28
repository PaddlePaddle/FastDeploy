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

from abc import ABC, abstractmethod

from paddle import Tensor

from fastdeploy.config import FDConfig


class LogitsProcessor(ABC):
    @abstractmethod
    def __init__(self, fd_config: FDConfig) -> None:
        raise NotImplementedError

    @abstractmethod
    def update_state(self, share_inputs: dict) -> None:
        """Called when there are new output tokens, prior to each forward pass.

        Each field in the `share_inputs` dict typically stores information for all request
        slots. It has a `stop_flags` array that indicates whether a slot currently has a
        running request (`False` means the slot is active). Therefore, it is recommended to
        filter entries by `stop_flags` to keep only data for the current batch.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits: Tensor) -> Tensor:
        """Apply LogitsProcessor to batch logits tensor.

        The updated tensor must be returned but may be modified in-place.
        """
        raise NotImplementedError
