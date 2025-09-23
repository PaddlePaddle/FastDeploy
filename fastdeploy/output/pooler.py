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

from typing import Any

import msgspec
import paddle


class PoolingSequenceGroupOutput(
    msgspec.Struct,
    omit_defaults=True,
    array_like=True,
):
    """The model output associated with a pooling sequence group."""

    # Annotated as Any to be compatible with msgspec
    # The actual type is in SequenceGroup.pooled_data
    data: Any

    def get_data_nbytes(self) -> int:
        if isinstance(self.data, paddle.Tensor):
            return self.data.numel() * self.data.element_size()
        elif hasattr(self.data, "nbytes"):
            return self.data.nbytes
        else:
            return 0

    def __repr__(self) -> str:
        return f"PoolingSequenceGroupOutput(data={self.data}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PoolingSequenceGroupOutput):
            raise NotImplementedError()
        return self.data == other.data


class PoolerOutput(msgspec.Struct, omit_defaults=True, array_like=True):
    """The output from a pooling operation in the pooling model."""

    outputs: list[PoolingSequenceGroupOutput]

    def get_data_nbytes(self) -> int:
        return sum(o.get_data_nbytes() for o in self.outputs)

    def __getitem__(self, idx: int) -> PoolingSequenceGroupOutput:
        return self.outputs[idx]

    def __setitem__(self, idx: int, value: PoolingSequenceGroupOutput):
        self.outputs[idx] = value

    def __len__(self):
        return len(self.outputs)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self.outputs == other.outputs
