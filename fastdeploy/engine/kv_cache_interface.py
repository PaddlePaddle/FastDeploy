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

import copy
from dataclasses import dataclass
from typing import List

from typing_extensions import Self


@dataclass
class KVCacheSpec:
    """
    A base class for specifying the KV cache format of one layer.
    """

    # number of tokens in a block
    block_size: int
    # the memory size used by each block in bytes.
    block_memory_used: int

    @classmethod
    def merge(cls, specs: List[Self]) -> Self:
        """
        Merge a List of KVCacheSpec objects into a single KVCacheSpec object.
        """
        # check List
        assert all(
            (spec.block_size == specs[0].block_size and spec.block_memory_used == specs[0].block_memory_used)
            for spec in specs[1:]
        ), "All layers in the model must share the same block_size."

        return copy.deepcopy(specs[0])


@dataclass
class AttentionSpec(KVCacheSpec):
    """ """

    num_kv_heads: int
    head_size: int
    dtype: str
