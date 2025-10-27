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

from enum import Enum

from fastdeploy.utils import get_logger

logger = get_logger("prefix_cache_manager", "cache_manager.log")


DISABLE_PREFIX_CACHE_MM_MODEL: set[str] = {
    "Ernie5ForCausalLM",
}


def is_mm_model_disable_prefix_cache(model_config):
    """
    check if the model architecture is in DISABLE_PREFIX_CACHE_MM_MODEL
    """
    return model_config._architecture in DISABLE_PREFIX_CACHE_MM_MODEL


class CacheStatus(Enum):
    """
    cache status enum class
    """

    GPU = 0
    SWAP2CPU = 1
    SWAP2GPU = 2
    CPU = 3


class BlockNode:
    """
    BlockNode: store the information of a block node
    """

    def __init__(
        self,
        node_id,
        input_ids,
        input_hash_value,
        depth,
        block_id,
        token_num,
        hash_value,
        last_used_time,
        parent=None,
        shared_count=1,
        reverved_dec_block_ids=[],
        cache_status=CacheStatus.GPU,
        is_persistent=False,
        persistent_shared_count=0,
    ):
        """
        Args:
            node_id: Unique identifier of the node
            depth: Depth of the node
            block_id: Assigned block ID (CPU block ID if on CPU, GPU block ID if on GPU)
            token_num: Number of tokens in the current block
            hash_value: Hash value of the current block
            last_used_time: Timestamp of last usage
            parent: Parent node
            shared_count: Reference count of requests currently using this node
            reserved_dec_block_ids: Pre-allocated block IDs reserved for decoding, formatted as [block_id, block_id,...]
            cache_status: Current cache state (USING, SWAP2CPU, SWAP2GPU, FREE)
            is_persistent: Whether the node is persistently stored
            persistent_shared_count: Reference count of persistent cache requests
        """

        self.node_id = node_id
        self.depth = depth
        self.parent = parent
        self.hash_value = hash_value
        self.token_num = token_num
        self.input_ids = input_ids
        self.input_hash_value = input_hash_value

        self.children = {}
        self.shared_count = shared_count
        self.last_used_time = last_used_time
        self.block_id = block_id
        self.reverved_dec_block_ids = reverved_dec_block_ids
        self.cache_status = cache_status
        self.is_persistent = is_persistent
        self.persistent_shared_count = persistent_shared_count
        self.req_id_set = set()

    def __lt__(self, other):
        """
        override the less than operator
        """
        if self.last_used_time < other.last_used_time:
            return True
        elif self.last_used_time > other.last_used_time:
            return False
        else:
            return self.depth > other.depth

    def __str__(self):
        """
        return node info
        """
        if self.parent is not None:
            parent_node_id = self.parent.node_id
        else:
            parent_node_id = None
        return (
            f"node_id {self.node_id}: depth {self.depth} hash_value {self.hash_value}"
            + f" shared_count {self.shared_count} is_gpu_leaf_node {self.is_gpu_leaf_node}"
            + f" is_cpu_leaf_node {self.is_cpu_leaf_node} block_id {self.block_id} "
            + f"has_in_gpu {self.has_in_gpu} "
            + f"cache_status {self.cache_status}  parent {parent_node_id} with children number "
            + f"{len(self.children)} req_id_set {self.req_id_set}"
        )

    @property
    def has_in_gpu(self):
        """
        check if the node has been allocated in GPU
        """
        return self.cache_status == CacheStatus.GPU

    def increment_shared_count(self):
        """
        increment shared count
        """
        self.shared_count += 1

    def decrement_shared_count(self):
        """
        decrement shared count
        """
        self.shared_count -= 1

    @property
    def is_cpu_leaf_node(self):
        """
        check if the node is a leaf node in CPU
        """
        if (self.cache_status == CacheStatus.CPU) and (len(self.children) == 0):
            return True
        return False

    @property
    def is_gpu_leaf_node(self):
        """
        check if the node is a leaf node in GPU
        """
        if self.has_in_gpu is False:
            return False
        else:
            if len(self.children) == 0:
                return True
            for child in self.children.values():
                if child.has_in_gpu is True:
                    return False
            return True
