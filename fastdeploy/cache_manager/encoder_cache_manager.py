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

from collections import OrderedDict

from fastdeploy.engine.request import ImagePosition
from fastdeploy.utils import get_logger

logger = get_logger("cache_messager", "cache_messager.log")


class EncoderCacheManager:
    """
    EncoderCacheManager is used to cache image feature data.
    """

    def __init__(self, max_encoder_cache):
        self.encoder_cache = OrderedDict()
        self.encoder_cache_size = 0
        self.max_encoder_cache = max_encoder_cache

    def apply_cache(self, mm_hashes: list[str], mm_positions: list[ImagePosition]):
        """
        apply_cache is used to apply the cache data to the server.
        """
        assert len(mm_hashes) == len(mm_positions), "mm_hashes and mm_positions must have the same length."

        evict_hashes = []
        for idx in range(len(mm_hashes)):
            if mm_hashes[idx] in self.encoder_cache:
                self.encoder_cache.move_to_end(mm_hashes[idx])
            else:
                if self.encoder_cache_size + mm_positions[idx].length >= self.max_encoder_cache:
                    evict_hashes = self.evict_cache(mm_positions[idx].length)
                self.encoder_cache[mm_hashes[idx]] = mm_positions[idx]
                self.encoder_cache_size += mm_positions[idx].length
        return evict_hashes

    def clear_cache(self):
        """
        clear_cache is used to clear the cache data.
        """
        evict_hashes = self.encoder_cache.keys()
        self.encoder_cache.clear()
        self.encoder_cache_size = 0
        return evict_hashes

    def evict_cache(self, token_nums: int):
        """
        evict_cache is used to evict the cache data.
        """
        freed_token_nums, evict_hashes = 0, []
        while freed_token_nums < token_nums:
            mm_hash, mm_position = self.encoder_cache.popitem(last=False)
            evict_hashes.append(mm_hash)
            freed_token_nums += mm_position.length
            self.encoder_cache_size -= mm_position.length
        return evict_hashes

    def has_mm_cache(self, mm_hashes: list[str]):
        """
        Check if the mm_hash is in the encoder_cache.
        """
        cache_state = [False] * len(mm_hashes)
        for idx in range(len(mm_hashes)):
            if mm_hashes[idx] in self.encoder_cache:
                cache_state[idx] = True
        return cache_state
