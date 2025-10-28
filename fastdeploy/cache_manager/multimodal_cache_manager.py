"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import pickle
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Tuple

import numpy as np
import zmq

from fastdeploy import envs
from fastdeploy.engine.request import ImagePosition
from fastdeploy.utils import get_logger

logger = get_logger("prefix_cache_manager", "cache_manager.log")


class MultimodalLRUCache(ABC):
    """
    General lru cache for multimodal data
    """

    def __init__(self, max_cache_size):
        self.cache = OrderedDict()
        self.current_cache_size = 0
        self.max_cache_size = max_cache_size

    def apply_cache(self, mm_hashes: list[str], mm_items: list[Any]) -> list[str]:
        """
        apply data cache, return evicted data
        """
        assert len(mm_hashes) == len(mm_items), "mm_hashes and mm_items should have same length"

        evicted_hashes = []
        for idx in range(len(mm_hashes)):
            if mm_hashes[idx] in self.cache:
                self.cache.move_to_end(mm_hashes[idx])
            else:
                item_size = self.get_item_size(mm_items[idx])
                if self.current_cache_size + item_size >= self.max_cache_size:
                    if item_size > self.max_cache_size:
                        # cannot be inserted even if we clear all cached data, skip it directly
                        continue
                    needed = item_size - (self.max_cache_size - self.current_cache_size)
                    evicted_hashes.extend(self.evict_cache(needed))
                self.cache[mm_hashes[idx]] = mm_items[idx]
                self.current_cache_size += item_size

        return evicted_hashes

    def evict_cache(self, needed: int) -> list[str]:
        """
        evict data cache with needed size
        """
        reduced_size, evicted_hashes = 0, []
        while reduced_size < needed and len(self.cache):
            mm_hash, mm_item = self.cache.popitem(last=False)
            evicted_hashes.append(mm_hash)
            reduced_size += self.get_item_size(mm_item)
            self.current_cache_size -= self.get_item_size(mm_item)

        return evicted_hashes

    def get_cache(self, mm_hashes: list[str]) -> list[Any]:
        """
        get cached data correspond to given hash values
        """
        mm_items = []
        for mm_hash in mm_hashes:
            if mm_hash not in self.cache:
                mm_items.append(None)
                continue
            mm_items.append(self.cache[mm_hash])

        return mm_items

    def clear_cache(self):
        """
        clear all cached data
        """
        evicted_hashes = list(self.cache.keys())
        self.cache.clear()
        self.current_cache_size = 0

        return evicted_hashes

    @abstractmethod
    def get_item_size(self, item: Any) -> int:
        raise NotImplementedError("Subclasses must define how to get size of an item")


class EncoderCacheManager(MultimodalLRUCache):
    """
    EncoderCacheManager is used to cache image features
    """

    def __init__(self, max_encoder_cache):
        super().__init__(max_encoder_cache)

    def get_item_size(self, item: ImagePosition) -> int:
        return item.length


class ProcessorCacheManager(MultimodalLRUCache):
    """
    ProcessorCacheManager is used to cache processed data
    """

    def __init__(self, max_processor_cache):
        super().__init__(max_processor_cache)

        self.context = zmq.Context()

        self.router = self.context.socket(zmq.ROUTER)
        self.router.setsockopt(zmq.SNDHWM, int(envs.FD_ZMQ_SNDHWM))
        self.router.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.router.setsockopt(zmq.SNDTIMEO, -1)
        self.router.bind("ipc:///dev/shm/processor_cache.ipc")

        self.poller = zmq.Poller()
        self.poller.register(self.router, zmq.POLLIN)

        self.handler_thread = threading.Thread(target=self.cache_request_handler, daemon=True)
        self.handler_thread.start()

    def get_item_size(self, item: Tuple[np.ndarray, dict]) -> int:
        return item[0].nbytes

    def cache_request_handler(self):
        try:
            while True:
                events = dict(self.poller.poll())

                if self.router in events:
                    client, _, content = self.router.recv_multipart()
                    req = pickle.loads(content)

                    if isinstance(req, tuple):
                        # apply cache request, in format of (mm_hashes, mm_items)
                        self.apply_cache(req[0], req[1])
                        logger.info(f"Apply processor cache of mm_hashes: {req[0]}")
                    else:
                        # get cache request
                        resp = self.get_cache(req)
                        logger.info(f"Get processor cache of mm_hashes: {req}")
                        self.router.send_multipart([client, b"", pickle.dumps(resp)])
        except Exception as e:
            logger.error(f"Error happened while handling processor cache request: {e}")
