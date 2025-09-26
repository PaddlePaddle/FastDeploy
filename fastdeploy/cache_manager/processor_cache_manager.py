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

import threading
from collections import OrderedDict

import pickle
import numpy as np
import zmq

from fastdeploy import envs
from fastdeploy.utils import get_logger

logger = get_logger("processor_cache_manager", "processor_cache_manager.log")


class ProcessorCacheManager:

    def __init__(self):
        self.processor_cache = OrderedDict()

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

    def apply_cache(self, mm_hashes: list[str], mm_items):
        """
        apply cache data
        """
        assert len(mm_hashes) == len(mm_items), "mm_hashes and mm_items must have the same length"

        for idx in range(len(mm_hashes)):
            if mm_hashes[idx] in self.processor_cache:
                continue
            self.processor_cache[mm_hashes[idx]] = mm_items[idx]

    def get_cache(self, mm_hashes: list[str]):
        """
        get cache correspond to given hash values
        """
        mm_items = []
        for mm_hash in mm_hashes:
            if mm_hash not in self.processor_cache:
                mm_items.append(None)
                continue
            mm_items.append(self.processor_cache[mm_hash])

        return mm_items
    
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