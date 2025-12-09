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

import asyncio
import json
import os
import unittest

from fastdeploy.inter_communicator.fmq import FMQ, Message

# Prepare environment config for testing
cfg = {
    "ipc_root": "/dev/shm",
    "io_threads": 1,
    "copy": False,
    "endpoints": {
        "test_queue": {"protocol": "ipc", "address": "/dev/shm/fmq_test_queue.ipc", "io_threads": 1, "copy": False},
        "test_topic": {"protocol": "ipc", "address": "/dev/shm/fmq_test_topic.ipc", "io_threads": 1, "copy": False},
    },
}
os.environ["FMQ_CONFIG_JSON"] = json.dumps(cfg)


class TestFMQ(unittest.TestCase):

    def setUp(self):
        self.fmq = FMQ()

    def test_queue_send_receive(self):
        async def run_test():
            producer = self.fmq.queue("test_queue", role="producer")
            consumer = self.fmq.queue("test_queue", role="consumer")

            test_data = b"hello world"
            await producer.put(test_data)
            msg = await consumer.get(timeout=1000)

            self.assertIsNotNone(msg)
            self.assertEqual(msg.payload, test_data)

        asyncio.run(run_test())

    def test_queue_large_shm_transfer(self):
        async def run_test():
            producer = self.fmq.queue("test_queue", role="producer")
            consumer = self.fmq.queue("test_queue", role="consumer")

            large_data = b"x" * (2 * 1024 * 1024)  # > 1MB
            await producer.put(large_data)
            msg = await consumer.get(timeout=1000)

            self.assertIsNotNone(msg)
            self.assertEqual(msg.payload, large_data)
            self.assertIsNotNone(msg.descriptor)

        asyncio.run(run_test())

    def test_topic_pub_sub(self):
        received = []

        async def run_test():
            topic = self.fmq.topic("test_topic")

            async def callback(msg: Message):
                received.append(msg.payload)

            await topic.sub(callback)
            await asyncio.sleep(0.1)  # allow SUB to connect

            await topic.pub("hello")
            await asyncio.sleep(0.2)

            self.assertIn("hello", received)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
