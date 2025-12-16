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

import unittest

from fastdeploy.inter_communicator.fmq import Message
from fastdeploy.inter_communicator.fmq_factory import FMQFactory as factory


class TestFMQFactory(unittest.IsolatedAsyncioTestCase):

    async def test_create_queues(self):
        """Test whether all producer/consumer queues can be created."""
        q1 = factory.q_a2e_producer()
        q2 = factory.q_a2e_consumer()
        q3 = factory.q_e2w_producer()
        q4 = factory.q_e2w_consumer()
        q5 = factory.q_w2e_producer()
        q6 = factory.q_w2e_consumer()
        q7 = factory.q_e2a_producer()
        q8 = factory.q_e2a_consumer()

        self.assertEqual(q1.name, "q_a2e")
        self.assertEqual(q2.name, "q_a2e")
        self.assertEqual(q3.name, "q_e2w")
        self.assertEqual(q4.name, "q_e2w")
        self.assertEqual(q5.name, "q_w2e")
        self.assertEqual(q6.name, "q_w2e")
        self.assertEqual(q7.name, "q_e2a")
        self.assertEqual(q8.name, "q_e2a")

        # 同一进程内 context 应相同
        self.assertIs(q1.context, q2.context)
        self.assertIs(q1.context, q3.context)

    async def test_message_roundtrip(self):
        """测试 producer → consumer 消息流转"""
        producer = factory.q_a2e_producer()
        consumer = factory.q_a2e_consumer()

        payload = {"k": "v"}

        await producer.put(payload)
        msg = await consumer.get(timeout=1500)

        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.payload, payload)

    async def test_multi_queue_independence(self):
        """测试多个队列互不干扰"""

        prod_a2e = factory.q_a2e_producer()
        cons_a2e = factory.q_a2e_consumer()

        prod_e2w = factory.q_e2w_producer()
        cons_e2w = factory.q_e2w_consumer()

        await prod_a2e.put("msg_api")
        await prod_e2w.put("msg_worker")

        msg1 = await cons_a2e.get(timeout=1500)
        msg2 = await cons_e2w.get(timeout=1500)

        self.assertEqual(msg1.payload, "msg_api")
        self.assertEqual(msg2.payload, "msg_worker")

    async def test_shared_context(self):
        """验证 FMQFactory 始终返回同一个 context (单进程)"""
        q1 = factory.q_a2e_producer()
        q2 = factory.q_e2w_consumer()
        q3 = factory.q_e2a_producer()

        self.assertIs(q1.context, q2.context)
        self.assertIs(q1.context, q3.context)


if __name__ == "__main__":
    unittest.main()
