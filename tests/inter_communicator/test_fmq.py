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
import time
import unittest

from fastdeploy.inter_communicator.fmq import FMQ, Descriptor, Message


class TestDescriptor(unittest.TestCase):

    def test_create_and_read(self):
        data = b"hello shared memory"
        desc = Descriptor.create(data)

        self.assertIsNotNone(desc.shm_name)
        self.assertEqual(desc.size, len(data))

        read = desc.read_and_unlink()
        self.assertEqual(read, data)

    def test_read_after_unlink(self):
        data = b"once"
        desc = Descriptor.create(data)

        _ = desc.read_and_unlink()
        again = desc.read_and_unlink()

        self.assertEqual(again, b"")


class TestMessage(unittest.TestCase):

    def test_serialize_deserialize(self):
        msg = Message(payload={"x": 1})
        raw = msg.serialize()
        new_msg = Message.deserialize(raw)

        self.assertEqual(new_msg.payload, {"x": 1})
        self.assertIsNone(new_msg.descriptor)

    def test_message_with_descriptor(self):
        data = b"x" * 1024
        desc = Descriptor.create(data)
        msg = Message(payload=None, descriptor=desc)

        raw = msg.serialize()
        new_msg = Message.deserialize(raw)

        self.assertIsNotNone(new_msg.descriptor)
        payload = new_msg.descriptor.read_and_unlink()
        self.assertEqual(payload, data)


class TestQueue(unittest.TestCase):

    def setUp(self):
        self.fmq = FMQ()

    def tearDown(self):
        self.fmq.destroy()

    def test_basic_put_get(self):
        name = "test_queue_basic"

        consumer = self.fmq.queue(name, role="consumer")
        producer = self.fmq.queue(name, role="producer")

        producer.put("hello")
        result = consumer.get()

        self.assertEqual(result, "hello")

    def test_large_bytes_shm(self):
        name = "test_queue_shm"

        consumer = self.fmq.queue(name, role="consumer")
        producer = self.fmq.queue(name, role="producer")

        data = b"x" * (2 * 1024 * 1024)
        producer.put(data, shm_threshold=1024)

        result = consumer.get()
        self.assertEqual(result, data)

    def test_wrong_role_put(self):
        q = self.fmq.queue("test_queue_wrong_put", role="consumer")
        with self.assertRaises(PermissionError):
            q.put("fail")

    def test_wrong_role_get(self):
        q = self.fmq.queue("test_queue_wrong_get", role="producer")
        with self.assertRaises(PermissionError):
            q.get()

    def test_cross_thread_usage_detected(self):
        producer = self.fmq.queue("test_queue_thread", role="producer")

        errors = []

        def target():
            try:
                producer.put("bad")
            except RuntimeError as e:
                errors.append(e)

        t = threading.Thread(target=target)
        t.start()
        t.join()

        self.assertEqual(len(errors), 1)
        producer.close()


class TestTopic(unittest.TestCase):

    def setUp(self):
        self.fmq = FMQ()

    def tearDown(self):
        self.fmq.destroy()

    def test_pub_sub(self):
        topic = self.fmq.topic("test_topic")

        received = []

        def callback(msg):
            received.append(msg.payload)

        topic.sub(callback)
        time.sleep(1)

        topic.pub("hello")
        topic.pub("world")

        time.sleep(1)

        self.assertIn("hello", received)
        self.assertIn("world", received)

        topic.stop_sub()

    def test_sub_start_twice(self):
        topic = self.fmq.topic("test_topic_twice")

        topic.sub(lambda _: None)
        with self.assertRaises(RuntimeError):
            topic.sub(lambda _: None)


class TestFMQLifecycle(unittest.TestCase):

    def test_create_destroy(self):
        fmq = FMQ()
        self.assertIsNotNone(fmq._context)

        fmq.destroy()
        self.assertIsNone(fmq._context)


if __name__ == "__main__":
    unittest.main()
