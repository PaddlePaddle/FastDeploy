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
import types
import unittest

import numpy as np
import paddle

if not hasattr(paddle, "compat"):
    paddle.compat = types.SimpleNamespace(enable_torch_proxy=lambda **_: None)

from fastdeploy import envs
from fastdeploy.engine.request import Request
from fastdeploy.inter_communicator.engine_worker_queue import EngineWorkerQueue
from fastdeploy.utils import to_numpy, to_tensor


class DummyTask:
    def __init__(self, images):
        self.multimodal_inputs = {"images": images}


class TestEngineWorkerQueue(unittest.TestCase):
    def _build_queue_pair(self):
        server = EngineWorkerQueue(address=("127.0.0.1", 0), is_server=True, num_client=1, client_id=0)
        client = EngineWorkerQueue(
            address=server.address,
            is_server=False,
            num_client=1,
            client_id=0,
        )
        return server, client

    def _cleanup_queue_pair(self, server):
        server.cleanup()

    def _set_list_after_delay(self, list_proxy, values, delay=0.01):
        def updater():
            time.sleep(delay)
            list_proxy[:] = values

        thread = threading.Thread(target=updater)
        thread.start()
        return thread

    def _set_value_after_delay(self, value_proxy, value, delay=0.01):
        def updater():
            time.sleep(delay)
            value_proxy.set(value)

        thread = threading.Thread(target=updater)
        thread.start()
        return thread

    def test_to_tensor_success(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)
        tasks = [task]
        to_tensor(tasks)

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_tensor_disabled(self):
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)
        tasks = [task]
        to_tensor(tasks)

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_tensor_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()
        tasks = [task]

        # 不应抛异常
        try:
            to_tensor(tasks)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_to_tensor_exception_handling(self):
        bad_task = DummyTask(images="not an array")
        bad_tasks = [bad_task]

        try:
            to_tensor(bad_tasks)
        except Exception as e:
            self.fail(f"Exception should be handled internally, but got: {e}")

    def test_to_numpy_success(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        # 构造 paddle.Tensor 输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        task = DummyTask(tensor_images)
        tasks = [task]
        to_numpy(tasks)

        # 验证转换为 numpy.ndarray
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_disabled(self):
        # 创建随机张量作为测试输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        # 创建模拟任务
        task = DummyTask(tensor_images)
        tasks = [task]

        # 调用转换方法(预期不会转换)
        to_numpy(tasks)

        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()
        tasks = [task]

        # 不应抛异常
        try:
            to_numpy(tasks)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_to_numpy_non_tensor_input(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        np_images = np.random.randn(2, 3, 224, 224)
        task = DummyTask(np_images)
        tasks = [task]

        to_numpy(tasks)

        # 非 Tensor 输入应保持为 numpy 数组
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_exception_handling(self):
        envs.FD_ENABLE_MAX_PREFILL = 1

        # 构造错误输入（让 .numpy() 抛异常）
        class BadTensor:
            def numpy(self):
                raise RuntimeError("mock error")

        bad_task = DummyTask(images=BadTensor())
        bad_tasks = [bad_task]

        try:
            to_numpy(bad_tasks)
        except Exception as e:
            self.fail(f"Exception should be handled internally, but got: {e}")

    def test_features_info_to_tensor(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        np_feature = paddle.randn([2, 3, 224, 224]).numpy()
        multimodal_inputs = {
            "image_features": [np_feature, np_feature],
        }
        req_dict = {
            "request_id": "req1",
            "multimodal_inputs": multimodal_inputs,
        }
        task = Request.from_dict(req_dict)
        to_tensor([task])

        # 验证已转换为tensor
        self.assertEqual(len(task.multimodal_inputs["image_features"]), 2)
        self.assertIsInstance(task.multimodal_inputs["image_features"][0], paddle.Tensor)
        self.assertIsInstance(task.multimodal_inputs["image_features"][1], paddle.Tensor)

    def test_features_info_to_numpy(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        tensor_feature = paddle.randn([2, 3, 224, 224])
        multimodal_inputs = {
            "video_features": [tensor_feature, tensor_feature],
        }
        req_dict = {
            "request_id": "req1",
            "multimodal_inputs": multimodal_inputs,
        }
        task = Request.from_dict(req_dict)
        to_numpy([task])

        # 验证已转换为ndarray
        self.assertEqual(len(task.multimodal_inputs["video_features"]), 2)
        self.assertIsInstance(task.multimodal_inputs["video_features"][0], np.ndarray)
        self.assertIsInstance(task.multimodal_inputs["video_features"][1], np.ndarray)

    def test_queue_exist_tasks_and_ports(self):
        server, client = self._build_queue_pair()
        try:
            self.assertIsNone(server.exist_tasks_intra_signal)
            self.assertFalse(client.exist_tasks())
            client.set_exist_tasks(True)
            self.assertTrue(client.exist_tasks())
            self.assertEqual(server.get_server_port(), server.address[1])
            with self.assertRaises(RuntimeError):
                client.get_server_port()
        finally:
            self._cleanup_queue_pair(server)

    def test_single_node_signal_updates(self):
        server = EngineWorkerQueue(address=("0.0.0.0", 0), is_server=True, num_client=1, client_id=0)
        try:
            self.assertFalse(server.exist_tasks())
            server.set_exist_tasks(True)
            self.assertTrue(server.exist_tasks())
            server.set_exist_tasks(False)
            self.assertFalse(server.exist_tasks())
        finally:
            server.cleanup()
            server.exist_tasks_intra_signal.clear()

    def test_put_get_tasks_and_clear_data(self):
        envs.FD_ENABLE_MAX_PREFILL = 0
        envs.FD_ENABLE_E2W_TENSOR_CONVERT = 0
        server, client = self._build_queue_pair()
        try:
            tasks = ["task-A"]
            client.put_tasks(tasks)
            self.assertEqual(client.num_tasks(), 1)
            fetched, all_read = client.get_tasks()
            self.assertTrue(all_read)
            self.assertEqual(fetched, [tasks])
            self.assertEqual(client.num_tasks(), 0)
            client.put_tasks(tasks)
            client.clear_data()
            self.assertEqual(list(client.client_read_flag), [1])
            self.assertEqual(client.num_tasks(), 0)
        finally:
            self._cleanup_queue_pair(server)

    def test_wait_loops_and_tensor_conversion(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        envs.FD_ENABLE_E2W_TENSOR_CONVERT = 0
        server, client = self._build_queue_pair()
        previous_device = paddle.get_device()
        paddle.set_device("cpu")
        try:
            np_images = paddle.randn([1, 3, 4, 4]).numpy()
            task = DummyTask(np_images)
            tasks = [[task]]
            client.client_read_flag[:] = [0]
            thread = self._set_list_after_delay(client.client_read_flag, [1])
            client.put_tasks(tasks)
            thread.join()
            self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

            client.client_get_connect_task_flag[:] = [0]
            thread = self._set_list_after_delay(client.client_get_connect_task_flag, [1])
            client.put_connect_rdma_task({"connect": "wait"})
            thread.join()

            client.can_put_next_connect_task_response_flag.set(0)
            thread = self._set_value_after_delay(client.can_put_next_connect_task_response_flag, 1)
            client.put_connect_rdma_task_response({"success": True})
            thread.join()

            client.connect_rdma_task_responses.append({"success": True})
            client.client_get_connect_task_response_flag[:] = [0]
            thread = self._set_list_after_delay(client.client_get_connect_task_response_flag, [1])
            client.get_connect_rdma_task_response()
            thread.join()

            client.client_read_info_flag[:] = [0]
            thread = self._set_list_after_delay(client.client_read_info_flag, [1])
            client.put_cache_info([{"cache": "wait"}])
            thread.join()

            client.can_put_next_send_cache_finished_flag.set(0)
            thread = self._set_value_after_delay(client.can_put_next_send_cache_finished_flag, 1)
            client.put_finished_req([["req-wait", {"status": "ok"}]])
            thread.join()

            client.finished_send_cache_list.append(["req-wait", {"error": "bad"}])
            client.client_get_finish_send_cache_flag[:] = [0]
            thread = self._set_list_after_delay(client.client_get_finish_send_cache_flag, [1])
            client.get_finished_req()
            thread.join()

            client.can_put_next_add_task_finished_flag.set(0)
            thread = self._set_value_after_delay(client.can_put_next_add_task_finished_flag, 1)
            client.put_finished_add_cache_task_req(["req-wait"])
            thread.join()

            client.finished_add_cache_task_list.append(["req-wait"])
            client.client_get_finished_add_cache_task_flag[:] = [0]
            thread = self._set_list_after_delay(client.client_get_finished_add_cache_task_flag, [1])
            client.get_finished_add_cache_task_req()
            thread.join()
        finally:
            paddle.set_device(previous_device)
            self._cleanup_queue_pair(server)

    def test_connect_rdma_task_flow(self):
        server, client = self._build_queue_pair()
        try:
            client.client_get_connect_task_flag[:] = [1]
            client.put_connect_rdma_task({"connect": "ok"})
            task, all_read = client.get_connect_rdma_task()
            self.assertTrue(all_read)
            self.assertEqual(task, {"connect": "ok"})
            self.assertEqual(list(client.connect_rdma_tasks), [])

            self.assertIsNone(client.get_connect_rdma_task_response())
            response = {"success": True}
            self.assertTrue(client.put_connect_rdma_task_response(response))
            client.connect_rdma_task_responses.append({"success": False})
            merged = client.get_connect_rdma_task_response()
            self.assertEqual(merged["success"], False)
            self.assertEqual(client.can_put_next_connect_task_response_flag.get(), 1)
        finally:
            self._cleanup_queue_pair(server)

    def test_cache_info_and_counts(self):
        server, client = self._build_queue_pair()
        try:
            client.client_read_info_flag[:] = [1]
            cache_info = [{"cache": "info"}]
            client.put_cache_info(cache_info)
            self.assertEqual(client.num_cache_infos(), 1)
            self.assertEqual(client.get_cache_info(), cache_info)
            self.assertEqual(client.num_cache_infos(), 0)
            self.assertEqual(client.get_cache_info(), [])
        finally:
            self._cleanup_queue_pair(server)

    def test_finished_req_flow(self):
        server, client = self._build_queue_pair()
        try:
            send_cache_result = [["req-1", {"status": "ok"}]]
            self.assertTrue(client.put_finished_req(send_cache_result))
            client.finished_send_cache_list.append(["req-1", {"error": "bad"}])
            response = client.get_finished_req()
            self.assertEqual(response, [["req-1", {"error": "bad"}]])
            self.assertEqual(client.get_finished_req(), [])
            self.assertEqual(client.can_put_next_send_cache_finished_flag.get(), 1)
        finally:
            self._cleanup_queue_pair(server)

    def test_finished_add_cache_task_req(self):
        server, client = self._build_queue_pair()
        try:
            req_ids = ["req-2"]
            self.assertTrue(client.put_finished_add_cache_task_req(req_ids))
            client.finished_add_cache_task_list.append(req_ids)
            self.assertEqual(client.get_finished_add_cache_task_req(), req_ids)
            self.assertEqual(client.get_finished_add_cache_task_req(), [])
            self.assertEqual(client.can_put_next_add_task_finished_flag.get(), 1)
        finally:
            self._cleanup_queue_pair(server)

    def test_disaggregated_queue(self):
        server, client = self._build_queue_pair()
        try:
            self.assertTrue(client.disaggregate_queue_empty())
            client.put_disaggregated_tasks({"item": 1})
            client.put_disaggregated_tasks({"item": 2})
            self.assertFalse(client.disaggregate_queue_empty())
            self.assertEqual(client.get_disaggregated_tasks(), [{"item": 1}, {"item": 2}])
            self.assertIsNone(client.get_disaggregated_tasks())
        finally:
            self._cleanup_queue_pair(server)

    def test_connect_retry_failure(self):
        dummy = EngineWorkerQueue.__new__(EngineWorkerQueue)

        class DummyManager:
            def connect(self):
                raise ConnectionRefusedError("refused")

        dummy.manager = DummyManager()
        dummy.address = ("127.0.0.1", 9999)
        with self.assertRaises(ConnectionError):
            dummy._connect_with_retry(max_retries=2, interval=0)


if __name__ == "__main__":
    unittest.main()
