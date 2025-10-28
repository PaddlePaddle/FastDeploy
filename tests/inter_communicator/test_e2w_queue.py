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

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.inter_communicator.engine_worker_queue import EngineWorkerQueue


class DummyTask:
    def __init__(self, images):
        self.multimodal_inputs = {"images": images}


class TestEngineWorkerQueue(unittest.TestCase):
    def test_to_tensor_success(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)
        tasks = ([task], 1)

        EngineWorkerQueue.to_tensor(tasks)

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_tensor_disabled(self):
        envs.FD_ENABLE_MAX_PREFILL = 0
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)
        tasks = ([task], 1)

        EngineWorkerQueue.to_tensor(tasks)

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_tensor_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()
        tasks = ([task], 1)

        # 不应抛异常
        try:
            EngineWorkerQueue.to_tensor(tasks)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_to_tensor_exception_handling(self):
        bad_task = DummyTask(images="not an array")
        bad_tasks = ([bad_task], 1)

        try:
            EngineWorkerQueue.to_tensor(bad_tasks)
        except Exception as e:
            self.fail(f"Exception should be handled internally, but got: {e}")

    def test_to_numpy_success(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        # 构造 paddle.Tensor 输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        task = DummyTask(tensor_images)
        tasks = [([task], 1)]

        EngineWorkerQueue.to_numpy(tasks)

        # 验证转换为 numpy.ndarray
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_disabled(self):
        # 禁用张量转换开关
        envs.FD_ENABLE_MAX_PREFILL = 0
        # 创建随机张量作为测试输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        # 创建模拟任务
        task = DummyTask(tensor_images)
        tasks = [([task], 1)]

        # 调用转换方法(预期不会转换)
        EngineWorkerQueue.to_numpy(tasks)

        # 因为开关关闭，应仍为 Tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_numpy_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()
        tasks = [([task], 1)]

        # 不应抛异常
        try:
            EngineWorkerQueue.to_numpy(tasks)
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_to_numpy_non_tensor_input(self):
        envs.FD_ENABLE_MAX_PREFILL = 1
        np_images = np.random.randn(2, 3, 224, 224)
        task = DummyTask(np_images)
        tasks = [([task], 1)]

        EngineWorkerQueue.to_numpy(tasks)

        # 非 Tensor 输入应保持为 numpy 数组
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_exception_handling(self):
        envs.FD_ENABLE_MAX_PREFILL = 1

        # 构造错误输入（让 .numpy() 抛异常）
        class BadTensor:
            def numpy(self):
                raise RuntimeError("mock error")

        bad_task = DummyTask(images=BadTensor())
        bad_tasks = [([bad_task], 1)]

        try:
            EngineWorkerQueue.to_numpy(bad_tasks)
        except Exception as e:
            self.fail(f"Exception should be handled internally, but got: {e}")


if __name__ == "__main__":
    unittest.main()
