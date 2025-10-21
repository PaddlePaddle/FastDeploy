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
        envs.FD_ENABLE_MM_TENSOR_CONVERT = 1
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)

        EngineWorkerQueue.to_tensor([task])

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_tensor_disabled(self):
        envs.FD_ENABLE_MM_TENSOR_CONVERT = 0
        # 模拟 numpy 数组输入（使用 paddle 转 numpy）
        np_images = paddle.randn([2, 3, 224, 224]).numpy()
        task = DummyTask(np_images)

        EngineWorkerQueue.to_tensor([task])

        # 验证已转换为tensor
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_tensor_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()

        # 不应抛异常
        try:
            EngineWorkerQueue.to_tensor([task])
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_to_tensor_exception_handling(self):
        bad_task = DummyTask(images="not an array")

        try:
            EngineWorkerQueue.to_tensor([bad_task])
        except Exception as e:
            self.fail(f"Exception should be handled internally, but got: {e}")

    def test_to_numpy_success(self):
        envs.FD_ENABLE_MM_TENSOR_CONVERT = 1
        # 构造 paddle.Tensor 输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        task = DummyTask(tensor_images)

        EngineWorkerQueue.to_numpy([task])

        # 验证转换为 numpy.ndarray
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_disabled(self):
        # 禁用张量转换开关
        envs.FD_ENABLE_MM_TENSOR_CONVERT = 0
        # 创建随机张量作为测试输入
        tensor_images = paddle.randn([2, 3, 224, 224])
        # 创建模拟任务
        task = DummyTask(tensor_images)

        # 调用转换方法(预期不会转换)
        EngineWorkerQueue.to_numpy([task])

        # 因为开关关闭，应仍为 Tensor
        self.assertIsInstance(task.multimodal_inputs["images"], paddle.Tensor)

    def test_to_numpy_no_multimodal_inputs(self):
        class NoMMTask:
            pass

        task = NoMMTask()

        # 不应抛异常
        try:
            EngineWorkerQueue.to_numpy([task])
        except Exception as e:
            self.fail(f"Unexpected exception raised: {e}")

    def test_to_numpy_non_tensor_input(self):
        envs.FD_ENABLE_MM_TENSOR_CONVERT = 1
        np_images = np.random.randn(2, 3, 224, 224)
        task = DummyTask(np_images)

        EngineWorkerQueue.to_numpy([task])

        # 非 Tensor 输入应保持为 numpy 数组
        self.assertIsInstance(task.multimodal_inputs["images"], np.ndarray)

    def test_to_numpy_exception_handling(self):
        envs.FD_ENABLE_MM_TENSOR_CONVERT = 1

        # 构造错误输入（让 .numpy() 抛异常）
        class BadTensor:
            def numpy(self):
                raise RuntimeError("mock error")

        bad_task = DummyTask(images=BadTensor())

        try:
            EngineWorkerQueue.to_numpy([bad_task])
        except Exception as e:
            self.fail(f"Exception should be handled internally, but got: {e}")


if __name__ == "__main__":
    unittest.main()
