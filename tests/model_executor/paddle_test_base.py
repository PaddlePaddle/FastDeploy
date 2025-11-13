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
Paddle-based Test Base Classes for Diffusion Models

提供Paddle特定的测试基础设施，替代torch相关的测试。
"""

import unittest
import paddle
import numpy as np
from typing import Optional, Dict, Any, Callable
import logging
import os
import importlib.util
import sys

# Import paddle_test_utils using importlib to avoid relative import issues
paddle_test_utils_path = os.path.join(os.path.dirname(__file__), 'paddle_test_utils.py')
spec = importlib.util.spec_from_file_location("paddle_test_utils", paddle_test_utils_path)
paddle_test_utils = importlib.util.module_from_spec(spec)
sys.modules['paddle_test_utils'] = paddle_test_utils
spec.loader.exec_module(paddle_test_utils)

PaddleTensorMock = paddle_test_utils.PaddleTensorMock
PaddleDeviceMock = paddle_test_utils.PaddleDeviceMock
PaddleOperationsMock = paddle_test_utils.PaddleOperationsMock
MockPaddleModel = paddle_test_utils.MockPaddleModel
PaddleTestHelper = paddle_test_utils.PaddleTestHelper

logger = logging.getLogger(__name__)


class PaddleTestCase(unittest.TestCase):
    """
    Paddle测试基类
    
    提供Paddle特定的测试方法和设置/拆卸。
    """
    
    def setUp(self) -> None:
        """测试前准备"""
        super().setUp()
        self.device_mock = PaddleDeviceMock()
        self.test_helper = PaddleTestHelper()
        
        # 记录初始设备状态
        self.initial_device = paddle.get_device()
        logger.info(f"setUp: Initial device = {self.initial_device}")
    
    def tearDown(self) -> None:
        """测试后清理"""
        # 恢复设备状态
        try:
            paddle.set_device(self.initial_device)
        except Exception:
            pass
        
        logger.info("tearDown: Device restored")
        super().tearDown()
    
    def assertTensorShape(self, tensor: paddle.Tensor, 
                         expected_shape: tuple) -> None:
        """断言张量形状"""
        # Paddle返回的shape可能是list，需要转换为tuple比较
        actual_shape = tuple(tensor.shape) if isinstance(tensor.shape, (list, tuple)) else tensor.shape
        self.assertEqual(actual_shape, expected_shape,
                        f"Expected shape {expected_shape}, got {actual_shape}")
    
    def assertTensorDtype(self, tensor: paddle.Tensor, 
                         expected_dtype) -> None:
        """断言张量数据类型"""
        self.assertEqual(tensor.dtype, expected_dtype,
                        f"Expected dtype {expected_dtype}, got {tensor.dtype}")
    
    def assertTensorEqual(self, tensor1: paddle.Tensor, 
                         tensor2: paddle.Tensor,
                         rtol: float = 1e-5,
                         atol: float = 1e-8) -> None:
        """
        断言两个张量相等
        
        Args:
            tensor1: 第一个张量
            tensor2: 第二个张量
            rtol: 相对容差
            atol: 绝对容差
        """
        self.assertTrue(
            paddle.allclose(tensor1, tensor2, rtol=rtol, atol=atol),
            f"Tensors are not equal.\n"
            f"tensor1 stats: min={paddle.min(tensor1)}, max={paddle.max(tensor1)}, mean={paddle.mean(tensor1)}\n"
            f"tensor2 stats: min={paddle.min(tensor2)}, max={paddle.max(tensor2)}, mean={paddle.mean(tensor2)}"
        )
    
    def assertTensorNumericallyStable(self, tensor: paddle.Tensor) -> None:
        """断言张量数值稳定性"""
        numpy_data = tensor.numpy()
        self.assertFalse(np.isnan(numpy_data).any(), 
                        "Tensor contains NaN values")
        self.assertFalse(np.isinf(numpy_data).any(),
                        "Tensor contains Inf values")
    
    def assertDeviceAvailable(self, device: str) -> None:
        """断言设备可用"""
        self.assertTrue(self.device_mock.is_available(device),
                       f"Device {device} not available")
    
    def create_tensor(self, shape: tuple, 
                     dtype: str = "float32",
                     init_type: str = "randn") -> paddle.Tensor:
        """创建测试张量"""
        return PaddleOperationsMock.create_tensor(shape, dtype, init_type)
    
    def create_tensor_mock(self, shape: tuple,
                          dtype=paddle.float32,
                          data: Optional[np.ndarray] = None) -> PaddleTensorMock:
        """创建张量Mock对象"""
        return PaddleTensorMock(shape, dtype, data)
    
    def create_model_mock(self, model_name: str = "test_model") -> MockPaddleModel:
        """创建模型Mock对象"""
        return MockPaddleModel(model_name)


class PaddleDiffusionTestCase(PaddleTestCase):
    """
    Paddle扩散模型测试基类
    
    提供扩散模型特定的测试方法。
    """
    
    def setUp(self) -> None:
        """扩散模型测试前准备"""
        super().setUp()
        self.config_mock = self.test_helper.create_diffusion_config_mock()
        logger.info("Diffusion model test setup complete")
    
    def assertValidPrompt(self, prompt: str) -> None:
        """断言prompt有效"""
        self.assertIsNotNone(prompt)
        self.assertGreater(len(prompt), 0)
        self.assertIsInstance(prompt, str)
    
    def assertValidLatent(self, latent: paddle.Tensor,
                         batch_size: int = 1,
                         channels: int = 4,
                         height: int = 64,
                         width: int = 64) -> None:
        """
        断言latent有效
        
        Args:
            latent: Latent张量
            batch_size: 批次大小
            channels: 通道数
            height: 高度
            width: 宽度
        """
        expected_shape = (batch_size, channels, height, width)
        self.assertTensorShape(latent, expected_shape)
        self.assertTensorNumericallyStable(latent)
    
    def assertValidEmbedding(self, embedding: paddle.Tensor,
                            batch_size: int = 1,
                            seq_length: int = 77,
                            hidden_size: int = 768) -> None:
        """
        断言embedding有效
        
        Args:
            embedding: 嵌入张量
            batch_size: 批次大小
            seq_length: 序列长度
            hidden_size: 隐藏大小
        """
        expected_shape = (batch_size, seq_length, hidden_size)
        self.assertTensorShape(embedding, expected_shape)
        self.assertTensorNumericallyStable(embedding)
    
    def assertValidImage(self, image: paddle.Tensor,
                        batch_size: int = 1,
                        channels: int = 3,
                        height: int = 512,
                        width: int = 512) -> None:
        """
        断言图像有效
        
        Args:
            image: 图像张量
            batch_size: 批次大小
            channels: 通道数
            height: 高度
            width: 宽度
        """
        expected_shape = (batch_size, channels, height, width)
        self.assertTensorShape(image, expected_shape)
        self.assertTensorNumericallyStable(image)
        
        # 检查像素值范围
        image_min = paddle.min(image)
        image_max = paddle.max(image)
        self.assertLessEqual(image_min, 1.0, "Image min value too high")
        self.assertGreaterEqual(image_max, 0.0, "Image max value too low")
    
    def create_diffusion_config_mock(self, **kwargs) -> Dict[str, Any]:
        """创建DiffusionConfig Mock"""
        return self.test_helper.create_diffusion_config_mock(**kwargs)
    
    def create_prompt(self, text: str = "a beautiful landscape") -> str:
        """创建测试prompt"""
        return text
    
    def create_latent_mock(self, batch_size: int = 1) -> paddle.Tensor:
        """创建模拟latent"""
        return self.create_tensor(
            (batch_size, 4, 64, 64),
            dtype="float32",
            init_type="randn"
        )
    
    def create_image_mock(self, batch_size: int = 1) -> paddle.Tensor:
        """创建模拟图像"""
        # 创建[0, 1]范围的图像
        image = paddle.rand((batch_size, 3, 512, 512), dtype=paddle.float32)
        return image
    
    def create_embedding_mock(self, batch_size: int = 1,
                             seq_length: int = 77,
                             hidden_size: int = 768) -> paddle.Tensor:
        """创建模拟embedding"""
        return self.create_tensor(
            (batch_size, seq_length, hidden_size),
            dtype="float32",
            init_type="randn"
        )


class PaddleSchedulerTestCase(PaddleTestCase):
    """
    Paddle调度器测试基类
    """
    
    def assertValidTimesteps(self, timesteps: paddle.Tensor,
                            num_steps: int) -> None:
        """断言timesteps有效"""
        self.assertEqual(len(timesteps), num_steps)
        # 检查timesteps是否递减
        self.assertTrue(paddle.all(timesteps[:-1] >= timesteps[1:]),
                       "Timesteps should be in descending order")
    
    def assertValidScheduleCoefficients(self, alphas: paddle.Tensor) -> None:
        """断言调度系数有效"""
        # 检查范围
        self.assertLessEqual(paddle.max(alphas), 1.0)
        self.assertGreaterEqual(paddle.min(alphas), 0.0)
        # 检查数值稳定性
        self.assertTensorNumericallyStable(alphas)


class PaddleInferenceTestCase(PaddleTestCase):
    """
    Paddle推理测试基类
    """
    
    def assertInferenceOutputValid(self, output: paddle.Tensor,
                                   expected_shape: Optional[tuple] = None,
                                   check_range: Optional[tuple] = None) -> None:
        """
        断言推理输出有效
        
        Args:
            output: 推理输出
            expected_shape: 期望形状
            check_range: 检查值范围 (min, max)
        """
        self.assertIsNotNone(output)
        self.assertTensorNumericallyStable(output)
        
        if expected_shape:
            self.assertTensorShape(output, expected_shape)
        
        if check_range:
            min_val, max_val = check_range
            self.assertGreaterEqual(paddle.min(output), min_val)
            self.assertLessEqual(paddle.max(output), max_val)
    
    def assertBatchInferenceConsistent(self, outputs: list) -> None:
        """断言批次推理输出一致"""
        self.assertGreater(len(outputs), 0)
        
        # 所有输出应该有相同的形状
        first_shape = outputs[0].shape
        for output in outputs[1:]:
            self.assertEqual(output.shape, first_shape)
        
        # 检查数值稳定性
        for i, output in enumerate(outputs):
            self.assertTensorNumericallyStable(output)

