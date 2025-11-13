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
Paddle Testing Utilities for Diffusion Models

这个模块提供了Paddle特定的测试工具，用于替代torch相关的mock和patch操作。
包含以下功能：
- Paddle张量操作模拟
- Paddle设备管理模拟
- Paddle模型管理模拟
- 数值稳定性验证
"""

import paddle
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from typing import Tuple, Optional, Dict, Any, Union, List
import logging

logger = logging.getLogger(__name__)


class PaddleTensorMock:
    """
    模拟Paddle张量用于测试
    
    提供与real paddle.Tensor相同的接口，但可以轻松控制行为用于测试。
    """
    
    def __init__(self, shape: Tuple[int, ...], dtype=paddle.float32, 
                 data: Optional[np.ndarray] = None, device: str = "cpu"):
        """
        初始化模拟张量
        
        Args:
            shape: 张量形状
            dtype: 数据类型
            data: 初始数据（如果为None则随机生成）
            device: 设备类型("cpu" 或 "gpu")
        """
        self.shape = shape
        self.dtype = dtype
        self.device = device
        
        if data is not None:
            self._data = paddle.to_tensor(data, dtype=dtype)
        else:
            if dtype == paddle.float32 or dtype == paddle.float16:
                self._data = paddle.randn(shape, dtype=dtype)
            else:
                self._data = paddle.randint(0, 100, shape)
    
    @property
    def data(self) -> paddle.Tensor:
        """获取实际的Paddle张量"""
        return self._data
    
    def numpy(self) -> np.ndarray:
        """转换为numpy数组"""
        return self._data.numpy()
    
    def copy(self) -> 'PaddleTensorMock':
        """复制张量"""
        new_mock = PaddleTensorMock(
            self.shape, self.dtype, self._data.numpy(), self.device
        )
        return new_mock
    
    def to(self, device: str) -> 'PaddleTensorMock':
        """移动到指定设备"""
        self.device = device
        return self
    
    def __repr__(self) -> str:
        return f"PaddleTensorMock(shape={self.shape}, dtype={self.dtype}, device={self.device})"


class PaddleDeviceMock:
    """
    模拟Paddle设备管理
    """
    
    def __init__(self):
        """初始化设备模拟"""
        self.current_device = "cpu"
        self.available_devices = ["cpu"]
        self.gpu_available = False
        
        try:
            if paddle.device.is_compiled_with_cuda():
                self.available_devices.append("gpu")
                self.gpu_available = True
        except Exception:
            pass
    
    def set_device(self, device: str) -> None:
        """设置当前设备"""
        if device not in self.available_devices:
            raise RuntimeError(f"Device {device} not available. Available: {self.available_devices}")
        self.current_device = device
        paddle.set_device(device)
    
    def get_device(self) -> str:
        """获取当前设备"""
        return self.current_device
    
    def is_available(self, device: str) -> bool:
        """检查设备是否可用"""
        return device in self.available_devices
    
    def __repr__(self) -> str:
        return f"PaddleDeviceMock(current={self.current_device}, available={self.available_devices})"


class PaddleOperationsMock:
    """
    模拟常用的Paddle操作
    """
    
    @staticmethod
    def create_tensor(shape: Tuple[int, ...], 
                     dtype: str = "float32",
                     init_type: str = "randn") -> paddle.Tensor:
        """创建张量"""
        paddle_dtype = getattr(paddle, dtype)
        
        if init_type == "randn":
            return paddle.randn(shape, dtype=paddle_dtype)
        elif init_type == "zeros":
            return paddle.zeros(shape, dtype=paddle_dtype)
        elif init_type == "ones":
            return paddle.ones(shape, dtype=paddle_dtype)
        elif init_type == "rand":
            return paddle.rand(shape, dtype=paddle_dtype)
        else:
            raise ValueError(f"Unknown init_type: {init_type}")
    
    @staticmethod
    def check_tensor_shape(tensor: paddle.Tensor, 
                          expected_shape: Tuple[int, ...]) -> bool:
        """验证张量形状"""
        # Paddle返回的shape可能是list，需要转换为tuple比较
        actual_shape = tuple(tensor.shape) if isinstance(tensor.shape, (list, tuple)) else tensor.shape
        return actual_shape == expected_shape
    
    @staticmethod
    def check_tensor_dtype(tensor: paddle.Tensor, 
                          expected_dtype) -> bool:
        """验证张量数据类型"""
        return tensor.dtype == expected_dtype
    
    @staticmethod
    def check_tensor_device(tensor: paddle.Tensor, 
                           expected_device: str) -> bool:
        """验证张量设备"""
        actual_place = tensor.place
        if expected_device == "cpu":
            return isinstance(actual_place, paddle.CPUPlace)
        elif expected_device == "gpu":
            return isinstance(actual_place, paddle.CUDAPlace)
        else:
            return True
    
    @staticmethod
    def check_numerical_stability(tensor: paddle.Tensor) -> Dict[str, Any]:
        """检查数值稳定性"""
        numpy_data = tensor.numpy()
        
        return {
            "has_nan": bool(np.isnan(numpy_data).any()),
            "has_inf": bool(np.isinf(numpy_data).any()),
            "min_value": float(np.min(numpy_data)),
            "max_value": float(np.max(numpy_data)),
            "mean_value": float(np.mean(numpy_data)),
            "std_value": float(np.std(numpy_data)),
        }


class MockPaddleModel:
    """
    模拟Paddle模型用于测试
    
    提供与真实Paddle模型相同的接口，用于单元测试。
    """
    
    def __init__(self, model_name: str = "mock_model", 
                 input_shape: Optional[Dict[str, Tuple]] = None,
                 output_shape: Optional[Dict[str, Tuple]] = None):
        """
        初始化模拟模型
        
        Args:
            model_name: 模型名称
            input_shape: 输入形状字典
            output_shape: 输出形状字典
        """
        self.model_name = model_name
        self.input_shape = input_shape or {"input": (1, 3, 512, 512)}
        self.output_shape = output_shape or {"output": (1, 1000)}
        self.input_tensors = {}
        self.output_tensors = {}
        self._initialize_tensors()
    
    def _initialize_tensors(self) -> None:
        """初始化输入输出张量"""
        for name, shape in self.input_shape.items():
            self.input_tensors[name] = PaddleTensorMock(shape, dtype=paddle.float32)
        
        for name, shape in self.output_shape.items():
            self.output_tensors[name] = PaddleTensorMock(shape, dtype=paddle.float32)
    
    def get_input_tensor(self, name: str) -> PaddleTensorMock:
        """获取输入张量"""
        if name not in self.input_tensors:
            raise ValueError(f"Input tensor {name} not found")
        return self.input_tensors[name]
    
    def get_output_tensor(self, name: str) -> PaddleTensorMock:
        """获取输出张量"""
        if name not in self.output_tensors:
            raise ValueError(f"Output tensor {name} not found")
        return self.output_tensors[name]
    
    def set_input_tensor(self, name: str, data: np.ndarray) -> None:
        """设置输入张量数据"""
        if name not in self.input_tensors:
            raise ValueError(f"Input tensor {name} not found")
        self.input_tensors[name]._data = paddle.to_tensor(data)
    
    def run(self) -> None:
        """运行模型（模拟推理）"""
        # 模拟推理过程
        for name in self.output_tensors:
            # 简单的线性变换模拟
            self.output_tensors[name]._data = paddle.randn(self.output_shape[name])
    
    def __repr__(self) -> str:
        return f"MockPaddleModel({self.model_name}, inputs={list(self.input_shape.keys())}, outputs={list(self.output_shape.keys())})"


class PaddleInferenceMock:
    """
    模拟Paddle推理接口
    """
    
    def __init__(self, model_file: Optional[str] = None, 
                 params_file: Optional[str] = None):
        """
        初始化推理模拟
        
        Args:
            model_file: 模型文件路径
            params_file: 参数文件路径
        """
        self.model_file = model_file
        self.params_file = params_file
        self.model = None
    
    def load_model(self) -> MockPaddleModel:
        """加载模型"""
        self.model = MockPaddleModel()
        logger.info(f"Loaded mock model from {self.model_file}")
        return self.model
    
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """运行推理"""
        if self.model is None:
            self.load_model()
        
        # 设置输入
        for name, data in inputs.items():
            self.model.set_input_tensor(name, data)
        
        # 运行模型
        self.model.run()
        
        # 获取输出
        outputs = {}
        for name in self.model.output_tensors:
            outputs[name] = self.model.get_output_tensor(name).numpy()
        
        return outputs


class PaddleTestHelper:
    """
    Paddle测试助手类
    
    提供常用的测试操作。
    """
    
    @staticmethod
    def create_diffusion_config_mock(model_type: str = "stable-diffusion",
                                    device: str = "cpu",
                                    use_fp16: bool = False) -> Mock:
        """创建DiffusionConfig的Mock对象"""
        mock_config = Mock()
        mock_config.model_type = model_type
        mock_config.device = device
        mock_config.use_fp16 = use_fp16
        mock_config.height = 512
        mock_config.width = 512
        mock_config.num_inference_steps = 20
        mock_config.guidance_scale = 7.5
        mock_config.max_batch_size = 1
        mock_config.enable_memory_optimization = False
        mock_config.enable_dynamic_shape = False
        return mock_config
    
    @staticmethod
    def assert_tensor_equal(tensor1: paddle.Tensor, 
                           tensor2: paddle.Tensor,
                           rtol: float = 1e-5,
                           atol: float = 1e-8) -> bool:
        """
        断言两个张量是否相等
        
        Args:
            tensor1: 第一个张量
            tensor2: 第二个张量
            rtol: 相对容差
            atol: 绝对容差
        
        Returns:
            是否相等
        """
        return bool(paddle.allclose(tensor1, tensor2, rtol=rtol, atol=atol))
    
    @staticmethod
    def assert_tensor_shape(tensor: paddle.Tensor, 
                           expected_shape: Tuple[int, ...]) -> bool:
        """断言张量形状"""
        return tensor.shape == expected_shape
    
    @staticmethod
    def print_tensor_stats(tensor: paddle.Tensor, name: str = "tensor") -> None:
        """打印张量统计信息"""
        numpy_data = tensor.numpy()
        logger.info(f"{name} Statistics:")
        logger.info(f"  Shape: {tensor.shape}")
        logger.info(f"  Dtype: {tensor.dtype}")
        logger.info(f"  Min: {np.min(numpy_data):.6f}")
        logger.info(f"  Max: {np.max(numpy_data):.6f}")
        logger.info(f"  Mean: {np.mean(numpy_data):.6f}")
        logger.info(f"  Std: {np.std(numpy_data):.6f}")


# 便利函数
def create_mock_device() -> PaddleDeviceMock:
    """创建Paddle设备模拟"""
    return PaddleDeviceMock()


def create_mock_tensor(shape: Tuple[int, ...], 
                      dtype: str = "float32") -> paddle.Tensor:
    """创建模拟张量"""
    return PaddleOperationsMock.create_tensor(shape, dtype)


def create_mock_model(model_name: str = "test_model") -> MockPaddleModel:
    """创建模拟模型"""
    return MockPaddleModel(model_name)


def verify_tensor_numerical_stability(tensor: paddle.Tensor) -> Dict[str, Any]:
    """验证张量数值稳定性"""
    return PaddleOperationsMock.check_numerical_stability(tensor)

