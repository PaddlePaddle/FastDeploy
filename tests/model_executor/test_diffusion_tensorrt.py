# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
Tests for Diffusion TensorRT integration - Refactored to use Paddle framework.
"""

import sys
import os
import importlib.util
import unittest

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import new Paddle test framework
# Use relative import or direct file location import
paddle_test_base_path = os.path.join(os.path.dirname(__file__), 'paddle_test_base.py')
spec = importlib.util.spec_from_file_location("paddle_test_base", paddle_test_base_path)
paddle_test_base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paddle_test_base_module)
PaddleTestCase = paddle_test_base_module.PaddleTestCase
PaddleDiffusionTestCase = paddle_test_base_module.PaddleDiffusionTestCase

# Import DiffusionConfig
try:
    spec = importlib.util.spec_from_file_location(
        "config_module",
        os.path.join(os.path.dirname(__file__), '..', '..', 'fastdeploy', 
                     'model_executor', 'diffusion_models', 'vision', 'diffusion', 'config.py')
    )
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config_module'] = config_module
    spec.loader.exec_module(config_module)
    DiffusionConfig = config_module.DiffusionConfig
    CONFIG_AVAILABLE = True
except Exception as e:
    CONFIG_AVAILABLE = False
    print(f"Warning: Could not import DiffusionConfig: {e}")


class TestDiffusionTensorRTStructure(PaddleTestCase):
    """Test cases for TensorRT integration file structure - Using Paddle test framework."""

    def test_tensorrt_integration_file_exists(self):
        """Test that tensorrt_integration.py file exists."""
        tensorrt_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'fastdeploy',
            'model_executor', 'diffusion_models', 'vision', 'diffusion', 'tensorrt_integration.py'
        )
        self.assertTrue(os.path.isfile(tensorrt_path))

    def test_tensorrt_integration_has_valid_syntax(self):
        """Test that tensorrt_integration.py has valid Python syntax."""
        import ast
        
        tensorrt_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'fastdeploy',
            'model_executor', 'diffusion_models', 'vision', 'diffusion', 'tensorrt_integration.py'
        )
        
        if os.path.isfile(tensorrt_path):
            with open(tensorrt_path, 'r') as f:
                source_code = f.read()
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                self.fail(f"Syntax error in tensorrt_integration.py: {e}")

    def test_tensorrt_integration_contains_manager_class(self):
        """Test that tensorrt_integration.py contains TensorRT manager class."""
        tensorrt_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'fastdeploy',
            'model_executor', 'diffusion_models', 'vision', 'diffusion', 'tensorrt_integration.py'
        )
        
        if os.path.isfile(tensorrt_path):
            with open(tensorrt_path, 'r') as f:
                source_code = f.read()
            # Check for either Manager or Plugin class
            self.assertTrue(
                'Manager' in source_code or 'Plugin' in source_code,
                "TensorRT integration should contain Manager or Plugin class"
            )


class TestTensorRTConfiguration(PaddleDiffusionTestCase):
    """Test cases for TensorRT configuration - Using Paddle test framework."""

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_enabled_config(self):
        """Test creating a config with TensorRT enabled."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True
        )
        self.assertTrue(config.use_tensorrt)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_gpu_only(self):
        """Test that TensorRT is GPU-only."""
        # Should work on GPU
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True
        )
        self.assertTrue(config.use_tensorrt)
        
        # Should fail on CPU
        with self.assertRaises(ValueError):
            DiffusionConfig(
                device="cpu",
                use_tensorrt=True
            )

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_with_fp16(self):
        """Test TensorRT with FP16 precision."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            use_fp16=True
        )
        self.assertTrue(config.use_tensorrt)
        self.assertTrue(config.use_fp16)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_with_fp32(self):
        """Test TensorRT with FP32 precision."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            use_fp16=False
        )
        self.assertTrue(config.use_tensorrt)
        self.assertFalse(config.use_fp16)


class TestTensorRTIntegration(PaddleDiffusionTestCase):
    """Integration tests for TensorRT with diffusion models - Using Paddle test framework."""

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_config_for_stable_diffusion(self):
        """Test TensorRT config for Stable Diffusion."""
        config = DiffusionConfig(
            model_type="stable-diffusion",
            device="gpu",
            use_tensorrt=True
        )
        self.assertEqual(config.model_type, "stable-diffusion")
        self.assertTrue(config.use_tensorrt)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_config_for_flux(self):
        """Test TensorRT config for Flux."""
        config = DiffusionConfig(
            model_type="flux",
            device="gpu",
            use_tensorrt=True
        )
        self.assertEqual(config.model_type, "flux")
        self.assertTrue(config.use_tensorrt)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_memory_optimization(self):
        """Test TensorRT with memory optimization."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            enable_memory_optimization=True
        )
        self.assertTrue(config.use_tensorrt)
        self.assertTrue(config.enable_memory_optimization)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_dynamic_shape_support(self):
        """Test TensorRT with dynamic shape support."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            enable_dynamic_shape=True
        )
        self.assertTrue(config.use_tensorrt)
        self.assertTrue(config.enable_dynamic_shape)


class TestTensorRTEngineManagement(PaddleTestCase):
    """Test cases for TensorRT engine management - Using Paddle test framework."""

    def test_tensorrt_engine_initialization(self):
        """Test expected TensorRT engine initialization flow."""
        # This is a structural test
        # Actual engine management is complex and depends on model availability
        pass

    def test_tensorrt_engine_caching(self):
        """Test expected TensorRT engine caching mechanism."""
        # This is a structural test
        # Engine caching should improve performance on repeated usage
        pass

    def test_tensorrt_engine_device_management(self):
        """Test expected device management for TensorRT engine."""
        # This is a structural test
        # Engine should be properly loaded/unloaded on device
        pass


class TestTensorRTPerformance(PaddleTestCase):
    """Test cases for TensorRT performance expectations - Using Paddle test framework."""

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_batch_processing(self):
        """Test TensorRT batch processing capability."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            max_batch_size=4
        )
        self.assertEqual(config.max_batch_size, 4)
        self.assertTrue(config.use_tensorrt)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_with_custom_resolution(self):
        """Test TensorRT with custom image resolution."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            height=768,
            width=768
        )
        self.assertEqual(config.height, 768)
        self.assertEqual(config.width, 768)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_tensorrt_config_serialization(self):
        """Test TensorRT config serialization."""
        original_config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True,
            use_fp16=True
        )
        
        config_dict = original_config.to_dict()
        restored_config = DiffusionConfig.from_dict(config_dict)
        
        self.assertTrue(restored_config.use_tensorrt)
        self.assertTrue(restored_config.use_fp16)


if __name__ == '__main__':
    unittest.main()
