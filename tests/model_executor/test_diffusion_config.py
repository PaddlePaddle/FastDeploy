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
Tests for DiffusionConfig class - Refactored to use Paddle framework.
"""

import sys
import os
import importlib.util

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import new Paddle test framework
# Use relative import or direct file location import
paddle_test_base_path = os.path.join(os.path.dirname(__file__), 'paddle_test_base.py')
spec = importlib.util.spec_from_file_location("paddle_test_base", paddle_test_base_path)
paddle_test_base_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paddle_test_base_module)
PaddleDiffusionTestCase = paddle_test_base_module.PaddleDiffusionTestCase

# Import DiffusionConfig directly without importing the full fastdeploy package
CONFIG_AVAILABLE = True
try:
    # Load the config module directly
    spec = importlib.util.spec_from_file_location(
        "config_module",
        os.path.join(os.path.dirname(__file__), '..', '..', 'fastdeploy', 
                     'model_executor', 'diffusion_models', 'vision', 'diffusion', 'config.py')
    )
    config_module = importlib.util.module_from_spec(spec)
    sys.modules['config_module'] = config_module
    spec.loader.exec_module(config_module)
    DiffusionConfig = config_module.DiffusionConfig
except Exception as e:
    CONFIG_AVAILABLE = False
    print(f"Warning: Could not import DiffusionConfig: {e}")
    raise


class TestDiffusionConfig(PaddleDiffusionTestCase):
    """Test cases for DiffusionConfig class - Using Paddle test framework."""

    # setUp is handled by parent class automatically

    def test_config_initialization_default(self):
        """Test DiffusionConfig initialization with default values."""
        config = DiffusionConfig()

        # Test default values
        self.assertIsNone(config.model_path)
        self.assertEqual(config.device, "gpu")
        self.assertEqual(config.use_fp16, True)
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.num_inference_steps, 20)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertEqual(config.enable_memory_optimization, True)
        self.assertEqual(config.enable_dynamic_shape, True)

    def test_config_initialization_custom(self):
        """Test DiffusionConfig initialization with custom values."""
        config = DiffusionConfig(
            model_path="/path/to/model",
            device="gpu",
            use_fp16=False,
            use_tensorrt=True,
            use_cinn=False,
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=12.0,
            max_batch_size=4,
            enable_memory_optimization=False,
            enable_dynamic_shape=False
        )

        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.device, "gpu")
        self.assertEqual(config.use_fp16, False)
        self.assertEqual(config.height, 1024)
        self.assertEqual(config.width, 1024)
        self.assertEqual(config.num_inference_steps, 50)
        self.assertEqual(config.guidance_scale, 12.0)
        self.assertEqual(config.max_batch_size, 4)
        self.assertTrue(config.use_tensorrt)
        self.assertFalse(config.use_cinn)
        self.assertFalse(config.enable_memory_optimization)
        self.assertFalse(config.enable_dynamic_shape)

    def test_config_validation(self):
        """Test DiffusionConfig validation methods."""
        config = DiffusionConfig()

        # Test validation was performed
        self.assertTrue(hasattr(config, '_validate_config'))
        # Test that invalid device raises error
        with self.assertRaises(ValueError):
            DiffusionConfig(device="invalid_device")

    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = DiffusionConfig()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn('model_path', config_dict)
        self.assertIn('device', config_dict)
        self.assertIn('use_fp16', config_dict)
        self.assertIn('height', config_dict)
        self.assertIn('width', config_dict)
        self.assertIn('num_inference_steps', config_dict)
        self.assertIn('guidance_scale', config_dict)
        self.assertIn('max_batch_size', config_dict)
        self.assertIn('use_tensorrt', config_dict)
        self.assertIn('enable_memory_optimization', config_dict)
        self.assertIn('enable_dynamic_shape', config_dict)

    def test_config_from_dict(self):
        """Test config from dictionary creation."""
        config_dict = {
            'model_path': '/path/to/model',
            'device': 'gpu',
            'use_fp16': True,
            'use_tensorrt': False,
            'height': 512,
            'width': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'max_batch_size': 2,
            'enable_memory_optimization': True,
            'enable_dynamic_shape': True
        }

        config = DiffusionConfig.from_dict(config_dict)

        self.assertEqual(config.model_path, '/path/to/model')
        self.assertEqual(config.device, 'gpu')
        self.assertEqual(config.use_fp16, True)
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.num_inference_steps, 20)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertEqual(config.max_batch_size, 2)
        self.assertFalse(config.use_tensorrt)
        self.assertTrue(config.enable_memory_optimization)
        self.assertTrue(config.enable_dynamic_shape)


if __name__ == '__main__':
    unittest.main()
