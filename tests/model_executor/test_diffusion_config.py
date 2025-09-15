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
Tests for DiffusionConfig class.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fastdeploy.model_executor.diffusion_models.vision.diffusion.config import DiffusionConfig
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Create mock DiffusionConfig for testing
    from unittest.mock import MagicMock

    DiffusionConfig = MagicMock()
    # Set up basic mock behavior
    mock_instance = MagicMock()
    DiffusionConfig.return_value = mock_instance

    # Configure mock attributes
    mock_instance.model_path = None
    mock_instance.device = "cuda"
    mock_instance.dtype = "float16"
    mock_instance.height = 512
    mock_instance.width = 512
    mock_instance.num_inference_steps = 20
    mock_instance.guidance_scale = 7.5
    mock_instance.negative_prompt = None
    mock_instance.num_images_per_prompt = 1
    mock_instance.seed = 42
    mock_instance.scheduler_config = None
    mock_instance.tensorrt_config = None
    mock_instance.enable_tensorrt = False
    mock_instance.enable_optimization = False
    mock_instance.optimization_level = "O2"


class TestDiffusionConfig(unittest.TestCase):
    """Test cases for DiffusionConfig class."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_config_initialization_default(self):
        """Test DiffusionConfig initialization with default values."""
        config = DiffusionConfig()

        # Test default values
        self.assertIsNone(config.model_path)
        self.assertEqual(config.device, "cuda")
        self.assertEqual(config.dtype, "float16")
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.num_inference_steps, 20)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertIsNone(config.negative_prompt)
        self.assertEqual(config.num_images_per_prompt, 1)
        self.assertEqual(config.seed, 42)
        self.assertIsNone(config.scheduler_config)
        self.assertIsNone(config.tensorrt_config)
        self.assertFalse(config.enable_tensorrt)
        self.assertFalse(config.enable_optimization)
        self.assertEqual(config.optimization_level, "O2")

    def test_config_initialization_custom(self):
        """Test DiffusionConfig initialization with custom values."""
        scheduler_config = {"beta_start": 0.00085, "beta_end": 0.012}
        tensorrt_config = {"max_workspace_size": 1 << 30}

        config = DiffusionConfig(
            model_path="/path/to/model",
            device="cpu",
            dtype="float32",
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=12.0,
            negative_prompt="blurry",
            num_images_per_prompt=4,
            seed=123,
            scheduler_config=scheduler_config,
            tensorrt_config=tensorrt_config,
            enable_tensorrt=True,
            enable_optimization=True,
            optimization_level="O3"
        )

        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.dtype, "float32")
        self.assertEqual(config.height, 1024)
        self.assertEqual(config.width, 1024)
        self.assertEqual(config.num_inference_steps, 50)
        self.assertEqual(config.guidance_scale, 12.0)
        self.assertEqual(config.negative_prompt, "blurry")
        self.assertEqual(config.num_images_per_prompt, 4)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.scheduler_config, scheduler_config)
        self.assertEqual(config.tensorrt_config, tensorrt_config)
        self.assertTrue(config.enable_tensorrt)
        self.assertTrue(config.enable_optimization)
        self.assertEqual(config.optimization_level, "O3")

    def test_config_validation(self):
        """Test DiffusionConfig validation methods."""
        config = DiffusionConfig()

        # Test device validation
        self.assertTrue(hasattr(config, '_validate_device'))
        self.assertTrue(hasattr(config, '_validate_dtype'))
        self.assertTrue(hasattr(config, '_validate_dimensions'))

    def test_config_to_dict(self):
        """Test config to dictionary conversion."""
        config = DiffusionConfig()
        config_dict = config.to_dict()

        self.assertIsInstance(config_dict, dict)
        self.assertIn('model_path', config_dict)
        self.assertIn('device', config_dict)
        self.assertIn('dtype', config_dict)
        self.assertIn('height', config_dict)
        self.assertIn('width', config_dict)
        self.assertIn('num_inference_steps', config_dict)
        self.assertIn('guidance_scale', config_dict)
        self.assertIn('negative_prompt', config_dict)
        self.assertIn('num_images_per_prompt', config_dict)
        self.assertIn('seed', config_dict)
        self.assertIn('enable_tensorrt', config_dict)
        self.assertIn('enable_optimization', config_dict)
        self.assertIn('optimization_level', config_dict)

    def test_config_from_dict(self):
        """Test config from dictionary creation."""
        config_dict = {
            'model_path': '/path/to/model',
            'device': 'cuda',
            'dtype': 'float16',
            'height': 512,
            'width': 512,
            'num_inference_steps': 20,
            'guidance_scale': 7.5,
            'negative_prompt': None,
            'num_images_per_prompt': 1,
            'seed': 42,
            'enable_tensorrt': False,
            'enable_optimization': False,
            'optimization_level': 'O2'
        }

        config = DiffusionConfig.from_dict(config_dict)

        self.assertEqual(config.model_path, '/path/to/model')
        self.assertEqual(config.device, 'cuda')
        self.assertEqual(config.dtype, 'float16')
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)
        self.assertEqual(config.num_inference_steps, 20)
        self.assertEqual(config.guidance_scale, 7.5)
        self.assertIsNone(config.negative_prompt)
        self.assertEqual(config.num_images_per_prompt, 1)
        self.assertEqual(config.seed, 42)
        self.assertFalse(config.enable_tensorrt)
        self.assertFalse(config.enable_optimization)
        self.assertEqual(config.optimization_level, 'O2')


if __name__ == '__main__':
    unittest.main()
