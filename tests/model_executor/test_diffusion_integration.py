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
Integration tests for diffusion models - focuses on DiffusionConfig.
"""

import unittest
import sys
import os
import importlib.util
from unittest.mock import Mock, patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import DiffusionConfig directly
DIFFUSION_AVAILABLE = True
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
except Exception as e:
    DIFFUSION_AVAILABLE = False
    print(f"Warning: Could not import DiffusionConfig: {e}")
    raise


class TestDiffusionIntegration(unittest.TestCase):
    """Integration tests for DiffusionConfig."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_diffusion_config_basic(self):
        """Test basic DiffusionConfig functionality."""
        config = DiffusionConfig(
            model_path="/tmp/test_model",
            device="gpu",
            use_fp16=True
        )

        self.assertEqual(config.model_path, "/tmp/test_model")
        self.assertEqual(config.device, "gpu")
        self.assertEqual(config.use_fp16, True)

    def test_config_with_custom_height_width(self):
        """Test DiffusionConfig with custom height and width."""
        config = DiffusionConfig(
            height=1024,
            width=1024,
            num_inference_steps=50,
            guidance_scale=12.0
        )
        self.assertEqual(config.height, 1024)
        self.assertEqual(config.width, 1024)
        self.assertEqual(config.num_inference_steps, 50)
        self.assertEqual(config.guidance_scale, 12.0)

    def test_config_device_validation(self):
        """Test that invalid devices are rejected."""
        with self.assertRaises(ValueError):
            DiffusionConfig(device="invalid_device")

    def test_config_model_type_validation(self):
        """Test that invalid model types are rejected."""
        with self.assertRaises(ValueError):
            DiffusionConfig(model_type="invalid_model")

    def test_config_valid_model_types(self):
        """Test that valid model types are accepted."""
        valid_types = ["stable-diffusion", "sdxl", "sd3", "flux"]
        for model_type in valid_types:
            config = DiffusionConfig(model_type=model_type)
            self.assertEqual(config.model_type, model_type)

    def test_config_inference_steps_validation(self):
        """Test that invalid inference steps are rejected."""
        with self.assertRaises(ValueError):
            DiffusionConfig(num_inference_steps=0)
        
        with self.assertRaises(ValueError):
            DiffusionConfig(num_inference_steps=-1)

    def test_config_dimensions_validation(self):
        """Test that invalid dimensions are rejected."""
        with self.assertRaises(ValueError):
            DiffusionConfig(height=0, width=512)
        
        with self.assertRaises(ValueError):
            DiffusionConfig(height=512, width=0)
        
        with self.assertRaises(ValueError):
            DiffusionConfig(height=-1, width=512)

    def test_config_tensorrt_gpu_only(self):
        """Test that TensorRT is GPU-only."""
        # TensorRT on GPU should work
        config = DiffusionConfig(device="gpu", use_tensorrt=True)
        self.assertTrue(config.use_tensorrt)
        
        # TensorRT on CPU should fail
        with self.assertRaises(ValueError):
            DiffusionConfig(device="cpu", use_tensorrt=True)

    def test_config_to_dict_integration(self):
        """Test config to_dict and from_dict integration."""
        original_config = DiffusionConfig(
            model_path="/path/to/model",
            model_type="flux",
            device="gpu",
            use_fp16=False,
            height=768,
            width=768,
            num_inference_steps=30,
            guidance_scale=9.0,
            max_batch_size=2
        )
        
        config_dict = original_config.to_dict()
        restored_config = DiffusionConfig.from_dict(config_dict)
        
        self.assertEqual(restored_config.model_path, original_config.model_path)
        self.assertEqual(restored_config.model_type, original_config.model_type)
        self.assertEqual(restored_config.device, original_config.device)
        self.assertEqual(restored_config.use_fp16, original_config.use_fp16)
        self.assertEqual(restored_config.height, original_config.height)
        self.assertEqual(restored_config.width, original_config.width)

    def test_config_cinn_xpu_compatibility(self):
        """Test config with CINN optimization on XPU."""
        config = DiffusionConfig(device="xpu", use_cinn=True)
        self.assertEqual(config.device, "xpu")
        self.assertTrue(config.use_cinn)

    def test_config_memory_optimization(self):
        """Test memory optimization settings."""
        config = DiffusionConfig(
            enable_memory_optimization=True,
            enable_dynamic_shape=True
        )
        self.assertTrue(config.enable_memory_optimization)
        self.assertTrue(config.enable_dynamic_shape)

    def test_config_extra_kwargs(self):
        """Test that extra kwargs are stored."""
        config = DiffusionConfig(
            custom_param1="value1",
            custom_param2=42
        )
        config_dict = config.to_dict()
        self.assertIn("custom_param1", config_dict)
        self.assertIn("custom_param2", config_dict)
        self.assertEqual(config_dict["custom_param1"], "value1")
        self.assertEqual(config_dict["custom_param2"], 42)


class TestDiffusionConfigDefaults(unittest.TestCase):
    """Test default configurations."""

    def test_default_device_is_gpu(self):
        """Test that default device is GPU."""
        config = DiffusionConfig()
        self.assertEqual(config.device, "gpu")

    def test_default_fp16_enabled(self):
        """Test that FP16 is enabled by default."""
        config = DiffusionConfig()
        self.assertTrue(config.use_fp16)

    def test_default_cinn_enabled(self):
        """Test that CINN is enabled by default."""
        config = DiffusionConfig()
        self.assertTrue(config.use_cinn)

    def test_default_batch_size_is_one(self):
        """Test that default batch size is 1."""
        config = DiffusionConfig()
        self.assertEqual(config.max_batch_size, 1)

    def test_default_resolution_is_512(self):
        """Test that default resolution is 512x512."""
        config = DiffusionConfig()
        self.assertEqual(config.height, 512)
        self.assertEqual(config.width, 512)

    def test_default_inference_steps_is_20(self):
        """Test that default inference steps is 20."""
        config = DiffusionConfig()
        self.assertEqual(config.num_inference_steps, 20)

    def test_default_guidance_scale_is_7_5(self):
        """Test that default guidance scale is 7.5."""
        config = DiffusionConfig()
        self.assertEqual(config.guidance_scale, 7.5)


if __name__ == '__main__':
    unittest.main()
