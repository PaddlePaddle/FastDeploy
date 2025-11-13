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
Tests for DiffusionPredictor class and file availability - Refactored to use Paddle framework.
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


class TestDiffusionPredictorStructure(PaddleTestCase):
    """Test cases for DiffusionPredictor structure - Using Paddle test framework."""

    def test_predictor_file_exists(self):
        """Test that predictor.py file exists."""
        predictor_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'fastdeploy',
            'model_executor', 'diffusion_models', 'vision', 'diffusion', 'predictor.py'
        )
        self.assertTrue(os.path.isfile(predictor_path))

    def test_predictor_file_has_valid_syntax(self):
        """Test that predictor.py file has valid Python syntax."""
        import ast
        
        predictor_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'fastdeploy',
            'model_executor', 'diffusion_models', 'vision', 'diffusion', 'predictor.py'
        )
        
        if os.path.isfile(predictor_path):
            with open(predictor_path, 'r') as f:
                source_code = f.read()
            try:
                ast.parse(source_code)
            except SyntaxError as e:
                self.fail(f"Syntax error in predictor.py: {e}")

    def test_predictor_contains_class_definition(self):
        """Test that predictor.py contains DiffusionPredictor class."""
        predictor_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'fastdeploy',
            'model_executor', 'diffusion_models', 'vision', 'diffusion', 'predictor.py'
        )
        
        if os.path.isfile(predictor_path):
            with open(predictor_path, 'r') as f:
                source_code = f.read()
            self.assertIn('class DiffusionPredictor', source_code)


class TestPredictorInterface(PaddleDiffusionTestCase):
    """Test cases for expected predictor interface - Using Paddle test framework."""

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_initialization(self):
        """Test predictor with DiffusionConfig."""
        config = DiffusionConfig(
            model_type="stable-diffusion",
            device="gpu"
        )
        self.assertIsNotNone(config)
        self.assertEqual(config.model_type, "stable-diffusion")

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_with_tensorrt(self):
        """Test predictor config with TensorRT."""
        config = DiffusionConfig(
            device="gpu",
            use_tensorrt=True
        )
        self.assertTrue(config.use_tensorrt)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_for_different_models(self):
        """Test predictor config for different model types."""
        model_types = ["stable-diffusion", "sd3", "flux"]
        for model_type in model_types:
            config = DiffusionConfig(model_type=model_type)
            self.assertEqual(config.model_type, model_type)

    def test_predictor_expected_methods(self):
        """Test that predictor should have expected methods."""
        # This is a structural expectation
        # The actual predictor class should implement:
        # - __init__(config: DiffusionConfig)
        # - predict(prompt: str) -> List[Image]
        # - load_model() -> bool
        # - unload_model() -> bool
        # - get_model_info() -> Dict
        pass


class TestDiffusionPredictorIntegration(PaddleDiffusionTestCase):
    """Integration tests for DiffusionPredictor - Using Paddle test framework."""

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_with_custom_dimensions(self):
        """Test predictor with custom image dimensions."""
        config = DiffusionConfig(
            height=768,
            width=768,
            num_inference_steps=30
        )
        self.assertEqual(config.height, 768)
        self.assertEqual(config.width, 768)
        self.assertEqual(config.num_inference_steps, 30)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_with_memory_optimization(self):
        """Test predictor config with memory optimization."""
        config = DiffusionConfig(
            enable_memory_optimization=True,
            enable_dynamic_shape=True,
            use_fp16=True
        )
        self.assertTrue(config.enable_memory_optimization)
        self.assertTrue(config.enable_dynamic_shape)
        self.assertTrue(config.use_fp16)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_batch_support(self):
        """Test predictor config with batch support."""
        config = DiffusionConfig(max_batch_size=4)
        self.assertEqual(config.max_batch_size, 4)

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_predictor_config_serialization(self):
        """Test predictor config serialization and deserialization."""
        original_config = DiffusionConfig(
            model_path="/models/stable-diffusion",
            model_type="stable-diffusion",
            device="gpu",
            height=512,
            width=512
        )
        
        config_dict = original_config.to_dict()
        restored_config = DiffusionConfig.from_dict(config_dict)
        
        self.assertEqual(restored_config.model_type, original_config.model_type)
        self.assertEqual(restored_config.device, original_config.device)
        self.assertEqual(restored_config.height, original_config.height)


class TestDiffusionPredictorErrorHandling(unittest.TestCase):
    """Test cases for predictor error handling."""

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_invalid_model_type_handling(self):
        """Test handling of invalid model types."""
        with self.assertRaises(ValueError):
            DiffusionConfig(model_type="invalid_type")

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_invalid_device_handling(self):
        """Test handling of invalid devices."""
        with self.assertRaises(ValueError):
            DiffusionConfig(device="invalid_device")

    @unittest.skipUnless(CONFIG_AVAILABLE, "DiffusionConfig not available")
    def test_invalid_dimensions_handling(self):
        """Test handling of invalid image dimensions."""
        with self.assertRaises(ValueError):
            DiffusionConfig(height=0, width=512)


if __name__ == '__main__':
    unittest.main()
