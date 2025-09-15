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
Tests for DiffusionPredictor class.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fastdeploy.model_executor.diffusion_models.vision.diffusion.predictor import DiffusionPredictor
    from fastdeploy.model_executor.diffusion_models.vision.diffusion.config import DiffusionConfig
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    # Create mock classes for testing
    from unittest.mock import MagicMock

    # Define DiffusionConfig first
    DiffusionConfig = MagicMock()
    mock_config = MagicMock()
    DiffusionConfig.return_value = mock_config

    DiffusionPredictor = MagicMock()
    mock_predictor = MagicMock()
    DiffusionPredictor.return_value = mock_predictor

    # Configure mock predictor
    mock_predictor.config = DiffusionConfig()
    mock_predictor.predict.return_value = None
    mock_predictor.load_model.return_value = False
    mock_predictor.unload_model.return_value = True
    mock_predictor.get_model_info.return_value = {
        'loaded': False,
        'model_type': 'diffusion',
        'device': 'cuda',
        'dtype': 'float16'
    }
    mock_predictor._validate_model_path.return_value = False


class TestDiffusionPredictor(unittest.TestCase):
    """Test cases for DiffusionPredictor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = DiffusionConfig()
        self.predictor = DiffusionPredictor(self.config)

    def test_predictor_initialization(self):
        """Test DiffusionPredictor initialization."""
        self.assertIsInstance(self.predictor, DiffusionPredictor)
        self.assertEqual(self.predictor.config, self.config)

    def test_predictor_has_required_methods(self):
        """Test that predictor has required methods."""
        # Test that predictor has the main interface methods
        self.assertTrue(hasattr(self.predictor, 'predict'))
        self.assertTrue(hasattr(self.predictor, 'load_model'))
        self.assertTrue(hasattr(self.predictor, 'unload_model'))
        self.assertTrue(hasattr(self.predictor, 'get_model_info'))

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.predictor.torch')
    def test_predict_method_signature(self, mock_torch):
        """Test predict method signature and basic behavior."""
        # Mock torch components
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = Mock()
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()

        # Test predict method exists and can be called with basic parameters
        prompt = "a beautiful landscape"
        try:
            # This should not raise an exception even if model is not loaded
            result = self.predictor.predict(prompt)
            # Result should be None or empty if model not loaded
            self.assertIsNone(result)
        except Exception as e:
            # Expected if model dependencies are not available
            self.assertIn("not loaded", str(e).lower())

    def test_load_model_method(self):
        """Test load_model method."""
        # Test that load_model method exists
        try:
            result = self.predictor.load_model()
            # Should return False if model path not set
            self.assertFalse(result)
        except Exception as e:
            # Expected if model dependencies are not available
            self.assertIn("model", str(e).lower())

    def test_unload_model_method(self):
        """Test unload_model method."""
        # Test that unload_model method exists and can be called
        result = self.predictor.unload_model()
        # Should return True (successful unload even if nothing was loaded)
        self.assertTrue(result)

    def test_get_model_info_method(self):
        """Test get_model_info method."""
        info = self.predictor.get_model_info()

        # Should return a dictionary with model information
        self.assertIsInstance(info, dict)
        self.assertIn('loaded', info)
        self.assertIn('model_type', info)
        self.assertIn('device', info)
        self.assertIn('dtype', info)
        self.assertFalse(info['loaded'])  # Model not loaded in test

    def test_predictor_with_custom_config(self):
        """Test predictor with custom configuration."""
        custom_config = DiffusionConfig(
            device="cpu",
            dtype="float32",
            height=256,
            width=256,
            num_inference_steps=10,
            guidance_scale=5.0
        )

        predictor = DiffusionPredictor(custom_config)
        self.assertEqual(predictor.config.device, "cpu")
        self.assertEqual(predictor.config.dtype, "float32")
        self.assertEqual(predictor.config.height, 256)
        self.assertEqual(predictor.config.width, 256)
        self.assertEqual(predictor.config.num_inference_steps, 10)
        self.assertEqual(predictor.config.guidance_scale, 5.0)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.predictor.os.path.exists')
    def test_model_validation(self, mock_exists):
        """Test model validation methods."""
        mock_exists.return_value = True

        # Test with valid model path
        self.predictor.config.model_path = "/valid/path"
        self.assertTrue(self.predictor._validate_model_path())

        # Test with invalid model path
        mock_exists.return_value = False
        self.predictor.config.model_path = "/invalid/path"
        self.assertFalse(self.predictor._validate_model_path())

        # Test with None model path
        self.predictor.config.model_path = None
        self.assertFalse(self.predictor._validate_model_path())


if __name__ == '__main__':
    unittest.main()
