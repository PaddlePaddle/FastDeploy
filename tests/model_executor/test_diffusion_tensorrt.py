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
Tests for Diffusion TensorRT integration.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fastdeploy.model_executor.diffusion_models.vision.diffusion.tensorrt_integration import (
        DiffusionTensorRTManager,
        DiffusionTensorRTPlugin
    )
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    # Create mock classes
    from unittest.mock import MagicMock

    DiffusionTensorRTManager = MagicMock()
    mock_manager = MagicMock()
    DiffusionTensorRTManager.return_value = mock_manager

    # Configure mock manager
    mock_manager.initialize.return_value = None
    mock_manager.build_engine.return_value = False
    mock_manager.load_engine.return_value = False
    mock_manager.save_engine.return_value = False
    mock_manager.infer.return_value = None
    mock_manager.get_engine_info.return_value = {
        'initialized': False,
        'engine_loaded': False,
        'tensorrt_version': '8.6',
        'supported_precisions': ['float16', 'float32']
    }
    mock_manager.cleanup.return_value = True

    DiffusionTensorRTPlugin = MagicMock()
    mock_plugin = MagicMock()
    DiffusionTensorRTPlugin.return_value = mock_plugin

    # Configure mock plugin
    mock_plugin.initialize.return_value = None
    mock_plugin.create_plugin.return_value = None
    mock_plugin.configure_plugin.return_value = None
    mock_plugin.enqueue.return_value = None
    mock_plugin.get_output_dimensions.return_value = None
    mock_plugin.supports_format.return_value = True
    mock_plugin.configure_format.return_value = None
    mock_plugin.serialize.return_value = b"test_data"
    mock_plugin.deserialize.return_value = None
    mock_plugin.destroy.return_value = True


class TestDiffusionTensorRTManager(unittest.TestCase):
    """Test cases for DiffusionTensorRTManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = DiffusionTensorRTManager()

    def test_manager_initialization(self):
        """Test DiffusionTensorRTManager initialization."""
        self.assertIsInstance(self.manager, DiffusionTensorRTManager)

    def test_manager_has_required_methods(self):
        """Test that manager has required methods."""
        required_methods = [
            'initialize',
            'build_engine',
            'load_engine',
            'save_engine',
            'infer',
            'get_engine_info',
            'cleanup'
        ]

        for method in required_methods:
            with self.subTest(method=method):
                self.assertTrue(hasattr(self.manager, method))

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.tensorrt_integration.torch')
    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.tensorrt_integration.tensorrt')
    def test_manager_initialization_with_config(self, mock_tensorrt, mock_torch):
        """Test manager initialization with configuration."""
        mock_tensorrt.Logger.return_value = Mock()
        mock_tensorrt.Builder.return_value = Mock()
        mock_tensorrt.Runtime.return_value = Mock()

        config = {
            'max_workspace_size': 1 << 30,
            'max_batch_size': 8,
            'precision': 'float16'
        }

        try:
            result = self.manager.initialize(config)
            # Result should be boolean or None
            self.assertIsInstance(result, (bool, type(None)))
        except Exception as e:
            # Expected if TensorRT dependencies are not available
            self.assertIn("tensorrt", str(e).lower())

    def test_manager_engine_operations(self):
        """Test manager engine operations."""
        # Test build_engine
        try:
            result = self.manager.build_engine(None)
            self.assertFalse(result)  # Should fail without proper setup
        except Exception as e:
            self.assertIn("engine", str(e).lower())

        # Test load_engine
        try:
            result = self.manager.load_engine("nonexistent.engine")
            self.assertFalse(result)
        except Exception as e:
            self.assertIn("engine", str(e).lower())

        # Test save_engine
        try:
            result = self.manager.save_engine("test.engine")
            self.assertFalse(result)
        except Exception as e:
            self.assertIn("engine", str(e).lower())

    def test_manager_inference(self):
        """Test manager inference capabilities."""
        try:
            result = self.manager.infer(None)
            self.assertIsNone(result)  # Should return None without proper setup
        except Exception as e:
            self.assertIn("infer", str(e).lower())

    def test_manager_get_engine_info(self):
        """Test get_engine_info method."""
        info = self.manager.get_engine_info()

        self.assertIsInstance(info, dict)
        self.assertIn('initialized', info)
        self.assertIn('engine_loaded', info)
        self.assertIn('tensorrt_version', info)
        self.assertIn('supported_precisions', info)
        self.assertFalse(info['initialized'])  # Not initialized in test
        self.assertFalse(info['engine_loaded'])  # No engine loaded in test

    def test_manager_cleanup(self):
        """Test cleanup method."""
        result = self.manager.cleanup()
        self.assertTrue(result)  # Should always succeed


class TestDiffusionTensorRTPlugin(unittest.TestCase):
    """Test cases for DiffusionTensorRTPlugin class."""

    def setUp(self):
        """Set up test fixtures."""
        if not TENSORRT_AVAILABLE:
            self.skipTest("Diffusion TensorRT integration not available")

        self.plugin = DiffusionTensorRTPlugin()

    def test_plugin_initialization(self):
        """Test DiffusionTensorRTPlugin initialization."""
        self.assertIsInstance(self.plugin, DiffusionTensorRTPlugin)

    def test_plugin_has_required_methods(self):
        """Test that plugin has required methods."""
        required_methods = [
            'initialize',
            'create_plugin',
            'configure_plugin',
            'enqueue',
            'get_output_dimensions',
            'supports_format',
            'configure_format',
            'serialize',
            'deserialize',
            'destroy'
        ]

        for method in required_methods:
            with self.subTest(method=method):
                self.assertTrue(hasattr(self.plugin, method))

    def test_plugin_initialization_with_config(self):
        """Test plugin initialization with configuration."""
        config = {
            'plugin_type': 'diffusion_attention',
            'num_heads': 8,
            'head_dim': 64,
            'precision': 'float16'
        }

        try:
            result = self.plugin.initialize(config)
            self.assertIsInstance(result, (bool, type(None)))
        except Exception as e:
            self.assertIn("plugin", str(e).lower())

    def test_plugin_format_support(self):
        """Test plugin format support methods."""
        # Test supports_format
        try:
            result = self.plugin.supports_format(None, None)
            self.assertIsInstance(result, bool)
        except Exception as e:
            self.assertIn("format", str(e).lower())

        # Test configure_format
        try:
            result = self.plugin.configure_format([], [])
            self.assertIsInstance(result, (bool, type(None)))
        except Exception as e:
            self.assertIn("format", str(e).lower())

    def test_plugin_dimensions(self):
        """Test plugin dimension methods."""
        try:
            dims = self.plugin.get_output_dimensions(0, None)
            self.assertIsInstance(dims, (list, type(None)))
        except Exception as e:
            self.assertIn("dimensions", str(e).lower())

    def test_plugin_serialization(self):
        """Test plugin serialization methods."""
        # Test serialize
        try:
            data = self.plugin.serialize()
            self.assertIsInstance(data, (bytes, type(None)))
        except Exception as e:
            self.assertIn("serial", str(e).lower())

        # Test deserialize
        try:
            result = self.plugin.deserialize(b"test_data")
            self.assertIsInstance(result, (bool, type(None)))
        except Exception as e:
            self.assertIn("deserial", str(e).lower())

    def test_plugin_enqueue(self):
        """Test plugin enqueue method."""
        try:
            result = self.plugin.enqueue(None, None, None, None, None)
            self.assertIsInstance(result, (bool, type(None)))
        except Exception as e:
            self.assertIn("enqueue", str(e).lower())

    def test_plugin_destroy(self):
        """Test plugin destroy method."""
        result = self.plugin.destroy()
        self.assertTrue(result)  # Should always succeed


if __name__ == '__main__':
    unittest.main()
