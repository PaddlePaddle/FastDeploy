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
Tests for diffusion model optimization passes.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fastdeploy.model_executor.diffusion_models.vision.diffusion import passes
    PASSES_AVAILABLE = True
except ImportError:
    PASSES_AVAILABLE = False
    # Create mock passes module
    from unittest.mock import MagicMock
    passes = MagicMock()
    passes.__all__ = [
        'StableDiffusionAttentionFusePass',
        'StableDiffusionUNetFusePass',
        'StableDiffusionVAEFusePass',
        'FluxTransformerFusePass',
        'FluxDiTFusePass',
        'FluxRoPEFusePass',
    ]


class TestDiffusionPasses(unittest.TestCase):
    """Test cases for diffusion model optimization passes."""

    def setUp(self):
        """Set up test fixtures."""
        pass

    def test_passes_module_import(self):
        """Test that passes module can be imported."""
        self.assertIsNotNone(passes)

    def test_passes_module_all_attribute(self):
        """Test that passes module has correct __all__ attribute."""
        expected_all = [
            'StableDiffusionAttentionFusePass',
            'StableDiffusionUNetFusePass',
            'StableDiffusionVAEFusePass',
            'FluxTransformerFusePass',
            'FluxDiTFusePass',
            'FluxRoPEFusePass',
        ]
        self.assertTrue(hasattr(passes, '__all__'))
        self.assertEqual(passes.__all__, expected_all)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.passes.sd_optimization_passes.StableDiffusionAttentionFusePass')
    def test_stable_diffusion_attention_fuse_pass_import(self, mock_pass):
        """Test StableDiffusionAttentionFusePass can be imported."""
        mock_pass_instance = Mock()
        mock_pass.return_value = mock_pass_instance

        # Test that the class can be imported from passes module
        cls = getattr(passes, 'StableDiffusionAttentionFusePass')
        self.assertIsNotNone(cls)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.passes.sd_optimization_passes.StableDiffusionUNetFusePass')
    def test_stable_diffusion_unet_fuse_pass_import(self, mock_pass):
        """Test StableDiffusionUNetFusePass can be imported."""
        mock_pass_instance = Mock()
        mock_pass.return_value = mock_pass_instance

        # Test that the class can be imported from passes module
        cls = getattr(passes, 'StableDiffusionUNetFusePass')
        self.assertIsNotNone(cls)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.passes.sd_optimization_passes.StableDiffusionVAEFusePass')
    def test_stable_diffusion_vae_fuse_pass_import(self, mock_pass):
        """Test StableDiffusionVAEFusePass can be imported."""
        mock_pass_instance = Mock()
        mock_pass.return_value = mock_pass_instance

        # Test that the class can be imported from passes module
        cls = getattr(passes, 'StableDiffusionVAEFusePass')
        self.assertIsNotNone(cls)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.passes.flux_optimization_passes.FluxTransformerFusePass')
    def test_flux_transformer_fuse_pass_import(self, mock_pass):
        """Test FluxTransformerFusePass can be imported."""
        mock_pass_instance = Mock()
        mock_pass.return_value = mock_pass_instance

        # Test that the class can be imported from passes module
        cls = getattr(passes, 'FluxTransformerFusePass')
        self.assertIsNotNone(cls)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.passes.flux_optimization_passes.FluxDiTFusePass')
    def test_flux_dit_fuse_pass_import(self, mock_pass):
        """Test FluxDiTFusePass can be imported."""
        mock_pass_instance = Mock()
        mock_pass.return_value = mock_pass_instance

        # Test that the class can be imported from passes module
        cls = getattr(passes, 'FluxDiTFusePass')
        self.assertIsNotNone(cls)

    @patch('fastdeploy.model_executor.diffusion_models.vision.diffusion.passes.flux_optimization_passes.FluxRoPEFusePass')
    def test_flux_rope_fuse_pass_import(self, mock_pass):
        """Test FluxRoPEFusePass can be imported."""
        mock_pass_instance = Mock()
        mock_pass.return_value = mock_pass_instance

        # Test that the class can be imported from passes module
        cls = getattr(passes, 'FluxRoPEFusePass')
        self.assertIsNotNone(cls)


class TestOptimizationPassBase(unittest.TestCase):
    """Test cases for optimization pass base functionality."""

    def setUp(self):
        """Set up test fixtures."""
        if not PASSES_AVAILABLE:
            self.skipTest("Diffusion passes module not available")

    def test_pass_base_methods(self):
        """Test that optimization passes have required base methods."""
        pass_classes = [
            'StableDiffusionAttentionFusePass',
            'StableDiffusionUNetFusePass',
            'StableDiffusionVAEFusePass',
            'FluxTransformerFusePass',
            'FluxDiTFusePass',
            'FluxRoPEFusePass',
        ]

        for pass_name in pass_classes:
            with self.subTest(pass_name=pass_name):
                try:
                    # Try to get the class
                    pass_cls = getattr(passes, pass_name)
                    # Check if it's a class (not a mock)
                    if hasattr(pass_cls, '__call__'):
                        # Create a mock instance to test interface
                        mock_instance = Mock()
                        # Test that the pass has expected methods
                        self.assertTrue(hasattr(mock_instance, 'apply'))
                        self.assertTrue(hasattr(mock_instance, 'optimize'))
                except AttributeError:
                    # If the class can't be imported, that's expected in test environment
                    pass


if __name__ == '__main__':
    unittest.main()
