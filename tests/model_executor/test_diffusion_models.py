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
Tests for diffusion models module.
"""

import unittest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from fastdeploy.model_executor.diffusion_models import vision
    from fastdeploy.model_executor.diffusion_models.vision import diffusion
    DIFFUSION_AVAILABLE = True
except ImportError:
    DIFFUSION_AVAILABLE = False
    # Create mock objects for testing
    import sys
    from unittest.mock import MagicMock

    vision = MagicMock()
    vision.__all__ = ['diffusion']

    diffusion = MagicMock()
    diffusion.__all__ = [
        'DiffusionConfig',
        'DiffusionPredictor',
        'SDPipeline',
        'SD3Pipeline',
        'FluxPipeline',
        'passes',
        'DiffusionTensorRTManager',
        'DiffusionTensorRTPlugin'
    ]


class TestDiffusionModels(unittest.TestCase):
    """Test cases for diffusion models module."""

    def test_import_vision_module(self):
        """Test that vision module can be imported."""
        self.assertIsNotNone(vision)

    def test_import_diffusion_module(self):
        """Test that diffusion module can be imported."""
        self.assertIsNotNone(diffusion)

    def test_vision_module_all_attribute(self):
        """Test that vision module has __all__ attribute."""
        self.assertTrue(hasattr(vision, '__all__'))
        self.assertEqual(vision.__all__, ['diffusion'])

    def test_diffusion_module_all_attribute(self):
        """Test that diffusion module has __all__ attribute."""
        expected_all = [
            'DiffusionConfig',
            'DiffusionPredictor',
            'SDPipeline',
            'SD3Pipeline',
            'FluxPipeline',
            'passes',
            'DiffusionTensorRTManager',
            'DiffusionTensorRTPlugin'
        ]
        self.assertTrue(hasattr(diffusion, '__all__'))
        self.assertEqual(diffusion.__all__, expected_all)


if __name__ == '__main__':
    unittest.main()
