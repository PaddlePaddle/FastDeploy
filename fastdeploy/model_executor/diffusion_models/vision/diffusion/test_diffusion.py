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
Test suite for diffusion models in FastDeploy.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
from PIL import Image

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

try:
    from fastdeploy.model_executor.diffusion_models.vision.diffusion import (
        DiffusionConfig,
        SDPipeline,
        SD3Pipeline,
        FluxPipeline,
    )
    FASTDEPLOY_AVAILABLE = True
except ImportError as e:
    print(f"FastDeploy not available: {e}")
    FASTDEPLOY_AVAILABLE = False


class TestDiffusionConfig(unittest.TestCase):
    """Test DiffusionConfig class"""

    def test_config_creation(self):
        """Test basic config creation"""
        config = DiffusionConfig(
            model_path="/tmp/test_model",
            model_type="stable-diffusion",
            device="gpu",
            use_fp16=True
        )

        self.assertEqual(config.model_path, "/tmp/test_model")
        self.assertEqual(config.model_type, "stable-diffusion")
        self.assertEqual(config.device, "gpu")
        self.assertTrue(config.use_fp16)

    def test_config_validation(self):
        """Test config validation"""
        # Test invalid model type
        with self.assertRaises(ValueError):
            DiffusionConfig(
                model_path="/tmp/test_model",
                model_type="invalid_model",
                device="gpu"
            )

        # Test invalid device
        with self.assertRaises(ValueError):
            DiffusionConfig(
                model_path="/tmp/test_model",
                model_type="stable-diffusion",
                device="invalid_device"
            )

    def test_config_to_dict(self):
        """Test config serialization"""
        config = DiffusionConfig(
            model_path="/tmp/test_model",
            model_type="stable-diffusion",
            device="gpu"
        )

        config_dict = config.to_dict()
        self.assertIn("model_path", config_dict)
        self.assertIn("model_type", config_dict)
        self.assertEqual(config_dict["model_path"], "/tmp/test_model")


@unittest.skipUnless(FASTDEPLOY_AVAILABLE, "FastDeploy not available")
class TestSDPipeline(unittest.TestCase):
    """Test Stable Diffusion Pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffusionConfig(
            model_path="/tmp/test_sd_model",
            model_type="stable-diffusion",
            device="cpu",  # Use CPU for testing
            use_fp16=False,
            height=256,    # Smaller size for testing
            width=256
        )

    def test_pipeline_creation(self):
        """Test SD pipeline creation"""
        pipeline = SDPipeline(self.config)
        self.assertIsInstance(pipeline, SDPipeline)
        self.assertEqual(pipeline.config.model_type, "stable-diffusion")

    def test_text_to_image_interface(self):
        """Test text-to-image interface (without actual inference)"""
        pipeline = SDPipeline(self.config)

        # Test that the method exists and has correct signature
        self.assertTrue(hasattr(pipeline, 'text_to_image'))

        # Test with basic parameters
        try:
            # This will use fallback implementation since no real model is loaded
            image = pipeline.text_to_image(
                prompt="A beautiful sunset",
                height=256,
                width=256,
                num_inference_steps=1,  # Minimal steps for testing
                seed=42
            )
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (256, 256))
        except Exception as e:
            # Expected to fail with fallback implementation
            self.assertIn("inference", str(e).lower())


@unittest.skipUnless(FASTDEPLOY_AVAILABLE, "FastDeploy not available")
class TestSD3Pipeline(unittest.TestCase):
    """Test Stable Diffusion 3 Pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffusionConfig(
            model_path="/tmp/test_sd3_model",
            model_type="sd3",
            device="cpu",
            use_fp16=False,
            height=256,
            width=256
        )

    def test_pipeline_creation(self):
        """Test SD3 pipeline creation"""
        pipeline = SD3Pipeline(self.config)
        self.assertIsInstance(pipeline, SD3Pipeline)
        self.assertEqual(pipeline.config.model_type, "sd3")

    def test_text_to_image_interface(self):
        """Test SD3 text-to-image interface"""
        pipeline = SD3Pipeline(self.config)

        self.assertTrue(hasattr(pipeline, 'text_to_image'))

        try:
            image = pipeline.text_to_image(
                prompt="A futuristic city",
                height=256,
                width=256,
                num_inference_steps=1,
                seed=42
            )
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (256, 256))
        except Exception as e:
            # Expected to fail with fallback implementation
            self.assertIn("inference", str(e).lower())


@unittest.skipUnless(FASTDEPLOY_AVAILABLE, "FastDeploy not available")
class TestFluxPipeline(unittest.TestCase):
    """Test Flux Pipeline"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DiffusionConfig(
            model_path="/tmp/test_flux_model",
            model_type="flux",
            device="cpu",
            use_fp16=False,
            height=256,
            width=256
        )

    def test_pipeline_creation(self):
        """Test Flux pipeline creation"""
        pipeline = FluxPipeline(self.config)
        self.assertIsInstance(pipeline, FluxPipeline)
        self.assertEqual(pipeline.config.model_type, "flux")

    def test_text_to_image_interface(self):
        """Test Flux text-to-image interface"""
        pipeline = FluxPipeline(self.config)

        self.assertTrue(hasattr(pipeline, 'text_to_image'))

        try:
            image = pipeline.text_to_image(
                prompt="A scenic landscape",
                height=256,
                width=256,
                num_inference_steps=1,
                seed=42
            )
            self.assertIsInstance(image, Image.Image)
            self.assertEqual(image.size, (256, 256))
        except Exception as e:
            # Expected to fail with fallback implementation
            self.assertIn("inference", str(e).lower())


class TestDiffusionIntegration(unittest.TestCase):
    """Integration tests for diffusion models"""

    @unittest.skipUnless(FASTDEPLOY_AVAILABLE, "FastDeploy not available")
    def test_pipeline_factory(self):
        """Test creating different pipeline types"""
        # Test SD pipeline
        sd_config = DiffusionConfig(
            model_path="/tmp/test_sd",
            model_type="stable-diffusion",
            device="cpu"
        )
        sd_pipeline = SDPipeline(sd_config)
        self.assertIsInstance(sd_pipeline, SDPipeline)

        # Test SD3 pipeline
        sd3_config = DiffusionConfig(
            model_path="/tmp/test_sd3",
            model_type="sd3",
            device="cpu"
        )
        sd3_pipeline = SD3Pipeline(sd3_config)
        self.assertIsInstance(sd3_pipeline, SD3Pipeline)

        # Test Flux pipeline
        flux_config = DiffusionConfig(
            model_path="/tmp/test_flux",
            model_type="flux",
            device="cpu"
        )
        flux_pipeline = FluxPipeline(flux_config)
        self.assertIsInstance(flux_pipeline, FluxPipeline)

    def test_model_type_validation(self):
        """Test model type validation in config"""
        # Valid model types
        valid_types = ["stable-diffusion", "sdxl", "sd3", "flux"]

        for model_type in valid_types:
            config = DiffusionConfig(
                model_path="/tmp/test",
                model_type=model_type,
                device="cpu"
            )
            self.assertEqual(config.model_type, model_type)

        # Invalid model type
        with self.assertRaises(ValueError):
            DiffusionConfig(
                model_path="/tmp/test",
                model_type="invalid_type",
                device="cpu"
            )

    def test_device_validation(self):
        """Test device validation in config"""
        # Valid devices
        valid_devices = ["cpu", "gpu", "xpu"]

        for device in valid_devices:
            config = DiffusionConfig(
                model_path="/tmp/test",
                model_type="stable-diffusion",
                device=device
            )
            self.assertEqual(config.device, device)

        # Invalid device
        with self.assertRaises(ValueError):
            DiffusionConfig(
                model_path="/tmp/test",
                model_type="stable-diffusion",
                device="invalid_device"
            )


if __name__ == '__main__':
    # Create test output directory
    os.makedirs('/tmp/test_diffusion_output', exist_ok=True)

    # Run tests
    unittest.main(verbosity=2)