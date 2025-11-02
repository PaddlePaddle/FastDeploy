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
Real inference tests for Flux model with actual model weights.
These tests require actual model files to be present.
"""

import unittest
import sys
import os
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import paddle
    import numpy as np
    from PIL import Image
    from fastdeploy.model_executor.diffusion_models.vision.diffusion import (
        DiffusionConfig,
        FluxPipeline,
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    print(f"Dependencies not available: {e}")
    # Create mock objects
    paddle = None
    np = None
    Image = None
    DiffusionConfig = None
    FluxPipeline = None


def check_model_exists(model_path):
    """Check if model files exist."""
    if not os.path.exists(model_path):
        return False
    
    # Check for expected subdirectories
    required_dirs = ['transformer', 'text_encoder', 'vae']
    for dir_name in required_dirs:
        dir_path = os.path.join(model_path, dir_name)
        if not os.path.exists(dir_path):
            return False
    
    return True


# Environment variable to specify model path
FLUX_MODEL_PATH = os.environ.get('FLUX_MODEL_PATH', None)
MODEL_AVAILABLE = FLUX_MODEL_PATH is not None and check_model_exists(FLUX_MODEL_PATH)


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Dependencies not available")
@unittest.skipIf(not MODEL_AVAILABLE, "Flux model not available - set FLUX_MODEL_PATH environment variable")
class TestFluxRealInference(unittest.TestCase):
    """Real inference tests with actual Flux model weights."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with real model."""
        cls.config = DiffusionConfig(
            model_path=FLUX_MODEL_PATH,
            model_type="flux",
            device="gpu",  # Use GPU for real inference
            use_fp16=True,
            max_batch_size=1,
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
        )
        
        cls.pipeline = FluxPipeline(cls.config)
        cls.output_dir = os.path.join(os.path.dirname(__file__), 'flux_test_outputs')
        os.makedirs(cls.output_dir, exist_ok=True)

    def test_real_text_to_image(self):
        """Test real text-to-image generation with Flux."""
        prompt = "a beautiful sunset over mountains, highly detailed, 4k"
        
        image = self.pipeline.text_to_image(
            prompt=prompt,
            negative_prompt="blurry, low quality",
            height=512,
            width=512,
            num_inference_steps=20,
            guidance_scale=3.5,
            seed=42
        )
        
        # Validate output
        self.assertIsInstance(image, Image.Image)
        self.assertEqual(image.size, (512, 512))
        
        # Save for manual inspection
        output_path = os.path.join(self.output_dir, 'test_sunset.png')
        image.save(output_path)
        
        print(f"✅ Real inference test passed - image saved to {output_path}")

    def test_real_model_quality_metrics(self):
        """Test generated image quality metrics."""
        prompt = "a photorealistic portrait of a cat"
        
        image = self.pipeline.text_to_image(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=20,
            seed=42
        )
        
        # Convert to array for analysis
        img_array = np.array(image)
        
        # Check image statistics
        mean = np.mean(img_array)
        std = np.std(img_array)
        
        # Image should have reasonable statistics
        self.assertGreater(mean, 10)  # Not too dark
        self.assertLess(mean, 245)     # Not too bright
        self.assertGreater(std, 20)   # Has variance
        
        # Save for inspection
        output_path = os.path.join(self.output_dir, 'test_cat.png')
        image.save(output_path)
        
        # Save statistics
        stats = {
            'mean': float(mean),
            'std': float(std),
            'min': int(img_array.min()),
            'max': int(img_array.max())
        }
        
        stats_path = os.path.join(self.output_dir, 'test_cat_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Quality metrics test passed - stats: {stats}")

    def test_real_batch_consistency(self):
        """Test consistency across multiple generations."""
        prompt = "a futuristic city at night"
        
        # Generate 3 images with same seed
        images = []
        for i in range(3):
            image = self.pipeline.text_to_image(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                seed=42  # Same seed
            )
            images.append(image)
            
            # Save each
            output_path = os.path.join(self.output_dir, f'test_consistency_{i}.png')
            image.save(output_path)
        
        # All should be identical with same seed
        arrays = [np.array(img) for img in images]
        
        # Check first two are identical
        np.testing.assert_array_equal(arrays[0], arrays[1])
        np.testing.assert_array_equal(arrays[1], arrays[2])
        
        print("✅ Batch consistency test passed")

    def test_real_different_resolutions(self):
        """Test generation at different resolutions."""
        prompt = "a beautiful landscape"
        
        resolutions = [
            (256, 256),
            (512, 512),
            (768, 768),
        ]
        
        for height, width in resolutions:
            image = self.pipeline.text_to_image(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=15,
                seed=42
            )
            
            self.assertEqual(image.size, (width, height))
            
            # Save
            output_path = os.path.join(self.output_dir, f'test_res_{height}x{width}.png')
            image.save(output_path)
            
            print(f"✅ Generated image at {height}x{width}")

    def test_real_guidance_scale_impact(self):
        """Test impact of different guidance scales."""
        prompt = "an artistic painting of flowers"
        
        guidance_scales = [1.0, 3.5, 7.0]
        
        for guidance_scale in guidance_scales:
            image = self.pipeline.text_to_image(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                guidance_scale=guidance_scale,
                seed=42
            )
            
            # Save with guidance scale in filename
            output_path = os.path.join(
                self.output_dir,
                f'test_guidance_{guidance_scale}.png'
            )
            image.save(output_path)
            
            print(f"✅ Generated with guidance scale {guidance_scale}")

    def test_real_inference_steps_impact(self):
        """Test impact of different inference steps."""
        prompt = "a serene lake at dawn"
        
        steps = [10, 20, 30]
        
        for num_steps in steps:
            image = self.pipeline.text_to_image(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=num_steps,
                seed=42
            )
            
            # Save
            output_path = os.path.join(
                self.output_dir,
                f'test_steps_{num_steps}.png'
            )
            image.save(output_path)
            
            print(f"✅ Generated with {num_steps} inference steps")


@unittest.skipIf(not DEPENDENCIES_AVAILABLE, "Dependencies not available")
@unittest.skipIf(not MODEL_AVAILABLE, "Flux model not available")
class TestFluxPerformanceBenchmark(unittest.TestCase):
    """Performance benchmarking with real model."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.config = DiffusionConfig(
            model_path=FLUX_MODEL_PATH,
            model_type="flux",
            device="gpu",
            use_fp16=True,
            height=512,
            width=512,
            num_inference_steps=20,
        )
        
        cls.pipeline = FluxPipeline(cls.config)

    def test_benchmark_throughput(self):
        """Benchmark throughput with real model."""
        import time
        
        prompts = [
            "a mountain landscape",
            "a city street at night",
            "a portrait of a person",
        ]
        
        total_start = time.time()
        
        for i, prompt in enumerate(prompts):
            start = time.time()
            
            image = self.pipeline.text_to_image(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=20,
                seed=i
            )
            
            elapsed = time.time() - start
            print(f"Image {i+1}: {elapsed:.2f}s")
        
        total_elapsed = time.time() - total_start
        avg_time = total_elapsed / len(prompts)
        
        print(f"✅ Throughput test - Average: {avg_time:.2f}s per image")
        
        # Performance assertion (adjust based on hardware)
        # This is a sanity check, not a hard requirement
        self.assertLess(avg_time, 60.0)  # Should complete within 60s


if __name__ == '__main__':
    # Run tests with verbosity
    if not MODEL_AVAILABLE:
        print("\n" + "="*70)
        print("NOTE: Real inference tests require actual Flux model weights.")
        print("Set the FLUX_MODEL_PATH environment variable to enable these tests.")
        print("Example:")
        print("  export FLUX_MODEL_PATH=/path/to/flux/model")
        print("  python test_flux_real_inference.py")
        print("="*70 + "\n")
    
    unittest.main(verbosity=2)
