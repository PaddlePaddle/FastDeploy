# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import os
import types
import unittest
from unittest import mock

from fastdeploy.model_executor.layers.utils import modules_to_convert


class MockModelConfig:
    """Mock ModelConfig for testing."""

    def __init__(
        self,
        quantization_config=None,
        prefix_name="model",
    ):
        self.quantization_config = quantization_config
        self.pretrained_config = types.SimpleNamespace(prefix_name=prefix_name)


class MockFDConfig:
    """Mock FDConfig for testing."""

    def __init__(self, model_config=None):
        self.model_config = model_config


class TestModulesToConvert(unittest.TestCase):
    """Unit tests for the modules_to_convert function."""

    def test_flashinfer_cutedsl_no_quantization_config(self):
        """Test when there is no quantization_config."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(quantization_config=None)
            fd_config = MockFDConfig(model_config=model_config)

            # Should return True when no exclude patterns
            self.assertTrue(modules_to_convert("model.layers.0", fd_config))

    def test_flashinfer_cutedsl_empty_quantization_config(self):
        """Test when quantization_config is empty."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(quantization_config={})
            fd_config = MockFDConfig(model_config=model_config)

            # Should return True when no exclude patterns
            self.assertTrue(modules_to_convert("model.layers.0", fd_config))

    def test_flashinfer_cutedsl_modules_to_not_convert(self):
        """Test with modules_to_not_convert in quantization_config."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(
                quantization_config={"modules_to_not_convert": ["model.layers.0.*", "model.embed_tokens"]}
            )
            fd_config = MockFDConfig(model_config=model_config)

            # Should return False for matching patterns
            self.assertFalse(modules_to_convert("model.layers.0.attention", fd_config))
            self.assertFalse(modules_to_convert("model.embed_tokens", fd_config))
            # Should return True for non-matching patterns
            self.assertTrue(modules_to_convert("model.layers.1.attention", fd_config))

    def test_flashinfer_cutedsl_ignore_pattern(self):
        """Test with ignore in quantization_config."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(quantization_config={"ignore": ["model.lm_head", "model.norm"]})
            fd_config = MockFDConfig(model_config=model_config)

            # Should return False for matching ignore patterns
            self.assertFalse(modules_to_convert("model.lm_head", fd_config))
            self.assertFalse(modules_to_convert("model.norm", fd_config))
            # Should return True for non-matching patterns
            self.assertTrue(modules_to_convert("model.layers.0", fd_config))

    def test_flashinfer_cutedsl_combined_patterns(self):
        """Test with both modules_to_not_convert and ignore."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(
                quantization_config={"modules_to_not_convert": ["model.layers.0.*"], "ignore": ["model.lm_head"]}
            )
            fd_config = MockFDConfig(model_config=model_config)

            # Should return False for patterns from both lists
            self.assertFalse(modules_to_convert("model.layers.0.attention", fd_config))
            self.assertFalse(modules_to_convert("model.lm_head", fd_config))
            # Should return True for non-matching patterns
            self.assertTrue(modules_to_convert("model.layers.1.attention", fd_config))

    def test_flashinfer_cutedsl_prefix_name_adaptation(self):
        """Test prefix name adaptation when pattern uses 'model' but actual prefix is different."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            # Model with prefix_name "ernie" instead of default "model"
            model_config = MockModelConfig(
                quantization_config={"modules_to_not_convert": ["model.layers.0.*"]}, prefix_name="ernie"
            )
            fd_config = MockFDConfig(model_config=model_config)

            # Pattern "model.layers.0.*" should match "ernie.layers.0.*"
            self.assertFalse(modules_to_convert("ernie.layers.0.attention", fd_config))
            # Should return True for non-matching layers
            self.assertTrue(modules_to_convert("ernie.layers.1.attention", fd_config))

    def test_flashinfer_cutedsl_prefix_name_adaptation_direct_match(self):
        """Test that direct match still works with prefix_name adaptation."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(
                quantization_config={"modules_to_not_convert": ["ernie.layers.0.*"]}, prefix_name="ernie"
            )
            fd_config = MockFDConfig(model_config=model_config)

            # Direct match should work
            self.assertFalse(modules_to_convert("ernie.layers.0.attention", fd_config))
            self.assertTrue(modules_to_convert("ernie.layers.1.attention", fd_config))

    def test_flashinfer_cutedsl_fnmatch_wildcard(self):
        """Test fnmatch wildcard patterns."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutedsl"}):
            model_config = MockModelConfig(
                quantization_config={"modules_to_not_convert": ["*.embed_*", "model.layers.*.v_proj"]}
            )
            fd_config = MockFDConfig(model_config=model_config)

            # Test wildcard patterns
            self.assertFalse(modules_to_convert("model.embed_tokens", fd_config))
            self.assertFalse(modules_to_convert("ernie.embed_positions", fd_config))
            self.assertFalse(modules_to_convert("model.layers.0.v_proj", fd_config))
            self.assertFalse(modules_to_convert("model.layers.5.v_proj", fd_config))
            # Should return True for non-matching
            self.assertTrue(modules_to_convert("model.layers.0.q_proj", fd_config))

    def test_other_backend_no_quantization_config(self):
        """Test other backends without quantization_config."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}):
            model_config = MockModelConfig(quantization_config=None)
            fd_config = MockFDConfig(model_config=model_config)

            # Should return True when no quantization_config
            self.assertTrue(modules_to_convert("model.layers.0", fd_config))

    def test_other_backend_modules_to_not_convert(self):
        """Test other backends with modules_to_not_convert."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}):
            model_config = MockModelConfig(
                quantization_config={"modules_to_not_convert": ["model.layers.0.*", "model.lm_head"]}
            )
            fd_config = MockFDConfig(model_config=model_config)

            # Should return False for matching patterns
            self.assertFalse(modules_to_convert("model.layers.0.attention", fd_config))
            self.assertFalse(modules_to_convert("model.lm_head", fd_config))
            # Should return True for non-matching patterns
            self.assertTrue(modules_to_convert("model.layers.1.attention", fd_config))

    def test_other_backend_empty_modules_to_not_convert(self):
        """Test other backends with empty modules_to_not_convert."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}):
            model_config = MockModelConfig(quantization_config={"modules_to_not_convert": []})
            fd_config = MockFDConfig(model_config=model_config)

            # Should return True when no patterns to exclude
            self.assertTrue(modules_to_convert("model.layers.0", fd_config))

    def test_other_backend_no_model_config_attribute(self):
        """Test when fd_config doesn't have proper model_config attributes."""
        with mock.patch.dict(os.environ, {"FD_MOE_BACKEND": "flashinfer-cutlass"}):
            # fd_config without model_config
            fd_config = types.SimpleNamespace()

            # Should raise AttributeError or handle gracefully
            # Based on the implementation, it will raise AttributeError
            with self.assertRaises(AttributeError):
                modules_to_convert("model.layers.0", fd_config)


if __name__ == "__main__":
    unittest.main()
