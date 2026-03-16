# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
Tests for InputPreprocessor.create_processor().

Why mock:
  - ModelConfig, ReasoningParserManager, ToolParserManager, and concrete processor
    classes all depend on model files or external resources not available in tests.
    We mock them at the import boundary to test InputPreprocessor's routing logic.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def _make_model_config(arch, enable_mm=False):
    cfg = SimpleNamespace(
        model="test_model",
        architectures=[arch],
        enable_mm=enable_mm,
    )
    return cfg


class TestInputPreprocessorBranching(unittest.TestCase):
    """Test that create_processor picks the right processor class based on architecture and flags."""

    def test_init_stores_params(self):
        from fastdeploy.input.preprocess import InputPreprocessor

        config = _make_model_config("LlamaForCausalLM")
        pp = InputPreprocessor(
            model_config=config,
            reasoning_parser="qwen3",
            tool_parser="ernie_x1",
            limit_mm_per_prompt={"image": 2},
        )
        self.assertEqual(pp.model_name_or_path, "test_model")
        self.assertEqual(pp.reasoning_parser, "qwen3")
        self.assertEqual(pp.tool_parser, "ernie_x1")
        self.assertEqual(pp.limit_mm_per_prompt, {"image": 2})

    def test_create_processor_text_normal_path(self):
        """Normal path: non-Ernie, non-MM arch creates a text DataProcessor."""
        from fastdeploy.input.preprocess import InputPreprocessor

        config = _make_model_config("LlamaForCausalLM", enable_mm=False)
        pp = InputPreprocessor(model_config=config)

        mock_dp = MagicMock()
        with (
            patch.dict("sys.modules", {"fastdeploy.plugins": None, "fastdeploy.plugins.input_processor": None}),
            patch("fastdeploy.input.preprocess.envs") as mock_envs,
            patch("fastdeploy.input.text_processor.DataProcessor", return_value=mock_dp),
        ):
            mock_envs.ENABLE_V1_DATA_PROCESSOR = False
            pp.create_processor()

        self.assertIs(pp.processor, mock_dp)

    def test_unsupported_mm_arch_raises(self):
        """When enable_mm=True and arch is unrecognized, should raise ValueError."""
        from fastdeploy.input.preprocess import InputPreprocessor

        config = _make_model_config("UnknownMMArch", enable_mm=True)
        pp = InputPreprocessor(model_config=config)

        with patch.dict("sys.modules", {"fastdeploy.plugins": None, "fastdeploy.plugins.input_processor": None}):
            with self.assertRaises(ValueError):
                pp.create_processor()


if __name__ == "__main__":
    unittest.main()
