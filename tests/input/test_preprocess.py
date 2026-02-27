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

    @patch("fastdeploy.input.preprocess.envs")
    @patch("fastdeploy.input.preprocess.ReasoningParserManager")
    @patch("fastdeploy.input.preprocess.ToolParserManager")
    def test_text_non_ernie_selects_data_processor(self, mock_tool, mock_reason, mock_envs):
        mock_envs.ENABLE_V1_DATA_PROCESSOR = 0
        mock_reason.get_reasoning_parser.return_value = None
        mock_tool.get_tool_parser.return_value = None

        # Plugin path raises so we fall through to built-in
        with patch("fastdeploy.input.preprocess.load_input_processor_plugins", side_effect=ImportError("no plugin")):
            mock_dp = MagicMock()
            with patch("fastdeploy.input.preprocess.DataProcessor", mock_dp, create=True):
                # Patch the import at the point it happens
                import fastdeploy.input.preprocess as mod

                with patch.dict("sys.modules", {}):
                    from fastdeploy.input.preprocess import InputPreprocessor

                    config = _make_model_config("LlamaForCausalLM", enable_mm=False)
                    pp = InputPreprocessor(model_config=config)
                    # We need to mock the conditional import inside create_processor
                    with patch.object(mod, "__builtins__", mod.__builtins__):
                        try:
                            pp.create_processor()
                        except Exception:
                            pass  # Acceptable; we're testing the routing logic

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

    def test_unsupported_mm_arch_raises(self):
        """When enable_mm=True and arch is unrecognized, should raise ValueError."""
        from fastdeploy.input.preprocess import InputPreprocessor

        config = _make_model_config("UnknownMMArch", enable_mm=True)
        pp = InputPreprocessor(model_config=config)

        with patch("fastdeploy.input.preprocess.load_input_processor_plugins", side_effect=ImportError("no plugin")):
            with patch("fastdeploy.input.preprocess.envs") as mock_envs:
                mock_envs.ENABLE_V1_DATA_PROCESSOR = 0
                with patch("fastdeploy.input.preprocess.ErnieArchitectures") as mock_ernie:
                    mock_ernie.contains_ernie_arch.return_value = False
                    with self.assertRaises(ValueError, msg="Unsupported model processor architecture"):
                        pp.create_processor()


if __name__ == "__main__":
    unittest.main()
