"""
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
"""

import unittest

from fastdeploy.model_executor.models.qwen3_vl.dfnrope.configuration import (
    Qwen3VisionTransformerConfig,
)


class TestQwen3VisionTransformerConfig(unittest.TestCase):
    """Test Qwen3VisionTransformerConfig class."""

    def test_model_type(self):
        """model_type is 'qwen3_vision_transformer'."""
        self.assertEqual(Qwen3VisionTransformerConfig.model_type, "qwen3_vision_transformer")

    def test_init_defaults(self):
        """__init__ with defaults sets all attributes correctly."""
        config = Qwen3VisionTransformerConfig()
        self.assertEqual(config.depth, 27)
        self.assertEqual(config.hidden_size, 1152)
        self.assertEqual(config.hidden_act, "gelu_tanh")
        self.assertEqual(config.intermediate_size, 4304)
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.in_channels, 3)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.spatial_merge_size, 2)
        self.assertEqual(config.temporal_patch_size, 2)
        self.assertEqual(config.out_hidden_size, 3584)
        self.assertEqual(config.num_position_embeddings, 2304)
        self.assertEqual(config.initializer_range, 0.02)
        self.assertEqual(config.deepstack_visual_indexes, [])
        self.assertEqual(config.tokens_per_second, 2)

    def test_init_custom(self):
        """__init__ with custom values stores them correctly."""
        config = Qwen3VisionTransformerConfig(
            depth=64,
            hidden_size=2560,
            hidden_act="silu",
            intermediate_size=8608,
            num_heads=32,
            in_channels=4,
            patch_size=14,
            spatial_merge_size=4,
            temporal_patch_size=4,
            out_hidden_size=7168,
            num_position_embeddings=4608,
            deepstack_visual_indexes=[3, 7, 11, 15],
            initializer_range=0.01,
            tokens_per_second=4,
        )
        self.assertEqual(config.depth, 64)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.intermediate_size, 8608)
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.in_channels, 4)
        self.assertEqual(config.patch_size, 14)
        self.assertEqual(config.spatial_merge_size, 4)
        self.assertEqual(config.temporal_patch_size, 4)
        self.assertEqual(config.out_hidden_size, 7168)
        self.assertEqual(config.num_position_embeddings, 4608)
        self.assertEqual(config.initializer_range, 0.01)
        self.assertEqual(config.deepstack_visual_indexes, [3, 7, 11, 15])
        self.assertEqual(config.tokens_per_second, 4)

    def test_deepstack_visual_indexes_none_becomes_empty_list(self):
        """deepstack_visual_indexes=None becomes empty list."""
        config = Qwen3VisionTransformerConfig(deepstack_visual_indexes=None)
        self.assertEqual(config.deepstack_visual_indexes, [])

    def test_inherits_pretrained_config(self):
        """Qwen3VisionTransformerConfig inherits from PretrainedConfig."""
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = Qwen3VisionTransformerConfig()
        self.assertIsInstance(config, PretrainedConfig)


if __name__ == "__main__":
    unittest.main()
