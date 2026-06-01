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

from fastdeploy.model_executor.models.qwen2_5_vl.dfnrope.configuration import (
    DFNRopeVisionTransformerConfig,
)


class TestDFNRopeVisionTransformerConfig(unittest.TestCase):
    """Test DFNRopeVisionTransformerConfig class."""

    def test_model_type(self):
        """model_type is 'DFNRope_vision_transformer'."""
        self.assertEqual(DFNRopeVisionTransformerConfig.model_type, "DFNRope_vision_transformer")

    def test_init_defaults(self):
        """__init__ with defaults sets all attributes correctly."""
        config = DFNRopeVisionTransformerConfig()
        self.assertEqual(config.depth, 32)
        self.assertEqual(config.hidden_size, 1280)
        self.assertEqual(config.out_hidden_size, 3584)
        self.assertEqual(config.intermediate_size, 3420)
        self.assertEqual(config.hidden_act, "silu")
        self.assertEqual(config.num_heads, 16)
        self.assertEqual(config.in_channels, 3)
        self.assertEqual(config.patch_size, 14)
        self.assertEqual(config.spatial_merge_size, 2)
        self.assertEqual(config.window_size, 112)
        self.assertEqual(config.fullatt_block_indexes, [7, 15, 23, 31])
        self.assertEqual(config.temporal_patch_size, 2)

    def test_init_custom(self):
        """__init__ with custom values stores them correctly."""
        config = DFNRopeVisionTransformerConfig(
            depth=64,
            hidden_size=2560,
            out_hidden_size=7168,
            intermediate_size=6840,
            hidden_act="gelu",
            num_heads=32,
            in_channels=4,
            patch_size=16,
            spatial_merge_size=4,
            window_size=224,
            fullatt_block_indexes=[15, 31, 47, 63],
            temporal_patch_size=4,
        )
        self.assertEqual(config.depth, 64)
        self.assertEqual(config.hidden_size, 2560)
        self.assertEqual(config.out_hidden_size, 7168)
        self.assertEqual(config.intermediate_size, 6840)
        self.assertEqual(config.hidden_act, "gelu")
        self.assertEqual(config.num_heads, 32)
        self.assertEqual(config.in_channels, 4)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.spatial_merge_size, 4)
        self.assertEqual(config.window_size, 224)
        self.assertEqual(config.fullatt_block_indexes, [15, 31, 47, 63])
        self.assertEqual(config.temporal_patch_size, 4)

    def test_inherits_pretrained_config(self):
        """DFNRopeVisionTransformerConfig inherits from PretrainedConfig."""
        from paddleformers.transformers.configuration_utils import PretrainedConfig

        config = DFNRopeVisionTransformerConfig()
        self.assertIsInstance(config, PretrainedConfig)


if __name__ == "__main__":
    unittest.main()
