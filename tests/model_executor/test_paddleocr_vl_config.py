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

from fastdeploy.model_executor.models.paddleocr_vl.config import (
    PaddleOCRConfig,
    PaddleOCRVisionConfig,
)


class TestPaddleOCRVisionConfig(unittest.TestCase):
    """Test PaddleOCRVisionConfig class."""

    def test_model_type(self):
        """model_type is 'paddleocr_vl'."""
        self.assertEqual(PaddleOCRVisionConfig.model_type, "paddleocr_vl")

    def test_init_defaults(self):
        """__init__ with defaults sets all attributes correctly."""
        config = PaddleOCRVisionConfig()
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 3072)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.num_channels, 3)
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.patch_size, 14)
        self.assertEqual(config.hidden_act, "gelu_pytorch_tanh")
        self.assertAlmostEqual(config.layer_norm_eps, 1e-6)
        self.assertEqual(config.attention_dropout, 0.0)
        self.assertEqual(config.spatial_merge_size, 2)
        self.assertEqual(config.temporal_patch_size, 2)
        self.assertEqual(config.tokens_per_second, 2)

    def test_init_custom(self):
        """__init__ with custom values stores them correctly."""
        config = PaddleOCRVisionConfig(
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_channels=4,
            image_size=448,
            patch_size=16,
            hidden_act="silu",
            layer_norm_eps=1e-5,
            attention_dropout=0.1,
            spatial_merge_size=4,
            temporal_patch_size=4,
            tokens_per_second=4,
        )
        self.assertEqual(config.hidden_size, 1024)
        self.assertEqual(config.intermediate_size, 4096)
        self.assertEqual(config.num_hidden_layers, 24)
        self.assertEqual(config.num_attention_heads, 16)
        self.assertEqual(config.num_channels, 4)
        self.assertEqual(config.image_size, 448)
        self.assertEqual(config.patch_size, 16)
        self.assertEqual(config.hidden_act, "silu")
        self.assertAlmostEqual(config.layer_norm_eps, 1e-5)
        self.assertEqual(config.attention_dropout, 0.1)
        self.assertEqual(config.spatial_merge_size, 4)
        self.assertEqual(config.temporal_patch_size, 4)
        self.assertEqual(config.tokens_per_second, 4)


class TestPaddleOCRConfig(unittest.TestCase):
    """Test PaddleOCRConfig class."""

    def test_model_type(self):
        """model_type is 'paddleocr_vl'."""
        self.assertEqual(PaddleOCRConfig.model_type, "paddleocr_vl")

    def test_init_defaults(self):
        """__init__ with defaults sets all attributes correctly."""
        config = PaddleOCRConfig()
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.intermediate_size, 11008)
        self.assertEqual(config.max_position_embeddings, 32768)
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertEqual(config.num_attention_heads, 2)
        self.assertEqual(config.image_token_id, 101304)
        self.assertEqual(config.video_token_id, 101305)
        self.assertEqual(config.vision_start_token_id, 101306)
        self.assertAlmostEqual(config.rms_norm_eps, 1e-6)
        self.assertFalse(config.use_cache)
        self.assertFalse(config.use_flash_attention)
        self.assertEqual(config.head_dim, 128)
        self.assertEqual(config.hidden_act, "silu")
        self.assertFalse(config.use_bias)
        self.assertEqual(config.rope_theta, 10000)
        self.assertTrue(config.weight_share_add_bias)
        self.assertEqual(config.ignored_index, -100)
        self.assertEqual(config.attention_probs_dropout_prob, 0.0)
        self.assertEqual(config.hidden_dropout_prob, 0.0)
        self.assertEqual(config.compression_ratio, 1.0)
        self.assertIsNone(config.num_key_value_heads)
        self.assertIsNone(config.max_sequence_length)
        # Hard-coded attributes
        self.assertTrue(config.fuse_rms_norm)
        self.assertTrue(config.use_sparse_flash_attn)
        self.assertFalse(config.use_var_len_flash_attn)
        self.assertEqual(config.scale_qk_coeff, 1.0)
        self.assertFalse(config.fuse_softmax_mask)
        self.assertFalse(config.use_sparse_head_and_loss_fn)
        self.assertFalse(config.use_recompute_loss_fn)
        self.assertFalse(config.use_fused_head_and_loss_fn)
        self.assertFalse(config.fuse_linear)
        self.assertFalse(config.token_balance_seqlen)
        self.assertTrue(config.use_rmsnorm)
        self.assertFalse(config.fuse_ln)
        self.assertFalse(config.cachekv_quant)
        self.assertFalse(config.fuse_swiglu)

    def test_init_with_vision_config_dict(self):
        """__init__ creates PaddleOCRVisionConfig from dict."""
        config = PaddleOCRConfig(vision_config={"hidden_size": 1024, "num_hidden_layers": 24})
        self.assertIsInstance(config.vision_config, PaddleOCRVisionConfig)
        self.assertEqual(config.vision_config.hidden_size, 1024)
        self.assertEqual(config.vision_config.num_hidden_layers, 24)

    def test_init_with_vision_config_none(self):
        """__init__ creates default PaddleOCRVisionConfig when None."""
        config = PaddleOCRConfig(vision_config=None)
        self.assertIsInstance(config.vision_config, PaddleOCRVisionConfig)
        self.assertEqual(config.vision_config.hidden_size, 768)

    def test_init_hidden_act_not_silu_raises(self):
        """__init__ raises NotImplementedError for non-silu hidden_act."""
        with self.assertRaises(NotImplementedError):
            PaddleOCRConfig(hidden_act="gelu")

    def test_sub_configs(self):
        """sub_configs maps vision_config to PaddleOCRVisionConfig."""
        self.assertIn("vision_config", PaddleOCRConfig.sub_configs)
        self.assertIs(PaddleOCRConfig.sub_configs["vision_config"], PaddleOCRVisionConfig)

    def test_base_model_tp_plan(self):
        """base_model_tp_plan contains expected keys."""
        plan = PaddleOCRConfig.base_model_tp_plan
        self.assertIn("layers.*.self_attn.q_proj", plan)
        self.assertEqual(plan["layers.*.self_attn.q_proj"], "colwise")
        self.assertEqual(plan["layers.*.mlp.down_proj"], "rowwise")


if __name__ == "__main__":
    unittest.main()
