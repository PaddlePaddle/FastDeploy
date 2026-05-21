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
from unittest.mock import MagicMock, patch

from fastdeploy.model_executor.layers.quantization.wint2 import WINT2Config


class TestWINT2ConfigInit(unittest.TestCase):
    """Test WINT2Config.__init__."""

    def test_init_sets_all_attributes(self):
        """__init__ sets all config attributes correctly."""
        config = WINT2Config(
            dense_quant_type="wint8",
            dense_quant_granularity="per_channel",
            moe_quant_type="w4w2",
            moe_w4_quant_type="wint4",
            moe_w4_quant_granularity="per_channel",
            moe_w4_quant_start_layer=0,
            moe_w4_quant_end_layer=6,
            moe_w2_quant_type="wint2",
            moe_w2_quant_granularity="pp_acc",
            moe_w2_quant_group_size=128,
            moe_w2_quant_start_layer=7,
            moe_w2_quant_end_layer=60,
        )
        self.assertEqual(config.quant_max_bound, 0)
        self.assertEqual(config.quant_min_bound, 0)
        self.assertEqual(config.quant_round_type, 0)
        self.assertEqual(config.dense_quant_type, "wint8")
        self.assertEqual(config.dense_quant_granularity, "per_channel")
        self.assertEqual(config.moe_quant_type, "w4w2")
        self.assertEqual(config.moe_w4_quant_type, "wint4")
        self.assertEqual(config.moe_w4_quant_granularity, "per_channel")
        self.assertEqual(config.moe_w4_quant_start_layer, 0)
        self.assertEqual(config.moe_w4_quant_end_layer, 6)
        self.assertEqual(config.moe_w2_quant_type, "wint2")
        self.assertEqual(config.moe_w2_quant_granularity, "pp_acc")
        self.assertEqual(config.moe_w2_quant_group_size, 128)
        self.assertEqual(config.moe_w2_quant_start_layer, 7)
        self.assertEqual(config.moe_w2_quant_end_layer, 60)


class TestWINT2ConfigName(unittest.TestCase):
    """Test WINT2Config.name."""

    def test_name_returns_wint2(self):
        """name() returns 'wint2'."""
        config = WINT2Config("a", "b", "c", "d", "e", 0, 1, "f", "g", 0, 0, 0)
        self.assertEqual(config.name(), "wint2")


class TestWINT2ConfigFromConfig(unittest.TestCase):
    """Test WINT2Config.from_config."""

    def test_from_config_defaults(self):
        """from_config uses defaults when config is empty."""
        config = WINT2Config.from_config({})
        self.assertEqual(config.dense_quant_type, "wint8")
        self.assertEqual(config.dense_quant_granularity, "per_channel")
        self.assertEqual(config.moe_quant_type, "w4w2")
        self.assertEqual(config.moe_w4_quant_type, "wint4")
        self.assertEqual(config.moe_w4_quant_granularity, "per_channel")
        self.assertEqual(config.moe_w4_quant_start_layer, 0)
        self.assertEqual(config.moe_w4_quant_end_layer, 6)
        self.assertEqual(config.moe_w2_quant_type, "wint2")
        self.assertEqual(config.moe_w2_quant_granularity, "pp_acc")
        self.assertEqual(config.moe_w2_quant_group_size, 0)
        self.assertEqual(config.moe_w2_quant_start_layer, 0)
        self.assertEqual(config.moe_w2_quant_end_layer, 0)

    def test_from_config_custom_values(self):
        """from_config extracts values from nested config dict."""
        cfg = {
            "dense_quant_type": "wint4",
            "dense_quant_granularity": "per_group",
            "moe_quant_config": {
                "quant_type": "w2w4",
                "moe_w4_quant_config": {
                    "quant_type": "wint4_gptq",
                    "quant_granularity": "per_group",
                    "quant_start_layer": 2,
                    "quant_end_layer": 10,
                },
                "moe_w2_quant_config": {
                    "quant_type": "wint2_gptq",
                    "quant_granularity": "per_group",
                    "quant_group_size": 64,
                    "quant_start_layer": 11,
                    "quant_end_layer": 50,
                },
            },
        }
        config = WINT2Config.from_config(cfg)
        self.assertEqual(config.dense_quant_type, "wint4")
        self.assertEqual(config.dense_quant_granularity, "per_group")
        self.assertEqual(config.moe_quant_type, "w2w4")
        self.assertEqual(config.moe_w4_quant_type, "wint4_gptq")
        self.assertEqual(config.moe_w4_quant_granularity, "per_group")
        self.assertEqual(config.moe_w4_quant_start_layer, 2)
        self.assertEqual(config.moe_w4_quant_end_layer, 10)
        self.assertEqual(config.moe_w2_quant_type, "wint2_gptq")
        self.assertEqual(config.moe_w2_quant_granularity, "per_group")
        self.assertEqual(config.moe_w2_quant_group_size, 64)
        self.assertEqual(config.moe_w2_quant_start_layer, 11)
        self.assertEqual(config.moe_w2_quant_end_layer, 50)


class TestWINT2ConfigGetQuantMethod(unittest.TestCase):
    """Test WINT2Config.get_quant_method."""

    @patch("fastdeploy.model_executor.layers.quantization.wint2.get_quantization_config")
    def test_get_quant_method_non_moe(self, mock_get_quant_config):
        """get_quant_method delegates to dense config for non-FusedMoE layers."""
        mock_dense_config = MagicMock()
        mock_dense_method = MagicMock()
        mock_dense_config.from_config.return_value.get_quant_method.return_value = mock_dense_method
        mock_get_quant_config.return_value = mock_dense_config

        config = WINT2Config.from_config({})
        layer = MagicMock()  # not a FusedMoE instance

        result = config.get_quant_method(layer)

        mock_get_quant_config.assert_called_once_with("wint8")
        self.assertIs(result, mock_dense_method)

    @patch("fastdeploy.model_executor.layers.quantization.wint2.get_quantization_config")
    def test_get_quant_method_moe_w4_layer(self, mock_get_quant_config):
        """get_quant_method delegates to w4 config for FusedMoE within w4 range."""
        from fastdeploy.model_executor.layers.moe import FusedMoE

        mock_w4_config = MagicMock()
        mock_w4_method = MagicMock()
        mock_w4_config.from_config.return_value.get_quant_method.return_value = mock_w4_method
        mock_get_quant_config.return_value = mock_w4_config

        config = WINT2Config.from_config({})  # moe_w4_quant_end_layer=6

        layer = MagicMock(spec=FusedMoE)
        layer.layer_idx = 3  # within w4 range (<=6)

        result = config.get_quant_method(layer)

        mock_get_quant_config.assert_called_once_with("wint4")
        self.assertIs(result, mock_w4_method)

    @patch(
        "fastdeploy.model_executor.layers.moe.fused_moe_wint2_backend.CutlassWint2FusedMoeMethod",
        create=True,
    )
    def test_get_quant_method_moe_w2_layer(self, mock_wint2_method_cls):
        """get_quant_method returns CutlassWint2FusedMoeMethod for layers beyond w4 range."""
        from fastdeploy.model_executor.layers.moe import FusedMoE

        mock_wint2_method_cls.return_value = "wint2_method_instance"

        config = WINT2Config.from_config({})  # moe_w4_quant_end_layer=6

        layer = MagicMock(spec=FusedMoE)
        layer.layer_idx = 10  # beyond w4 range (>6)

        result = config.get_quant_method(layer)

        mock_wint2_method_cls.assert_called_once_with(config)
        self.assertEqual(result, "wint2_method_instance")


if __name__ == "__main__":
    unittest.main()
