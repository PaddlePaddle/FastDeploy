"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

import json
import unittest
from unittest.mock import Mock, mock_open, patch

from paddleformers.transformers.configuration_utils import PretrainedConfig

from fastdeploy.config import ModelConfig


class TestConfig(unittest.TestCase):
    def test_model_config_init(self):
        # Mock external dependencies
        with (
            patch(
                "paddleformers.transformers.configuration_utils.PretrainedConfig.get_config_dict"
            ) as mock_get_config_dict,
            patch("paddleformers.transformers.configuration_utils.PretrainedConfig.from_dict") as mock_from_dict,
        ):

            # Setup mock return values
            mock_pretrained_config = {
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "vocab_size": 30000,
                "text_config": {"some_text_key": "some_text_value"},
                "rope_scaling": {"mrope_section": [0.5]},
                "num_key_value_heads": 8,
            }
            mock_get_config_dict.return_value = (mock_pretrained_config, "mock_path")
            mock_from_dict.return_value = Mock(spec=PretrainedConfig)
            mock_from_dict.return_value.to_dict.return_value = mock_pretrained_config

            # Test args
            test_args = {
                "model": "mock_model",
                "ori_vocab_size": 32000,
                "think_end_id": 1,
                "image_patch_id": 2,
                "max_logprobs": 25000,
                "runner": "generate",
                "convert": "none",
            }

            # Initialize ModelConfig
            model_config = ModelConfig(test_args)

            # Assertions
            mock_get_config_dict.assert_called_once_with(test_args["model"])
            mock_from_dict.assert_called_once_with(mock_pretrained_config)

            # Test attributes from args
            self.assertEqual(model_config.model, test_args["model"])
            self.assertEqual(model_config.ori_vocab_size, test_args["ori_vocab_size"])
            self.assertEqual(model_config.think_end_id, test_args["think_end_id"])
            self.assertEqual(model_config.im_patch_id, test_args["image_patch_id"])

            # Test attributes from pretrained_config
            self.assertEqual(model_config.hidden_size, mock_pretrained_config["hidden_size"])
            self.assertEqual(model_config.num_attention_heads, mock_pretrained_config["num_attention_heads"])
            self.assertEqual(model_config.some_text_key, "some_text_value")  # from text_config

            # Test default values from PRETRAINED_INIT_CONFIGURATION
            self.assertEqual(model_config.top_p, 1.0)
            self.assertEqual(model_config.min_length, 1)

            # Test calculated head_dim
            self.assertEqual(
                model_config.head_dim,
                mock_pretrained_config["hidden_size"] // mock_pretrained_config["num_attention_heads"],
            )

            # Test rope_3d and freq_allocation
            self.assertTrue(model_config.rope_3d)
            self.assertEqual(model_config.freq_allocation, mock_pretrained_config["rope_scaling"]["mrope_section"][0])

            # Test max_logprobs validation
            self.assertEqual(model_config.max_logprobs, test_args["max_logprobs"])
            test_args_invalid_logprobs = test_args.copy()
            test_args_invalid_logprobs["max_logprobs"] = 35000  # Greater than ori_vocab_size
            with self.assertRaises(ValueError):
                ModelConfig(test_args_invalid_logprobs)

            test_args_negative_logprobs = test_args.copy()
            test_args_negative_logprobs["max_logprobs"] = -2  # Less than -1
            with self.assertRaises(ValueError):
                ModelConfig(test_args_negative_logprobs)

            # Test _post_init calls
            # For now, just check if it runs without error.
            # More specific tests for _get_runner_type and _get_convert_type are below.
            self.assertTrue(hasattr(model_config, "is_unified_ckpt"))
            self.assertTrue(hasattr(model_config, "runner_type"))
            self.assertTrue(hasattr(model_config, "convert_type"))

    def test_model_config_runner_type_auto(self):
        # Mock ModelRegistry and get_pooling_config
        with (
            patch("fastdeploy.model_executor.models.model_base.ModelRegistry") as mock_model_registry_cls,
            patch("fastdeploy.transformer_utils.config.get_pooling_config") as mock_get_pooling_config,
        ):

            mock_registry = mock_model_registry_cls.return_value
            mock_registry.get_supported_archs.return_value = ["TestCausalLM"]
            mock_registry.is_pooling_model.return_value = False
            mock_registry.is_text_generation_model.return_value = True
            mock_get_pooling_config.return_value = None

            # Test auto resolution for generate model
            test_args = {"model": "test_model", "runner": "auto", "architectures": ["TestCausalLM"]}
            model_config = ModelConfig(test_args)
            self.assertEqual(model_config.runner_type, "generate")

            # Test auto resolution for pooling model
            mock_registry.is_pooling_model.return_value = True
            mock_registry.is_text_generation_model.return_value = False
            mock_get_pooling_config.return_value = {"pooling_type": "mean"}
            model_config = ModelConfig(test_args)
            self.assertEqual(model_config.runner_type, "pooling")

            # Test architecture suffix matching
            mock_registry.is_pooling_model.return_value = False
            mock_registry.is_text_generation_model.return_value = False
            mock_get_pooling_config.return_value = None
            test_args["architectures"] = ["SomeModelForCausalLM"]
            model_config = ModelConfig(test_args)
            self.assertEqual(model_config.runner_type, "generate")

            test_args["architectures"] = ["SomeEmbeddingModel"]
            model_config = ModelConfig(test_args)
            self.assertEqual(model_config.runner_type, "pooling")

    def test_model_config_convert_type_auto(self):
        with patch("fastdeploy.model_executor.models.model_base.ModelRegistry") as mock_model_registry_cls:
            mock_registry = mock_model_registry_cls.return_value
            mock_registry.get_supported_archs.return_value = ["TestCausalLM"]
            mock_registry.is_text_generation_model.return_value = True
            mock_registry.is_pooling_model.return_value = False

            # Test auto resolution when runner_type is generate
            test_args = {"model": "test_model", "convert": "auto", "architectures": ["TestCausalLM"]}
            model_config = ModelConfig(test_args)
            model_config.runner_type = "generate"
            model_config.architectures = ["TestCausalLM"]  # Set directly for this test
            self.assertEqual(model_config.convert_type, "none")

            # Test auto resolution when runner_type is pooling
            mock_registry.is_pooling_model.return_value = True
            mock_registry.is_text_generation_model.return_value = False
            model_config.runner_type = "pooling"
            model_config.architectures = ["TestPoolingModel"]  # Set directly for this test
            self.assertEqual(model_config.convert_type, "none")

            # Test architecture suffix matching for convert type
            model_config.architectures = ["SomeForTextEncoding"]
            model_config.runner_type = "pooling"
            model_config.convert = "auto"
            self.assertEqual(model_config.convert_type, "embed")

            # Test default for pooling if no specific match
            model_config.architectures = ["SomeOtherModel"]
            model_config.runner_type = "pooling"
            model_config.convert = "auto"
            self.assertEqual(model_config.convert_type, "embed")

    def test_model_config_override_name_from_config(self):
        # Test unified_ckpt and infer_model_mp_num
        model_config = ModelConfig({"model": "test_model"})
        model_config.is_unified_ckpt = False
        model_config.infer_model_mp_num = 4
        model_config.override_name_from_config()
        self.assertEqual(model_config.tensor_parallel_size, 4)
        self.assertFalse(hasattr(model_config, "infer_model_mp_num"))

        # Test num_hidden_layers and remove_tail_layer
        model_config = ModelConfig({"model": "test_model"})
        model_config.num_hidden_layers = 10
        model_config.runner = "generate"  # Should apply if not pooling
        model_config.remove_tail_layer = True
        model_config.override_name_from_config()
        self.assertEqual(model_config.num_hidden_layers, 9)

        model_config = ModelConfig({"model": "test_model"})
        model_config.num_hidden_layers = 10
        model_config.runner = "generate"
        model_config.remove_tail_layer = 2
        model_config.override_name_from_config()
        self.assertEqual(model_config.num_hidden_layers, 8)

        # Test mla_use_absorb default
        model_config = ModelConfig({"model": "test_model"})
        self.assertFalse(hasattr(model_config, "mla_use_absorb"))
        model_config.override_name_from_config()
        self.assertFalse(model_config.mla_use_absorb)

        # Test moe_num_experts
        model_config = ModelConfig({"model": "test_model"})
        model_config.num_experts = 8
        model_config.moe_num_experts = None
        model_config.override_name_from_config()
        self.assertEqual(model_config.moe_num_experts, 8)

        model_config = ModelConfig({"model": "test_model"})
        model_config.n_routed_experts = 16
        model_config.moe_num_experts = None
        model_config.override_name_from_config()
        self.assertEqual(model_config.moe_num_experts, 16)

        # Test moe_num_shared_experts
        model_config = ModelConfig({"model": "test_model"})
        model_config.n_shared_experts = 4
        model_config.moe_num_shared_experts = None
        model_config.override_name_from_config()
        self.assertEqual(model_config.moe_num_shared_experts, 4)

    def test_model_config_read_from_env(self):
        # Test default values and environment variable override
        with patch.dict("os.environ", {}, clear=True):
            model_config = ModelConfig({"model": "test_model"})
            model_config.read_from_env()
            self.assertEqual(model_config.compression_ratio, 1.0)
            self.assertEqual(model_config.rope_theta, 10000)

        with patch.dict("os.environ", {"COMPRESSION_RATIO": "0.5", "ROPE_THETA": "20000"}, clear=True):
            model_config = ModelConfig({"model": "test_model"})
            model_config.read_from_env()
            self.assertEqual(model_config.compression_ratio, 0.5)
            self.assertEqual(model_config.rope_theta, 20000)

    def test_model_config_read_model_config(self):
        # Mock os.path.exists and json.load
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps({"torch_dtype": "bfloat16"}))),
            patch("json.load", return_value={"torch_dtype": "bfloat16"}),
        ):
            model_config = ModelConfig({"model": "/path/to/model"})
            model_config.read_model_config()
            self.assertEqual(model_config.model_format, "torch")

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                mock_open(read_data=json.dumps({"dtype": "float16", "transformers_version": "4.57.0"})),
            ),
            patch("json.load", return_value={"dtype": "float16", "transformers_version": "4.57.0"}),
        ):
            model_config = ModelConfig({"model": "/path/to/model"})
            model_config.read_model_config()
            self.assertEqual(model_config.model_format, "torch")

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open",
                mock_open(read_data=json.dumps({"dtype": "float16", "transformers_version": "4.50.0"})),
            ),
            patch("json.load", return_value={"dtype": "float16", "transformers_version": "4.50.0"}),
        ):
            model_config = ModelConfig({"model": "/path/to/model"})
            model_config.read_model_config()
            self.assertEqual(model_config.model_format, "paddle")

        with (
            patch("os.path.exists", return_value=True),
            patch(
                "builtins.open", mock_open(read_data=json.dumps({"quantization_config": {"quant_method": "mxfp4"}}))
            ),
            patch("json.load", return_value={"quantization_config": {"quant_method": "mxfp4"}}),
        ):
            model_config = ModelConfig({"model": "/path/to/model"})
            model_config.read_model_config()
            self.assertEqual(model_config.model_format, "torch")

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps({"torch_dtype": "bfloat16", "dtype": "float16"}))),
            patch("json.load", return_value={"torch_dtype": "bfloat16", "dtype": "float16"}),
        ):
            model_config = ModelConfig({"model": "/path/to/model"})
            with self.assertRaises(ValueError):
                model_config.read_model_config()

        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps({"unknown_key": "unknown_value"}))),
            patch("json.load", return_value={"unknown_key": "unknown_value"}),
        ):
            model_config = ModelConfig({"model": "/path/to/model"})
            with self.assertRaises(ValueError):
                model_config.read_model_config()

        with patch("os.path.exists", return_value=False):
            model_config = ModelConfig({"model": "/path/to/model"})
            model_config.read_model_config()
            self.assertFalse(hasattr(model_config, "model_format"))


if __name__ == "__main__":
    unittest.main()
