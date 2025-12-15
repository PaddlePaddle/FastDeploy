"""
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

import unittest
from unittest.mock import MagicMock, patch

from fastdeploy.rl.rollout_model import (
    BaseRLModel,
    Ernie4_5_MoeForCausalLMRL,
    Glm4MoeForCausalLMRL,
    Qwen2ForCausalLMRL,
    Qwen3ForCausalLMRL,
    Qwen3MoeForCausalLMRL,
    RolloutModel,
)

# Note: This test requires dependencies like paddleformers, paddle, etc.
# In CI environment, these should be available.
# For local testing without dependencies, you may need to install them or use CI.


class TestBaseRLModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.base_model = BaseRLModel()

    def test_name(self):
        """Test name class method"""
        self.assertEqual(BaseRLModel.name(), "BaseRLModel")

    def test_update_base_mappings(self):
        """Test _update_base_mappings method"""
        self.base_model._update_base_mappings("test_base")
        self.assertIn("test_base.embed_tokens.embeddings.weight", self.base_model.infer_to_train_mapping)
        self.assertIn("lm_head.linear.weight", self.base_model.infer_to_train_mapping)

    def test_complete_missing_mappings(self):
        """Test _complete_missing_mappings method"""
        self.base_model.state_dict = MagicMock(return_value={"layer1.weight": None, "layer2.weight": None})
        self.base_model.infer_to_train_mapping = {"layer1.weight": "layer1.weight"}
        self.base_model._complete_missing_mappings()
        self.assertIn("layer2.weight", self.base_model.infer_to_train_mapping)

    def test_get_quantization_infer_keys(self):
        """Test get_quantization_infer_keys method"""
        self.base_model.fd_config = MagicMock()
        self.base_model.fd_config.quant_config = MagicMock()
        self.base_model.fd_config.quant_config.name.return_value = "wint8"
        self.base_model.state_dict = MagicMock(return_value={"layer.weight": None, "layer.weight_scale": None})

        keys = self.base_model.get_quantization_infer_keys()
        self.assertIsInstance(keys, list)

        # Test unsupported quantization
        self.base_model.fd_config.quant_config.name.return_value = "unsupported"
        with self.assertRaises(ValueError):
            self.base_model.get_quantization_infer_keys()


class TestRolloutModel(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_rollout_config = MagicMock()
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.architectures = ["ErnieMoeForCausalLM"]
        self.mock_rollout_config.initialize.return_value = self.mock_fd_config

    @patch("fastdeploy.rl.rollout_model.ModelRegistry")
    def test_init(self, mock_registry):
        """Test RolloutModel initialization"""
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model_cls.return_value = mock_model
        mock_registry.get_class.return_value = mock_model_cls

        rollout_model = RolloutModel(self.mock_rollout_config)
        self.assertIsNotNone(rollout_model)
        self.assertIsNotNone(rollout_model.rollout_model)

    @patch("fastdeploy.rl.rollout_model.ModelRegistry")
    def test_get_name_mappings_to_training(self, mock_registry):
        """Test get_name_mappings_to_training method"""
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.get_name_mappings_to_training = MagicMock(return_value={"key1": "value1"})
        mock_model_cls.return_value = mock_model
        mock_registry.get_class.return_value = mock_model_cls

        rollout_model = RolloutModel(self.mock_rollout_config)
        mappings = rollout_model.get_name_mappings_to_training()
        self.assertEqual(mappings, {"key1": "value1"})

    @patch("fastdeploy.rl.rollout_model.ModelRegistry")
    def test_get_quantization_infer_keys(self, mock_registry):
        """Test get_quantization_infer_keys method"""
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_model.get_quantization_infer_keys = MagicMock(return_value=["key1", "key2"])
        mock_model_cls.return_value = mock_model
        mock_registry.get_class.return_value = mock_model_cls

        rollout_model = RolloutModel(self.mock_rollout_config)
        keys = rollout_model.get_quantization_infer_keys()
        self.assertEqual(keys, ["key1", "key2"])

    @patch("fastdeploy.rl.rollout_model.ModelRegistry")
    def test_state_dict(self, mock_registry):
        """Test state_dict method"""
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.eval.return_value = None
        mock_state_dict = {"layer1.weight": None, "layer2.weight": None}
        mock_model.state_dict.return_value = mock_state_dict
        mock_model_cls.return_value = mock_model
        mock_registry.get_class.return_value = mock_model_cls

        rollout_model = RolloutModel(self.mock_rollout_config)
        state_dict = rollout_model.state_dict()
        self.assertEqual(state_dict, mock_state_dict)


class TestErnie4_5_MoeForCausalLMRL(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.moe_layer_start_index = 8
        self.mock_fd_config.model_config.num_hidden_layers = 32
        self.mock_fd_config.model_config.moe_num_experts = 8
        self.mock_fd_config.model_config.moe_use_aux_free = False
        self.mock_fd_config.parallel_config = MagicMock()
        self.mock_fd_config.parallel_config.tensor_parallel_size = 1

    @patch("fastdeploy.rl.rollout_model.Ernie4_5_MoeForCausalLM.__init__")
    def test_init(self, mock_base_init):
        """Test Ernie4_5_MoeForCausalLMRL initialization"""
        mock_base_init.return_value = None
        model = Ernie4_5_MoeForCausalLMRL(self.mock_fd_config)
        self.assertIsNotNone(model)

    @patch("fastdeploy.rl.rollout_model.Ernie4_5_MoeForCausalLM.__init__")
    def test_name(self, mock_base_init):
        """Test name class method"""
        mock_base_init.return_value = None
        model = Ernie4_5_MoeForCausalLMRL(self.mock_fd_config)
        self.assertEqual(model.name(), "Ernie4_5_MoeForCausalLMRL")

    @patch("fastdeploy.rl.rollout_model.Ernie4_5_MoeForCausalLM.__init__")
    def test_get_name_mappings_to_training(self, mock_base_init):
        """Test get_name_mappings_to_training method"""
        mock_base_init.return_value = None
        model = Ernie4_5_MoeForCausalLMRL(self.mock_fd_config)
        # Initialize BaseRLModel attributes since __init__ was mocked
        # Use object.__setattr__ to bypass Paddle's custom __setattr__
        object.__setattr__(model, "_mappings_built", False)
        object.__setattr__(model, "infer_to_train_mapping", {})
        object.__setattr__(model, "fd_config", self.mock_fd_config)
        model.state_dict = MagicMock(return_value={"ernie.layers.8.mlp.gate.weight": None})

        mappings = model.get_name_mappings_to_training()
        self.assertIsInstance(mappings, dict)
        self.assertTrue(model._mappings_built)

        # Test cached mappings
        mappings2 = model.get_name_mappings_to_training()
        self.assertEqual(mappings, mappings2)


class TestQwen2ForCausalLMRL(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.num_hidden_layers = 32

    @patch("fastdeploy.rl.rollout_model.Qwen2ForCausalLM.__init__")
    def test_init(self, mock_base_init):
        """Test Qwen2ForCausalLMRL initialization"""
        mock_base_init.return_value = None
        model = Qwen2ForCausalLMRL(self.mock_fd_config)
        self.assertIsNotNone(model)

    @patch("fastdeploy.rl.rollout_model.Qwen2ForCausalLM.__init__")
    def test_name(self, mock_base_init):
        """Test name class method"""
        mock_base_init.return_value = None
        model = Qwen2ForCausalLMRL(self.mock_fd_config)
        self.assertEqual(model.name(), "Qwen2ForCausalLMRL")

    @patch("fastdeploy.rl.rollout_model.Qwen2ForCausalLM.__init__")
    def test_get_name_mappings_to_training(self, mock_base_init):
        """Test get_name_mappings_to_training method"""
        mock_base_init.return_value = None
        model = Qwen2ForCausalLMRL(self.mock_fd_config)
        # Initialize BaseRLModel attributes since __init__ was mocked
        # Use object.__setattr__ to bypass Paddle's custom __setattr__
        object.__setattr__(model, "_mappings_built", False)
        object.__setattr__(model, "infer_to_train_mapping", {})
        object.__setattr__(model, "fd_config", self.mock_fd_config)
        model.state_dict = MagicMock(return_value={"qwen2.layers.0.mlp.up_gate_proj.weight": None})

        mappings = model.get_name_mappings_to_training()
        self.assertIsInstance(mappings, dict)


class TestQwen3ForCausalLMRL(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.num_hidden_layers = 32

    @patch("fastdeploy.rl.rollout_model.Qwen3ForCausalLM.__init__")
    def test_init(self, mock_base_init):
        """Test Qwen3ForCausalLMRL initialization"""
        mock_base_init.return_value = None
        model = Qwen3ForCausalLMRL(self.mock_fd_config)
        self.assertIsNotNone(model)

    @patch("fastdeploy.rl.rollout_model.Qwen3ForCausalLM.__init__")
    def test_name(self, mock_base_init):
        """Test name class method"""
        mock_base_init.return_value = None
        model = Qwen3ForCausalLMRL(self.mock_fd_config)
        self.assertEqual(model.name(), "Qwen3ForCausalLMRL")

    @patch("fastdeploy.rl.rollout_model.Qwen3ForCausalLM.__init__")
    def test_get_name_mappings_to_training(self, mock_base_init):
        """Test get_name_mappings_to_training method"""
        mock_base_init.return_value = None
        model = Qwen3ForCausalLMRL(self.mock_fd_config)
        # Initialize BaseRLModel attributes since __init__ was mocked
        # Use object.__setattr__ to bypass Paddle's custom __setattr__
        object.__setattr__(model, "_mappings_built", False)
        object.__setattr__(model, "infer_to_train_mapping", {})
        object.__setattr__(model, "fd_config", self.mock_fd_config)
        model.state_dict = MagicMock(return_value={"model.layers.0.mlp.up_gate_proj.weight": None})

        mappings = model.get_name_mappings_to_training()
        self.assertIsInstance(mappings, dict)


class TestQwen3MoeForCausalLMRL(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.num_hidden_layers = 32
        self.mock_fd_config.model_config.num_experts = 8
        self.mock_fd_config.model_config.moe_use_aux_free = False

    @patch("fastdeploy.rl.rollout_model.Qwen3MoeForCausalLM.__init__")
    def test_init(self, mock_base_init):
        """Test Qwen3MoeForCausalLMRL initialization"""
        mock_base_init.return_value = None
        model = Qwen3MoeForCausalLMRL(self.mock_fd_config)
        self.assertIsNotNone(model)

    @patch("fastdeploy.rl.rollout_model.Qwen3MoeForCausalLM.__init__")
    def test_name(self, mock_base_init):
        """Test name class method"""
        mock_base_init.return_value = None
        model = Qwen3MoeForCausalLMRL(self.mock_fd_config)
        self.assertEqual(model.name(), "Qwen3MoeForCausalLMRL")

    @patch("fastdeploy.rl.rollout_model.Qwen3MoeForCausalLM.__init__")
    def test_get_name_mappings_to_training(self, mock_base_init):
        """Test get_name_mappings_to_training method"""
        mock_base_init.return_value = None
        model = Qwen3MoeForCausalLMRL(self.mock_fd_config)
        # Initialize BaseRLModel attributes since __init__ was mocked
        # Use object.__setattr__ to bypass Paddle's custom __setattr__
        object.__setattr__(model, "_mappings_built", False)
        object.__setattr__(model, "infer_to_train_mapping", {})
        object.__setattr__(model, "fd_config", self.mock_fd_config)
        model.state_dict = MagicMock(return_value={"model.layers.0.mlp.gate.weight": None})

        mappings = model.get_name_mappings_to_training()
        self.assertIsInstance(mappings, dict)


class TestGlm4MoeForCausalLMRL(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.mock_fd_config = MagicMock()
        self.mock_fd_config.model_config = MagicMock()
        self.mock_fd_config.model_config.num_hidden_layers = 32
        self.mock_fd_config.model_config.n_routed_experts = 8
        self.mock_fd_config.model_config.first_k_dense_replace = 0

    @patch("fastdeploy.rl.rollout_model.Glm4MoeForCausalLM.__init__")
    def test_init(self, mock_base_init):
        """Test Glm4MoeForCausalLMRL initialization"""
        mock_base_init.return_value = None
        model = Glm4MoeForCausalLMRL(self.mock_fd_config)
        self.assertIsNotNone(model)

    @patch("fastdeploy.rl.rollout_model.Glm4MoeForCausalLM.__init__")
    def test_name(self, mock_base_init):
        """Test name class method"""
        mock_base_init.return_value = None
        model = Glm4MoeForCausalLMRL(self.mock_fd_config)
        self.assertEqual(model.name(), "Glm4MoeForCausalLMRL")

    @patch("fastdeploy.rl.rollout_model.Glm4MoeForCausalLM.__init__")
    def test_get_name_mappings_to_training(self, mock_base_init):
        """Test get_name_mappings_to_training method"""
        mock_base_init.return_value = None
        model = Glm4MoeForCausalLMRL(self.mock_fd_config)
        # Initialize BaseRLModel attributes since __init__ was mocked
        # Use object.__setattr__ to bypass Paddle's custom __setattr__
        object.__setattr__(model, "_mappings_built", False)
        object.__setattr__(model, "infer_to_train_mapping", {})
        object.__setattr__(model, "fd_config", self.mock_fd_config)
        model.state_dict = MagicMock(return_value={"model.layers.0.mlp.gate.weight": None})

        mappings = model.get_name_mappings_to_training()
        self.assertIsInstance(mappings, dict)


if __name__ == "__main__":
    unittest.main()
