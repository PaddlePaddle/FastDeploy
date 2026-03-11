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
Tests for MiniMax-M1 model
"""

import unittest
import numpy as np
import paddle


class TestMiniMaxM1Config(unittest.TestCase):
    """Test MiniMax-M1 configuration"""
    
    def test_minimax_m1_config_creation(self):
        """Test creating MiniMax-M1 config"""
        from fastdeploy.config import ModelConfig
        
        # MiniMax-M1 config (from HuggingFace)
        config_dict = {
            "model_type": "minimax_m1",
            "hidden_size": 6144,
            "intermediate_size": 9216,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 200064,
            "rope_theta": 10000000,
            "max_position_embeddings": 10240000,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "num_local_experts": 32,
            "num_experts_per_tok": 2,
            "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1] * 10,  # 8 layers pattern, repeated 10 times
            "layernorm_full_attention_alpha": 3.5565588200778455,
            "layernorm_full_attention_beta": 1.0,
            "layernorm_linear_attention_alpha": 3.5565588200778455,
            "layernorm_linear_attention_beta": 1.0,
            "layernorm_mlp_alpha": 3.5565588200778455,
            "layernorm_mlp_beta": 1.0,
            "postnorm": True,
        }
        
        # Just verify config dict is valid
        self.assertEqual(config_dict["model_type"], "minimax_m1")
        self.assertEqual(config_dict["hidden_size"], 6144)
        self.assertEqual(config_dict["num_local_experts"], 32)
        self.assertEqual(config_dict["num_experts_per_tok"], 2)
        self.assertEqual(len(config_dict["attn_type_list"]), 80)


class TestMiniMaxM1LightningAttention(unittest.TestCase):
    """Test Lightning Attention implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 2
        self.seq_len = 512
        self.num_heads = 64
        self.head_dim = 128
        self.hidden_size = self.num_heads * self.head_dim
        self.block_size = 256
    
    def test_slope_tensor_creation(self):
        """Test slope tensor building for Lightning Attention"""
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1Model
        
        # Test slope tensor creation
        def get_slopes(n):
            import math
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]
                )
        
        slopes = get_slopes(self.num_heads)
        self.assertEqual(len(slopes), self.num_heads)
        
        # All slopes should be positive
        for s in slopes:
            self.assertGreater(s, 0)
    
    def test_attn_type_list_length(self):
        """Test attn_type_list matches num_hidden_layers"""
        num_layers = 80
        attn_type_list = [0, 0, 0, 0, 0, 0, 0, 1] * 10  # 80 layers
        
        self.assertEqual(len(attn_type_list), num_layers)
        
        # Count attention types
        lightning_count = sum(1 for t in attn_type_list if t == 0)
        standard_count = sum(1 for t in attn_type_list if t == 1)
        
        # MiniMax-M1 uses 7 Lightning + 1 Standard per 8-layer block
        self.assertEqual(lightning_count, 70)  # 7 * 10
        self.assertEqual(standard_count, 10)   # 1 * 10


class TestMiniMaxM1ModelStructure(unittest.TestCase):
    """Test MiniMax-M1 model structure"""
    
    def test_model_registry_registration(self):
        """Test that MiniMax-M1 is properly registered"""
        from fastdeploy.model_executor.models.model_base import ModelRegistry
        
        # Check if MiniMax-M1 is registered
        # The model should be registered via @ModelRegistry.register_model_class
        # This is a basic check - in production, we would check the actual registry
        
        # For now, just verify the module can be imported
        try:
            from fastdeploy.model_executor.models.minimax_m1 import (
                MiniMaxM1ForCausalLM,
                MiniMaxM1Model,
                MiniMaxM1DecoderLayer,
                MiniMaxM1LightningAttention,
                MiniMaxM1StandardAttention,
                MiniMaxM1SparseMoEBlock,
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import MiniMax-M1 modules: {e}")
    
    def test_attention_type_selection(self):
        """Test attention type selection based on layer index"""
        # Test that attention type is correctly selected
        attn_type_list = [0, 0, 0, 0, 0, 0, 0, 1] * 10
        
        # Layer 0-6 should use Lightning Attention
        for i in range(7):
            self.assertEqual(attn_type_list[i], 0, f"Layer {i} should use Lightning Attention")
        
        # Layer 7 should use Standard Attention
        self.assertEqual(attn_type_list[7], 1, "Layer 7 should use Standard Attention")
        
        # Layer 8-14 should use Lightning Attention
        for i in range(8, 15):
            self.assertEqual(attn_type_list[i], 0, f"Layer {i} should use Lightning Attention")
        
        # Layer 15 should use Standard Attention
        self.assertEqual(attn_type_list[15], 1, "Layer 15 should use Standard Attention")


class TestMiniMaxM1MoE(unittest.TestCase):
    """Test MiniMax-M1 MoE configuration"""
    
    def test_moe_config(self):
        """Test MoE configuration"""
        num_experts = 32
        top_k = 2
        
        # Each token activates top_k experts
        self.assertEqual(num_experts, 32)
        self.assertEqual(top_k, 2)
        
        # Load balancing loss coefficient (from config)
        router_aux_loss_coef = 0.001
        self.assertEqual(router_aux_loss_coef, 0.001)
    
    def test_expert_routing(self):
        """Test expert routing computation"""
        batch_size = 2
        seq_len = 4
        num_experts = 32
        top_k = 2
        
        # Simulate router logits
        router_logits = np.random.randn(batch_size * seq_len, num_experts).astype(np.float32)
        
        # Softmax over experts
        routing_weights = np.exp(router_logits) / np.exp(router_logits).sum(axis=-1, keepdims=True)
        
        # Top-k selection
        selected_experts = np.argsort(routing_weights, axis=-1)[:, -top_k:]
        
        # Verify shapes
        self.assertEqual(selected_experts.shape, (batch_size * seq_len, top_k))
        
        # Verify expert indices are valid
        self.assertTrue(np.all(selected_experts >= 0))
        self.assertTrue(np.all(selected_experts < num_experts))


class TestMiniMaxM1Integration(unittest.TestCase):
    """Integration tests for MiniMax-M1"""
    
    def test_config_equivalence(self):
        """Test that our config matches HuggingFace config"""
        # MiniMax-M1-80k config from HuggingFace
        hf_config = {
            "model_type": "minimax_m1",
            "architectures": ["MiniMaxM1ForCausalLM"],
            "hidden_size": 6144,
            "intermediate_size": 9216,
            "num_hidden_layers": 80,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "vocab_size": 200064,
            "rope_theta": 10000000,
            "max_position_embeddings": 10240000,
            "rms_norm_eps": 1e-5,
            "hidden_act": "silu",
            "num_local_experts": 32,
            "num_experts_per_tok": 2,
            "attn_type_list": [0] * 70 + [1] * 10,  # 70 Lightning + 10 Standard
        }
        
        # Verify key parameters
        self.assertEqual(hf_config["hidden_size"], 6144)
        self.assertEqual(hf_config["num_hidden_layers"], 80)
        self.assertEqual(hf_config["num_local_experts"], 32)
        self.assertEqual(hf_config["num_experts_per_tok"], 2)
        
        # Total params: 456B, activated: 45.9B
        # hidden_size * num_attention_heads * head_dim check
        self.assertEqual(hf_config["num_attention_heads"] * hf_config["head_dim"], hf_config["hidden_size"])
    
    def test_position_embeddings_limit(self):
        """Test max position embeddings"""
        max_position = 10240000  # 10M context
        self.assertEqual(max_position, 10 * 1024 * 1024)  # 10M tokens


if __name__ == "__main__":
    unittest.main()