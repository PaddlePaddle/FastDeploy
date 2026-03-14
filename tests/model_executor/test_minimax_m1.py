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

import unittest
import paddle
import numpy as np
from types import SimpleNamespace

from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode

class TestMiniMaxM1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set seed for reproducibility
        paddle.seed(42)
        
        # Create a standard mock configuration
        cls.model_config = SimpleNamespace(
            model_type="minimax_m1",
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=128,
            vocab_size=1000,
            num_local_experts=4,
            num_experts_per_tok=2,
            attn_type_list=[0, 1],
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            pad_token_id=0,
            hidden_act="silu",
            rope_theta=10000.0,
            shared_intermediate_size=512,
            layernorm_linear_attention_alpha=3.556,
            layernorm_linear_attention_beta=1.0,
            layernorm_full_attention_alpha=3.556,
            layernorm_full_attention_beta=1.0,
            layernorm_mlp_alpha=3.556,
            layernorm_mlp_beta=1.0,
            model_format="paddle",
            tie_word_embeddings=False,
            is_quantized=False,
            is_mtp=False,
        )
        
        cls.fd_config = SimpleNamespace()
        cls.fd_config.model_config = cls.model_config
        cls.fd_config.scheduler_config = SimpleNamespace(max_num_seqs=256, max_num_batched_tokens=256, max_model_len=2048, splitwise_role="mixed")
        cls.fd_config.parallel_config = SimpleNamespace(tensor_parallel_size=1, tensor_parallel_rank=0, expert_parallel_size=1, expert_parallel_rank=0, pp_size=1, pp_rank=0, use_sequence_parallel_moe=False, tp_group=None)
        cls.fd_config.cache_config = SimpleNamespace(kv_cache_dtype="fp16", block_size=16, num_gpu_blocks=100)
        cls.fd_config.device_config = SimpleNamespace()
        cls.fd_config.quant_config = SimpleNamespace(quant_type=None, weight_quant_method=None, quant_round_type=0, get_quant_method=lambda x: None, name=lambda: "")
        cls.fd_config.routing_replay_config = SimpleNamespace(enable_routing_replay=False)
        cls.fd_config.plas_attention_config = None
        cls.fd_config.mla_attention_config = None
        cls.fd_config.reasoning_config = None

    def test_prefill_forward(self):
        """Test MiniMax-M1 prefill forward pass on GPU"""
        model = MiniMaxM1ForCausalLM(self.fd_config)
        
        batch_size, seq_len = 2, 16
        total_tokens = batch_size * seq_len
        input_ids = paddle.randint(0, 1000, shape=[total_tokens])
        
        forward_meta = ForwardMeta(
            ids_remove_padding=input_ids,
            forward_mode=ForwardMode.EXTEND,
            cu_seqlens_q=paddle.to_tensor([0, seq_len, total_tokens], dtype="int32"),
            cu_seqlens_k=paddle.to_tensor([0, seq_len, total_tokens], dtype="int32"),
        )
        
        output = model(input_ids, forward_meta)
        self.assertEqual(output.shape, [total_tokens, self.model_config.hidden_size])
        print(f"Prefill Output Shape: {output.shape} - OK")

    def test_decode_forward(self):
        """Test MiniMax-M1 decode forward pass on GPU"""
        model = MiniMaxM1ForCausalLM(self.fd_config)
        
        batch_size = 2
        input_ids = paddle.randint(0, 1000, shape=[batch_size])
        
        forward_meta = ForwardMeta(
            ids_remove_padding=input_ids,
            forward_mode=ForwardMode.DECODE,
            cu_seqlens_q=paddle.to_tensor([0, 1, 2], dtype="int32"),
            cu_seqlens_k=paddle.to_tensor([0, 1, 2], dtype="int32"),
        )
        
        output = model(input_ids, forward_meta)
        self.assertEqual(output.shape, [batch_size, self.model_config.hidden_size])
        print(f"Decode Output Shape: {output.shape} - OK")

if __name__ == "__main__":
    unittest.main()
