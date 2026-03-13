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
MiniMax-M1 Model Execution Test
"""

import os
import paddle
import numpy as np
from types import SimpleNamespace
from fastdeploy.config import FDConfig
from fastdeploy.model_executor.forward_meta import ForwardMeta, ForwardMode
from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM

# Set seed for reproducibility
paddle.seed(42)

def create_mock_config():
    """Create a minimal configuration for MiniMax-M1"""
    model_config = SimpleNamespace(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=1000,
        num_local_experts=4,
        num_experts_per_tok=2,
        attn_type_list=[0, 1], # 1 Lightning + 1 Standard
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        hidden_act="silu",
        rope_theta=10000.0,
        shared_intermediate_size=512,
        layernorm_linear_attention_alpha=3.5,
        layernorm_linear_attention_beta=1.0,
        layernorm_full_attention_alpha=3.5,
        layernorm_full_attention_beta=1.0,
        layernorm_mlp_alpha=3.5,
        layernorm_mlp_beta=1.0,
    )
    
    parallel_config = SimpleNamespace(
        tensor_parallel_size=1,
        tensor_parallel_rank=0,
        expert_parallel_size=1,
    )
    
    fd_config = FDConfig()
    fd_config.model_config = model_config
    fd_config.parallel_config = parallel_config
    
    return fd_config

def test_forward_prefill():
    """Test model forward in prefill mode"""
    print("Testing Prefill Forward...")
    fd_config = create_mock_config()
    model = MiniMaxM1ForCausalLM(fd_config)
    
    batch_size = 2
    seq_len = 16
    total_tokens = batch_size * seq_len
    
    input_ids = paddle.randint(0, 1000, shape=[total_tokens])
    
    # Create ForwardMeta
    forward_meta = ForwardMeta(
        ids_remove_padding=input_ids,
        forward_mode=ForwardMode.EXTEND,
        cu_seqlens_q=paddle.to_tensor([0, seq_len, total_tokens], dtype="int32"),
        cu_seqlens_k=paddle.to_tensor([0, seq_len, total_tokens], dtype="int32"),
    )
    
    # Forward pass
    output = model(input_ids, forward_meta)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == [total_tokens, fd_config.model_config.hidden_size]
    print("Prefill Test Passed!")

def test_forward_decode():
    """Test model forward in decode mode"""
    print("Testing Decode Forward...")
    fd_config = create_mock_config()
    model = MiniMaxM1ForCausalLM(fd_config)
    
    batch_size = 2
    # In decode mode, each request has 1 token
    input_ids = paddle.randint(0, 1000, shape=[batch_size])
    
    # Create ForwardMeta
    forward_meta = ForwardMeta(
        ids_remove_padding=input_ids,
        forward_mode=ForwardMode.DECODE,
        cu_seqlens_q=paddle.to_tensor([0, 1, 2], dtype="int32"),
        cu_seqlens_k=paddle.to_tensor([0, 1, 2], dtype="int32"),
    )
    
    # Forward pass
    output = model(input_ids, forward_meta)
    
    print(f"Output shape: {output.shape}")
    assert output.shape == [batch_size, fd_config.model_config.hidden_size]
    print("Decode Test Passed!")

if __name__ == "__main__":
    try:
        test_forward_prefill()
        test_forward_decode()
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
