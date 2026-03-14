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
# Compatibility for Paddle versions missing 'compat', 'enable_torch_proxy', or newer functionals
if not hasattr(paddle, 'compat'):
    paddle.compat = type('obj', (object,), {'enable_torch_proxy': lambda *args, **kwargs: None})()
elif not hasattr(paddle.compat, 'enable_torch_proxy'):
    paddle.compat.enable_torch_proxy = lambda *args, **kwargs: None

if not hasattr(paddle, 'enable_compat'):
    paddle.enable_compat = lambda *args, **kwargs: None

# Mock Backend for dy2static if missing
try:
    import paddle.jit.dy2static.utils as jit_utils
    if not hasattr(jit_utils, 'Backend'):
        jit_utils.Backend = type('obj', (object,), {'CINN': 'CINN', 'SOT': 'SOT', 'PIR': 'PIR'})()
except ImportError:
    pass

# Mock swiglu if missing (used in some FD layers)
if not hasattr(paddle.nn.functional, 'swiglu'):
    def mock_swiglu(x, y=None):
        if y is None:
            x, y = paddle.chunk(x, 2, axis=-1)
        return paddle.nn.functional.silu(x) * y
    paddle.nn.functional.swiglu = mock_swiglu

# Mock fused_rms_norm if missing
if not hasattr(paddle.incubate.nn.functional, 'fused_rms_norm'):
    if not hasattr(paddle.incubate.nn, 'functional'):
        paddle.incubate.nn.functional = type('obj', (object,), {})()
    paddle.incubate.nn.functional.fused_rms_norm = lambda x, w, eps, **kwargs: (paddle.nn.functional.rms_norm(x, w.shape, w, eps), None)

# Fix paddleformers import
try:
    import paddleformers.transformers as pt
    if not hasattr(pt, 'PretrainedModel'):
        from paddleformers.transformers.model_utils import PretrainedModel
        pt.PretrainedModel = PretrainedModel
except ImportError:
    pass

import sys
# Mock missing C++ extensions in distributed.communication
import fastdeploy.distributed.communication as comm
comm.decode_alltoall_transpose = lambda *args, **kwargs: None
comm.tensor_model_parallel_all_reduce = lambda x: x

import importlib
original_import = importlib.import_module
def hacked_import(name, package=None):
    try:
        return original_import(name, package)
    except ImportError as e:
        if 'fastdeploy.model_executor.models.' in name and 'minimax_m1' not in name:
            print(f"Ignoring ImportError for {name}: {e}")
            return type('obj', (object,), {})()
        raise e
importlib.import_module = hacked_import

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
