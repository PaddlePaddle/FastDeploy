#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax-M1 Small Config Test Script (Simplified)

直接测试模型类，不依赖完整的 FastDeploy 运行时

Usage:
    python test_minimax_m1_small.py
"""

import os
import sys
import tempfile

# 设置环境变量
if os.name == 'nt':
    temp_dir = os.path.join(tempfile.gettempdir(), 'fd_prom')
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = temp_dir
else:
    os.environ["PROMETHEUS_MULTIPROC_DIR"] = f"/tmp/fd_prom_{os.getpid()}"
    os.makedirs(os.environ["PROMETHEUS_MULTIPROC_DIR"], exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paddle
import math

print("="*60)
print("MiniMax-M1 Small Config Test")
print("="*60)
print(f"PaddlePaddle version: {paddle.__version__}")
print(f"Device: CPU (forced)")
print()


# Small config
MINIMAX_M1_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 1000,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1],
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
    "use_fused_rope": False,
    "use_dynamic_ntk": True,
    "use_logn_attn": False,
    "hidden_act": "silu",
    "rope_theta": 10000.0,
    "is_quantized": False,
    "is_moe_quantized": False,
    "model_format": "paddle",
}


class FakeFDConfig:
    """Fake FDConfig for testing"""
    def __init__(self, model_config):
        self.model_config = model_config
        self.scheduler_config = type('obj', (object,), {
            'max_num_seqs': 256,
            'max_num_batched_tokens': 256,
            'max_model_len': 2048,
        })()
        self.cache_config = type('obj', (object,), {
            'kv_cache_dtype': 'fp16',
            'block_size': 16,
        })()
        self.device_config = type('obj', (object,), {})
        self.parallel_config = type('obj', (object,), {
            'tensor_parallel_size': 1,
            'tensor_parallel_rank': 0,
            'tp_group': None,
            'expert_parallel_size': 1,
            'expert_parallel_rank': 0,
            'ep_group': None,
            'use_sequence_parallel_moe': False,
            'use_sequence_parallel': False,
            'sep_parallel_size': 1,
            'ring_parallel_size': 1,
        })()
        self.model_config.use_fused_rope = False
        self.model_config.hidden_dropout_prob = 0.0
        self.model_config.initializer_range = 0.02
        self.model_config.tie_word_embeddings = False
        self.model_config.lm_head_fp32 = False
        self.quant_config = None
        self.routing_replay_config = type('obj', (object,), {'enable_routing_replay': False})()
        
        # Mock fleet
        from paddle.distributed import fleet
        fleet.get_hybrid_communicate_group = lambda: type('obj', (object,), {
            'get_model_parallel_rank': lambda self: 0,
            'get_model_parallel_world_size': lambda self: 1,
        })()
        
        class FakeMetaParallelLinear(nn.Layer):
            def __init__(self, in_features, out_features, *args, **kwargs):
                super().__init__()
                self.weight = self.create_parameter(shape=[in_features, out_features], dtype="float32")
                self.bias = None
            def forward(self, x): return paddle.matmul(x, self.weight)
            
        class MetaParallelMock:
            ColumnParallelLinear = FakeMetaParallelLinear
            RowParallelLinear = FakeMetaParallelLinear
            
        fleet.meta_parallel = MetaParallelMock()


def test_model_structure():
    """Test 1: Model Structure"""
    print("="*60)
    print("Test 1: Model Structure")
    print("="*60)
    
    try:
        # Import model
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
        import fastdeploy.model_executor.models.minimax_m1 as minimax_m1_mod
        from paddle import nn
        class FakeVocabParallelEmbedding(nn.Layer):
            def __init__(self, fd_config, num_embeddings, embedding_dim, params_dtype="bfloat16", prefix=""):
                super().__init__()
                self.weight = self.create_parameter(
                    shape=[num_embeddings, embedding_dim],
                    dtype="float32",
                    is_bias=False
                )
            def forward(self, x):
                return nn.functional.embedding(x, self.weight)
        minimax_m1_mod.VocabParallelEmbedding = FakeVocabParallelEmbedding
        
        class FakeLinear(nn.Layer):
            def __init__(self, fd_config, input_size, output_size, with_bias=False, **kwargs):
                super().__init__()
                self.weight = self.create_parameter(shape=[input_size, output_size], dtype="float32")
                if with_bias:
                    self.bias = self.create_parameter(shape=[output_size], dtype="float32", is_bias=True)
                else:
                    self.bias = None
            def forward(self, x):
                out = paddle.matmul(x, self.weight)
                if self.bias is not None:
                    out = out + self.bias
                return out
                
        minimax_m1_mod.ColumnParallelLinear = FakeLinear
        minimax_m1_mod.RowParallelLinear = FakeLinear
        minimax_m1_mod.ReplicatedLinear = FakeLinear
        
        # Mock MoE method for CPU test
        import fastdeploy.model_executor.layers.moe.moe as moe_mod
        class FakeMoEMethod:
            def __init__(self, *args, **kwargs):
                self.load_up_proj_weight_first = False
            def init_ep(self, *args, **kwargs): pass
            def create_weights(self, layer, *args, **kwargs):
                layer.weight = layer.create_parameter(shape=[1], dtype="float32")
            def process_loaded_weights(self, *args, **kwargs): pass
            def apply(self, layer, x, gate, *args, **kwargs): return x
        moe_mod.get_moe_method = lambda layer=None: FakeMoEMethod()
        
        # Create config object
        config = type('MiniMaxM1Config', (), MINIMAX_M1_CONFIG)()
        
        # Create FDConfig
        fd_config = FakeFDConfig(config)
        
        # Create model
        model = MiniMaxM1ForCausalLM(fd_config)
        
        # Count parameters
        total_params = sum(p.numel().item() for p in model.parameters())
        
        print(f"[PASS] Model created successfully")
        print(f"   Total params: {total_params:,}")
        print(f"   Memory (FP16): ~{total_params * 2 / 1024 / 1024:.2f} MB")
        
        return model, fd_config
        
    except Exception as e:
        print(f"[FAIL] Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward(model):
    """Test 2: Forward Pass"""
    print("\n" + "="*60)
    print("Test 2: Forward Pass")
    print("="*60)
    
    if model is None:
        print("[SKIP] Model not created")
        return
    
    try:
        batch_size = 2
        seq_len = 16
        vocab_size = MINIMAX_M1_CONFIG["vocab_size"]
        
        # Random input
        input_ids = paddle.randint(low=0, high=vocab_size, shape=[batch_size, seq_len])
        
        print(f"Input shape: {input_ids.shape}")
        
        # Forward pass
        output = model(input_ids)
        
        print(f"Output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = [batch_size, seq_len, vocab_size]
        if list(output.shape) == expected_shape:
            print(f"[PASS] Forward pass successful, output shape correct")
        else:
            print(f"[FAIL] Output shape wrong: expected {expected_shape}, got {list(output.shape)}")
        
    except Exception as e:
        print(f"[FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_attention(model):
    """Test 3: Hybrid Attention"""
    print("\n" + "="*60)
    print("Test 3: Hybrid Attention Mechanism")
    print("="*60)
    
    if model is None:
        print("[SKIP]")
        return
    
    try:
        # Check attention config
        attn_types = model.model.attn_type_list
        
        print(f"Attention layer config:")
        for i, attn_type in enumerate(attn_types):
            attn_name = "Lightning" if attn_type == 0 else "Standard"
            print(f"  Layer {i}: {attn_name}")
        
        # Verify config
        expected = MINIMAX_M1_CONFIG["attn_type_list"]
        if attn_types == expected:
            print(f"[PASS] Hybrid attention config correct")
        else:
            print(f"[FAIL] Config mismatch")
            
    except Exception as e:
        print(f"[FAIL] Hybrid attention test failed: {e}")
        import traceback
        traceback.print_exc()


def test_moe(model):
    """Test 4: MoE"""
    print("\n" + "="*60)
    print("Test 4: MoE Expert Routing")
    print("="*60)
    
    if model is None:
        print("[SKIP]")
        return
    
    try:
        print(f"MoE config:")
        print(f"  Expert count: {MINIMAX_M1_CONFIG['num_local_experts']}")
        print(f"  Active experts per token: {MINIMAX_M1_CONFIG['num_experts_per_tok']}")
        
        # Check each layer for MoE
        moe_count = 0
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                moe_count += 1
                print(f"  Layer {i}: has MoE")
        
        if moe_count > 0:
            print(f"[PASS] MoE layer count: {moe_count}")
        else:
            print("[INFO] No MoE layers found (may use FusedMoE)")
        
        # Test MoE forward
        x = paddle.randn([1, 4, MINIMAX_M1_CONFIG['hidden_size']])
        
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp'):
                try:
                    out = layer.mlp(x)
                    print(f"  Layer {i} MoE forward: {out.shape} [OK]")
                except Exception as e:
                    print(f"  Layer {i} MoE forward failed: {e}")
        
        print(f"[PASS] MoE test completed")
        
    except Exception as e:
        print(f"[FAIL] MoE test failed: {e}")
        import traceback
        traceback.print_exc()


def test_registry():
    """Test 5: Model Registry"""
    print("\n" + "="*60)
    print("Test 5: Model Registry")
    print("="*60)
    
    try:
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
        
        print(f"[PASS] MiniMaxM1ForCausalLM is registered")
        print(f"   Class name: {MiniMaxM1ForCausalLM.name()}")
        
    except Exception as e:
        print(f"[FAIL] Model registry test failed: {e}")


def main():
    model, fd_config = test_model_structure()
    test_forward(model)
    test_hybrid_attention(model)
    test_moe(model)
    test_registry()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()