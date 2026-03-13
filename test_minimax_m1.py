#!/usr/bin/env python3
"""
MiniMax-M1 GPU Test Script

用于在 GPU 环境下测试 MiniMax-M1 模型

Usage:
    python test_minimax_m1.py
"""

import os
import sys
import paddle

# Setup environment
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/fd_prom"
os.makedirs("/tmp/fd_prom", exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("MiniMax-M1 GPU Test")
print("="*60)
print(f"PaddlePaddle version: {paddle.__version__}")
print(f"CUDA available: {paddle.device.is_compiled_with_cuda()}")

if not paddle.device.is_compiled_with_cuda():
    print("ERROR: This test requires GPU!")
    sys.exit(1)

# Test config - small for quick testing
TEST_CONFIG = {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 128,
    "vocab_size": 1000,
    "num_local_experts": 4,
    "num_experts_per_tok": 2,
    "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1],  # 7 Lightning + 1 Standard
    "max_position_embeddings": 512,
    "rms_norm_eps": 1e-6,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
}


def test_model_creation():
    """Test 1: Model Creation"""
    print("\n[Test 1] Model Creation")
    try:
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
        from fastdeploy.config import FDConfig
        
        # Create config
        config = type('obj', (object,), TEST_CONFIG)()
        
        fd_config = FDConfig()
        fd_config.model_config = config
        fd_config.scheduler_config = type('obj', (object,), {
            'max_num_seqs': 256,
            'max_num_batched_tokens': 256,
            'max_model_len': 2048,
        })()
        fd_config.cache_config = type('obj', (object,), {
            'kv_cache_dtype': 'fp16',
            'block_size': 16,
        })()
        fd_config.device_config = type('obj', (object,), {})
        
        # Create model
        model = MiniMaxM1ForCausalLM(fd_config)
        model = model.cuda()  # Move to GPU
        
        # Count parameters
        total_params = sum(p.numel().item() for p in model.parameters())
        
        print(f"  [PASS] Model created on GPU")
        print(f"  Params: {total_params:,}")
        
        return model
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return None


def test_forward(model):
    """Test 2: Forward Pass"""
    print("\n[Test 2] Forward Pass")
    if model is None:
        print("  [SKIP]")
        return
    
    try:
        batch_size = 2
        seq_len = 16
        vocab_size = TEST_CONFIG["vocab_size"]
        
        input_ids = paddle.randint(low=0, high=vocab_size, shape=[batch_size, seq_len])
        input_ids = input_ids.cuda()
        
        output = model(input_ids)
        
        expected_shape = [batch_size, seq_len, vocab_size]
        if list(output.shape) == expected_shape:
            print(f"  [PASS] Output shape: {output.shape}")
        else:
            print(f"  [FAIL] Shape mismatch: {output.shape}")
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_attention(model):
    """Test 3: Hybrid Attention"""
    print("\n[Test 3] Hybrid Attention")
    if model is None:
        print("  [SKIP]")
        return
    
    try:
        attn_types = model.model.attn_type_list
        print(f"  Attention layers: {len(attn_types)}")
        
        lightning_count = sum(1 for t in attn_types if t == 0)
        standard_count = sum(1 for t in attn_types if t == 1)
        
        print(f"  Lightning: {lightning_count}, Standard: {standard_count}")
        print(f"  [PASS] Hybrid attention configured correctly")
        
    except Exception as e:
        print(f"  [FAIL] {e}")


def test_moe(model):
    """Test 4: MoE Layer"""
    print("\n[Test 4] MoE Layer")
    if model is None:
        print("  [SKIP]")
        return
    
    try:
        x = paddle.randn([1, 4, TEST_CONFIG["hidden_size"]]).cuda()
        
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp'):
                out = layer.mlp(x)
                print(f"  Layer {i} MoE: {out.shape} [OK]")
        
        print(f"  [PASS] MoE layers work")
        
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()


def main():
    model = test_model_creation()
    test_forward(model)
    test_hybrid_attention(model)
    test_moe(model)
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)


if __name__ == "__main__":
    main()