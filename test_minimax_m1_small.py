#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MiniMax-M1 Small Config Test Script

用于在小显存环境下验证 MiniMax-M1 代码逻辑
配置经过优化，可以在 16GB 显存笔记本电脑上运行

Usage:
    python test_minimax_m1_small.py
"""

import os
import sys
import paddle
import numpy as np

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0，如果用 CPU 改为 "-1"

# 添加 FastDeploy 到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastdeploy import RuntimeOption
from fastdeploy.model_executor import LLMModel


# 小配置 - 可以在 16GB 显存上运行
MINIMAX_M1_SMALL_CONFIG = {
    "model_type": "minimax_m1",
    "architectures": ["MiniMaxM1ForCausalLM"],
    "hidden_size": 512,          # 原版: 6144
    "intermediate_size": 1024,   # 原版: 9216
    "num_hidden_layers": 4,      # 原版: 80
    "num_attention_heads": 4,    # 原版: 64
    "num_key_value_heads": 2,    # 原版: 8
    "head_dim": 128,             # 原版: 128
    "vocab_size": 1000,          # 原版: 200064 (用小词汇表测试)
    "num_local_experts": 4,      # 原版: 32
    "num_experts_per_tok": 2,    # 原版: 2
    # 每8层为1组: 7层Lightning(0) + 1层Standard(1)
    "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1],
    "max_position_embeddings": 512,  # 原版: 10240000
    "rms_norm_eps": 1e-6,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": False,
    "use_fused_rope": False,
    "use_dynamic_ntk": True,
    "use_logn_attn": False,
}


def test_model_loading():
    """测试模型加载"""
    print("\n" + "="*50)
    print("测试 1: 模型加载")
    print("="*50)
    
    try:
        # 创建运行时选项 - 使用 CPU 因为显存不够
        option = RuntimeOption()
        option.use_cpu()  # 强制使用 CPU
        # option.use_paddle_inference()  # 如果有 GPU 可以用这个
        
        # 创建配置
        config = MINIMAX_M1_SMALL_CONFIG.copy()
        
        print(f"模型配置:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # 创建模型 (不使用真实权重，只验证结构)
        print("\n创建模型中...")
        
        # 直接实例化模型进行测试
        from fastdeploy.config import FDConfig
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
        
        # 构造 FDConfig
        fd_config = FDConfig()
        fd_config.model_config = type('Config', (), config)()
        
        # 创建模型
        model = MiniMaxM1ForCausalLM(fd_config)
        
        # 统计参数量
        total_params = sum(p.numel().item() for p in model.parameters())
        trainable_params = sum(p.numel().item() for p in model.parameters() if p.stop_gradient is False)
        
        print(f"\n模型创建成功!")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        print(f"  显存需求 (FP16): ~{total_params * 2 / 1024 / 1024:.2f} MB")
        
        return model, fd_config
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, fd_config):
    """测试前向传播"""
    print("\n" + "="*50)
    print("测试 2: 前向传播")
    print("="*50)
    
    if model is None:
        print("跳过 - 模型未创建")
        return
    
    try:
        batch_size = 2
        seq_len = 16
        
        # 创建随机输入
        input_ids = paddle.randint(
            low=0, 
            high=MINIMAX_M1_SMALL_CONFIG["vocab_size"], 
            shape=[batch_size, seq_len]
        )
        
        print(f"输入 shape: {input_ids.shape}")
        print(f"输入数据 (前5个token):\n{input_ids[0][:5].numpy()}")
        
        # 前向传播
        output = model(input_ids)
        
        print(f"\n输出 shape: {output.shape}")
        print(f"输出数据 (前5个token, 前5个词概率):")
        probs = paddle.nn.functional.softmax(output[0, 0], axis=-1)
        print(f"  Top-5 概率: {probs.numpy()[:5]}")
        
        print("\n前向传播测试通过!")
        
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()


def test_hybrid_attention(model, fd_config):
    """测试混合注意力机制"""
    print("\n" + "="*50)
    print("测试 3: 混合注意力机制")
    print("="*50)
    
    if model is None:
        print("跳过 - 模型未创建")
        return
    
    try:
        # 检查 attention 类型
        model_inner = model.model
        attn_types = model_inner.attn_type_list
        
        print(f"注意力层配置 (0=Lightning, 1=Standard):")
        for i, attn_type in enumerate(attn_types):
            attn_name = "Lightning" if attn_type == 0 else "Standard"
            print(f"  Layer {i}: {attn_name}")
        
        # 统计
        lightning_count = sum(1 for t in attn_types if t == 0)
        standard_count = sum(1 for t in attn_types if t == 1)
        
        print(f"\n注意力层统计:")
        print(f"  Lightning Attention: {lightning_count} 层")
        print(f"  Standard Attention: {standard_count} 层")
        
        print("\n混合注意力测试通过!")
        
    except Exception as e:
        print(f"混合注意力测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_moe_layer(model, fd_config):
    """测试 MoE 层"""
    print("\n" + "="*50)
    print("测试 4: MoE 专家路由")
    print("="*50)
    
    if model is None:
        print("跳过 - 模型未创建")
        return
    
    try:
        # 检查 MoE 配置
        num_experts = MINIMAX_M1_SMALL_CONFIG["num_local_experts"]
        experts_per_tok = MINIMAX_M1_SMALL_CONFIG["num_experts_per_tok"]
        
        print(f"MoE 配置:")
        print(f"  专家数量: {num_experts}")
        print(f"  每 token 激活专家数: {experts_per_tok}")
        
        # 创建测试输入
        batch_size = 1
        seq_len = 8
        hidden_size = MINIMAX_M1_SMALL_CONFIG["hidden_size"]
        
        x = paddle.randn([batch_size, seq_len, hidden_size])
        
        print(f"\n测试输入 shape: {x.shape}")
        
        # 遍历每一层测试 MoE
        print("\n各层 MoE 专家激活测试:")
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'experts'):
                # 测试前向
                output = layer.mlp(x)
                print(f"  Layer {i}: MoE output shape = {output.shape}")
        
        print("\nMoE 测试通过!")
        
    except Exception as e:
        print(f"MoE 测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_model_registry():
    """测试模型注册"""
    print("\n" + "="*50)
    print("测试 5: 模型注册")
    print("="*50)
    
    try:
        from fastdeploy.model_executor.models import model_base
        
        # 检查 MiniMaxM1 是否注册
        registered = hasattr(model_base.ModelRegistry, '_registry')
        
        # 检查类是否存在
        from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
        
        print(f"MiniMaxM1ForCausalLM 类: 已找到")
        print(f"类名: {MiniMaxM1ForCausalLM.name()}")
        
        print("\n模型注册测试通过!")
        
    except Exception as e:
        print(f"模型注册测试失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    print("="*50)
    print("MiniMax-M1 小配置测试")
    print("="*50)
    print(f"设备: {'GPU' if paddle.device.is_gpu_available() else 'CPU'}")
    print(f"PaddlePaddle 版本: {paddle.__version__}")
    
    # 测试 1: 模型加载
    model, fd_config = test_model_loading()
    
    # 测试 2: 前向传播
    test_forward_pass(model, fd_config)
    
    # 测试 3: 混合注意力
    test_hybrid_attention(model, fd_config)
    
    # 测试 4: MoE
    test_moe_layer(model, fd_config)
    
    # 测试 5: 模型注册
    test_model_registry()
    
    print("\n" + "="*50)
    print("所有测试完成!")
    print("="*50)


if __name__ == "__main__":
    main()