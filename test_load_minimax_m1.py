# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
使用 FastDeploy 加载 MiniMax-M1 模型

使用方法:
    python test_load_minimax_m1.py

注意: 
    - 需要先安装 FastDeploy 依赖: pip install fastapi uvicorn
    - 需要有 MiniMax-M1 模型权重
    - Windows 下需要设置 TEMP 环境变量
"""

import os
import sys

# Windows 设置 TEMP
if sys.platform == 'win32':
    os.environ['TEMP'] = os.environ.get('TEMP', 'C:\\Temp')
    os.makedirs(os.environ['TEMP'], exist_ok=True)

# 设置 PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("使用 FastDeploy 加载 MiniMax-M1 模型")
print("=" * 60)


def test_load_with_fdconfig():
    """测试用 FDConfig 加载模型"""
    print("\n=== 测试: 使用 FDConfig 加载模型 ===")
    
    try:
        # 尝试导入 FastDeploy 核心模块
        print("[1] 导入 FastDeploy 模块...")
        from fastdeploy.config import FDConfig, ModelConfig, LoadConfig
        print("    [OK] FDConfig 导入成功")
        
        # 检查模型注册
        print("\n[2] 检查模型注册...")
        from fastdeploy.model_executor.models.model_base import ModelRegistry
        
        # 尝试获取 MiniMax-M1 类
        try:
            model_cls = ModelRegistry.get_class("MiniMaxM1ForCausalLM")
            print(f"    [OK] 找到模型类: {model_cls.__name__}")
        except Exception as e:
            print(f"    [WARN] ModelRegistry 中未找到: {e}")
            print("    尝试直接导入...")
            from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
            print(f"    [OK] 直接导入成功: {MiniMaxM1ForCausalLM.__name__}")
        
        # 创建测试配置
        print("\n[3] 创建测试配置...")
        model_config = ModelConfig(
            model="/path/to/minimax-m1-model",  # 需要替换为实际路径
            architectures=["MiniMaxM1ForCausalLM"],
            hidden_size=6144,
            intermediate_size=9216,
            num_hidden_layers=2,  # 减少用于测试
            num_attention_heads=64,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=200064,
            model_type="minimax_m1",
        )
        print(f"    [OK] ModelConfig 创建成功")
        print(f"         hidden_size: {model_config.hidden_size}")
        print(f"         num_hidden_layers: {model_config.num_hidden_layers}")
        print(f"         model_type: {model_config.model_type}")
        
        return True
        
    except ImportError as e:
        print(f"    [ERROR] 导入失败: {e}")
        print("\n    请先安装依赖:")
        print("    pip install fastapi uvicorn paddlepaddle")
        return False
    except Exception as e:
        print(f"    [ERROR] {type(e).__name__}: {e}")
        return False


def test_model_path():
    """测试模型路径配置"""
    print("\n=== 测试: 模型路径配置 ===")
    
    # MiniMax-M1 模型可能的路径
    possible_paths = [
        # 本地路径
        "D:/models/minimax-m1",
        "D:/models/MiniMax-M1-80k",
        "C:/models/minimax-m1",
        "./minimax-m1-model",
        
        # HuggingFace 风格
        "MiniMaxAI/MiniMax-M1-80k",
        "minimaxai/MiniMax-M1",
    ]
    
    print("可能的模型路径:")
    for path in possible_paths:
        exists = os.path.exists(path) if not path.startswith("MiniMaxAI") else "[HuggingFace]"
        print(f"  - {path} ({'存在' if exists else exists})")
    
    # 检查环境变量
    print("\n环境变量中的模型路径:")
    env_model_path = os.environ.get("MODEL_PATH", "")
    if env_model_path:
        print(f"  MODEL_PATH: {env_model_path}")
    else:
        print("  MODEL_PATH: 未设置")


def create_sample_config():
    """创建示例配置文件"""
    print("\n=== 创建示例 config.json ===")
    
    config_json = {
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
        "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1] * 10,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "postnorm": True,
    }
    
    import json
    
    # 保存到文件
    config_path = "minimax_m1_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_json, f, indent=2)
    
    print(f"[OK] 示例配置已保存到: {config_path}")
    print("\n要使用此配置加载模型:")
    print(f"  1. 下载 MiniMax-M1 模型权重")
    print(f"  2. 将 config.json 放入模型目录")
    print(f"  3. 使用 FastDeploy 加载:")
    print("     from fastdeploy import FastDeployEngine")
    print(f"     engine = FastDeployEngine(model='{config_path}')")


def main():
    """主函数"""
    print("\n步骤 1: 测试 FastDeploy 模块加载")
    success1 = test_load_with_fdconfig()
    
    print("\n步骤 2: 检查模型路径")
    test_model_path()
    
    print("\n步骤 3: 创建示例配置")
    create_sample_config()
    
    print("\n" + "=" * 60)
    if success1:
        print("FastDeploy 模块加载成功!")
    else:
        print("需要安装 FastDeploy 依赖")
    print("=" * 60)


if __name__ == "__main__":
    main()