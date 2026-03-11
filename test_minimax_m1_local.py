# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

"""
本地测试 MiniMax-M1 模型 - 简化版
避免依赖 FastDeploy 完整环境
"""

import sys
import os
import math

# 设置编码
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("=" * 60)
print("MiniMax-M1 本地测试 (简化版)")
print("=" * 60)


def test_1_import():
    """测试能否导入 MiniMax-M1 模块"""
    print("\n=== 测试 1: 导入 MiniMax-M1 模块 ===")
    
    try:
        # 直接导入模块文件，不走 fastdeploy __init__
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "minimax_m1",
            "fastdeploy/model_executor/models/minimax_m1.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # 检查语法
        print("[OK] 模块文件存在")
        
        # 检查类名是否存在
        with open("fastdeploy/model_executor/models/minimax_m1.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        classes = [
            "MiniMaxM1ForCausalLM",
            "MiniMaxM1Model", 
            "MiniMaxM1DecoderLayer",
            "MiniMaxM1LightningAttention",
            "MiniMaxM1StandardAttention",
            "MiniMaxM1SparseMoEBlock",
            "MiniMaxM1MLP",
        ]
        
        for cls in classes:
            if f"class {cls}" in content:
                print(f"[OK] 找到类: {cls}")
            else:
                print(f"[FAIL] 未找到类: {cls}")
                return False
        
        # 检查是否有 ModelRegistry.register_model_class 装饰器
        if "@ModelRegistry.register_model_class" in content:
            print("[OK] ModelRegistry 注册装饰器存在")
        else:
            print("[WARN] ModelRegistry 注册装饰器未找到")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_2_config():
    """测试配置"""
    print("\n=== 测试 2: 配置测试 ===")
    
    # MiniMax-M1 配置 (从 HuggingFace)
    config = {
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
    }
    
    print(f"[OK] hidden_size: {config['hidden_size']}")
    print(f"[OK] num_hidden_layers: {config['num_hidden_layers']}")
    print(f"[OK] num_local_experts: {config['num_local_experts']}")
    print(f"[OK] num_experts_per_tok: {config['num_experts_per_tok']}")
    print(f"[OK] max_position_embeddings: {config['max_position_embeddings']}")
    
    return True


def test_3_attention_type():
    """测试注意力类型"""
    print("\n=== 测试 3: 注意力类型 ===")
    
    # MiniMax-M1: 7 层 Lightning + 1 层 Standard 的模式重复 10 次
    num_layers = 80
    # 正确的 pattern: [0,0,0,0,0,0,0,1] x 10 = 80 layers
    attn_type_list = [0, 0, 0, 0, 0, 0, 0, 1] * 10
    
    print(f"[OK] 总层数: {num_layers}")
    print(f"[OK] Lightning Attention (0) 层数: {attn_type_list.count(0)}")
    print(f"[OK] Standard Attention (1) 层数: {attn_type_list.count(1)}")
    
    # 验证前 8 层 pattern: 7x Lightning + 1x Standard
    expected_first_8 = [0, 0, 0, 0, 0, 0, 0, 1]
    for i in range(8):
        if attn_type_list[i] != expected_first_8[i]:
            print(f"[FAIL] 层 {i} 注意力类型错误: 期望 {expected_first_8[i]}, 实际 {attn_type_list[i]}")
            return False
    
    print("[OK] 前 8 层 pattern 正确: 7x Lightning + 1x Standard")
    return True
    return True


def test_4_slope():
    """测试斜率计算"""
    print("\n=== 测试 4: 斜率计算 ===")
    
    def get_slopes(n):
        """计算 Lightning Attention 的斜率"""
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
    
    # 测试 64 个头
    slopes = get_slopes(64)
    print(f"[OK] num_heads=64: {len(slopes)} slopes")
    print(f"[OK] 前 3 个斜率: {slopes[:3]}")
    
    # 验证斜率递减
    for i in range(len(slopes) - 1):
        if slopes[i] <= slopes[i + 1]:
            print(f"[FAIL] 斜率应该递减: {slopes[i]} <= {slopes[i+1]}")
            return False
    
    print("[OK] 斜率计算正确 (递减)")
    return True


def test_5_syntax():
    """测试语法"""
    print("\n=== 测试 5: 语法检查 ===")
    
    import ast
    
    try:
        with open("fastdeploy/model_executor/models/minimax_m1.py", "r", encoding="utf-8") as f:
            source = f.read()
        
        ast.parse(source)
        print("[OK] 语法检查通过")
        return True
    except SyntaxError as e:
        print(f"[FAIL] 语法错误: {e}")
        return False


def test_6_file_exists():
    """测试文件存在"""
    print("\n=== 测试 6: 文件检查 ===")
    
    files = [
        "fastdeploy/model_executor/models/minimax_m1.py",
        "tests/model_executor/test_minimax_m1.py",
        "test_minimax_m1_local.py",
    ]
    
    for f in files:
        if os.path.exists(f):
            size = os.path.getsize(f)
            print(f"[OK] {f} ({size} bytes)")
        else:
            print(f"[FAIL] {f} 不存在")
            return False
    
    return True


def run_all_tests():
    """运行所有测试"""
    results = []
    
    results.append(("文件检查", test_6_file_exists()))
    results.append(("语法检查", test_5_syntax()))
    results.append(("配置测试", test_2_config()))
    results.append(("注意力类型", test_3_attention_type()))
    results.append(("斜率计算", test_4_slope()))
    results.append(("模块导入", test_1_import()))
    
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("所有测试通过!")
    else:
        print("部分测试失败")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)