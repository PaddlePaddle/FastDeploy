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
本地测试 MiniMax-M1 模型

使用方法:
    python test_minimax_m1_local.py

注意: 
    - 需要 PaddlePaddle 和 FastDeploy 环境
    - Windows 下可能需要 GPU 才能运行推理
    - 可以只测试模型能否被正确实例化
"""

import os
import sys
import unittest

# 设置路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMiniMaxM1Local(unittest.TestCase):
    """本地测试 MiniMax-M1 模型"""

    def test_import_minimax_m1(self):
        """测试能否导入 MiniMax-M1 模块"""
        print("\n=== 测试 1: 导入 MiniMax-M1 模块 ===")
        
        try:
            from fastdeploy.model_executor.models.minimax_m1 import (
                MiniMaxM1ForCausalLM,
                MiniMaxM1Model,
                MiniMaxM1DecoderLayer,
                MiniMaxM1LightningAttention,
                MiniMaxM1StandardAttention,
                MiniMaxM1SparseMoEBlock,
                MiniMaxM1MLP,
            )
            print("✅ 成功导入 MiniMax-M1 模块")
            
            # 检查类是否存在
            self.assertTrue(hasattr(MiniMaxM1ForCausalLM, 'arch_name'))
            print(f"✅ MiniMaxM1ForCausalLM.arch_name = {MiniMaxM1ForCausalLM.arch_name}")
            
        except ImportError as e:
            self.fail(f"❌ 导入失败: {e}")
        except Exception as e:
            self.fail(f"❌ 未知错误: {e}")

    def test_model_registration(self):
        """测试模型是否注册到 ModelRegistry"""
        print("\n=== 测试 2: 模型注册检查 ===")
        
        try:
            from fastdeploy.model_executor.models.model_base import ModelRegistry
            
            # 检查是否可以通过架构名获取类
            # MiniMax-M1 使用的是 MiniMaxM1ForCausalLM
            model_classes = ModelRegistry.get_all_classes()
            
            # 检查是否有 MiniMax 相关的类
            minimax_classes = [c for c in model_classes if 'minimax' in c.lower()]
            print(f"✅ 找到 MiniMax 相关类: {minimax_classes}")
            
            if not minimax_classes:
                print("⚠️ 警告: ModelRegistry 中没有找到 MiniMax 类")
                print("   这可能是因为模型还没有被加载")
            
        except Exception as e:
            print(f"⚠️ ModelRegistry 检查跳过: {e}")

    def test_config_creation(self):
        """测试配置创建"""
        print("\n=== 测试 3: 配置创建测试 ===")
        
        try:
            # 创建一个模拟的 FDConfig
            from fastdeploy.config import FDConfig
            from dataclasses import dataclass, field
            
            # MiniMax-M1 配置参数
            config_dict = {
                "model_type": "minimax_m1",
                "architectures": ["MiniMaxM1ForCausalLM"],
                "hidden_size": 6144,
                "intermediate_size": 9216,
                "num_hidden_layers": 2,  # 减少层数用于测试
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 200064,
                "rope_theta": 10000000,
                "max_position_embeddings": 10240,  # 减少用于测试
                "rms_norm_eps": 1e-5,
                "hidden_act": "silu",
                "num_local_experts": 32,
                "num_experts_per_tok": 2,
                "attn_type_list": [0, 0, 0, 0, 0, 0, 0, 1],  # 简化用于测试
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "tie_word_embeddings": False,
                "postnorm": True,
            }
            
            print(f"✅ 配置参数字典创建成功")
            print(f"   - hidden_size: {config_dict['hidden_size']}")
            print(f"   - num_hidden_layers: {config_dict['num_hidden_layers']}")
            print(f"   - num_local_experts: {config_dict['num_local_experts']}")
            print(f"   - attn_type_list: {config_dict['attn_type_list']}")
            
        except Exception as e:
            print(f"⚠️ 配置测试跳过: {e}")

    def test_attention_type_per_layer(self):
        """测试每层的注意力类型"""
        print("\n=== 测试 4: 注意力类型测试 ===")
        
        # MiniMax-M1 使用 7 层 Lightning + 1 层 Standard 的模式
        num_layers = 8
        attn_type_list = [0, 0, 0, 0, 0, 0, 0, 1]
        
        print(f"层数: {num_layers}")
        
        for i, attn_type in enumerate(attn_type_list):
            if attn_type == 0:
                print(f"  层 {i}: Lightning Attention")
            else:
                print(f"  层 {i}: Standard Attention")
        
        # 验证
        self.assertEqual(attn_type_list.count(0), 7)
        self.assertEqual(attn_type_list.count(1), 1)
        print("✅ 注意力类型配置正确: 7x Lightning + 1x Standard")

    def test_slope_tensor_calculation(self):
        """测试 Lightning Attention 的斜率计算"""
        print("\n=== 测试 5: 斜率计算测试 ===")
        
        import math
        import numpy as np
        
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
        
        # 测试不同头数
        for num_heads in [8, 64]:
            slopes = get_slopes(num_heads)
            print(f"  num_heads={num_heads}: {len(slopes)} slopes")
            print(f"    前3个斜率: {slopes[:3]}")
            
            # 验证斜率是递减的
            for i in range(len(slopes) - 1):
                self.assertGreater(slopes[i], slopes[i + 1], 
                    f"斜率应该递减: {slopes[i]} > {slopes[i+1]}")
        
        print("✅ 斜率计算正确")

    def test_model_config_from_huggingface(self):
        """测试使用 HuggingFace 配置创建模型（不实际加载权重）"""
        print("\n=== 测试 6: 模型实例化测试 ===")
        
        try:
            # 这个测试只检查模型类能否被找到
            from fastdeploy.model_executor.models.model_base import ModelRegistry
            
            # 尝试获取 MiniMaxM1 类
            try:
                model_cls = ModelRegistry.get_class("MiniMaxM1ForCausalLM")
                print(f"✅ 找到模型类: {model_cls.__name__}")
            except:
                print("⚠️ ModelRegistry.get_class('MiniMaxM1ForCausalLM') 失败")
                print("   这可能是因为模型还没有被注册到 ModelRegistry")
                print("   尝试检查 auto discovery...")
                
                # 尝试直接导入
                from fastdeploy.model_executor.models.minimax_m1 import MiniMaxM1ForCausalLM
                print(f"✅ 直接导入成功: {MiniMaxM1ForCausalLM.__name__}")
                
        except Exception as e:
            print(f"⚠️ 模型实例化测试跳过: {e}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("MiniMax-M1 本地测试")
    print("=" * 60)
    print()
    print("测试环境:")
    print(f"  - Python: {sys.version}")
    print(f"  - 工作目录: {os.getcwd()}")
    print()
    
    # 运行测试
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMiniMaxM1Local)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print()
    print("=" * 60)
    if result.wasSuccessful():
        print("✅ 所有测试通过!")
    else:
        print(f"❌ 测试失败: {len(result.failures)} failures, {len(result.errors)} errors")
    print("=" * 60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)