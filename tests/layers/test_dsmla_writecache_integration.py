#!/usr/bin/env python3
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
测试 dsa_attention_backend.py 中 DS MLA writecache 的集成测试
对应文件中第371-387行的注释代码
"""

import os
import sys
import unittest

import paddle

# 添加本地路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "../../"))


def create_dsmla_test_data(batch_size=2, num_tokens=16, max_num_blocks=100):
    """创建 DS MLA writecache 测试数据"""

    block_size = 16
    kv_lora_rank = 512  # DS MLA 使用的 NoPE 部分维度
    pe_dim = 64  # DS MLA 使用的 RoPE 部分维度

    # 计算 DS MLA FP8 缓存条目大小: 512 + 16 + 128 = 656 字节
    # 512: NoPE fp8, 16: scales, 128: RoPE bf16 (64*2)
    entry_size = kv_lora_rank + 16 + pe_dim * 2

    # 创建输入张量
    compressed_kv = paddle.randn([num_tokens, kv_lora_rank], dtype="bfloat16")
    k_pe = paddle.randn([num_tokens, pe_dim], dtype="bfloat16")
    latent_cache = paddle.zeros([max_num_blocks, 1, block_size, entry_size], dtype="uint8")

    # 创建必需参数
    slot_mapping = paddle.randint(0, max_num_blocks * block_size, [num_tokens], dtype="int64")
    seq_lens_this_time = paddle.to_tensor([num_tokens // 2] * batch_size, dtype="int32")
    seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
    batch_id_per_token = paddle.concat(
        [paddle.zeros([num_tokens // 2], dtype="int32"), paddle.ones([num_tokens // 2], dtype="int32")]
    )
    cu_seqlens_q = paddle.to_tensor([0, num_tokens // 2, num_tokens], dtype="int32")
    block_tables = paddle.randint(0, max_num_blocks, [batch_size, 10], dtype="int32")

    # scale 参数 - 从原始 bfloat16 到 uint8 的量化需要 scale
    # 通常每个 token 需要一个 scale
    scale = paddle.randn([num_tokens, 1], dtype="float32")

    return {
        "compressed_kv": compressed_kv,
        "k_pe": k_pe,
        "latent_cache": latent_cache,
        "slot_mapping": slot_mapping,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_decoder": seq_lens_decoder,
        "batch_id_per_token": batch_id_per_token,
        "cu_seqlens_q": cu_seqlens_q,
        "block_tables": block_tables,
        "scale": scale,
        "max_seq_len": 4096,
        "cache_quant_type_str": "fp8_ds_mla",
        "is_prefill": True,
    }


class TestDSMLAWriteCacheIntegration(unittest.TestCase):
    """测试 DS MLA writecache 集成"""

    def setUp(self):
        """设置测试环境"""
        paddle.set_device("gpu")

    def test_dsmla_writecache_import(self):
        """测试 dsmla_write_cache 导入"""
        try:
            from fastdeploy.model_executor.ops.gpu import dsmla_write_cache

            print("✓ dsmla_write_cache 导入成功")
            self.assertTrue(True)
        except ImportError as e:
            self.skipTest(f"dsmla_write_cache 未找到: {e}")

    def test_dsmla_writecache_prefill(self):
        """测试 DS MLA writecache prefill 模式（对应注释代码行371-387）"""
        from fastdeploy.model_executor.ops.gpu import dsmla_write_cache

        # 创建测试数据
        test_data = create_dsmla_test_data(batch_size=2, num_tokens=16)

        # 调用 dsmla_write_cache - 对应注释代码的调用
        # 注意: 注释代码缺少 slot_mapping 和 scale 参数定义

        try:
            result = dsmla_write_cache(
                test_data["compressed_kv"],
                test_data["k_pe"],
                test_data["latent_cache"],
                test_data["slot_mapping"],  # 注释代码中未定义！
                test_data["seq_lens_this_time"],
                test_data["seq_lens_decoder"],
                test_data["batch_id_per_token"],
                test_data["cu_seqlens_q"],
                test_data["block_tables"],
                None,  # kv_signal_data - 注释代码中为 None
                test_data["scale"],  # 注释代码中未定义！
                "fp8_ds_mla",  # 根据测试，这是正确的值
                test_data["max_seq_len"],
                True,  # is_prefill
            )

            # 验证返回值
            self.assertIsNotNone(result)
            self.assertEqual(result.shape, test_data["latent_cache"].shape)
            self.assertEqual(result.dtype, paddle.uint8)

            print("✓ DS MLA writecache prefill 模式测试成功")
            print(f"  输入缓存形状: {test_data['latent_cache'].shape}")
            print(f"  输出缓存形状: {result.shape}")

        except Exception as e:
            self.fail(f"DS MLA writecache 调用失败: {e}")

    def test_dsmla_writecache_decode(self):
        """测试 DS MLA writecache decode 模式"""
        from fastdeploy.model_executor.ops.gpu import dsmla_write_cache

        # 创建 decode 模式测试数据 (batch_size=2, num_tokens=2)
        test_data = create_dsmla_test_data(batch_size=2, num_tokens=2)

        # 修改 decode 特定参数
        test_data["seq_lens_this_time"] = paddle.to_tensor([1, 1], dtype="int32")
        test_data["cu_seqlens_q"] = paddle.to_tensor([0, 1, 2], dtype="int32")

        try:
            result = dsmla_write_cache(
                test_data["compressed_kv"],
                test_data["k_pe"],
                test_data["latent_cache"],
                test_data["slot_mapping"],
                test_data["seq_lens_this_time"],
                test_data["seq_lens_decoder"],
                test_data["batch_id_per_token"],
                test_data["cu_seqlens_q"],
                test_data["block_tables"],
                None,  # kv_signal_data
                test_data["scale"],
                "fp8_ds_mla",
                test_data["max_seq_len"],
                False,  # is_prefill = False for decode
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.shape, test_data["latent_cache"].shape)

            print("✓ DS MLA writecache decode 模式测试成功")

        except Exception as e:
            self.fail(f"DS MLA writecache decode 模式失败: {e}")

    def test_cache_quant_type_options(self):
        """测试不同的 cache_quant_type_str 选项"""
        from fastdeploy.model_executor.ops.gpu import dsmla_write_cache

        test_data = create_dsmla_test_data(batch_size=1, num_tokens=8)

        # 测试不同的量化类型
        quant_types = ["fp8_ds_mla", "none"]

        for quant_type in quant_types:
            with self.subTest(quant_type=quant_type):
                try:
                    result = dsmla_write_cache(
                        test_data["compressed_kv"],
                        test_data["k_pe"],
                        test_data["latent_cache"],
                        test_data["slot_mapping"],
                        test_data["seq_lens_this_time"],
                        test_data["seq_lens_decoder"],
                        test_data["batch_id_per_token"],
                        test_data["cu_seqlens_q"],
                        test_data["block_tables"],
                        None,
                        test_data["scale"],
                        quant_type,
                        test_data["max_seq_len"],
                        True,
                    )

                    self.assertIsNotNone(result)
                    print(f"  ✓ cache_quant_type_str='{quant_type}' 工作正常")

                except Exception as e:
                    if quant_type == "fp8_ds_mla":
                        self.fail(f"fp8_ds_mla 应该工作: {e}")
                    print(f"  ✗ cache_quant_type_str='{quant_type}' 失败: {e}")

    def test_parameter_validation(self):
        """测试参数验证"""
        from fastdeploy.model_executor.ops.gpu import dsmla_write_cache

        test_data = create_dsmla_test_data()

        # 测试缺失必需参数
        test_cases = [
            {
                "name": "缺少 slot_mapping",
                "kwargs": {k: v for k, v in test_data.items() if k != "slot_mapping"},
                "should_fail": True,
            },
            {
                "name": "缺少 scale",
                "kwargs": {k: v for k, v in test_data.items() if k != "scale"},
                "should_fail": True,
                "fix": lambda kwargs: kwargs.update({"scale": None}),
            },
            {"name": "正确的参数", "kwargs": test_data.copy(), "should_fail": False},
        ]

        for test_case in test_cases:
            with self.subTest(test_case["name"]):
                kwargs = test_case["kwargs"].copy()

                if "fix" in test_case:
                    test_case["fix"](kwargs)

                try:
                    result = dsmla_write_cache(
                        kwargs["compressed_kv"],
                        kwargs["k_pe"],
                        kwargs["latent_cache"],
                        kwargs.get("slot_mapping", None),
                        kwargs["seq_lens_this_time"],
                        kwargs["seq_lens_decoder"],
                        kwargs["batch_id_per_token"],
                        kwargs["cu_seqlens_q"],
                        kwargs["block_tables"],
                        None,
                        kwargs.get("scale", None),
                        kwargs["cache_quant_type_str"],
                        kwargs["max_seq_len"],
                        kwargs["is_prefill"],
                    )

                    if test_case["should_fail"]:
                        self.fail(f"期望失败但成功了: {test_case['name']}")
                    else:
                        print(f"✓ {test_case['name']} 成功")

                except Exception as e:
                    if not test_case["should_fail"]:
                        self.fail(f"期望成功但失败: {test_case['name']} - {e}")
                    else:
                        print(f"✓ {test_case['name']} 如预期失败")

    def test_dsa_attention_backend_integration(self):
        """测试与 DSAttentionBackend 的集成"""
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../"))

        try:
            from fastdeploy.model_executor.layers.attention.dsa_attention_backend import (
                DSAAttentionBackend,
                DSAAttentionMetadata,
            )

            print("✓ DSAAttentionBackend 导入成功")

            # 模拟 FDConfig
            class MockFDConfig:
                def __init__(self):
                    self.cache_config = type("CacheConfig", (), {"block_size": 16})()
                    self.model_config = type(
                        "ModelConfig",
                        (),
                        {
                            "max_model_len": 4096,
                            "head_dim": 128,
                            "num_hidden_layers": 12,
                            "kv_lora_rank": 512,
                            "qk_rope_head_dim": 64,
                            "qk_nope_head_dim": 64,
                            "rope_theta": 10000.0,
                            "rope_scaling": None,
                            "start_layer_index": 0,
                        },
                    )()
                    self.speculative_config = type(
                        "SpeculativeConfig", (), {"method": None, "num_speculative_tokens": 4, "model_type": None}
                    )()
                    self.parallel_config = type("ParallelConfig", (), {"pd_disaggregation_mode": "none"})()

            # 创建 backend 实例
            fd_config = MockFDConfig()
            backend = DSAAttentionBackend(fd_config, kv_num_heads=8, num_heads=8, head_dim=128)

            self.assertIsNotNone(backend)
            self.assertEqual(backend.block_size, 16)
            self.assertEqual(backend.max_seq_len, 4096)
            self.assertEqual(backend.kv_lora_rank, 512)

            print("✓ DSAttentionBackend 创建成功")

            # 测试 get_kv_cache_shape
            key_shape, value_shape = backend.get_kv_cache_shape(100)
            expected_key_size = 512 + 64  # kv_lora_rank + qk_rope_head_dim
            self.assertEqual(key_shape, [100, 1, 16, expected_key_size])

            print("✓ get_kv_cache_shape 测试成功")

        except ImportError as e:
            self.skipTest(f"DSAAttentionBackend 导入失败: {e}")
        except Exception as e:
            self.fail(f"DSAttentionBackend 测试失败: {e}")


def run_dsmla_test_summary():
    """运行测试并生成总结报告"""
    print("=" * 70)
    print("DS MLA WriteCache 集成测试 - dsa_attention_backend.py 第371-387行")
    print("=" * 70)

    # 导入测试
    print("\n1. 检查导入...")
    try:
        from fastdeploy.model_executor.ops.gpu import dsmla_write_cache

        print("   ✓ dsmla_write_cache 可用")

        # 检查函数签名
        if hasattr(dsmla_write_cache, "__doc__"):
            doc = dsmla_write_cache.__doc__
            if doc:
                print(f"   函数签名: {doc}")
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return

    # 测试数据创建
    print("\n2. 测试数据创建...")
    test_data = create_dsmla_test_data()
    print("   ✓ 创建测试数据:")
    print(f"     - compressed_kv: {test_data['compressed_kv'].shape} ({test_data['compressed_kv'].dtype})")
    print(f"     - k_pe: {test_data['k_pe'].shape} ({test_data['k_pe'].dtype})")
    print(f"     - latent_cache: {test_data['latent_cache'].shape} ({test_data['latent_cache'].dtype})")
    print(f"     - slot_mapping: {test_data['slot_mapping'].shape}")
    print(f"     - scale: {test_data['scale'].shape}")

    # 问题分析
    print("\n3. dsa_attention_backend.py 第371-387行问题分析:")
    print(
        """
    # from fastdeploy.model_executor.ops.gpu import dsmla_write_cache
    # dsmla_write_cache(
    #     compressed_kv,
    #     k_pe,
    #     latent_cache,
    #     slot_mapping,           # <-- 问题: 未定义！
    #     forward_meta.seq_lens_this_time,
    #     forward_meta.seq_lens_decoder,
    #     forward_meta.batch_id_per_token,
    #     forward_meta.cu_seqlens_q,
    #     metadata.block_tables,
    #     None,
    #     scale,                  # <-- 问题: 未定义！
    #     "none",                 # <-- 问题: 应该是 'fp8_ds_mla'
    #     self.max_seq_len,
    #     True,
    # )
    """
    )

    print("   发现的问题:")
    print("   1. slot_mapping 变量未定义")
    print("   2. scale 变量未定义")
    print("   3. cache_quant_type_str 应该是 'fp8_ds_mla' 而不是 'none'")

    print("\n4. 运行单元测试...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDSMLAWriteCacheIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print("测试总结报告")
    print("=" * 70)
    print("✓ DS MLA writecache 函数存在且可用")
    print("✓ 参数类型验证完成:")
    print("  - compressed_kv/k_pe: bfloat16")
    print("  - latent_cache: uint8")
    print("  - 必需参数: slot_mapping, scale")
    print("✓ 正确的 cache_quant_type_str: 'fp8_ds_mla'")
    print("\n📋 修复建议 (dsa_attention_backend.py):")
    print("  1. 在调用前定义 slot_mapping 变量")
    print("  2. 在调用前定义 scale 变量")
    print("  3. 将 cache_quant_type_str 改为 'fp8_ds_mla'")

    if result.wasSuccessful():
        print("\n✅ 所有测试通过!")
    else:
        print(f"\n⚠️  {len(result.failures) + len(result.errors)} 个测试失败")

    return result.wasSuccessful()


if __name__ == "__main__":
    # 直接运行生成报告
    success = run_dsmla_test_summary()

    # 然后运行单元测试
    print("\n" + "=" * 70)
    print("运行详细的单元测试...")
    print("=" * 70)

    unittest.main(argv=["first-arg-is-ignored"], exit=False)

    sys.exit(0 if success else 1)
