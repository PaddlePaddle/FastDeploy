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
测试 Flash Attention V2 和 V3 的确定性

检查不同 Flash Attention backend 的确定性表现：
- FA2 (Flash Attention V2)
- FA3 (Flash Attention V3)
"""

import unittest

import paddle
import paddle.nn.functional as F

print("=" * 70)
print(" FLASH ATTENTION V2/V3 DETERMINISM TEST")
print("=" * 70)
print()

# 基础配置
BATCH_SIZE = 2
NUM_HEADS = 32
HEAD_DIM = 64
SEQ_LEN = 2048


class TestFlashAttentionDeterminism(unittest.TestCase):
    """测试 Flash Attention 的确定性"""

    def setUp(self):
        """Set up test environment and save current flash attn version"""
        self.device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.set_device(self.device)

        # Save current flash attn version to restore later
        self._saved_flash_attn_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
            "FLAGS_flash_attn_version"
        ]

    def tearDown(self):
        """Restore original flash attn version after each test"""
        paddle.set_flags({"FLAGS_flash_attn_version": self._saved_flash_attn_version})

    def test_fa2_determinism(self):
        """测试 Flash Attention V2 的确定性"""
        print("\n[1] Flash Attention V2 确定性测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        # 设置 Flash Attention V2
        paddle.set_flags({"FLAGS_flash_attn_version": 2})

        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        key = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        value = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        results = []
        for i in range(5):
            paddle.device.synchronize()
            result = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )
            results.append(result.clone())

        # 使用完全相等检查
        all_equal = True
        for i in range(1, 5):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  结果: FA2 是确定性的（完全相等）")
        else:
            print("  结果: FA2 不是完全确定性的")

        self.assertTrue(all_equal, "FA2 results are not equal")

    def test_fa3_determinism(self):
        """测试 Flash Attention V3 的确定性"""
        print("\n[2] Flash Attention V3 确定性测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        # 检查 GPU 架构是否支持 FA3
        prop = paddle.device.cuda.get_device_properties()
        sm_version = prop.major * 10 + prop.minor

        if sm_version < 89 or sm_version >= 100:
            self.skipTest(f"Flash Attention V3 requires SM89+ but SM100-. Current: SM{sm_version}")

        # 设置 Flash Attention V3
        paddle.set_flags({"FLAGS_flash_attn_version": 3})

        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        key = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        value = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        results = []
        for i in range(5):
            paddle.device.synchronize()
            result = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )
            results.append(result.clone())

        # 使用完全相等检查
        all_equal = True
        for i in range(1, 5):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  结果: FA3 是确定性的（完全相等）")
        else:
            print("  结果: FA3 不是完全确定性的")

        self.assertTrue(all_equal, "FA3 results are not equal")

    def test_fa2_different_batch_sizes(self):
        """测试 FA2 不同批大小的确定性（批量不变性）

        验证同一条 request 的 q,k,v 在不同 batch size 下计算，结果仍然一致。
        """
        print("\n[3] FA2 不同批大小批量不变性测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        paddle.set_flags({"FLAGS_flash_attn_version": 2})

        # 生成一份固定的数据（最大 batch size）
        MAX_BATCH_SIZE = 8
        paddle.seed(42)
        full_query = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        full_key = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        full_value = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        # 计算不同 batch size 下的结果，比较第 0 个 request 的结果
        reference_result = None
        batch_sizes = [1, 2, 4, 8]
        all_equal = True

        for bs in batch_sizes:
            # 取前 bs 个 request 的数据
            query = full_query[:bs]
            key = full_key[:bs]
            value = full_value[:bs]

            result = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )

            # 取第 0 个 request 的结果
            current_result = result[0:1]

            if reference_result is None:
                reference_result = current_result.clone()
                is_equal = True
            else:
                is_equal = paddle.equal(reference_result, current_result).all().item()

            print(f"  Batch {bs}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  结果: FA2 是批量不变的（同一 request 在不同 batch 下结果一致）")
        else:
            print("  结果: FA2 不是批量不变的（不同 batch 下同一 request 结果不同）")

    def test_fa3_different_batch_sizes(self):
        """测试 FA3 不同批大小的确定性（批量不变性）

        验证同一条 request 的 q,k,v 在不同 batch size 下计算，结果仍然一致。
        """
        print("\n[4] FA3 不同批大小批量不变性测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        prop = paddle.device.cuda.get_device_properties()
        sm_version = prop.major * 10 + prop.minor

        if sm_version < 89 or sm_version >= 100:
            self.skipTest(f"Flash Attention V3 requires SM89+ but SM100-. Current: SM{sm_version}")

        paddle.set_flags({"FLAGS_flash_attn_version": 3})

        # 生成一份固定的数据（最大 batch size）
        MAX_BATCH_SIZE = 8
        paddle.seed(42)
        full_query = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        full_key = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        full_value = paddle.randn([MAX_BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        # 计算不同 batch size 下的结果，比较第 0 个 request 的结果
        reference_result = None
        batch_sizes = [1, 2, 4, 8]
        all_equal = True

        for bs in batch_sizes:
            # 取前 bs 个 request 的数据
            query = full_query[:bs]
            key = full_key[:bs]
            value = full_value[:bs]

            result = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )

            # 取第 0 个 request 的结果
            current_result = result[0:1]

            if reference_result is None:
                reference_result = current_result.clone()
                is_equal = True
            else:
                is_equal = paddle.equal(reference_result, current_result).all().item()

            print(f"  Batch {bs}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  结果: FA3 是批量不变的（同一 request 在不同 batch 下结果一致）")
        else:
            print("  结果: FA3 不是批量不变的（不同 batch 下同一 request 结果不同）")

    def test_fa2_different_sequence_lengths(self):
        """测试 FA2 不同序列长度的确定性"""
        print("\n[5] FA2 不同序列长度测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        paddle.set_flags({"FLAGS_flash_attn_version": 2})

        seq_lengths = [16, 32, 64, 128, 256, 512]
        all_deterministic = []

        for seq_len in seq_lengths:
            paddle.seed(42)
            query = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype="float16")
            key = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype="float16")
            value = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype="float16")

            result1 = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )
            result2 = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )

            is_equal = paddle.equal(result1, result2).all().item()
            all_deterministic.append(is_equal)
            print(f"  Seq_len={seq_len:4d}: equal={is_equal}")

        all_equal = all(all_deterministic)
        if all_equal:
            print("  结果: FA2 所有序列长度都是确定性的")
        else:
            print("  结果: FA2 部分序列长度不确定")

    def test_fa3_different_sequence_lengths(self):
        """测试 FA3 不同序列长度的确定性"""
        print("\n[6] FA3 不同序列长度测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        prop = paddle.device.cuda.get_device_properties()
        sm_version = prop.major * 10 + prop.minor

        if sm_version < 89 or sm_version >= 100:
            self.skipTest(f"Flash Attention V3 requires SM89+ but SM100-. Current: SM{sm_version}")

        paddle.set_flags({"FLAGS_flash_attn_version": 3})

        seq_lengths = [16, 32, 64, 128, 256, 512]
        all_deterministic = []

        for seq_len in seq_lengths:
            paddle.seed(42)
            query = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype="float16")
            key = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype="float16")
            value = paddle.randn([BATCH_SIZE, NUM_HEADS, seq_len, HEAD_DIM], dtype="float16")

            result1 = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )
            result2 = F.scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False, backend="flash"
            )

            is_equal = paddle.equal(result1, result2).all().item()
            all_deterministic.append(is_equal)
            print(f"  Seq_len={seq_len:4d}: equal={is_equal}")

        all_equal = all(all_deterministic)
        if all_equal:
            print("  结果: FA3 所有序列长度都是确定性的")
        else:
            print("  结果: FA3 部分序列长度不确定")

    def test_fa2_vs_fa3_comparison(self):
        """比较 FA2 和 FA3 的结果（仅用于了解差异，非确定性测试）"""
        print("\n[7] FA2 vs FA3 结果对比")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        prop = paddle.device.cuda.get_device_properties()
        sm_version = prop.major * 10 + prop.minor

        if sm_version < 89 or sm_version >= 100:
            self.skipTest(f"Flash Attention V3 requires SM89+ but SM100-. Current: SM{sm_version}")

        # 相同输入
        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        key = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        value = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        # FA2
        paddle.set_flags({"FLAGS_flash_attn_version": 2})
        result_fa2 = F.scaled_dot_product_attention(
            query, key, value, is_causal=False, enable_gqa=False, backend="flash"
        )

        # FA3
        paddle.set_flags({"FLAGS_flash_attn_version": 3})
        result_fa3 = F.scaled_dot_product_attention(
            query, key, value, is_causal=False, enable_gqa=False, backend="flash"
        )

        is_equal = paddle.equal(result_fa2, result_fa3).all().item()
        max_diff = paddle.abs(result_fa2.cast("float32") - result_fa3.cast("float32")).max().item()

        print(f"  FA2 vs FA3 equal: {is_equal}")
        print(f"  FA2 vs FA3 max_diff: {max_diff:.2e}")

        if is_equal:
            print("  结果: FA2 和 FA3 产生完全相同的输出")
        else:
            print("  结果: FA2 和 FA3 输出不同（预期行为，不同实现可能精度不同）")

    def test_fa2_causal_mask(self):
        """测试 FA2 因果掩码模式的确定性"""
        print("\n[8] FA2 因果掩码测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        paddle.set_flags({"FLAGS_flash_attn_version": 2})

        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        key = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        value = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        results = []
        for i in range(5):
            paddle.device.synchronize()
            result = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=False, backend="flash"
            )
            results.append(result.clone())

        all_equal = True
        for i in range(1, 5):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  结果: FA2 因果掩码模式是确定性的")
        else:
            print("  结果: FA2 因果掩码模式不是确定性的")

        self.assertTrue(all_equal, "FA2 causal mode not deterministic")

    def test_fa3_causal_mask(self):
        """测试 FA3 因果掩码模式的确定性"""
        print("\n[9] FA3 因果掩码测试")
        print("-" * 70)

        if not paddle.is_compiled_with_cuda():
            self.skipTest("Flash Attention requires CUDA")

        prop = paddle.device.cuda.get_device_properties()
        sm_version = prop.major * 10 + prop.minor

        if sm_version < 89 or sm_version >= 100:
            self.skipTest(f"Flash Attention V3 requires SM89+ but SM100-. Current: SM{sm_version}")

        paddle.set_flags({"FLAGS_flash_attn_version": 3})

        paddle.seed(42)
        query = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        key = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")
        value = paddle.randn([BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM], dtype="float16")

        results = []
        for i in range(5):
            paddle.device.synchronize()
            result = F.scaled_dot_product_attention(
                query, key, value, is_causal=True, enable_gqa=False, backend="flash"
            )
            results.append(result.clone())

        all_equal = True
        for i in range(1, 5):
            is_equal = paddle.equal(results[0], results[i]).all().item()
            print(f"  Run 1 vs Run {i+1}: equal={is_equal}")
            if not is_equal:
                all_equal = False

        if all_equal:
            print("  结果: FA3 因果掩码模式是确定性的")
        else:
            print("  结果: FA3 因果掩码模式不是确定性的")

        self.assertTrue(all_equal, "FA3 causal mode not deterministic")


if __name__ == "__main__":
    # 运行测试
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFlashAttentionDeterminism)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print()
    print("=" * 70)
    print(" 测试结果总结")
    print("=" * 70)
    print(f"  通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print(f"  跳过: {len(result.skipped)}")
    print(f"  总计: {result.testsRun}")
    print()

    if result.wasSuccessful():
        print(" ✓ 所有运行测试通过！")
    else:
        print(" ✗ 有测试失败或错误")

    print("=" * 70)
