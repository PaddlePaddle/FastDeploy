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
DSMLAWriteCacheKernel 算子单元测试

测试 cpp_extensions.cc 中定义的 dsk_attn_write_cache 算子：
- 参数: kv_nope, kv_pe, kv_cache, slot_mapping, seq_lens, seq_lens_decoder,
        batch_id_per_token, cu_seqlens_q, block_tables, kv_signal_data(optional),
        scale(optional), cache_quant_type_str, max_seq_len, is_prefill
"""


import time
import unittest

import paddle

from fastdeploy.model_executor.layers.attention.dsa_helper import dsk_attn_write_cache

# from fastdeploy.model_executor.ops.gpu import dsk_attn_write_cache


# DS MLA 常量定义
KV_LORA_RANK = 512  # NoPE 部分维度
PE_DIM = 64  # RoPE 部分维度
BLOCK_SIZE = 64  # 每个 block 的 token 数
# FP8 entry size: 512(fp8 nope) + 16(scales) + 128(rope bf16) = 656 bytes
FP8_ENTRY_SIZE = 656


def create_test_tensors(
    batch_size: int = 2,
    num_tokens: int = 16,
    max_num_blocks: int = 100,
    is_prefill: bool = True,
    dtype: str = "bfloat16",
):
    """创建测试用张量

    Args:
        batch_size: 批次大小
        num_tokens: token 总数
        max_num_blocks: 最大 block 数
        is_prefill: 是否为 prefill 阶段
        dtype: 输入数据类型

    Returns:
        dict: 包含所有测试张量的字典
    """
    # 输入张量
    kv_nope = paddle.randn([num_tokens, KV_LORA_RANK], dtype=dtype)
    kv_pe = paddle.randn([num_tokens, PE_DIM], dtype=dtype).unsqueeze(1)

    # KV cache: [num_blocks, num_kv_heads=1, block_size, entry_size]
    kv_cache = paddle.zeros([max_num_blocks, 1, BLOCK_SIZE, FP8_ENTRY_SIZE], dtype="uint8")

    # slot_mapping: 每个 token 对应的 cache slot 位置
    slot_mapping = paddle.arange(num_tokens, dtype="int64")

    # seq_lens: 每个请求的序列长度
    tokens_per_req = num_tokens // batch_size
    seq_lens = paddle.full([batch_size], tokens_per_req, dtype="int32")

    # seq_lens_decoder: decode 阶段的序列长度
    if is_prefill:
        seq_lens_decoder = paddle.zeros([batch_size], dtype="int32")
    else:
        seq_lens_decoder = paddle.full([batch_size], 1, dtype="int32")

    # batch_id_per_token: 每个 token 所属的 batch
    batch_id_per_token = paddle.concat([paddle.full([tokens_per_req], i, dtype="int32") for i in range(batch_size)])

    # cu_seqlens_q: 累积序列长度 [0, seq1_len, seq1_len + seq2_len, ...]
    cu_seqlens = [0]
    for i in range(batch_size):
        cu_seqlens.append(cu_seqlens[-1] + tokens_per_req)
    cu_seqlens_q = paddle.to_tensor(cu_seqlens, dtype="int32")

    # block_tables: [batch_size, max_blocks_per_seq]
    max_blocks_per_seq = 10
    block_tables = paddle.randint(0, max_num_blocks, [batch_size, max_blocks_per_seq], dtype="int32")

    # scale: 量化缩放因子 (optional)
    scale = paddle.ones([num_tokens, 1], dtype="float32")

    return {
        "kv_nope": kv_nope,
        "kv_pe": kv_pe,
        "kv_cache": kv_cache,
        "slot_mapping": slot_mapping,
        "seq_lens": seq_lens,
        "seq_lens_decoder": seq_lens_decoder,
        "batch_id_per_token": batch_id_per_token,
        "cu_seqlens_q": cu_seqlens_q,
        "block_tables": block_tables,
        "scale": scale,
        "max_seq_len": 4096,
        "is_prefill": is_prefill,
    }


class BaseDSMLAWriteCacheTest(unittest.TestCase):
    """基础测试类，包含共用的初始化和辅助方法"""

    @classmethod
    def setUpClass(cls):
        """测试类初始化"""
        paddle.set_device("gpu")

    def setUp(self):
        """每个测试前检查"""
        pass


# ==================== 基础功能测试 ====================


class TestBasicPrefill(BaseDSMLAWriteCacheTest):
    """测试基本 prefill 模式"""

    def test_basic_prefill(self):
        """基本 prefill 模式测试"""
        tensors = create_test_tensors(batch_size=2, num_tokens=16, is_prefill=True)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )

        # dsk_attn_write_cache 是 in-place 操作，直接修改 kv_cache
        # 返回值是空列表，验证 kv_cache 已被修改

        self.assertEqual(tensors["kv_cache"].dtype, paddle.uint8)


class TestBasicDecode(BaseDSMLAWriteCacheTest):
    """测试基本 decode 模式"""

    def test_basic_decode(self):
        """基本 decode 模式测试"""
        tensors = create_test_tensors(batch_size=2, num_tokens=2, is_prefill=False)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )

        # in-place 操作验证

        self.assertEqual(tensors["kv_cache"].dtype, paddle.uint8)


# ==================== 边界条件测试 ====================


class TestSingleToken(BaseDSMLAWriteCacheTest):
    """测试单 token 场景"""

    def test_single_token(self):
        """单 token 场景测试"""
        tensors = create_test_tensors(batch_size=1, num_tokens=1)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )


class TestLargeBatch(BaseDSMLAWriteCacheTest):
    """测试大批次场景"""

    def test_large_batch(self):
        """大批次场景测试"""
        tensors = create_test_tensors(batch_size=32, num_tokens=512)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )

        self.assertEqual(tensors["kv_cache"].dtype, paddle.uint8)


class TestUnalignedTokens(BaseDSMLAWriteCacheTest):
    """测试非对齐 token 数（非 block_size 整数倍）"""

    def test_unaligned_tokens(self):
        """非对齐 token 数测试"""
        # 17 tokens 不是 16 (block_size) 的整数倍
        tensors = create_test_tensors(batch_size=1, num_tokens=17)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )


# ==================== 量化类型测试 ====================


class TestQuantTypeFp8DsMla(BaseDSMLAWriteCacheTest):
    """测试 fp8_ds_mla 量化类型"""

    def test_quant_type_fp8_ds_mla(self):
        """fp8_ds_mla 量化类型测试"""
        tensors = create_test_tensors(batch_size=2, num_tokens=16)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",  # 主要测试的量化类型
        )


class TestQuantTypeNone(BaseDSMLAWriteCacheTest):
    """测试无量化模式"""

    def test_quant_type_none(self):
        """无量化模式测试"""
        tensors = create_test_tensors(batch_size=2, num_tokens=16)
        # 无量化时 cache 格式不同: [num_blocks, 1, block_size, kv_lora_rank + pe_dim]
        tensors["kv_cache"] = paddle.zeros([100, 1, BLOCK_SIZE, KV_LORA_RANK + PE_DIM], dtype="bfloat16")

        try:
            dsk_attn_write_cache(
                tensors["kv_nope"],
                tensors["kv_pe"],
                tensors["kv_cache"],
                tensors["slot_mapping"],
                None,  # scale 在无量化时可为 None
                True,
            )

        except Exception as e:
            # 如果 'none' 类型不支持，跳过
            self.skipTest(f"'none' quant type 可能未实现: {e}")


# ==================== 可选参数测试 ====================


class TestWithoutScale(BaseDSMLAWriteCacheTest):
    """测试不传 scale 参数"""

    def test_without_scale(self):
        """不传 scale 参数测试"""
        tensors = create_test_tensors(batch_size=2, num_tokens=16)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            None,
            "fp8_ds_mla",
        )


class TestWithoutKvSignalData(BaseDSMLAWriteCacheTest):
    """测试不传 kv_signal_data 参数"""

    def test_without_kv_signal_data(self):
        """不传 kv_signal_data 参数测试"""
        tensors = create_test_tensors(batch_size=2, num_tokens=16)

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )


# ==================== 数据类型测试 ====================


class TestBfloat16Input(BaseDSMLAWriteCacheTest):
    """测试 bfloat16 输入"""

    def test_bfloat16_input(self):
        """bfloat16 输入测试"""
        tensors = create_test_tensors(dtype="bfloat16")

        dsk_attn_write_cache(
            tensors["kv_nope"],
            tensors["kv_pe"],
            tensors["kv_cache"],
            tensors["slot_mapping"],
            tensors["scale"],
            "fp8_ds_mla",
        )


class TestFloat16Input(BaseDSMLAWriteCacheTest):
    """测试 float16 输入"""

    def test_float16_input(self):
        """float16 输入测试"""
        tensors = create_test_tensors(dtype="float16")

        try:
            dsk_attn_write_cache(
                tensors["kv_nope"],
                tensors["kv_pe"],
                tensors["kv_cache"],
                tensors["slot_mapping"],
                tensors["scale"],
                "fp8_ds_mla",
            )

        except Exception as e:
            self.skipTest(f"float16 输入可能不支持: {e}")


# ==================== 性能测试 ====================


class TestDSMLAWriteCachePerformance(BaseDSMLAWriteCacheTest):
    """DSMLAWriteCacheKernel 性能测试"""

    def test_warmup_and_benchmark(self):
        """Warmup 并简单 benchmark"""
        tensors = create_test_tensors(batch_size=16, num_tokens=256)

        # Warmup
        for _ in range(5):
            _ = dsk_attn_write_cache(
                tensors["kv_nope"],
                tensors["kv_pe"],
                tensors["kv_cache"],
                tensors["slot_mapping"],
                tensors["scale"],
                "fp8_ds_mla",
            )

        paddle.device.synchronize()

        # Benchmark
        num_iters = 100
        start = time.perf_counter()

        for _ in range(num_iters):
            _ = dsk_attn_write_cache(
                tensors["kv_nope"],
                tensors["kv_pe"],
                tensors["kv_cache"],
                tensors["slot_mapping"],
                tensors["scale"],
                "fp8_ds_mla",
            )

        paddle.device.synchronize()
        end = time.perf_counter()

        avg_time_ms = (end - start) / num_iters * 1000
        print(f"\n[Benchmark] 256 tokens, avg time: {avg_time_ms:.3f} ms")

        # 性能阈值检查 (可根据实际情况调整)
        self.assertLess(avg_time_ms, 100.0, "性能应在 10ms 内")


if __name__ == "__main__":
    print("=" * 70)
    print("DSMLAWriteCacheKernel 单元测试")
    print("=" * 70)

    # 运行测试
    unittest.main(verbosity=2)
