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
radix_topk_ragged_transform 精度测试

对比算子输出与 paddle.topk 的结果
使用 unittest.TestCase 框架
"""

import unittest

import paddle

from fastdeploy.model_executor.ops.gpu import radix_topk_ragged_transform


class BaseTestRadixTopk(unittest.TestCase):
    """基础测试类，包含共用的辅助方法"""

    def setUp(self):
        """测试前准备"""
        paddle.set_device("gpu")

    def get_reference_topk(self, input_pd, lengths_pd, offsets_pd, top_k, q_num_heads):
        """
        使用 paddle.topk 生成参考结果
        注意：算子输出的索引是相对于 offsets 的偏移量（0-based 相对索引）

        Args:
            input_pd: (num_rows, max_len)
            lengths_pd: (batch_size,) - 每个batch的长度
            offsets_pd: (num_rows,) - 每一行的偏移基点
            top_k: k值
            q_num_heads: query head数量

        Returns:
            ref_indices: (num_rows, top_k) - 参考索引（相对于 offset 的偏移），长度不足的部分用-1填充
        """
        num_rows = input_pd.shape[0]
        ref_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        offsets = offsets_pd.numpy()

        for row_idx in range(num_rows):
            batch_idx = row_idx // q_num_heads
            length = lengths_pd[batch_idx].item()
            offset = offsets[row_idx]

            if length == 0:
                continue

            row_data = input_pd[row_idx, :length]

            if length <= top_k:
                # 长度不足top_k，按顺序返回所有索引（相对于 offset）
                ref_indices[row_idx, :length] = paddle.arange(offset, offset + length, dtype="int32")
            else:
                # 长度足够，使用 paddle.topk 获取最大的top_k个值的索引
                topk_vals, topk_inds = paddle.topk(row_data, top_k)
                # 加上 offset 作为基点
                ref_indices[row_idx, :top_k] = topk_inds + offset

        return ref_indices

    def compare_indices(self, custom_output, ref_output):
        """
        对比两个索引矩阵

        Args:
            custom_output: 算子输出
            ref_output: 参考输出

        Returns:
            是否完全匹配
        """
        # 转换为 numpy 进行比较
        custom_np = custom_output.numpy() if isinstance(custom_output, paddle.Tensor) else custom_output
        ref_np = ref_output.numpy() if isinstance(ref_output, paddle.Tensor) else ref_output

        # 对每一行进行比较：提取有效索引（非-1）后排序后比较
        num_rows = custom_np.shape[0]
        matches = 0
        mismatches_detail = []

        for row_idx in range(num_rows):
            # 提取有效索引（非-1）
            custom_valid = custom_np[row_idx]
            custom_valid = custom_valid[custom_valid != -1]

            ref_valid = ref_np[row_idx]
            ref_valid = ref_valid[ref_valid != -1]

            # 排序后比较
            custom_sorted = sorted(custom_valid.tolist())
            ref_sorted = sorted(ref_valid.tolist())

            if custom_sorted == ref_sorted:
                matches += 1
            else:
                mismatches_detail.append((row_idx, custom_sorted, ref_sorted))

        total = num_rows
        accuracy = matches / total * 100 if total > 0 else 0

        print(f"  行匹配数: {matches}/{total} ({accuracy:.2f}%)")

        if matches == total:
            return True
        else:
            print("  不匹配详情（前3行）:")
            for row_idx, custom_sorted, ref_sorted in mismatches_detail[:3]:
                print(f"    行 {row_idx}: custom={custom_sorted}, ref={ref_sorted}")
            return False


class TestPrefillMode(BaseTestRadixTopk):
    """测试 Prefill 模式"""

    def test_prefill_mode(self):
        """
        Prefill 模式测试

        场景：多个 query head，每个 batch 有长度信息，使用 lengths 参数
        """
        paddle.seed(2025)

        num_rows = 32
        max_len = 256
        top_k = 8
        q_num_heads = 4
        batch_size = num_rows // q_num_heads

        # 使用 paddle 构造数据
        input_pd = paddle.randn([num_rows, max_len], dtype="float32")
        offsets_pd = paddle.arange(num_rows, dtype="int32")
        lengths_pd = paddle.randint(16, max_len, [batch_size], dtype="int32")

        # 调用算子
        output_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        radix_topk_ragged_transform(
            input_pd, output_indices, offsets_pd, lengths_pd, None, None, None, None, 0, top_k, q_num_heads
        )

        # 获取参考结果
        ref_indices = self.get_reference_topk(input_pd, lengths_pd, offsets_pd, top_k, q_num_heads)

        # 对比结果
        result = self.compare_indices(output_indices, ref_indices)
        self.assertTrue(result, "Prefill 模式测试失败")


class TestDecodeMode(BaseTestRadixTopk):
    """测试 Decode 模式"""

    def test_decode_mode(self):
        """
        Decode 模式测试

        场景：使用 seq_len_decoder 和 batch_id_per_token 参数
        长度 = seq_len_decoder + 1
        """
        paddle.seed(2025)

        batch_size = 2
        kv_head = 1  # decode 模式下，每个 batch 只有一个新 token
        num_rows = batch_size * kv_head  # = batch_size
        max_len = 1024
        top_k = 8

        # 使用 paddle 构造数据
        input_pd = paddle.randn([num_rows, max_len], dtype="float32")

        # 生成 cu_seqlens_q: 每个 batch 在打平的 query 中的偏移量
        # 在 decode 模式下，每个 batch 只有一个新 token，所以 cu_seqlens_q = [0, 1, 2, ..., batch_size]
        cu_seqlens_q_pd = paddle.concat(
            [
                paddle.zeros([1], dtype="int32"),
                paddle.cumsum(paddle.ones([batch_size], dtype="int32")).astype("int32"),
            ],
            axis=0,
        )

        lengths_pd = paddle.full([num_rows], 0, dtype="int32")  # unused
        seq_len_decoder_pd = paddle.randint(16, 128, [batch_size], dtype="int32")

        # 调用算子（不使用 block_tables，让它按照 prefill 模式类似的逻辑工作）
        output_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        radix_topk_ragged_transform(
            input_pd,
            output_indices,
            cu_seqlens_q_pd,
            lengths_pd,  # unused
            seq_len_decoder_pd,
            None,  # batch_id_per_token
            None,  # block_tables
            None,  # buffer
            0,  # max_block_per_seq
            top_k,
            kv_head,
        )

        # Decode 模式下，长度 = seq_len_decoder + 1
        decode_lengths = seq_len_decoder_pd + 1

        # 获取参考结果（注意：num_rows = batch_size * kv_head）
        ref_indices = self.get_reference_topk(input_pd, decode_lengths, cu_seqlens_q_pd, top_k, kv_head)

        # 对比结果
        result = self.compare_indices(output_indices, ref_indices)
        self.assertTrue(result, "Decode 模式测试失败")


class TestEdgeLengthZero(BaseTestRadixTopk):
    """测试边界情况：length == 0"""

    def test_edge_length_zero(self):
        """
        边界情况：所有序列长度为 0

        预期：所有输出都应该是 -1
        """
        paddle.seed(2025)

        num_rows = 4
        max_len = 64
        top_k = 8
        q_num_heads = 1

        input_pd = paddle.randn([num_rows, max_len], dtype="float32")
        offsets_pd = paddle.arange(num_rows, dtype="int32")
        lengths_pd = paddle.full([num_rows], 0, dtype="int32")

        output_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        radix_topk_ragged_transform(
            input_pd, output_indices, offsets_pd, lengths_pd, None, None, None, None, 0, top_k, q_num_heads
        )

        # 预期结果：全是 -1
        ref_indices = paddle.full([num_rows, top_k], -1, dtype="int32")

        # 对比结果
        result = self.compare_indices(output_indices, ref_indices)
        self.assertTrue(result, "length == 0 测试失败")


class TestEdgeLengthLessThanTopk(BaseTestRadixTopk):
    """测试边界情况：length < top_k"""

    def test_edge_length_less_than_topk(self):
        """
        边界情况：序列长度小于 top_k

        预期：返回所有有效元素的索引，其余填充 -1
        """
        paddle.seed(2025)

        num_rows = 4
        max_len = 64
        top_k = 8
        q_num_heads = 1

        input_pd = paddle.randn([num_rows, max_len], dtype="float32")
        offsets_pd = paddle.arange(num_rows, dtype="int32")
        lengths_pd = paddle.full([num_rows], top_k // 2, dtype="int32")  # 长度为 4

        output_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        radix_topk_ragged_transform(
            input_pd, output_indices, offsets_pd, lengths_pd, None, None, None, None, 0, top_k, q_num_heads
        )

        # 获取参考结果
        ref_indices = self.get_reference_topk(input_pd, lengths_pd, offsets_pd, top_k, q_num_heads)

        # 对比结果
        result = self.compare_indices(output_indices, ref_indices)
        self.assertTrue(result, "length < top_k 测试失败")


class TestEdgeLengthEqualTopk(BaseTestRadixTopk):
    """测试边界情况：length == top_k"""

    def test_edge_length_equal_topk(self):
        """
        边界情况：序列长度等于 top_k

        预期：当 length == top_k 时，应返回所有元素的索引
        """
        paddle.seed(2025)

        num_rows = 4
        max_len = 64
        top_k = 8
        q_num_heads = 1

        input_pd = paddle.randn([num_rows, max_len], dtype="float32")
        offsets_pd = paddle.arange(num_rows, dtype="int32")
        lengths_pd = paddle.full([num_rows], top_k, dtype="int32")

        output_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        radix_topk_ragged_transform(
            input_pd, output_indices, offsets_pd, lengths_pd, None, None, None, None, 0, top_k, q_num_heads
        )

        # 获取参考结果
        ref_indices = self.get_reference_topk(input_pd, lengths_pd, offsets_pd, top_k, q_num_heads)

        # 对比结果
        result = self.compare_indices(output_indices, ref_indices)
        self.assertTrue(result, "length == top_k 测试失败")


class TestLargeScale(BaseTestRadixTopk):
    """测试大规模数据"""

    def test_large_scale(self):
        """
        大规模数据测试

        场景：大数据量和大 k 值
        - 128 行
        - 2048 长度
        - top_k = 32
        - 8 个 query head
        """
        paddle.seed(2025)

        num_rows = 128
        max_len = 2048
        top_k = 32
        q_num_heads = 8
        batch_size = num_rows // q_num_heads

        input_pd = paddle.randn([num_rows, max_len], dtype="float32")
        offsets_pd = paddle.arange(num_rows, dtype="int32")
        lengths_pd = paddle.randint(64, max_len, [batch_size], dtype="int32")

        output_indices = paddle.full([num_rows, top_k], -1, dtype="int32")
        radix_topk_ragged_transform(
            input_pd, output_indices, offsets_pd, lengths_pd, None, None, None, None, 0, top_k, q_num_heads
        )

        # 获取参考结果
        ref_indices = self.get_reference_topk(input_pd, lengths_pd, offsets_pd, top_k, q_num_heads)

        # 对比结果
        result = self.compare_indices(output_indices, ref_indices)
        self.assertTrue(result, "大规模数据测试失败")


if __name__ == "__main__":
    unittest.main(verbosity=2)
