"""
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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.triton_ops.indexer_update_attn_mask_offsets import (
    update_indexer_attn_mask_offsets,
)


def ref_update_attn_mask_offsets(seq_lens_this_time, seq_lens_encoder, cu_seqlens_k):
    """Python 参考实现，与 deepseek_v3.py 中的原始循环语义对齐。
    返回 attn_mask_offsets: [num_tokens * 2], 偶数位=start, 奇数位=end

    注意：cu_seqlens_k 在 Indexer 中是 Q 侧的累积长度（cumsum of seq_lens_this_time），
    因此 num_tokens 应取 sum(seq_lens_this_time) 而非 cu_seqlens_k[-1]。
    """
    num_tokens = int(sum(int(s.numpy()) for s in seq_lens_this_time))
    result = np.zeros(num_tokens * 2, dtype=np.int32)

    bsz = len(seq_lens_this_time)
    for i in range(bsz):
        if int(seq_lens_encoder[i].numpy()) > 0:
            token_start_k = int(cu_seqlens_k[i].numpy())
            token_end_k = int(cu_seqlens_k[i + 1].numpy())
            for t in range(token_start_k, token_end_k):
                result[t * 2] = token_start_k  # start: 本 batch 的 k 起始偏移
                result[t * 2 + 1] = t + 1  # end: 当前 token 全局索引 + 1
    return result


def make_cu_seqlens(seq_lens):
    """从每个序列的长度列表构造 cu_seqlens（前缀和）。"""
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    return paddle.to_tensor(cu, dtype=paddle.int32)


class TestIndexerUpdateAttnMaskOffsets(unittest.TestCase):

    def _run_and_compare(self, seq_lens_this_time_list, seq_lens_encoder_list, k_lens_list):
        """构造输入，运行 Triton kernel 和参考实现，对比结果。"""
        seq_lens_this_time = paddle.to_tensor(seq_lens_this_time_list, dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor(seq_lens_encoder_list, dtype=paddle.int32)
        cu_seqlens_k = make_cu_seqlens(k_lens_list)

        num_tokens = int(sum(seq_lens_this_time_list))
        ids_remove_padding = paddle.zeros([num_tokens], dtype=paddle.int32)

        triton_out = update_indexer_attn_mask_offsets(
            ids_remove_padding,
            seq_lens_this_time,
            seq_lens_encoder,
            cu_seqlens_k,
        ).numpy()

        ref_out = ref_update_attn_mask_offsets(seq_lens_this_time, seq_lens_encoder, cu_seqlens_k)

        np.testing.assert_array_equal(
            triton_out,
            ref_out,
            err_msg=f"Mismatch!\ntriton: {triton_out}\nref:    {ref_out}",
        )

    # ------------------------------------------------------------------
    # 基础功能用例
    # ------------------------------------------------------------------

    def test_single_prefill_seq(self):
        """单条 prefill 序列，4 个 token。"""
        self._run_and_compare(
            seq_lens_this_time_list=[4],
            seq_lens_encoder_list=[4],
            k_lens_list=[4],
        )

    def test_single_token_prefill(self):
        """边界：只有 1 个 token 的 prefill 序列。"""
        self._run_and_compare(
            seq_lens_this_time_list=[1],
            seq_lens_encoder_list=[1],
            k_lens_list=[1],
        )

    def test_single_decode_seq(self):
        """单条 decode 序列，所有输出应全为 0（decode 路径不写 offsets）。"""
        self._run_and_compare(
            seq_lens_this_time_list=[1],
            seq_lens_encoder_list=[0],
            k_lens_list=[1],
        )

    # ------------------------------------------------------------------
    # 多 batch 用例
    # ------------------------------------------------------------------

    def test_multi_prefill_batch(self):
        """多条 prefill 序列，长度不同。"""
        self._run_and_compare(
            seq_lens_this_time_list=[3, 5, 2],
            seq_lens_encoder_list=[3, 5, 2],
            k_lens_list=[3, 5, 2],
        )

    def test_all_decode_batch(self):
        """全 decode batch，所有偶数/奇数位均应为 0。
        decode 请求的 k_len = seq_lens_this_time（Q 侧长度），不是 KV cache 历史长度。
        """
        self._run_and_compare(
            seq_lens_this_time_list=[1, 1, 1],
            seq_lens_encoder_list=[0, 0, 0],
            k_lens_list=[1, 1, 1],
        )

    def test_mixed_prefill_decode_batch(self):
        """混合 batch：第 0 条是 prefill，第 1 条是 decode，第 2 条是 prefill。
        decode 请求 k_len = seq_lens_this_time = 1（Q 侧长度）。
        """
        self._run_and_compare(
            seq_lens_this_time_list=[4, 1, 3],
            seq_lens_encoder_list=[4, 0, 3],
            k_lens_list=[4, 1, 3],
        )

    # ------------------------------------------------------------------
    # 数值正确性校验
    # ------------------------------------------------------------------

    def test_prefill_ks_ke_values(self):
        """精确验证 prefill token 的 start/end 值。

        场景：bsz=1, seq=[0,1,2], k_start=0
        期望：
            token 0: start=0, end=1
            token 1: start=0, end=2
            token 2: start=0, end=3
        """
        seq_lens_this_time = paddle.to_tensor([3], dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor([3], dtype=paddle.int32)
        cu_seqlens_k = paddle.to_tensor([0, 3], dtype=paddle.int32)
        ids_remove_padding = paddle.zeros([3], dtype=paddle.int32)

        out = update_indexer_attn_mask_offsets(
            ids_remove_padding, seq_lens_this_time, seq_lens_encoder, cu_seqlens_k
        ).numpy()

        # [ks0, ke0, ks1, ke1, ks2, ke2]
        expected = np.array([0, 1, 0, 2, 0, 3], dtype=np.int32)
        np.testing.assert_array_equal(out, expected)

    def test_prefill_with_nonzero_k_start(self):
        """验证 k_start 非 0 时 start 偏移正确传播。

        场景：bsz=2，第 0 条 decode（q_len=1），第 1 条 prefill（q_len=3）
        cu_seqlens_k = cumsum(seq_lens_this_time) = [0, 1, 4]
        第 1 条 k_start=1，全局 token 索引 = 1,2,3
        期望：
            token 0 (decode): start=0, end=0
            token 1: start=1, end=2
            token 2: start=1, end=3
            token 3: start=1, end=4
        """
        seq_lens_this_time = paddle.to_tensor([1, 3], dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor([0, 3], dtype=paddle.int32)
        cu_seqlens_k = paddle.to_tensor([0, 1, 4], dtype=paddle.int32)
        ids_remove_padding = paddle.zeros([4], dtype=paddle.int32)  # 1+3 tokens

        out = update_indexer_attn_mask_offsets(
            ids_remove_padding, seq_lens_this_time, seq_lens_encoder, cu_seqlens_k
        ).numpy()

        # token 0 (decode): [0, 0]
        # token 1,2,3 (prefill): start=1; end=2,3,4
        expected = np.array([0, 0, 1, 2, 1, 3, 1, 4], dtype=np.int32)
        np.testing.assert_array_equal(out, expected)

    def test_decode_tokens_remain_zero(self):
        """decode token 对应的位置必须保持 0，不被 kernel 改写。"""
        seq_lens_this_time = paddle.to_tensor([1, 4], dtype=paddle.int32)
        seq_lens_encoder = paddle.to_tensor([0, 4], dtype=paddle.int32)
        cu_seqlens_k = paddle.to_tensor([0, 1, 5], dtype=paddle.int32)
        ids_remove_padding = paddle.zeros([5], dtype=paddle.int32)  # 1+4 tokens

        out = update_indexer_attn_mask_offsets(
            ids_remove_padding, seq_lens_this_time, seq_lens_encoder, cu_seqlens_k
        ).numpy()

        # 第 0 条 decode：token 0 的 start/end 均应为 0
        self.assertEqual(out[0], 0, "decode token start should be 0")
        self.assertEqual(out[1], 0, "decode token end should be 0")

    # ------------------------------------------------------------------
    # 大序列压力用例
    # ------------------------------------------------------------------

    def test_large_seq_len(self):
        """较长序列，确保 BLOCK_M 分块循环逻辑正确。"""
        seq_len = 512
        self._run_and_compare(
            seq_lens_this_time_list=[seq_len],
            seq_lens_encoder_list=[seq_len],
            k_lens_list=[seq_len],
        )

    def test_large_batch(self):
        """较大 batch，验证多 program 并行结果正确。"""
        bsz = 32
        seq_lens = [8] * bsz
        self._run_and_compare(
            seq_lens_this_time_list=seq_lens,
            seq_lens_encoder_list=seq_lens,
            k_lens_list=seq_lens,
        )

    def test_large_mixed_batch(self):
        """大规模混合 batch，交替 prefill/decode。
        decode 请求 k_len = seq_lens_this_time = 1。
        """
        bsz = 20
        seq_lens_this_time = [6 if i % 2 == 0 else 1 for i in range(bsz)]
        seq_lens_encoder = [6 if i % 2 == 0 else 0 for i in range(bsz)]
        k_lens = seq_lens_this_time  # cu_seqlens_k = cumsum(seq_lens_this_time)
        self._run_and_compare(seq_lens_this_time, seq_lens_encoder, k_lens)


if __name__ == "__main__":
    unittest.main()
