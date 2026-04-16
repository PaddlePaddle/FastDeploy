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
验证 DeepseekV3ForCausalLM.pre_process 中计算的 slot_mapping
与独立辅助函数 compute_slot_mapping 的结果一致。

测试策略：
  pre_process 的 slot_mapping 计算逻辑等价于：
      block_idx    = position_ids // block_size
      block_ids    = block_tables[batch_id_per_token, block_idx]
      block_offset = position_ids % block_size
      slot_mapping = (block_ids * block_size + block_offset).cast(int64)

  compute_slot_mapping 封装了完全相同的公式。
  因此，只需用相同的 position_ids / block_tables / batch_id_per_token 分别调用
  两段逻辑，断言结果相等即可。

  为了获得与 pre_process 完全一致的 position_ids，测试直接调用
  get_position_ids_and_mask_encoder_batch（pre_process 内部也调用它），
  而不需要实例化整个模型。
"""

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import get_position_ids_and_mask_encoder_batch

# ---------------------------------------------------------------------------
# 被测辅助函数（与用户提供的代码完全一致）
# ---------------------------------------------------------------------------


def compute_slot_mapping(
    block_tables: paddle.Tensor,  # [num_reqs, max_blocks_per_req]
    positions: paddle.Tensor,  # [num_tokens]
    batch_id_per_token: paddle.Tensor,  # [num_tokens]
    block_size: int,
) -> paddle.Tensor:
    """
    计算 slot_mapping

    公式: slot = block_id * block_size + offset_in_block
    """
    block_idx = positions // block_size
    block_ids = block_tables[batch_id_per_token, block_idx]
    block_offset = positions % block_size
    slot_mapping = block_ids * block_size + block_offset
    return slot_mapping.cast(paddle.int64)


# ---------------------------------------------------------------------------
# 与 pre_process 中相同的内联 slot_mapping 计算逻辑（抽成函数便于复用）
# ---------------------------------------------------------------------------


def _pre_process_slot_mapping(
    block_tables: paddle.Tensor,
    batch_id_per_token: paddle.Tensor,
    position_ids: paddle.Tensor,
    block_size: int,
) -> paddle.Tensor:
    """复刻 pre_process 中的 slot_mapping 计算，不依赖模型对象。"""
    block_idx = position_ids // block_size
    block_ids = block_tables[batch_id_per_token, block_idx]
    block_offset = position_ids % block_size
    return (block_ids * block_size + block_offset).cast(paddle.int64)


# ---------------------------------------------------------------------------
# 测试工具函数
# ---------------------------------------------------------------------------


def _build_position_ids(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_decoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
) -> paddle.Tensor:
    """调用 custom op 得到 position_ids（与 pre_process 完全一致的路径）。"""
    total_len = int(seq_lens_encoder.numpy().sum() + seq_lens_this_time.numpy().sum())
    position_ids = paddle.zeros([total_len], dtype="int32")
    get_position_ids_and_mask_encoder_batch(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        position_ids,
    )
    return position_ids


def _build_batch_id_per_token(
    seq_lens_encoder: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
) -> paddle.Tensor:
    """根据 encoder 序列长度和本次处理长度构建 batch_id_per_token。"""
    enc = seq_lens_encoder.numpy().tolist()
    dec_this = seq_lens_this_time.numpy().tolist()
    batch_ids = []
    for bid, (e, d) in enumerate(zip(enc, dec_this)):
        batch_ids.extend([bid] * (e + d))
    return paddle.to_tensor(batch_ids, dtype="int32")


# ---------------------------------------------------------------------------
# 单测类
# ---------------------------------------------------------------------------


class TestPreProcessSlotMappingConsistency(unittest.TestCase):
    """验证 pre_process 内联逻辑与 compute_slot_mapping 的一致性。"""

    def setUp(self):
        paddle.set_device("gpu")

    def _assert_slot_mapping_equal(
        self,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        block_tables,
        block_size,
        test_name="",
    ):
        """通用断言：两种计算路径的 slot_mapping 完全相等。"""
        position_ids = _build_position_ids(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time)
        batch_id_per_token = _build_batch_id_per_token(seq_lens_encoder, seq_lens_this_time)

        # pre_process 内联逻辑
        ref = _pre_process_slot_mapping(block_tables, batch_id_per_token, position_ids, block_size)

        # compute_slot_mapping
        got = compute_slot_mapping(block_tables, position_ids, batch_id_per_token, block_size)

        np.testing.assert_array_equal(
            ref.numpy(),
            got.numpy(),
            err_msg=f"[{test_name}] slot_mapping mismatch",
        )

    # ------------------------------------------------------------------
    # case 1: 纯 prefill（batch_size=2，无 decode）
    # ------------------------------------------------------------------
    def test_pure_prefill(self):
        """两条请求均处于 prefill 阶段。"""
        block_size = 4
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([3, 5], dtype="int32")

        # block_tables: [2 reqs, 4 blocks each]，随机填充合法 block id
        block_tables = paddle.to_tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype="int32")

        self._assert_slot_mapping_equal(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            block_tables,
            block_size,
            test_name="pure_prefill",
        )

    # ------------------------------------------------------------------
    # case 2: 纯 decode（每条请求本次只处理 1 个 token）
    # ------------------------------------------------------------------
    def test_pure_decode(self):
        """两条请求均处于 decode 阶段，seq_lens_decoder 反映历史已填充长度。"""
        block_size = 8
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        # decode 历史已填充：req0=7 tokens，req1=15 tokens
        seq_lens_decoder = paddle.to_tensor([7, 15], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([1, 1], dtype="int32")

        block_tables = paddle.to_tensor([[5, 6, 0, 0], [30, 31, 0, 0]], dtype="int32")

        self._assert_slot_mapping_equal(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            block_tables,
            block_size,
            test_name="pure_decode",
        )

    # ------------------------------------------------------------------
    # case 3: 混合（prefill + decode）batch
    # ------------------------------------------------------------------
    def test_mixed_prefill_decode(self):
        """batch 中同时包含 prefill 和 decode 请求（MIXED 模式）。"""
        block_size = 4
        # req0: prefill（encoder=3, this_time=3）
        # req1: decode（decoder=5, this_time=1）
        seq_lens_encoder = paddle.to_tensor([3, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([1, 5], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([1, 1], dtype="int32")

        block_tables = paddle.to_tensor([[100, 101, 102, 103], [200, 201, 202, 203]], dtype="int32")

        self._assert_slot_mapping_equal(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            block_tables,
            block_size,
            test_name="mixed_prefill_decode",
        )

    # ------------------------------------------------------------------
    # case 4: block_size=1（每个 block 仅容纳 1 个 token，边界条件）
    # ------------------------------------------------------------------
    def test_block_size_one(self):
        """block_size=1 时 block_offset 恒为 0，slot == block_id。"""
        block_size = 1
        seq_lens_encoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0, 0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([4, 3], dtype="int32")

        # 每个 token 占一个 block
        block_tables = paddle.to_tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
            dtype="int32",
        )

        self._assert_slot_mapping_equal(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            block_tables,
            block_size,
            test_name="block_size_one",
        )

    # ------------------------------------------------------------------
    # case 5: 较大 batch_size，验证批量正确性
    # ------------------------------------------------------------------
    def test_large_batch(self):
        """batch_size=8，混合 prefill 和 decode，较多 token。"""
        block_size = 16
        bsz = 8
        np.random.seed(0)

        enc_lens = np.random.randint(0, 32, size=bsz).astype(np.int32)
        dec_lens = np.random.randint(0, 16, size=bsz).astype(np.int32)
        this_lens = np.random.randint(1, 8, size=bsz).astype(np.int32)

        # 确保 decoder 历史不超过 block_tables 能索引的范围
        max_blocks = 8
        block_tables_np = np.random.randint(0, 1024, size=(bsz, max_blocks), dtype=np.int32)

        # 检查 this_lens 对应的最大位置不超过 max_blocks * block_size
        total_pos = enc_lens + dec_lens + this_lens - 1
        cap = max_blocks * block_size - 1
        # 超出容量的请求截断到安全范围
        this_lens = np.where(total_pos > cap, np.maximum(1, cap - enc_lens - dec_lens + 1), this_lens)
        this_lens = np.maximum(this_lens, 1).astype(np.int32)

        seq_lens_encoder = paddle.to_tensor(enc_lens)
        seq_lens_decoder = paddle.to_tensor(dec_lens)
        seq_lens_this_time = paddle.to_tensor(this_lens)
        block_tables = paddle.to_tensor(block_tables_np)

        self._assert_slot_mapping_equal(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            block_tables,
            block_size,
            test_name="large_batch",
        )

    # ------------------------------------------------------------------
    # case 6: 单请求，精确验证数值
    # ------------------------------------------------------------------
    def test_single_request_exact_values(self):
        """
        单请求 prefill，block_size=4，seq_len=6。
        手工推导期望值并验证。

        seq_lens_encoder=0, seq_lens_decoder=0, seq_lens_this_time=6
        => position_ids = [0,1,2,3,4,5]
        block_tables[0] = [10, 20, 30, ...]

        slot[i] = block_tables[0, pos//4] * 4 + pos%4
          pos=0 -> block=10, off=0 -> slot=40
          pos=1 -> block=10, off=1 -> slot=41
          pos=2 -> block=10, off=2 -> slot=42
          pos=3 -> block=10, off=3 -> slot=43
          pos=4 -> block=20, off=0 -> slot=80
          pos=5 -> block=20, off=1 -> slot=81
        """
        block_size = 4
        seq_lens_encoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_decoder = paddle.to_tensor([0], dtype="int32")
        seq_lens_this_time = paddle.to_tensor([6], dtype="int32")
        block_tables = paddle.to_tensor([[10, 20, 30, 40]], dtype="int32")

        position_ids = _build_position_ids(seq_lens_encoder, seq_lens_decoder, seq_lens_this_time)
        batch_id_per_token = _build_batch_id_per_token(seq_lens_encoder, seq_lens_this_time)

        expected = np.array([40, 41, 42, 43, 80, 81], dtype=np.int64)

        ref = _pre_process_slot_mapping(block_tables, batch_id_per_token, position_ids, block_size)
        got = compute_slot_mapping(block_tables, position_ids, batch_id_per_token, block_size)

        np.testing.assert_array_equal(ref.numpy(), expected, err_msg="pre_process mismatch expected")
        np.testing.assert_array_equal(got.numpy(), expected, err_msg="compute_slot_mapping mismatch expected")
        np.testing.assert_array_equal(ref.numpy(), got.numpy(), err_msg="two paths mismatch")


if __name__ == "__main__":
    unittest.main()
