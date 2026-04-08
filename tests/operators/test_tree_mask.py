"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import math
import time
import unittest

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.incubate.nn.functional import fused_rms_norm

from fastdeploy.model_executor.layers.attention.ops import (
    append_attention,
    get_block_shape_and_split_kv_block,
)

np.random.seed(0)
paddle.seed(0)

import os
from contextlib import contextmanager


@contextmanager
def temp_env(key, value):
    old_val = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_val


class TestTreeMask(unittest.TestCase):
    def setUp(self):
        # TODO(liuzichang): If set q_head=32 or bsz=128, some case will fail.
        paddle.seed(0)
        self.max_seq_len = 32768
        self.encoder_max_partition_size = self.max_seq_len
        self.max_partition_size = self.max_seq_len

        self.max_dec_len = 1024
        self.bsz = 64
        self.run_time = 3
        self.warm_up = 1
        self.block_size = 64
        self.head_dim = 128
        self.num_q_head = 20
        self.num_kv_head = 4
        self.use_qknorm = True
        self.dtype = "bfloat16"

        self.rope_3d = False
        self.use_neox_rotary_style = False
        self.CURRENT_Q = [None]
        self.TOTAL_K = []
        self.TOTAL_V = []

        # Initialize cache and block tables
        block_num_per_seq = (self.max_seq_len + self.block_size - 1) // self.block_size
        max_block_num = block_num_per_seq * self.bsz
        cache_shape = (
            max_block_num,
            self.num_kv_head,
            self.block_size,
            self.head_dim,
        )

        self.cache_k = paddle.zeros(shape=cache_shape).astype(self.dtype)
        self.cache_v = paddle.zeros(shape=cache_shape).astype(self.dtype)

        self.block_tables = paddle.zeros(shape=(self.bsz, block_num_per_seq), dtype="int32")

        free_list = list(range(max_block_num - 1, -1, -1))

        for i in range(self.bsz):
            need_block_num = (self.max_seq_len + self.block_size - 1) // self.block_size
            for j in range(need_block_num):
                block_id = free_list.pop()
                self.block_tables[i, j] = block_id

    def tearDown(self):
        self.CURRENT_Q = [None]
        self.TOTAL_K = []
        self.TOTAL_V = []

    def split_qkv(self, qkv, bsz, seq_len):
        qkv = qkv.reshape([bsz, seq_len, -1, self.head_dim])
        q = qkv[:, :, : self.num_q_head, :]
        self.CURRENT_Q[0] = q

        k = qkv[:, :, self.num_q_head : self.num_q_head + self.num_kv_head, :]
        self.TOTAL_K.append(k)

        v = qkv[:, :, self.num_q_head + self.num_kv_head :, :]
        self.TOTAL_V.append(v)

    def get_padding_offset(self, bsz, seq_lens_this_time, seq_lens_decoder):
        batch_id_per_token = []
        cu_seqlens_q = paddle.zeros(shape=(bsz + 1), dtype="int32")
        cu_seqlens_k = paddle.zeros(shape=(bsz + 1), dtype="int32")
        cum_seq_len_q = 0
        cum_seq_len_k = 0
        for i in range(bsz):
            seq_len_now = seq_lens_this_time[i]
            seq_len_dec_now = seq_lens_decoder[i]
            for j in range(seq_len_now):
                batch_id_per_token.append(i)
            cum_seq_len_q += seq_len_now
            cum_seq_len_k += seq_len_now + seq_len_dec_now
            cu_seqlens_q[i + 1] = cum_seq_len_q
            cu_seqlens_k[i + 1] = cum_seq_len_k
        return paddle.to_tensor(batch_id_per_token, dtype="int32"), cu_seqlens_q, cu_seqlens_k

    def ref_attention(self, q, k, v, mask, use_qknorm=False):
        if use_qknorm:
            q = q.reshape([-1, self.head_dim])
            q = fused_rms_norm(q.astype("float32"), self.q_norm_weight_tensor, None, 1e-6)[0].astype(self.dtype)
            q = q.reshape([self.bsz, -1, self.num_q_head, self.head_dim])
        q = q.transpose([0, 2, 1, 3])
        if len(k) > 1:
            k = paddle.concat(k, axis=1)
        else:
            k = k[0]
        if use_qknorm:
            k = k.reshape([-1, self.head_dim])
            k = fused_rms_norm(k.astype("float32"), self.k_norm_weight_tensor, None, 1e-6)[0].astype(self.dtype)
            k = k.reshape([self.bsz, -1, self.num_kv_head, self.head_dim])
        k = k.transpose([0, 2, 1, 3])
        if len(v) > 1:
            v = paddle.concat(v, axis=1)
        else:
            v = v[0]
        v = v.transpose([0, 2, 1, 3])
        total_len = k.shape[2]

        scores = (
            q.reshape([self.bsz, self.num_kv_head, -1, self.head_dim])
            @ k.transpose([0, 1, 3, 2])
            * (1.0 / math.sqrt(self.head_dim))
        )
        scores = scores.reshape([self.bsz, self.num_q_head, -1, total_len])

        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                mask = mask.unsqueeze(1)
            scores = paddle.add(scores, mask)
        weights = F.softmax(scores, axis=-1)

        o = weights.reshape([self.bsz, self.num_kv_head, -1, total_len]) @ v
        return (
            o.reshape([self.bsz, self.num_q_head, -1, self.head_dim])
            .transpose([0, 2, 1, 3])
            .reshape([-1, self.num_q_head, self.head_dim])
        )

    def run_append_c16_attention(
        self, q_len, kv_len, prefill=False, attn_mask=None, use_qknorm=False, mask_offset=None, qkv=None
    ):
        if prefill:
            seq_lens_enc = [
                q_len,
            ] * self.bsz
        else:
            seq_lens_enc = [
                0,
            ] * self.bsz

        seq_lens_dec = [
            kv_len,
        ] * self.bsz
        seq_lens_cur = [
            q_len,
        ] * self.bsz
        token_num = sum(seq_lens_cur)
        decoder_step_token_num = 1 if prefill else q_len

        seq_lens_encoder = paddle.to_tensor(seq_lens_enc, "int32")
        seq_lens_this_time = paddle.to_tensor(seq_lens_cur, "int32")
        seq_lens_decoder = paddle.to_tensor(seq_lens_dec, "int32")

        batch_id_per_token, cu_seqlens_q, cu_seqlens_k = self.get_padding_offset(
            self.bsz, seq_lens_this_time, seq_lens_decoder
        )
        if qkv is None:
            qkv_varlen_shape = [token_num, (self.num_q_head + 2 * self.num_kv_head) * self.head_dim]
            qkv = paddle.randn(shape=qkv_varlen_shape).astype(self.dtype)
            self.split_qkv(qkv, self.bsz, q_len)

        rotary_embs_shape = [
            2,
            1,
            self.max_seq_len,
            1,
            self.head_dim if self.use_neox_rotary_style else self.head_dim // 2,
        ]
        rotary_embs = paddle.randn(shape=rotary_embs_shape).astype("float32")
        rotary_embs[0, :, :, :, :] = 1
        rotary_embs[1, :, :, :, :] = 0

        cache_k_scale = None
        cache_v_scale = None
        cache_k_out_scale = None
        cache_v_out_scale = None

        encoder_block_shape_q = 64
        decoder_block_shape_q = 16
        group_size = self.num_q_head // self.num_kv_head
        decode_max_tile_size = (
            1024 * self.bsz * (decoder_step_token_num * group_size + decoder_block_shape_q - 1) / decoder_block_shape_q
        )
        encode_max_tile_size = (
            self.bsz * (self.max_seq_len * group_size + encoder_block_shape_q - 1) / encoder_block_shape_q
        )
        kv_max_tile_size = self.bsz * (self.max_seq_len + self.block_size - 1) / self.block_size

        decoder_batch_ids = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        decoder_tile_ids_per_batch = paddle.full([int(decode_max_tile_size)], 0, dtype="int32")
        decoder_num_blocks = paddle.full([1], 0, dtype="int32").pin_memory()
        decoder_num_blocks_device = paddle.full([1], 0, dtype="int32")
        decoder_chunk_size_device = paddle.full([1], 64, dtype="int32")
        max_len_tensor_cpu = paddle.full([9], 0, dtype="int32").cpu()
        encoder_batch_ids = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
        encoder_tile_ids_per_batch = paddle.full([int(encode_max_tile_size)], 0, dtype="int32")
        encoder_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
        kv_batch_ids = paddle.full([int(kv_max_tile_size)], 0, dtype="int32")
        kv_tile_ids_per_batch = paddle.full([int(kv_max_tile_size)], 0, dtype="int32")
        kv_num_blocks_x_cpu = paddle.full([1], 0, dtype="int32").cpu()
        q_norm_weight = np.ones([self.head_dim])
        k_norm_weight = np.ones([self.head_dim])
        self.q_norm_weight_tensor = paddle.to_tensor(q_norm_weight, dtype="float32")
        self.k_norm_weight_tensor = paddle.to_tensor(k_norm_weight, dtype="float32")
        paddle.device.synchronize()
        get_block_shape_and_split_kv_block(
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            decoder_batch_ids,
            decoder_tile_ids_per_batch,
            decoder_num_blocks,
            decoder_num_blocks_device,
            decoder_chunk_size_device,
            max_len_tensor_cpu,
            encoder_batch_ids,
            encoder_tile_ids_per_batch,
            encoder_num_blocks_x_cpu,
            kv_batch_ids,
            kv_tile_ids_per_batch,
            kv_num_blocks_x_cpu,
            encoder_block_shape_q,
            decoder_block_shape_q,
            self.num_q_head // self.num_kv_head,
            self.block_size,
        )
        s_time = 0
        for i in range(self.run_time + self.warm_up):
            if i == self.warm_up:
                s_time = time.time()
            out = append_attention(
                qkv,
                self.cache_k,
                self.cache_v,
                seq_lens_encoder,
                seq_lens_decoder,
                seq_lens_this_time,
                batch_id_per_token,
                cu_seqlens_q,
                self.block_tables,
                encoder_batch_ids,
                encoder_tile_ids_per_batch,
                encoder_num_blocks_x_cpu,
                kv_batch_ids,
                kv_tile_ids_per_batch,
                kv_num_blocks_x_cpu,
                decoder_batch_ids,
                decoder_tile_ids_per_batch,
                decoder_num_blocks,
                max_len_tensor_cpu,
                rotary_embs,
                attn_mask,
                None,  # qkv_bias
                None,  # qkv_out_scales
                cache_k_scale,
                cache_v_scale,
                cache_k_out_scale,
                cache_v_out_scale,
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # linear_shift
                None,  # linear_smooth
                mask_offset,  # mask_offset
                None,  # kv_signal_data
                self.q_norm_weight_tensor if use_qknorm else None,  # q_norm_weight
                self.k_norm_weight_tensor if use_qknorm else None,  # k_norm_weight
                None,  # sinks
                1e-6,
                "bf16",
                "none",
                self.use_neox_rotary_style,
                self.rope_3d,
                self.max_seq_len,
                0.0,
                0.0,
                -1.0,
                encoder_block_shape_q,
                decoder_block_shape_q,
                self.max_partition_size,
                self.encoder_max_partition_size,
                decoder_step_token_num,
                True if mask_offset is None else False,
                decoder_step_token_num > 1,
                0,
            )
            paddle.device.synchronize()
        e_time = time.time()
        print(f"mean infer time: {np.mean((e_time - s_time) * 1000 / self.run_time):.2f}")
        return out.reshape([token_num, self.num_q_head, self.head_dim])

    def test_naive_speculative_decoding(self):
        """
        在 speculative mode 下，测试 Attention 在 causal_mask 下的功能
        """
        prefill_len = 8192
        dec_len_q = 5
        total_len = prefill_len + dec_len_q
        mask = paddle.tril(paddle.ones((self.bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
        mask = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))
        self.run_append_c16_attention(prefill_len, 0, True, use_qknorm=self.use_qknorm)
        dec_out = self.run_append_c16_attention(dec_len_q, prefill_len, False, use_qknorm=self.use_qknorm)

        ref_out = self.ref_attention(self.CURRENT_Q[0], self.TOTAL_K, self.TOTAL_V, mask, use_qknorm=self.use_qknorm)
        np.testing.assert_allclose(
            ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
        )

    def test_mask(self):
        """
        在 speculative mode 下，测试 Attention 在传入 mask 下的功能
        """
        prefill_len = 8192
        dec_len_q = 5
        total_len = prefill_len + dec_len_q
        mask = paddle.tril(paddle.ones((self.bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
        mask_ref = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))

        mask_append_attn = mask[:, :, prefill_len:]
        mask_append_attn = paddle.where(
            mask_append_attn == 1,
            paddle.full_like(mask_append_attn, fill_value=False, dtype=bool),
            paddle.full_like(mask_append_attn, fill_value=True, dtype=bool),
        )

        self.run_append_c16_attention(prefill_len, 0, True)
        dec_out = self.run_append_c16_attention(dec_len_q, prefill_len, False, mask_append_attn)

        ref_out = self.ref_attention(self.CURRENT_Q[0], self.TOTAL_K, self.TOTAL_V, mask_ref)

        np.testing.assert_allclose(
            ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
        )

    def test_tree_mask(self):
        """
        在 speculative mode 下，测试 Attention 在传入 tree mask 下的功能
        """
        prefill_len = 8192
        dec_len_q = 5
        total_len = prefill_len + dec_len_q
        mask = paddle.tril(paddle.ones((self.bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
        mask[:, 2, prefill_len + 1] = 0
        mask[:, 3, prefill_len + 2] = 0
        mask[:, 4, prefill_len + 1] = 0
        mask[:, 4, prefill_len + 3] = 0

        mask_ref = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))

        mask_append_attn = mask[:, :, prefill_len:]
        mask_append_attn = paddle.where(
            mask_append_attn == 1,
            paddle.full_like(mask_append_attn, fill_value=False, dtype=bool),
            paddle.full_like(mask_append_attn, fill_value=True, dtype=bool),
        )

        self.run_append_c16_attention(prefill_len, 0, True)
        dec_out = self.run_append_c16_attention(dec_len_q, prefill_len, False, mask_append_attn)
        ref_out = self.ref_attention(self.CURRENT_Q[0], self.TOTAL_K, self.TOTAL_V, mask_ref)
        np.testing.assert_allclose(
            ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
        )

    def test_mask_offset(self):
        """
        在 speculative mode 下，测试 Attention 在传入 mask_offset 下的功能
        """
        prefill_len = 8192
        dec_len_q = 5
        total_len = prefill_len + dec_len_q
        mask = paddle.tril(paddle.ones((self.bsz, dec_len_q, total_len), dtype="float32"), diagonal=prefill_len)
        mask = paddle.where(mask == 1, paddle.zeros_like(mask), paddle.full_like(mask, fill_value=float("-inf")))
        self.run_append_c16_attention(prefill_len, 0, True, use_qknorm=self.use_qknorm)

        mask_offset = paddle.tile(
            paddle.tensor(
                [0, prefill_len + 1, 0, prefill_len + 2, 0, prefill_len + 3, 0, prefill_len + 4, 0, prefill_len + 5],
                dtype="int32",
            ),
            [self.bsz],
        ).astype("int32")
        dec_out = self.run_append_c16_attention(
            dec_len_q, prefill_len, False, use_qknorm=self.use_qknorm, mask_offset=mask_offset
        )

        ref_out = self.ref_attention(self.CURRENT_Q[0], self.TOTAL_K, self.TOTAL_V, mask, use_qknorm=self.use_qknorm)
        np.testing.assert_allclose(
            ref_out.astype("float32").numpy(), dec_out.astype("float32").numpy(), rtol=1e-03, atol=5e-03
        )

    def test_consistency_with_multi_tokens(self):
        """
        严格测试投机解码多token功能，包含 qkv_norm/mask_offset，通过对比：
        (A) 完整 5 步推理中取出第 1~3 步的结果
        (B) 单独跑第 1~3 步推理的结果
        若一致则投机解码 Attention 计算正确。
        """
        # -----------------------------
        # 基础参数
        # -----------------------------
        import os

        os.environ["FLAGS_max_partition_size"] = "131072"

        prefill_len = 8192
        dec_len_q = 5
        dec_start, dec_end = 1, 3
        dec_len = dec_end - dec_start

        head_num = self.num_q_head + 2 * self.num_kv_head
        qkv_dim = head_num * self.head_dim

        # -----------------------------
        # Prefill 阶段（构建 KV Cache）
        # -----------------------------
        with temp_env("FLAGS_max_partition_size", "131072"):
            self.run_append_c16_attention(
                prefill_len,  # q_len
                0,  # q_start
                True,  # is_prefill
                use_qknorm=self.use_qknorm,
            )

        # -----------------------------
        # 构造 mask_offset：每步两项 [0, prefill+i]
        # -----------------------------
        pattern = []
        for i in range(dec_len_q):
            pattern.extend([0, prefill_len + i + 1])
        mask_offset = paddle.tile(paddle.to_tensor(pattern, dtype="int32"), [self.bsz])

        # 截取 dec_start ~ dec_end 的偏移字段
        mask_offset_slice = (
            mask_offset.reshape([self.bsz, -1])[:, 2 * dec_start : 2 * dec_end].reshape([-1]).astype("int32")
        )

        # -----------------------------
        # 构造完整 qkv（5 步）
        # -----------------------------
        qkv_full = paddle.randn([self.bsz * dec_len_q, qkv_dim], dtype=self.dtype)

        # -----------------------------
        # 运行完整 5 步 attention
        # （前三个参数依然必须位置传参）
        # -----------------------------
        with temp_env("FLAGS_max_partition_size", "131072"):
            dec_out_full = self.run_append_c16_attention(
                dec_len_q,  # q_len
                prefill_len,  # q_start
                False,  # is_prefill
                use_qknorm=self.use_qknorm,
                mask_offset=mask_offset,
                qkv=qkv_full,
            )

        # -----------------------------
        # 构造截取后的 qkv：只保留第 1~3 步
        # -----------------------------
        qkv_slice = qkv_full.reshape([self.bsz, dec_len_q, qkv_dim])[:, dec_start:dec_end, :].reshape([-1, qkv_dim])

        # -----------------------------
        # 单独跑 1~3 步 attention
        # -----------------------------
        with temp_env("FLAGS_max_partition_size", "131072"):
            dec_out_slice = self.run_append_c16_attention(
                dec_len,  # q_len
                prefill_len + dec_start,  # q_start
                False,  # is_prefill
                use_qknorm=self.use_qknorm,
                mask_offset=mask_offset_slice,
                qkv=qkv_slice,
            )

        # -----------------------------
        # 从 full 结果中提取第 1~3 步的输出并对齐
        # -----------------------------
        dec_out_full_range = (
            dec_out_full.reshape([self.bsz, dec_len_q, -1, self.head_dim])[:, dec_start:dec_end, :, :]
            .reshape([self.bsz * dec_len, -1, self.head_dim])
            .astype("float32")
            .numpy()
        )

        # -----------------------------
        # 两种模式输出必须完全一致
        # -----------------------------
        np.testing.assert_array_equal(
            dec_out_full_range,
            dec_out_slice.astype("float32").numpy(),
        )


if __name__ == "__main__":
    unittest.main()
