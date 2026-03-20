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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_pre_process


def speculate_pre_process_ref(
    input_ids,
    seq_lens,
    draft_tokens,
    seq_lens_encoder,
    max_seq_len,
    max_draft_tokens_per_batch,
    real_bsz,
    token_num,
):
    """
    Python reference implementation for SpeculatePreProcessKernel.

    Returns:
        ids_remove_padding:         int64[token_num]
        batch_id_per_token:         int32[token_num]
        cu_seqlens_q:               int32[real_bsz + 1]
        cu_seqlens_k:               int32[real_bsz + 1]
        seq_lens_output:            int32[real_bsz]
        cu_seq_lens_q_output:       int32[real_bsz + 1]
        batch_id_per_token_output:  int32[real_bsz * max_draft_tokens_per_batch]
        real_output_token_num:      int32[1]
    """
    # --- Part 1: ids_remove_padding, batch_id_per_token, cu_seqlens_q/k ---
    ids_remove_padding = np.zeros(token_num, dtype=np.int64)
    batch_id_per_token = np.zeros(token_num, dtype=np.int32)
    cu_seqlens_q = np.zeros(real_bsz + 1, dtype=np.int32)
    cu_seqlens_k = np.zeros(real_bsz + 1, dtype=np.int32)

    cum = 0
    for bi in range(real_bsz):
        cum += seq_lens[bi]
        cu_seqlens_q[bi + 1] = cum
        cu_seqlens_k[bi + 1] = cum

        start = cum - seq_lens[bi]
        for i in range(seq_lens[bi]):
            tgt = start + i
            if max_draft_tokens_per_batch > 0 and seq_lens_encoder[bi] <= 0:
                src = bi * max_draft_tokens_per_batch + i
                ids_remove_padding[tgt] = draft_tokens[src]
            else:
                src = bi * max_seq_len + i
                ids_remove_padding[tgt] = input_ids[src]
            batch_id_per_token[tgt] = bi

    # --- Part 2: seq_lens_output ---
    seq_lens_output = np.zeros(real_bsz, dtype=np.int32)
    for bid in range(real_bsz):
        if seq_lens[bid] == 0:
            seq_lens_output[bid] = 0
        elif seq_lens[bid] == 1:
            seq_lens_output[bid] = 1
        elif seq_lens_encoder[bid] != 0:
            seq_lens_output[bid] = 1
        else:
            seq_lens_output[bid] = seq_lens[bid]

    # --- Part 3: cu_seq_lens_q_output, batch_id_per_token_output, real_output_token_num ---
    cu_seq_lens_q_output = np.zeros(real_bsz + 1, dtype=np.int32)
    batch_id_per_token_output = np.zeros(real_bsz * max_draft_tokens_per_batch, dtype=np.int32)

    cum_output = 0
    for bi in range(real_bsz):
        cum_output += seq_lens_output[bi]
        cu_seq_lens_q_output[bi + 1] = cum_output

        start_out = cum_output - seq_lens_output[bi]
        for i in range(seq_lens_output[bi]):
            batch_id_per_token_output[start_out + i] = bi

    real_output_token_num = np.array([cum_output], dtype=np.int32)

    return (
        ids_remove_padding,
        batch_id_per_token,
        cu_seqlens_q,
        cu_seqlens_k,
        seq_lens_output,
        cu_seq_lens_q_output,
        batch_id_per_token_output,
        real_output_token_num,
    )


def build_inputs(
    real_bsz,
    max_seq_len,
    max_draft_tokens,
    seq_lens_list,
    seq_lens_encoder_list,
    draft_tokens_data=None,
    input_ids_data=None,
    seed=42,
):
    """
    Helper to build test inputs from explicit seq_lens and seq_lens_encoder lists.
    draft_tokens_data and input_ids_data are optional; if None, random data is used.
    """
    rng = np.random.default_rng(seed)
    seq_lens = np.array(seq_lens_list, dtype=np.int32)
    seq_lens_encoder = np.array(seq_lens_encoder_list, dtype=np.int32)
    seq_lens_decoder = np.zeros(real_bsz, dtype=np.int32)  # not used in kernel logic

    token_num = int(np.sum(seq_lens))

    if input_ids_data is not None:
        input_ids = np.array(input_ids_data, dtype=np.int64).reshape(real_bsz, max_seq_len)
    else:
        input_ids = rng.integers(1, 1000, size=(real_bsz, max_seq_len), dtype=np.int64)

    if draft_tokens_data is not None:
        draft_tokens = np.array(draft_tokens_data, dtype=np.int64).reshape(real_bsz, max_draft_tokens)
    else:
        draft_tokens = rng.integers(1, 1000, size=(real_bsz, max_draft_tokens), dtype=np.int64)

    return {
        "input_ids": input_ids,
        "seq_lens": seq_lens,
        "draft_tokens": draft_tokens,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "max_seq_len": max_seq_len,
        "max_draft_tokens": max_draft_tokens,
        "token_num": token_num,
        "real_bsz": real_bsz,
    }


def run_and_compare(tc, inputs):
    """
    Call GPU op and Python reference, compare all outputs.
    tc: unittest.TestCase instance (for assertion messages).
    """
    real_bsz = inputs["real_bsz"]
    max_seq_len = inputs["max_seq_len"]
    max_draft_tokens = inputs["max_draft_tokens"]
    token_num = inputs["token_num"]

    t_input_ids = paddle.to_tensor(inputs["input_ids"], dtype="int64")
    t_seq_lens = paddle.to_tensor(inputs["seq_lens"], dtype="int32")
    t_draft_tokens = paddle.to_tensor(inputs["draft_tokens"], dtype="int64")
    t_seq_lens_encoder = paddle.to_tensor(inputs["seq_lens_encoder"], dtype="int32")
    t_seq_lens_decoder = paddle.to_tensor(inputs["seq_lens_decoder"], dtype="int32")

    gpu_outs = speculate_pre_process(
        token_num, t_input_ids, t_seq_lens, t_draft_tokens, t_seq_lens_encoder, t_seq_lens_decoder
    )

    ref_outs = speculate_pre_process_ref(
        input_ids=inputs["input_ids"].reshape(-1),
        seq_lens=inputs["seq_lens"],
        draft_tokens=inputs["draft_tokens"].reshape(-1),
        seq_lens_encoder=inputs["seq_lens_encoder"],
        max_seq_len=max_seq_len,
        max_draft_tokens_per_batch=max_draft_tokens,
        real_bsz=real_bsz,
        token_num=token_num,
    )

    output_names = [
        "ids_remove_padding",
        "batch_id_per_token",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "cu_seq_lens_q_output",
        "batch_id_per_token_output",
        "real_output_token_num",
    ]
    # GPU op returns 7 tensors; ref returns 8 (with seq_lens_output at index 4).
    # GPU output order: ids_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k,
    #                   cu_seq_lens_q_output, batch_id_per_token_output, real_output_token_num
    # Ref output order: ids_remove_padding, batch_id_per_token, cu_seqlens_q, cu_seqlens_k,
    #                   seq_lens_output, cu_seq_lens_q_output, batch_id_per_token_output, real_output_token_num
    ref_indices = [0, 1, 2, 3, 5, 6, 7]  # skip seq_lens_output (index 4) for direct comparison

    for name, gpu_idx, ref_idx in zip(output_names, range(7), ref_indices):
        gpu_val = gpu_outs[gpu_idx].numpy()
        ref_val = ref_outs[ref_idx]
        # Trim batch_id_per_token_output to the valid portion (real_output_token_num)
        # The kernel only writes valid positions; beyond that the content is undefined.
        if name == "batch_id_per_token_output":
            valid_len = int(ref_outs[7][0])  # real_output_token_num
            gpu_val = gpu_val[:valid_len]
            ref_val = ref_val[:valid_len]
        np.testing.assert_allclose(
            gpu_val,
            ref_val,
            err_msg=f"Mismatch in output '{name}'",
        )


class TestSpeculatePreProcess(unittest.TestCase):
    """Unit tests for speculate_pre_process custom operator."""

    # ----------------------------------------------------------------
    # Test 1: mixed batch covering all 4 seq_lens_output branches
    #   bid=0: seq_lens=0             => output=0 (skip)
    #   bid=1: seq_lens=1, encoder=0  => output=1, read draft_tokens
    #   bid=2: seq_lens=5, encoder=3  => output=1, read input_ids (prefill)
    #   bid=3: seq_lens=4, encoder=0  => output=4, read draft_tokens (decode)
    #   bid=4: seq_lens=1, encoder=2  => output=1, read input_ids (prefill single)
    #   bid=5: seq_lens=8, encoder=0  => output=8, read draft_tokens (decode saturated)
    # ----------------------------------------------------------------
    def test_mixed_batch_all_branches(self):
        inputs = build_inputs(
            real_bsz=6,
            max_seq_len=16,
            max_draft_tokens=8,
            seq_lens_list=[0, 1, 5, 4, 1, 8],
            seq_lens_encoder_list=[0, 0, 3, 0, 2, 0],
        )
        run_and_compare(self, inputs)

    # ----------------------------------------------------------------
    # Test 2: token_num=0 early return — verify no crash, 7 outputs
    # ----------------------------------------------------------------
    def test_all_zero_seq_lens(self):
        real_bsz = 3
        t_input_ids = paddle.zeros([real_bsz, 8], dtype="int64")
        t_seq_lens = paddle.zeros([real_bsz], dtype="int32")
        t_draft_tokens = paddle.zeros([real_bsz, 4], dtype="int64")
        t_seq_lens_encoder = paddle.zeros([real_bsz], dtype="int32")
        t_seq_lens_decoder = paddle.zeros([real_bsz], dtype="int32")

        gpu_outs = speculate_pre_process(
            0, t_input_ids, t_seq_lens, t_draft_tokens, t_seq_lens_encoder, t_seq_lens_decoder
        )
        self.assertEqual(len(gpu_outs), 7)
        self.assertIsNotNone(gpu_outs[-3])
        self.assertIsNotNone(gpu_outs[-2])
        self.assertIsNotNone(gpu_outs[-1])
        # test copy
        fake_cu_seqlens_q_output = paddle.empty([real_bsz + 1], dtype="int32")
        fake_batch_id_per_token_output = paddle.empty([real_bsz], dtype="int32")
        fake_cu_seqlens_q_output.copy_(gpu_outs[-3])
        fake_batch_id_per_token_output.copy_(gpu_outs[-2])
        # test slice
        fake_batch_id_per_token_output[: gpu_outs[-1].item()]

    # ----------------------------------------------------------------
    # Test 3: exact token values — manually verify ids_remove_padding
    #   bid=0: encoder=0 (decode) => draft_tokens[0][0:3] = [10,11,12]
    #   bid=1: encoder=5 (prefill) => input_ids[1][0:2] = [200,201]
    # ----------------------------------------------------------------
    def test_exact_token_values(self):
        inputs = build_inputs(
            real_bsz=2,
            max_seq_len=4,
            max_draft_tokens=4,
            seq_lens_list=[3, 2],
            seq_lens_encoder_list=[0, 5],
            draft_tokens_data=[[10, 11, 12, 13], [20, 21, 22, 23]],
            input_ids_data=[[100, 101, 102, 103], [200, 201, 202, 203]],
        )

        t_input_ids = paddle.to_tensor(inputs["input_ids"], dtype="int64")
        t_seq_lens = paddle.to_tensor(inputs["seq_lens"], dtype="int32")
        t_draft_tokens = paddle.to_tensor(inputs["draft_tokens"], dtype="int64")
        t_seq_lens_encoder = paddle.to_tensor(inputs["seq_lens_encoder"], dtype="int32")
        t_seq_lens_decoder = paddle.to_tensor(inputs["seq_lens_decoder"], dtype="int32")

        gpu_outs = speculate_pre_process(
            int(np.sum(inputs["seq_lens"])),
            t_input_ids,
            t_seq_lens,
            t_draft_tokens,
            t_seq_lens_encoder,
            t_seq_lens_decoder,
        )

        np.testing.assert_allclose(gpu_outs[0].numpy(), [10, 11, 12, 200, 201])
        np.testing.assert_allclose(gpu_outs[1].numpy(), [0, 0, 0, 1, 1])
        np.testing.assert_allclose(gpu_outs[2].numpy(), [0, 3, 5])
        np.testing.assert_allclose(gpu_outs[6].numpy(), [4])  # real_output_token_num = 3+1

    # ----------------------------------------------------------------
    # Test 4: random stress test (2 configs covering small & medium batch)
    # ----------------------------------------------------------------
    def test_random_configs(self):
        configs = [
            {"real_bsz": 7, "max_seq_len": 32, "max_draft_tokens": 8, "seed": 200},
            {"real_bsz": 32, "max_seq_len": 128, "max_draft_tokens": 16, "seed": 400},
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                rng = np.random.default_rng(cfg["seed"])
                real_bsz = cfg["real_bsz"]
                max_draft = cfg["max_draft_tokens"]
                seq_lens_list = rng.integers(0, max_draft + 1, size=real_bsz).tolist()
                seq_lens_encoder_list = rng.integers(0, 3, size=real_bsz).tolist()

                inputs = build_inputs(
                    real_bsz=real_bsz,
                    max_seq_len=cfg["max_seq_len"],
                    max_draft_tokens=max_draft,
                    seq_lens_list=seq_lens_list,
                    seq_lens_encoder_list=seq_lens_encoder_list,
                    seed=cfg["seed"],
                )
                if inputs["token_num"] == 0:
                    continue
                run_and_compare(self, inputs)


if __name__ == "__main__":
    unittest.main()
