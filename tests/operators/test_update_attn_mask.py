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

import os
import unittest

import numpy as np
import paddle

# 请确保你的编译后 op 在这个路径下可导入
from fastdeploy.model_executor.ops.gpu import update_attn_mask_offsets


def py_update_attn_mask_offsets_op(
    ids_remove_padding_len,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    cu_seqlens_q,
    attn_mask_offsets_full,
    attn_mask_offsets_decoder,
    is_block_step,
    decode_states,
    mask_rollback,
):
    """
    Python-side reference op that mirrors the CUDA kernel you provided (latest version).
    - ids_remove_padding_len: 总的去padding后 token 数（用于算 batch_seq_lens）
    - seq_lens_*: 1D numpy int32 arrays (len == bsz)
    - cu_seqlens_q: 1D numpy int32 prefix sums (len == bsz)
    - attn_mask_offsets_full: numpy array shape (bsz, max_model_len)
    - attn_mask_offsets_decoder: 1D numpy int32 (bsz,)
    - is_block_step: 1D bool array (bsz,)
    - decode_states: numpy int32 array shape (bsz, decode_states_len)
    - mask_rollback: 1D numpy int32 (bsz,) or shape (bsz,1)
    Returns:
      attn_mask_offsets_ref (1D int32 length batch_seq_lens * 2),
      decode_states_ref (bsz x decode_states_len int32)
    """
    # normalize inputs
    seq_lens_this_time = np.array(seq_lens_this_time, dtype=np.int32).reshape(-1)
    seq_lens_encoder = np.array(seq_lens_encoder, dtype=np.int32).reshape(-1)
    seq_lens_decoder = np.array(seq_lens_decoder, dtype=np.int32).reshape(-1)
    cu_seqlens_q = np.array(cu_seqlens_q, dtype=np.int32).reshape(-1)
    is_block_step = np.array(is_block_step, dtype=bool).reshape(-1)
    attn_mask_offsets_full = np.array(attn_mask_offsets_full, dtype=np.int32)
    attn_mask_offsets_decoder = np.array(attn_mask_offsets_decoder, dtype=np.int32).reshape(-1)
    decode_states = np.array(decode_states, dtype=np.int32).copy()
    mask_rollback = np.array(mask_rollback, dtype=np.int32).reshape(-1)

    bsz = int(seq_lens_this_time.shape[0])
    total_seq = int(np.sum(seq_lens_this_time))
    decode_states_len = int(decode_states.shape[1])

    # CUDA creates paddle::full({batch_seq_lens * 2}, 0)
    attn_mask_offsets = np.zeros((total_seq * 2,), dtype=np.int32)

    for bid in range(bsz):
        if is_block_step[bid]:
            # skip update for this batch entry
            continue

        seq_len_this = int(seq_lens_this_time[bid])
        seq_len_enc = int(seq_lens_encoder[bid])
        seq_len_dec = int(seq_lens_decoder[bid])
        query_start = int(cu_seqlens_q[bid])
        # pointer-like views in C++: attn_mask_offsets_full_now, decode_states_now
        full_now = attn_mask_offsets_full[bid]
        decode_now = decode_states[bid]  # this is a view into decode_states

        # stop: both zero => do nothing
        if seq_len_enc == 0 and seq_len_dec == 0:
            continue

        # prefill path (encoder > 0)
        if seq_len_enc > 0:
            for i in range(seq_len_this):
                # vision generate phase check: (*decode_states_now == 2 && seq_len_decoder > 0)
                # In C++ code they used '*decode_states_now == 2' — meaning first element compare.
                if decode_now.size > 0 and decode_now[0] == 2 and seq_len_dec > 0:
                    attn_mask_offsets[(query_start + i) * 2 + 1] = seq_len_dec + seq_len_this
                else:
                    # attn_mask_offsets_full_now[i] + 1
                    attn_mask_offsets[(query_start + i) * 2 + 1] = int(full_now[i]) + 1
            # done prefill branch
            continue

        # decoder path (seq_len_decoder > 0)
        if seq_len_dec > 0:
            # subtract mask rollback
            rollback = int(mask_rollback[bid]) if bid < mask_rollback.shape[0] else 0
            attn_mask_offsets_decoder[bid] = int(attn_mask_offsets_decoder[bid]) - rollback
            start = int(attn_mask_offsets_decoder[bid])

            for i in range(seq_len_this):
                attn_mask_offsets[(query_start + i) * 2 + 1] = start + 1 + i

            # advance decoder offset
            attn_mask_offsets_decoder[bid] = int(attn_mask_offsets_decoder[bid]) + seq_len_this

            # speculative decoding: if seq_len_this > 1 then set decode_states_now[i] accordingly
            if seq_len_this > 1:
                for i in range(decode_states_len):
                    decode_now[i] = 0 if i < seq_len_this else -1
            # done decoder branch
            continue

    return attn_mask_offsets, decode_states


class UpdateAttnMaskOffsetsTestCase(unittest.TestCase):
    def setUp(self):
        # If GPU available, use it. But we don't hard require CUDA here; op itself must be callable.
        # Ensure Paddle uses GPU if available to match operator placement
        try:
            paddle.set_device("gpu")
        except Exception:
            paddle.set_device("cpu")

    def _call_and_compare(
        self,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        is_block_step,
        max_model_len=8,
        decode_states_len=4,
        vision_generate=False,
    ):
        # build numpy inputs
        seq_lens_this_time = np.array(seq_lens_this_time, dtype=np.int32).reshape(-1)
        seq_lens_encoder = np.array(seq_lens_encoder, dtype=np.int32).reshape(-1)
        seq_lens_decoder = np.array(seq_lens_decoder, dtype=np.int32).reshape(-1)
        bsz = seq_lens_this_time.shape[0]
        total_seq = int(np.sum(seq_lens_this_time))
        cu_seqlens_q = np.zeros((bsz,), dtype=np.int32)
        if bsz > 1:
            cu_seqlens_q[1:] = np.cumsum(seq_lens_this_time[:-1])

        # attn_mask_offsets_full: shape (bsz, max_model_len)
        attn_mask_offsets_full = np.arange(bsz * max_model_len, dtype=np.int32).reshape(bsz, max_model_len)

        # attn_mask_offsets_decoder initial (use seq_lens_decoder as seed for deterministic test)
        attn_mask_offsets_decoder = np.array(seq_lens_decoder, dtype=np.int32).copy()

        # decode_states initial
        decode_states = np.full((bsz, decode_states_len), -1, dtype=np.int32)
        if vision_generate:
            decode_states[:, 0] = 2  # make first element 2 to trigger vision phase

        mask_rollback = np.zeros((bsz,), dtype=np.int32)

        # ids_remove_padding: length = total_seq (only length used by op)
        ids_remove_padding = paddle.randint(low=0, high=10, shape=[total_seq], dtype="int32")
        decode_states_tensor = paddle.to_tensor(decode_states, dtype="int32")
        # prepare paddle tensors and call the compiled op
        out = update_attn_mask_offsets(
            ids_remove_padding,
            paddle.to_tensor(seq_lens_this_time, dtype="int32"),
            paddle.to_tensor(seq_lens_encoder, dtype="int32"),
            paddle.to_tensor(seq_lens_decoder, dtype="int32"),
            paddle.to_tensor(cu_seqlens_q, dtype="int32"),
            paddle.to_tensor(attn_mask_offsets_full, dtype="int32"),
            paddle.to_tensor(attn_mask_offsets_decoder, dtype="int32"),
            paddle.to_tensor(np.array(is_block_step, dtype=bool).reshape(-1), dtype="bool"),
            decode_states_tensor,
            paddle.to_tensor(mask_rollback, dtype="int32"),
        )

        # op returns [attn_mask_offsets, decode_states_out] per your PD_BUILD_STATIC_OP outputs
        if isinstance(out, (list, tuple)):
            op_attn_mask_offsets = out[0].numpy().astype(np.int32).reshape(-1)
            op_decode_states = out[1].numpy().astype(np.int32)
        else:
            # Some bindings might return single tensor and inplace decode_states update
            # Try to handle that case: assume attn_mask_offsets returned and decode_states was mutated inplace.
            op_attn_mask_offsets = out.numpy().astype(np.int32).reshape(-1)
            # fetch decode_states by re-creating input decode_states tensor? best effort:
            # (we passed decode_states as a paddle tensor; in operator we passed a copy, but PD set inplace mapping
            #  so many builds will actually give decode_states_out as second output; this block is fallback.)
            op_decode_states = decode_states_tensor.numpy()

        # compute python reference outputs
        ref_attn_mask_offsets, ref_decode_states = py_update_attn_mask_offsets_op(
            ids_remove_padding_len=total_seq,
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            cu_seqlens_q=cu_seqlens_q,
            attn_mask_offsets_full=attn_mask_offsets_full,
            attn_mask_offsets_decoder=attn_mask_offsets_decoder.copy(),
            is_block_step=np.array(is_block_step, dtype=bool).reshape(-1),
            decode_states=decode_states.copy(),
            mask_rollback=mask_rollback,
        )

        # optionally print debug if env var set
        if os.environ.get("ATTN_MASK_TEST_DEBUG", "0") == "1":
            print("=== DEBUG ===")
            print("seq_lens_this_time:", seq_lens_this_time)
            print("seq_lens_encoder:", seq_lens_encoder)
            print("seq_lens_decoder:", seq_lens_decoder)
            print("cu_seqlens_q:", cu_seqlens_q)
            print("ref_attn_mask_offsets:", ref_attn_mask_offsets)
            print("op_attn_mask_offsets:", op_attn_mask_offsets)
            print("ref_decode_states:", ref_decode_states)
            print("op_decode_states:", op_decode_states)
            print("=============")

        # shape checks
        self.assertEqual(
            op_attn_mask_offsets.shape,
            ref_attn_mask_offsets.shape,
            f"attn_mask_offsets shape mismatch: op {op_attn_mask_offsets.shape}, ref {ref_attn_mask_offsets.shape}",
        )
        # element-wise equality
        np.testing.assert_array_equal(op_attn_mask_offsets, ref_attn_mask_offsets)
        np.testing.assert_array_equal(op_decode_states, ref_decode_states)

    # --- Test cases below (cover branches) ---

    def test_stop_case(self):
        # stop: both encoder and decoder are zero -> nothing written (all zeros)
        self._call_and_compare(
            seq_lens_this_time=[1],
            seq_lens_encoder=[0],
            seq_lens_decoder=[0],
            is_block_step=[False],
            max_model_len=4,
            decode_states_len=2,
        )

    def test_prefill_case(self):
        # prefill: encoder > 0, should copy attn_mask_offsets_full[i] + 1 into positions ((q+i)*2+1)
        self._call_and_compare(
            seq_lens_this_time=[3],
            seq_lens_encoder=[3],
            seq_lens_decoder=[0],
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=4,
        )

    def test_vision_generate_prefill(self):
        # vision generate: decode_states[0] == 2 and seq_len_decoder > 0 triggers alternate write
        self._call_and_compare(
            seq_lens_this_time=[2],
            seq_lens_encoder=[2],
            seq_lens_decoder=[5],  # >0 to activate vision branch
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=4,
            vision_generate=True,
        )

    def test_decoder_case(self):
        # decoder path: should write attn_mask_offsets_decoder - rollback + 1 .. +seq_len_this_time-1
        self._call_and_compare(
            seq_lens_this_time=[2],
            seq_lens_encoder=[0],
            seq_lens_decoder=[7],
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=6,
        )

    def test_mixed_batch_case(self):
        # mixed batch with different statuses
        self._call_and_compare(
            seq_lens_this_time=[2, 4, 1],
            seq_lens_encoder=[0, 4, 0],
            seq_lens_decoder=[5, 0, 1],
            is_block_step=[False, False, False],
            max_model_len=12,
            decode_states_len=2,
        )


if __name__ == "__main__":
    unittest.main()
