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

import os
import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import update_attn_mask_offsets

# 固定随机种子，保证测试可复现
np.random.seed(2023)
paddle.seed(2023)


def py_update_attn_mask_offsets_ref(
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


def generate_test_data(
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    is_block_step,
    max_model_len=8,
    decode_states_len=4,
    vision_generate=False,
):
    """
    Generate test data for both CPU and XPU execution.
    """
    seq_lens_this_time = np.array(seq_lens_this_time, dtype=np.int32).reshape(-1)
    seq_lens_encoder = np.array(seq_lens_encoder, dtype=np.int32).reshape(-1)
    seq_lens_decoder = np.array(seq_lens_decoder, dtype=np.int32).reshape(-1)
    is_block_step = np.array(is_block_step, dtype=bool).reshape(-1)

    bsz = seq_lens_this_time.shape[0]
    total_seq = int(np.sum(seq_lens_this_time))

    # cu_seqlens_q: prefix sum
    cu_seqlens_q = np.zeros((bsz,), dtype=np.int32)
    if bsz > 1:
        cu_seqlens_q[1:] = np.cumsum(seq_lens_this_time[:-1])

    # attn_mask_offsets_full: shape (bsz, max_model_len)
    attn_mask_offsets_full = np.arange(bsz * max_model_len, dtype=np.int32).reshape(bsz, max_model_len)

    # attn_mask_offsets_decoder: initial values
    attn_mask_offsets_decoder = np.array(seq_lens_decoder, dtype=np.int32).copy()

    # decode_states: initial values
    decode_states = np.full((bsz, decode_states_len), -1, dtype=np.int32)
    if vision_generate:
        decode_states[:, 0] = 2  # Trigger vision generate phase

    # mask_rollback
    mask_rollback = np.zeros((bsz,), dtype=np.int32)

    # ids_remove_padding: length = total_seq
    ids_remove_padding = np.random.randint(0, 10, [total_seq], dtype=np.int32)

    return {
        "ids_remove_padding": ids_remove_padding,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "cu_seqlens_q": cu_seqlens_q,
        "attn_mask_offsets_full": attn_mask_offsets_full,
        "attn_mask_offsets_decoder": attn_mask_offsets_decoder,
        "is_block_step": is_block_step,
        "decode_states": decode_states,
        "mask_rollback": mask_rollback,
        "max_model_len": max_model_len,
        "decode_states_len": decode_states_len,
        "total_seq": total_seq,
    }


def to_paddle_tensors(data, device):
    """Convert numpy arrays to paddle tensors on specified device."""
    paddle.set_device(device)
    return {
        "ids_remove_padding": paddle.to_tensor(data["ids_remove_padding"]),
        "seq_lens_this_time": paddle.to_tensor(data["seq_lens_this_time"]),
        "seq_lens_encoder": paddle.to_tensor(data["seq_lens_encoder"]),
        "seq_lens_decoder": paddle.to_tensor(data["seq_lens_decoder"]),
        "cu_seqlens_q": paddle.to_tensor(data["cu_seqlens_q"]),
        "attn_mask_offsets_full": paddle.to_tensor(data["attn_mask_offsets_full"]),
        "attn_mask_offsets_decoder": paddle.to_tensor(data["attn_mask_offsets_decoder"]),
        "is_block_step": paddle.to_tensor(data["is_block_step"]),
        "decode_states": paddle.to_tensor(data["decode_states"]),
        "mask_rollback": paddle.to_tensor(data["mask_rollback"]),
    }


def execute_op(tensors):
    """Execute the update_attn_mask_offsets operator."""
    result = update_attn_mask_offsets(
        tensors["ids_remove_padding"],
        tensors["seq_lens_this_time"],
        tensors["seq_lens_encoder"],
        tensors["seq_lens_decoder"],
        tensors["cu_seqlens_q"],
        tensors["attn_mask_offsets_full"],
        tensors["attn_mask_offsets_decoder"],
        tensors["is_block_step"],
        tensors["decode_states"],
        tensors["mask_rollback"],
    )
    return result


class TestUpdateAttnMaskOffsetsXPU(unittest.TestCase):
    """
    XPU unit test for update_attn_mask_offsets operator.
    Compares CPU wrapper and XPU kernel results.
    """

    def assert_results_equal(self, result_cpu, result_xpu, tensors_cpu, tensors_xpu):
        """Assert CPU and XPU results are equal."""
        # Extract attn_mask_offsets
        if isinstance(result_cpu, (list, tuple)):
            attn_mask_offsets_cpu = result_cpu[0].numpy()
        else:
            attn_mask_offsets_cpu = result_cpu.numpy()

        if isinstance(result_xpu, (list, tuple)):
            attn_mask_offsets_xpu = result_xpu[0].numpy()
        else:
            attn_mask_offsets_xpu = result_xpu.numpy()

        # Compare attn_mask_offsets
        np.testing.assert_array_equal(
            attn_mask_offsets_cpu, attn_mask_offsets_xpu, err_msg="attn_mask_offsets mismatch between CPU and XPU!"
        )

        # Compare inplace tensors
        decode_states_cpu = tensors_cpu["decode_states"].numpy()
        decode_states_xpu = tensors_xpu["decode_states"].numpy()
        np.testing.assert_array_equal(
            decode_states_cpu, decode_states_xpu, err_msg="decode_states mismatch between CPU and XPU!"
        )

        mask_rollback_cpu = tensors_cpu["mask_rollback"].numpy()
        mask_rollback_xpu = tensors_xpu["mask_rollback"].numpy()
        np.testing.assert_array_equal(
            mask_rollback_cpu, mask_rollback_xpu, err_msg="mask_rollback mismatch between CPU and XPU!"
        )

        print("✅ CPU and XPU results are identical!")

    def _run_test_case(
        self,
        seq_lens_this_time,
        seq_lens_encoder,
        seq_lens_decoder,
        is_block_step,
        max_model_len=8,
        decode_states_len=4,
        vision_generate=False,
    ):
        """Run a single test case comparing CPU, XPU and Python reference."""
        # Generate test data
        data = generate_test_data(
            seq_lens_this_time=seq_lens_this_time,
            seq_lens_encoder=seq_lens_encoder,
            seq_lens_decoder=seq_lens_decoder,
            is_block_step=is_block_step,
            max_model_len=max_model_len,
            decode_states_len=decode_states_len,
            vision_generate=vision_generate,
        )

        # Compute Python reference
        ref_attn_mask_offsets, ref_decode_states = py_update_attn_mask_offsets_ref(
            ids_remove_padding_len=data["total_seq"],
            seq_lens_this_time=data["seq_lens_this_time"].copy(),
            seq_lens_encoder=data["seq_lens_encoder"].copy(),
            seq_lens_decoder=data["seq_lens_decoder"].copy(),
            cu_seqlens_q=data["cu_seqlens_q"].copy(),
            attn_mask_offsets_full=data["attn_mask_offsets_full"].copy(),
            attn_mask_offsets_decoder=data["attn_mask_offsets_decoder"].copy(),
            is_block_step=data["is_block_step"].copy(),
            decode_states=data["decode_states"].copy(),
            mask_rollback=data["mask_rollback"].copy(),
        )

        # Convert to tensors
        tensors_cpu = to_paddle_tensors(data, "cpu")
        tensors_xpu = to_paddle_tensors(data, "xpu:0")

        # Execute on CPU and XPU
        paddle.set_device("cpu")
        result_cpu = execute_op(tensors_cpu)

        paddle.set_device("xpu:0")
        result_xpu = execute_op(tensors_xpu)

        # Extract results
        if isinstance(result_cpu, (list, tuple)):
            attn_mask_offsets_cpu = result_cpu[0].numpy()
        else:
            attn_mask_offsets_cpu = result_cpu.numpy()

        if isinstance(result_xpu, (list, tuple)):
            attn_mask_offsets_xpu = result_xpu[0].numpy()
        else:
            attn_mask_offsets_xpu = result_xpu.numpy()

        decode_states_cpu = tensors_cpu["decode_states"].numpy()
        decode_states_xpu = tensors_xpu["decode_states"].numpy()

        attn_mask_offsets_decoder_cpu = tensors_cpu["attn_mask_offsets_decoder"].numpy()
        attn_mask_offsets_decoder_xpu = tensors_xpu["attn_mask_offsets_decoder"].numpy()

        # Debug output
        if os.environ.get("ATTN_MASK_TEST_DEBUG", "0") == "1":
            print("=== DEBUG ===")
            print("seq_lens_this_time:", data["seq_lens_this_time"])
            print("seq_lens_encoder:", data["seq_lens_encoder"])
            print("seq_lens_decoder:", data["seq_lens_decoder"])
            print("Initial attn_mask_offsets_decoder:", data["attn_mask_offsets_decoder"])
            print("After CPU attn_mask_offsets_decoder:", attn_mask_offsets_decoder_cpu)
            print("After XPU attn_mask_offsets_decoder:", attn_mask_offsets_decoder_xpu)
            print("ref_attn_mask_offsets:", ref_attn_mask_offsets)
            print("cpu_attn_mask_offsets:", attn_mask_offsets_cpu)
            print("xpu_attn_mask_offsets:", attn_mask_offsets_xpu)
            print("ref_decode_states:", ref_decode_states)
            print("cpu_decode_states:", decode_states_cpu)
            print("xpu_decode_states:", decode_states_xpu)
            print("=============")

        # Compare Python reference with CPU
        np.testing.assert_array_equal(
            ref_attn_mask_offsets, attn_mask_offsets_cpu, err_msg="Python reference vs CPU mismatch!"
        )
        np.testing.assert_array_equal(
            ref_decode_states, decode_states_cpu, err_msg="Python reference decode_states vs CPU mismatch!"
        )

        # Compare Python reference with XPU
        np.testing.assert_array_equal(
            ref_attn_mask_offsets, attn_mask_offsets_xpu, err_msg="Python reference vs XPU mismatch!"
        )
        np.testing.assert_array_equal(
            ref_decode_states, decode_states_xpu, err_msg="Python reference decode_states vs XPU mismatch!"
        )

        # Compare CPU and XPU results
        self.assert_results_equal(result_cpu, result_xpu, tensors_cpu, tensors_xpu)

    # --- Test cases ---

    def test_stop_case(self):
        """Test stop status: both encoder and decoder are zero."""
        print("\nRunning test: test_stop_case")
        self._run_test_case(
            seq_lens_this_time=[1],
            seq_lens_encoder=[0],
            seq_lens_decoder=[0],
            is_block_step=[False],
            max_model_len=4,
            decode_states_len=2,
        )

    def test_prefill_case(self):
        """Test prefill status: encoder > 0."""
        print("\nRunning test: test_prefill_case")
        self._run_test_case(
            seq_lens_this_time=[3],
            seq_lens_encoder=[3],
            seq_lens_decoder=[0],
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=4,
        )

    def test_vision_generate_prefill(self):
        """Test vision generate phase: decode_states[0] == 2 and seq_len_decoder > 0."""
        print("\nRunning test: test_vision_generate_prefill")
        self._run_test_case(
            seq_lens_this_time=[2],
            seq_lens_encoder=[2],
            seq_lens_decoder=[5],
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=4,
            vision_generate=True,
        )

    def test_decoder_case(self):
        """Test decoder status: seq_len_decoder > 0."""
        print("\nRunning test: test_decoder_case")
        self._run_test_case(
            seq_lens_this_time=[2],
            seq_lens_encoder=[0],
            seq_lens_decoder=[7],
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=6,
        )

    def test_speculative_decoding(self):
        """Test speculative decoding: seq_len_this_time > 1 in decode phase."""
        print("\nRunning test: test_speculative_decoding")
        self._run_test_case(
            seq_lens_this_time=[3],
            seq_lens_encoder=[0],
            seq_lens_decoder=[5],
            is_block_step=[False],
            max_model_len=8,
            decode_states_len=4,
        )

    def test_mixed_batch_case(self):
        """Test mixed batch with different statuses."""
        print("\nRunning test: test_mixed_batch_case")
        self._run_test_case(
            seq_lens_this_time=[2, 4, 1],
            seq_lens_encoder=[0, 4, 0],
            seq_lens_decoder=[5, 0, 1],
            is_block_step=[False, False, False],
            max_model_len=12,
            decode_states_len=2,
        )

    def test_block_step_skip(self):
        """Test is_block_step=True should skip processing."""
        print("\nRunning test: test_block_step_skip")
        self._run_test_case(
            seq_lens_this_time=[2, 3],
            seq_lens_encoder=[2, 0],
            seq_lens_decoder=[0, 5],
            is_block_step=[True, False],
            max_model_len=8,
            decode_states_len=4,
        )

    def test_large_batch(self):
        """Test with larger batch size."""
        print("\nRunning test: test_large_batch")
        bsz = 16
        self._run_test_case(
            seq_lens_this_time=np.random.randint(1, 5, [bsz]).tolist(),
            seq_lens_encoder=np.random.randint(0, 2, [bsz]).tolist(),
            seq_lens_decoder=np.random.randint(0, 10, [bsz]).tolist(),
            is_block_step=[False] * bsz,
            max_model_len=16,
            decode_states_len=8,
        )


if __name__ == "__main__":
    unittest.main()
