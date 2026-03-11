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
Unit tests for unified_update_model_status kernel.

Kernel semantics (from unified_update_model_status.cu):
  - Launched as <<<1, 1024>>>, one thread per batch slot (max_bsz <= 1024).
  - real_bsz = seq_lens_this_time.shape[0], max_bsz = stop_flags.shape[0].
  - has_running_seqs is a CPU tensor (copied to GPU, kernel writes, copied back).
  - Padding slots (batch_id >= real_bsz): only counted as stopped, NO state modified.
  - Stopped/paused real slots: set stop_flags=true, seq_lens_decoder=0,
    seq_lens_this_time=0, step_output_len=0.
  - Running slots: EOS detection → state update → token_ids_all write → next input setup.
"""

import unittest
from typing import Any, Dict

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import unified_update_model_status

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
CPU_PLACE = paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers — tensor creation / kernel invocation / output extraction
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict → paddle tensors. has_running_seqs goes to CPU."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif k == "has_running_seqs":
            # Kernel host function: has_running_seqs.copy_to(GPU) → kernel → copy_to(CPU)
            paddle_inputs[k] = paddle.to_tensor(v, place=CPU_PLACE)
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs


def run_kernel(paddle_inputs: Dict[str, Any], inputs: Dict[str, Any]):
    """Call unified_update_model_status kernel."""
    unified_update_model_status(
        paddle_inputs["seq_lens_encoder"],
        paddle_inputs["seq_lens_decoder"],
        paddle_inputs["has_running_seqs"],
        paddle_inputs["step_input_ids"],
        paddle_inputs["adaptive_step_input_len"],
        paddle_inputs["step_output_ids"],
        paddle_inputs["step_output_len"],
        paddle_inputs["stop_flags"],
        paddle_inputs["seq_lens_this_time"],
        paddle_inputs["is_paused"],
        paddle_inputs["mask_rollback"],
        paddle_inputs["token_ids_all"],
        paddle_inputs["prompt_lens"],
        paddle_inputs["step_idx"],
        paddle_inputs["end_tokens"],
        paddle_inputs["max_dec_len"],
        inputs["is_naive_mode"],
        inputs["prefill_one_step_stop"],
    )


# All 12 in-place output keys (from SetInplaceMap in .cu)
OUTPUT_KEYS = [
    "seq_lens_encoder",
    "seq_lens_decoder",
    "has_running_seqs",
    "step_input_ids",
    "step_output_ids",
    "step_output_len",
    "stop_flags",
    "seq_lens_this_time",
    "mask_rollback",
    "token_ids_all",
    "step_idx",
    # adaptive_step_input_len is in InplaceMap but kernel never writes it
]


def get_outputs(paddle_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract ALL in-place-modified tensors back to numpy."""
    return {k: paddle_inputs[k].numpy() for k in OUTPUT_KEYS}


# ============================================================
# Layer 2: Input generation
# ============================================================


def gen_inputs(
    real_bsz: int = 8,
    max_step_tokens: int = 16,
    max_model_len: int = 256,
    seed: int = 42,
    is_naive_mode: bool = False,
    prefill_one_step_stop: bool = False,
) -> Dict[str, Any]:
    """Generate randomized test inputs for unified_update_model_status kernel.

    Shapes follow the kernel contract:
      - real_bsz = seq_lens_this_time.shape[0]
      - max_bsz  = stop_flags.shape[0]  (= real_bsz + padding)
      - is_paused.shape[0] = max_bsz
    """
    rng = np.random.default_rng(seed)
    max_bsz = real_bsz + 4  # padding slots

    # Per-slot arrays (size=max_bsz)
    seq_lens_encoder = rng.integers(0, 5, size=max_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(10, 100, size=max_bsz, dtype=np.int32)
    step_input_ids = rng.integers(0, 1000, size=(max_bsz, max_step_tokens), dtype=np.int64)
    adaptive_step_input_len = rng.integers(1, max_step_tokens + 1, size=max_bsz, dtype=np.int32)
    step_output_ids = rng.integers(0, 1000, size=(max_bsz, max_step_tokens), dtype=np.int64)
    step_output_len = rng.integers(1, max_step_tokens + 1, size=max_bsz, dtype=np.int32)
    stop_flags = np.zeros(max_bsz, dtype=bool)
    # Randomly stop a few real slots
    stop_flags[rng.choice(real_bsz, size=min(2, real_bsz), replace=False)] = True
    # Padding slots (batch_id >= real_bsz) must be stopped — kernel accesses
    # seq_lens_this_time[batch_id] which is only sized real_bsz
    stop_flags[real_bsz:] = True
    is_paused = np.zeros(max_bsz, dtype=bool)
    mask_rollback = np.zeros(max_bsz, dtype=np.int32)
    prompt_lens = rng.integers(10, 50, size=max_bsz, dtype=np.int64)
    token_ids_all = rng.integers(0, 1000, size=(max_bsz, max_model_len), dtype=np.int64)
    step_idx = rng.integers(0, 50, size=max_bsz, dtype=np.int64)
    max_dec_len = rng.integers(100, 200, size=max_bsz, dtype=np.int64)

    # Per-real-batch arrays (size=real_bsz)
    seq_lens_this_time = rng.integers(1, max_step_tokens + 1, size=real_bsz, dtype=np.int32)

    # Scalar / small tensors
    has_running_seqs = np.array([True], dtype=bool)
    end_tokens = rng.integers(1, 1000, size=4, dtype=np.int64)

    return {
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "has_running_seqs": has_running_seqs,
        "step_input_ids": step_input_ids,
        "adaptive_step_input_len": adaptive_step_input_len,
        "step_output_ids": step_output_ids,
        "step_output_len": step_output_len,
        "stop_flags": stop_flags,
        "seq_lens_this_time": seq_lens_this_time,
        "is_paused": is_paused,
        "mask_rollback": mask_rollback,
        "token_ids_all": token_ids_all,
        "prompt_lens": prompt_lens,
        "step_idx": step_idx,
        "end_tokens": end_tokens,
        "max_dec_len": max_dec_len,
        # Scalar configs
        "real_bsz": real_bsz,
        "max_bsz": max_bsz,
        "max_step_tokens": max_step_tokens,
        "max_model_len": max_model_len,
        "is_naive_mode": is_naive_mode,
        "prefill_one_step_stop": prefill_one_step_stop,
    }


# ============================================================
# Layer 3: Reference implementation (1:1 with CUDA kernel)
# ============================================================


def reference_impl(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference of unified_update_model_status_kernel.

    Line references are to unified_update_model_status.cu.
    """
    # Deep-copy all mutable in-place tensors
    seq_lens_encoder = inputs["seq_lens_encoder"].copy()
    seq_lens_decoder = inputs["seq_lens_decoder"].copy()
    step_output_len = inputs["step_output_len"].copy()
    stop_flags = inputs["stop_flags"].copy()
    seq_lens_this_time = inputs["seq_lens_this_time"].copy()
    mask_rollback = inputs["mask_rollback"].copy()
    token_ids_all = inputs["token_ids_all"].copy()
    step_idx = inputs["step_idx"].copy()
    step_input_ids = inputs["step_input_ids"].copy()
    step_output_ids = inputs["step_output_ids"].copy()

    # Read-only inputs
    real_bsz = inputs["real_bsz"]
    max_bsz = inputs["max_bsz"]
    max_model_len = inputs["max_model_len"]
    is_naive_mode = inputs["is_naive_mode"]
    prefill_one_step_stop = inputs["prefill_one_step_stop"]
    end_tokens = inputs["end_tokens"]
    num_end_tokens = len(end_tokens)
    max_dec_len = inputs["max_dec_len"]
    prompt_lens = inputs["prompt_lens"]
    is_paused = inputs["is_paused"]

    # Block-level stop count for has_running_seqs reduction (line 175)
    stop_count = 0

    for batch_id in range(max_bsz):
        # --- line 68-75: Read state ---
        cur_seq_len_encoder = int(seq_lens_encoder[batch_id])
        cur_seq_len_decoder = int(seq_lens_decoder[batch_id])
        cur_stop_flag = bool(stop_flags[batch_id])
        output_len = 0
        cur_step_idx = int(step_idx[batch_id])
        cur_is_paused = bool(is_paused[batch_id])

        # line 77
        is_running = not cur_stop_flag and not cur_is_paused

        # --- line 80-86: Compute output length ---
        if is_running:
            output_len = 1 if is_naive_mode else int(step_output_len[batch_id])

        # --- line 89-110: EOS detection ---
        if is_running and output_len > 0:
            hit_stop = False
            for i in range(output_len):
                cur_step_idx += 1  # line 94
                token = int(step_output_ids[batch_id, i])  # line 95
                is_eos = any(token == end_tokens[j] for j in range(num_end_tokens))  # line 96
                max_len_hit = cur_step_idx >= int(max_dec_len[batch_id])  # line 97

                if is_eos or max_len_hit:  # line 99
                    if not is_eos:
                        step_output_ids[batch_id, i] = end_tokens[0]  # line 100
                    output_len = i + 1  # line 101
                    cur_stop_flag = True  # line 102
                    hit_stop = True  # line 103
                    break  # line 104

            # line 108-110
            if not hit_stop and prefill_one_step_stop and cur_seq_len_encoder > 0:
                cur_stop_flag = True

        # --- line 114-166: Update state and write back ---
        if is_running:
            if cur_stop_flag:
                # line 115-119
                stop_count += 1
                if output_len == 0:
                    cur_seq_len_decoder = 0  # line 117
                stop_flags[batch_id] = True  # line 118
                mask_rollback[batch_id] = 0  # line 119
            elif cur_seq_len_encoder == 0:
                # line 120-122
                cur_seq_len_decoder += output_len  # line 121
                mask_rollback[batch_id] = int(seq_lens_this_time[batch_id]) - output_len  # line 122
            else:
                # line 123-124 (encoder > 0, not stopped)
                mask_rollback[batch_id] = 0

            # line 127-130: Fold encoder into decoder
            if cur_seq_len_encoder > 0:
                cur_seq_len_decoder += cur_seq_len_encoder  # line 128
                cur_seq_len_encoder = 0  # line 129

            # line 132-135: Write back scalar state
            seq_lens_encoder[batch_id] = cur_seq_len_encoder
            seq_lens_decoder[batch_id] = cur_seq_len_decoder
            step_output_len[batch_id] = output_len
            step_idx[batch_id] = cur_step_idx

            # line 138-145: Write history to token_ids_all
            if cur_step_idx > 0 and output_len > 0:
                base = int(prompt_lens[batch_id])
                for i in range(output_len):
                    # token_ids_all_now[cur_step_idx - i] = output_ids[output_len - 1 - i]
                    write_idx = base + cur_step_idx - i
                    if 0 <= write_idx < max_model_len:
                        token_ids_all[batch_id, write_idx] = step_output_ids[batch_id, output_len - 1 - i]

            # line 148-151: Setup next step_input_ids
            if output_len > 0:
                step_input_ids[batch_id, 0] = step_output_ids[batch_id, output_len - 1]

            # line 153-155: naive_mode → seq_lens_this_time
            if is_naive_mode:
                seq_lens_this_time[batch_id] = 0 if cur_stop_flag else 1

        elif batch_id >= real_bsz:
            # line 156-158: Padding slot — only count, don't modify state
            stop_count += 1
        else:
            # line 159-166: Stopped or paused real slot
            stop_count += 1
            stop_flags[batch_id] = True  # line 162
            seq_lens_decoder[batch_id] = 0  # line 163
            seq_lens_this_time[batch_id] = 0  # line 164
            step_output_len[batch_id] = 0  # line 165

    # line 177-179: has_running_seqs = stop_sum < max_bsz
    has_running_seqs = np.array([stop_count < max_bsz], dtype=bool)

    return {
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "has_running_seqs": has_running_seqs,
        "step_input_ids": step_input_ids,
        "step_output_ids": step_output_ids,
        "step_output_len": step_output_len,
        "stop_flags": stop_flags,
        "seq_lens_this_time": seq_lens_this_time,
        "mask_rollback": mask_rollback,
        "token_ids_all": token_ids_all,
        "step_idx": step_idx,
    }


# ============================================================
# Layer 4a: TEST_CONFIGS
# ============================================================

TEST_CONFIGS = [
    # --- basic mode coverage ---
    {
        "name": "mtp_mode",
        "real_bsz": 8,
        "max_step_tokens": 16,
        "max_model_len": 256,
        "seed": 42,
        "is_naive_mode": False,
    },
    {
        "name": "naive_mode",
        "real_bsz": 8,
        "max_step_tokens": 16,
        "max_model_len": 256,
        "seed": 42,
        "is_naive_mode": True,
    },
    # --- batch size ---
    {
        "name": "small_batch",
        "real_bsz": 1,
        "max_step_tokens": 8,
        "max_model_len": 128,
        "seed": 42,
        "is_naive_mode": False,
    },
    {
        "name": "large_batch",
        "real_bsz": 32,
        "max_step_tokens": 16,
        "max_model_len": 512,
        "seed": 42,
        "is_naive_mode": False,
    },
    # --- prefill_one_step_stop ---
    {
        "name": "prefill_one_step_stop",
        "real_bsz": 8,
        "max_step_tokens": 8,
        "max_model_len": 128,
        "seed": 42,
        "is_naive_mode": False,
        "prefill_one_step_stop": True,
    },
    # --- different seeds for randomized coverage ---
    {
        "name": "seed_100",
        "real_bsz": 8,
        "max_step_tokens": 16,
        "max_model_len": 256,
        "seed": 100,
        "is_naive_mode": False,
    },
    {
        "name": "seed_200_naive",
        "real_bsz": 8,
        "max_step_tokens": 16,
        "max_model_len": 256,
        "seed": 200,
        "is_naive_mode": True,
    },
]


# ============================================================
# Layer 4b: Test suite
# ============================================================


class TestUnifiedUpdateModelStatus(unittest.TestCase):

    def setUp(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("Requires CUDA")

    # ------ shared helpers ------

    def _run_and_get(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs, inputs)
        return get_outputs(paddle_inputs)

    def _check_all_outputs(self, inputs: Dict[str, Any], outputs: Dict[str, np.ndarray]):
        """Compare ALL output tensors against reference + sanity checks."""
        ref = reference_impl(inputs)
        for key in OUTPUT_KEYS:
            if not np.array_equal(outputs[key], ref[key]):
                diff_mask = outputs[key] != ref[key]
                diff_indices = np.argwhere(diff_mask)
                for idx in diff_indices[:10]:  # print first 10 mismatches
                    idx_tuple = tuple(idx)
                    print(
                        f"  [{key}] mismatch at {idx_tuple}: "
                        f"gpu={outputs[key][idx_tuple]}  ref={ref[key][idx_tuple]}"
                    )
                    if key == "token_ids_all":
                        bid = idx_tuple[0]
                        print(
                            f"    batch_id={bid}, prompt_lens={inputs['prompt_lens'][bid]}, "
                            f"step_idx(input)={inputs['step_idx'][bid]}, "
                            f"step_idx(gpu)={outputs['step_idx'][bid]}, "
                            f"step_idx(ref)={ref['step_idx'][bid]}, "
                            f"step_output_len(gpu)={outputs['step_output_len'][bid]}, "
                            f"step_output_len(ref)={ref['step_output_len'][bid]}, "
                            f"stop_flags(input)={inputs['stop_flags'][bid]}, "
                            f"is_paused={inputs['is_paused'][bid]}, "
                            f"seq_lens_encoder={inputs['seq_lens_encoder'][bid]}"
                        )
                np.testing.assert_array_equal(outputs[key], ref[key], err_msg=f"{key} mismatch")

        # Sanity: running slots must have encoder zeroed
        for i in range(inputs["real_bsz"]):
            if not inputs["stop_flags"][i] and not inputs["is_paused"][i]:
                self.assertEqual(outputs["seq_lens_encoder"][i], 0, f"Running slot {i} should have encoder=0")
        self.assertTrue(np.all(outputs["seq_lens_decoder"] >= 0), "negative seq_lens_decoder")
        self.assertTrue(np.all(outputs["step_output_len"] >= 0), "negative step_output_len")
        self.assertTrue(np.all(outputs["step_idx"] >= 0), "negative step_idx")

    def _run_full_test(self, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
        inputs = gen_inputs(**config)
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        return outputs

    # ------ test cases ------

    def test_configs(self):
        """Run all TEST_CONFIGS via subTest."""
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                self._run_full_test(test_cfg)

    def test_eos_detection(self):
        """EOS token at position 2 should truncate output_len to 3."""
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        eos_token = int(inputs["end_tokens"][0])
        inputs["step_output_ids"][0, 2] = eos_token
        inputs["step_output_len"][:] = [5, 3, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_max_dec_len_stop(self):
        """step_idx near max_dec_len should trigger stop and replace with end_tokens[0]."""
        # Use large max_model_len to avoid token_ids_all overflow:
        # kernel doesn't bounds-check prompt_lens + step_idx < max_model_len
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=512, seed=42)
        inputs["step_idx"][:] = [95, 50, 0, 0, 0, 0]
        inputs["max_dec_len"][:] = 100
        inputs["step_output_len"][:] = [10, 5, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_paused_slots(self):
        """Paused slots should be treated as stopped/paused (decoder=0, output_len=0)."""
        inputs = gen_inputs(real_bsz=4, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["is_paused"][:] = [True, True, False, False, False, False, False, False]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_all_stopped(self):
        """All slots stopped → has_running_seqs should be False."""
        inputs = gen_inputs(real_bsz=4, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["stop_flags"][:] = True
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_encoder_to_decoder(self):
        """Encoder length should fold into decoder: decoder += encoder, encoder → 0."""
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["seq_lens_encoder"][:] = [10, 0, 0, 0, 0, 0]
        inputs["seq_lens_decoder"][:] = [20, 30, 0, 0, 0, 0]
        inputs["step_output_len"][:] = [5, 3, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_token_ids_all_writing(self):
        """token_ids_all should be written at prompt_lens + step_idx positions."""
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["step_idx"][:] = [10, 20, 0, 0, 0, 0]
        inputs["prompt_lens"][:] = [5, 5, 0, 0, 0, 0]
        inputs["step_output_len"][:] = [3, 2, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        inputs["seq_lens_encoder"][:] = 0
        # Use end_tokens that won't collide with output_ids
        inputs["end_tokens"][:] = [9990, 9991, 9992, 9993]
        inputs["max_dec_len"][:] = 10000
        inputs["step_output_ids"][0, :3] = [100, 200, 300]
        inputs["step_output_ids"][1, :2] = [400, 500]
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_zero_output_len(self):
        """Running slot with output_len=0 in MTP mode: output_len stays 0."""
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["step_output_len"][:] = [0, 5, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_prefill_one_step_stop_with_encoder(self):
        """prefill_one_step_stop + encoder>0 should stop even without EOS."""
        inputs = gen_inputs(real_bsz=4, max_step_tokens=8, max_model_len=128, seed=42, prefill_one_step_stop=True)
        inputs["seq_lens_encoder"][:] = [5, 0, 0, 0, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        # Ensure no accidental EOS hit
        inputs["end_tokens"][:] = [9990, 9991, 9992, 9993]
        inputs["max_dec_len"][:] = 10000
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_mask_rollback(self):
        """mask_rollback = seq_lens_this_time - output_len for running decode slots."""
        inputs = gen_inputs(real_bsz=4, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        inputs["seq_lens_encoder"][:] = 0  # All decode slots
        inputs["seq_lens_this_time"][:] = [6, 4, 8, 3]
        inputs["step_output_len"][:] = [3, 2, 5, 1, 0, 0, 0, 0]
        # Avoid EOS/max_dec_len hits
        inputs["end_tokens"][:] = [9990, 9991, 9992, 9993]
        inputs["max_dec_len"][:] = 10000
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)


if __name__ == "__main__":
    unittest.main()
