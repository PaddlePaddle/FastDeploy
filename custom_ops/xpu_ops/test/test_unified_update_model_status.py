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
  - Running slots: EOS detection -> state update -> token_ids_all write -> next input setup.
"""

import unittest
from typing import Any, Dict

import numpy as np
import paddle

from fastdeploy.model_executor.ops.xpu import unified_update_model_status

CUDA_PLACE = paddle.XPUPlace(0)
CPU_PLACE = paddle.CPUPlace()


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict -> paddle tensors. has_running_seqs goes to CPU."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif k == "has_running_seqs":
            paddle_inputs[k] = paddle.to_tensor(v, place=CPU_PLACE)
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs


def run_kernel(paddle_inputs: Dict[str, Any]):
    """Call unified_update_model_status kernel."""
    unified_update_model_status(
        paddle_inputs["seq_lens_encoder"],
        paddle_inputs["seq_lens_decoder"],
        paddle_inputs["has_running_seqs"],
        paddle_inputs["step_input_ids"],
        paddle_inputs["step_output_ids"],
        paddle_inputs["step_output_len"],
        paddle_inputs["stop_flags"],
        paddle_inputs["seq_lens_this_time"],
        paddle_inputs["is_paused"],
        paddle_inputs["token_ids_all"],
        paddle_inputs["prompt_lens"],
        paddle_inputs["step_idx"],
        paddle_inputs["end_tokens"],
        paddle_inputs["max_dec_len"],
    )


OUTPUT_KEYS = [
    "seq_lens_encoder",
    "seq_lens_decoder",
    "has_running_seqs",
    "step_input_ids",
    "step_output_ids",
    "step_output_len",
    "stop_flags",
    "seq_lens_this_time",
    "token_ids_all",
    "step_idx",
]


def get_outputs(paddle_inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract ALL in-place-modified tensors back to numpy."""
    return {k: paddle_inputs[k].numpy() for k in OUTPUT_KEYS}


def gen_inputs(
    real_bsz: int = 8,
    max_step_tokens: int = 16,
    max_model_len: int = 256,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    max_bsz = real_bsz + 4

    seq_lens_encoder = rng.integers(0, 5, size=max_bsz, dtype=np.int32)
    seq_lens_decoder = rng.integers(10, 100, size=max_bsz, dtype=np.int32)
    step_input_ids = rng.integers(0, 1000, size=(max_bsz, max_step_tokens), dtype=np.int64)
    step_output_ids = rng.integers(0, 1000, size=(max_bsz, max_step_tokens), dtype=np.int64)
    step_output_len = rng.integers(1, max_step_tokens + 1, size=max_bsz, dtype=np.int32)
    stop_flags = np.zeros(max_bsz, dtype=bool)
    stop_flags[rng.choice(real_bsz, size=min(2, real_bsz), replace=False)] = True
    stop_flags[real_bsz:] = True
    is_paused = np.zeros(max_bsz, dtype=bool)
    prompt_lens = rng.integers(10, 50, size=max_bsz, dtype=np.int64)
    token_ids_all = rng.integers(0, 1000, size=(max_bsz, max_model_len), dtype=np.int64)
    step_idx = rng.integers(0, 50, size=max_bsz, dtype=np.int64)
    max_dec_len = rng.integers(100, 200, size=max_bsz, dtype=np.int64)

    seq_lens_this_time = rng.integers(1, max_step_tokens + 1, size=real_bsz, dtype=np.int32)

    has_running_seqs = np.array([True], dtype=bool)
    end_tokens = rng.integers(1, 1000, size=4, dtype=np.int64)

    return {
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "has_running_seqs": has_running_seqs,
        "step_input_ids": step_input_ids,
        "step_output_ids": step_output_ids,
        "step_output_len": step_output_len,
        "stop_flags": stop_flags,
        "seq_lens_this_time": seq_lens_this_time,
        "is_paused": is_paused,
        "token_ids_all": token_ids_all,
        "prompt_lens": prompt_lens,
        "step_idx": step_idx,
        "end_tokens": end_tokens,
        "max_dec_len": max_dec_len,
        "real_bsz": real_bsz,
        "max_bsz": max_bsz,
        "max_step_tokens": max_step_tokens,
        "max_model_len": max_model_len,
    }


def reference_impl(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference of unified_update_model_status_kernel (GPU version)."""
    seq_lens_encoder = inputs["seq_lens_encoder"].copy()
    seq_lens_decoder = inputs["seq_lens_decoder"].copy()
    step_output_len = inputs["step_output_len"].copy()
    stop_flags = inputs["stop_flags"].copy()
    seq_lens_this_time = inputs["seq_lens_this_time"].copy()
    token_ids_all = inputs["token_ids_all"].copy()
    step_idx = inputs["step_idx"].copy()
    step_input_ids = inputs["step_input_ids"].copy()
    step_output_ids = inputs["step_output_ids"].copy()

    real_bsz = inputs["real_bsz"]
    max_bsz = inputs["max_bsz"]
    max_model_len = inputs["max_model_len"]
    end_tokens = inputs["end_tokens"]
    num_end_tokens = len(end_tokens)
    max_dec_len = inputs["max_dec_len"]
    prompt_lens = inputs["prompt_lens"]
    is_paused = inputs["is_paused"]

    stop_count = 0

    for batch_id in range(max_bsz):
        cur_seq_len_encoder = int(seq_lens_encoder[batch_id])
        cur_seq_len_decoder = int(seq_lens_decoder[batch_id])
        cur_stop_flag = bool(stop_flags[batch_id])
        output_len = int(step_output_len[batch_id])
        cur_step_idx = int(step_idx[batch_id])
        cur_is_paused = bool(is_paused[batch_id])

        is_running = not cur_stop_flag and not cur_is_paused

        # EOS detection
        if is_running and output_len > 0:
            for i in range(output_len):
                cur_step_idx += 1
                token = int(step_output_ids[batch_id, i])
                is_eos = any(token == end_tokens[j] for j in range(num_end_tokens))
                max_len_hit = cur_step_idx >= int(max_dec_len[batch_id])

                if is_eos or max_len_hit:
                    if not is_eos:
                        step_output_ids[batch_id, i] = end_tokens[0]
                    output_len = i + 1
                    cur_stop_flag = True
                    break

        if is_running:
            if cur_seq_len_encoder > 0:
                cur_seq_len_decoder += cur_seq_len_encoder
                cur_seq_len_encoder = 0
            elif cur_seq_len_decoder > 0:
                cur_seq_len_decoder += output_len

            if cur_stop_flag:
                stop_count += 1
                stop_flags[batch_id] = True

            seq_lens_encoder[batch_id] = cur_seq_len_encoder
            seq_lens_decoder[batch_id] = cur_seq_len_decoder
            step_output_len[batch_id] = output_len
            step_idx[batch_id] = cur_step_idx

            # Write history to token_ids_all
            if output_len > 0:
                prompt_len = int(prompt_lens[batch_id])
                if prompt_len + cur_step_idx < max_model_len:
                    base = cur_step_idx - output_len
                    for i in range(output_len):
                        token_ids_all[batch_id, prompt_len + base + i] = step_output_ids[batch_id, i]

            # Prepare next step input
            if output_len > 0:
                step_input_ids[batch_id, 0] = step_output_ids[batch_id, output_len - 1]

        elif batch_id >= real_bsz:
            stop_count += 1
        else:
            stop_count += 1
            stop_flags[batch_id] = True
            seq_lens_encoder[batch_id] = 0
            seq_lens_decoder[batch_id] = 0
            seq_lens_this_time[batch_id] = 0
            step_output_len[batch_id] = 0

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
        "token_ids_all": token_ids_all,
        "step_idx": step_idx,
    }


TEST_CONFIGS = [
    {"name": "basic", "real_bsz": 8, "max_step_tokens": 16, "max_model_len": 256, "seed": 42},
    {"name": "small_batch", "real_bsz": 1, "max_step_tokens": 8, "max_model_len": 128, "seed": 42},
    {"name": "large_batch", "real_bsz": 32, "max_step_tokens": 16, "max_model_len": 512, "seed": 42},
    {"name": "seed_100", "real_bsz": 8, "max_step_tokens": 16, "max_model_len": 256, "seed": 100},
    {"name": "seed_200", "real_bsz": 8, "max_step_tokens": 16, "max_model_len": 256, "seed": 200},
]


class TestUnifiedUpdateModelStatus(unittest.TestCase):

    def setUp(self):
        if not paddle.is_compiled_with_xpu():
            self.skipTest("Requires XPU")

    def _run_and_get(self, inputs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs)
        return get_outputs(paddle_inputs)

    def _check_all_outputs(self, inputs: Dict[str, Any], outputs: Dict[str, np.ndarray]):
        ref = reference_impl(inputs)
        for key in OUTPUT_KEYS:
            np.testing.assert_array_equal(outputs[key], ref[key], err_msg=f"{key} mismatch")

    def _run_full_test(self, config: Dict[str, Any]):
        inputs = gen_inputs(**config)
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_configs(self):
        for cfg in TEST_CONFIGS:
            with self.subTest(name=cfg["name"]):
                test_cfg = {k: v for k, v in cfg.items() if k != "name"}
                self._run_full_test(test_cfg)

    def test_eos_detection(self):
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        eos_token = int(inputs["end_tokens"][0])
        inputs["step_output_ids"][0, 2] = eos_token
        inputs["step_output_len"][:] = [5, 3, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_max_dec_len_stop(self):
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=512, seed=42)
        inputs["step_idx"][:] = [95, 50, 0, 0, 0, 0]
        inputs["max_dec_len"][:] = 100
        inputs["step_output_len"][:] = [10, 5, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_paused_slots(self):
        inputs = gen_inputs(real_bsz=4, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["is_paused"][:] = [True, True, False, False, False, False, False, False]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_all_stopped(self):
        inputs = gen_inputs(real_bsz=4, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["stop_flags"][:] = True
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_encoder_to_decoder(self):
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["seq_lens_encoder"][:] = [10, 0, 0, 0, 0, 0]
        inputs["seq_lens_decoder"][:] = [20, 30, 0, 0, 0, 0]
        inputs["step_output_len"][:] = [5, 3, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_token_ids_all_writing(self):
        inputs = gen_inputs(real_bsz=2, max_step_tokens=8, max_model_len=128, seed=42)
        inputs["step_idx"][:] = [10, 20, 0, 0, 0, 0]
        inputs["prompt_lens"][:] = [5, 5, 0, 0, 0, 0]
        inputs["step_output_len"][:] = [3, 2, 0, 0, 0, 0]
        inputs["stop_flags"][: inputs["real_bsz"]] = False
        inputs["is_paused"][:] = False
        inputs["seq_lens_encoder"][:] = 0
        inputs["end_tokens"][:] = [9990, 9991, 9992, 9993]
        inputs["max_dec_len"][:] = 10000
        inputs["step_output_ids"][0, :3] = [100, 200, 300]
        inputs["step_output_ids"][1, :2] = [400, 500]
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)


if __name__ == "__main__":
    unittest.main()
