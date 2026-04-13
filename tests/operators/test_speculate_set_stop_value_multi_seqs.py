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
from typing import Any, Dict

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import speculate_set_stop_value_multi_seqs

CUDA_PLACE = paddle.CUDAPlace(0) if paddle.is_compiled_with_cuda() else paddle.CPUPlace()
CPU_PLACE = paddle.CPUPlace()


# ============================================================
# Layer 1: Helpers — tensor creation / kernel invocation / output extraction
# ============================================================


def to_paddle_inputs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy dict -> paddle tensors. All tensors are on GPU."""
    paddle_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, (int, bool, float, str)):
            paddle_inputs[k] = v
        elif v is not None:
            paddle_inputs[k] = paddle.to_tensor(v, place=CUDA_PLACE)
        else:
            paddle_inputs[k] = None
    return paddle_inputs


def run_kernel(paddle_inputs):
    """Call the CUDA kernel."""
    speculate_set_stop_value_multi_seqs(
        paddle_inputs["accept_tokens"],
        paddle_inputs["accept_num"],
        paddle_inputs["token_ids_all"],
        paddle_inputs["prompt_lens"],
        paddle_inputs["step_idx"],
        paddle_inputs["stop_flags"],
        paddle_inputs["seq_lens"],
        paddle_inputs["stop_seqs"],
        paddle_inputs["stop_seqs_len"],
        paddle_inputs["end_ids"],
        paddle_inputs["min_tokens"],
    )


def get_outputs(paddle_inputs) -> Dict[str, np.ndarray]:
    """Extract all in-place-modified tensors back to numpy."""
    keys = ["accept_tokens", "accept_num"]
    return {k: paddle_inputs[k].numpy() for k in keys}


# ============================================================
# Layer 2: Input generation
# ============================================================


def gen_inputs(
    real_bsz=2,
    accept_tokens_len=5,
    max_model_len=32,
    stop_seqs_bs=2,
    stop_seqs_max_len=4,
    prompt_len_range=(0, 5),
    step_idx_range=(5, 15),
    accept_num_range=(1, 4),
    vocab_size=100,
    seed=42,
) -> Dict[str, Any]:
    """Generate randomized test inputs matching kernel shapes/dtypes."""
    rng = np.random.default_rng(seed)

    prompt_lens = rng.integers(prompt_len_range[0], prompt_len_range[1], size=(real_bsz, 1)).astype("int64")
    step_idx = rng.integers(step_idx_range[0], step_idx_range[1], size=(real_bsz,)).astype("int64")
    accept_num = rng.integers(
        accept_num_range[0], min(accept_num_range[1], accept_tokens_len) + 1, size=(real_bsz,)
    ).astype("int32")

    # token_ids_all: [bsz, max_model_len] — fill with random tokens
    token_ids_all = rng.integers(1, vocab_size, size=(real_bsz, max_model_len)).astype("int64")

    # accept_tokens: [bsz, accept_tokens_len] — first accept_num[i] slots are valid
    accept_tokens = np.zeros((real_bsz, accept_tokens_len), dtype="int64")
    for i in range(real_bsz):
        accept_tokens[i, : accept_num[i]] = rng.integers(1, vocab_size, size=accept_num[i])

    stop_flags = np.zeros(real_bsz, dtype="bool")
    seq_lens = (step_idx + accept_num).astype("int32")

    # stop_seqs: [bsz, stop_seqs_bs, stop_seqs_max_len]
    stop_seqs = rng.integers(1, vocab_size, size=(real_bsz, stop_seqs_bs, stop_seqs_max_len)).astype("int64")
    # stop_seqs_len: [bsz, stop_seqs_bs] — 0 means disabled
    stop_seqs_len = np.zeros((real_bsz, stop_seqs_bs), dtype="int32")

    end_ids = np.array([-1], dtype="int64")
    min_tokens = np.zeros(real_bsz, dtype="int64")

    return {
        "accept_tokens": accept_tokens,
        "accept_num": accept_num,
        "token_ids_all": token_ids_all,
        "prompt_lens": prompt_lens,
        "step_idx": step_idx,
        "stop_flags": stop_flags,
        "seq_lens": seq_lens,
        "stop_seqs": stop_seqs,
        "stop_seqs_len": stop_seqs_len,
        "end_ids": end_ids,
        "min_tokens": min_tokens,
        # Config params for reference
        "real_bsz": real_bsz,
        "accept_tokens_len": accept_tokens_len,
        "max_model_len": max_model_len,
        "stop_seqs_bs": stop_seqs_bs,
        "stop_seqs_max_len": stop_seqs_max_len,
    }


# ============================================================
# Layer 3: Reference implementation (pure Python/NumPy)
# ============================================================


def reference_spec_set_stop_value_multi_seqs(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Python reference — must match CUDA kernel logic exactly.

    token_ids_all 布局 (新 step_idx 语义):
      pre_ids_now[k] = 第 k 个 output token (k >= 0, 0-indexed)
      最后一个 output token 在 pre_ids_now[step_idx - 1]
      step_idx = 历史已生成的 token 数量

    核心设计:
      1. accept_idx 从 -1 开始，-1 表示检查 pre_ids 末尾（上一轮延迟的情况）
      2. 主循环检查 accept_idx <= accept_num - 2
      3. 匹配成功时: 保留 stop_seq 所有 token，在其后追加 eos
    """
    accept_tokens = inputs["accept_tokens"].copy()
    accept_num = inputs["accept_num"].copy()
    stop_flags = inputs["stop_flags"].copy()
    token_ids_all = inputs["token_ids_all"]
    prompt_lens = inputs["prompt_lens"]
    step_idx = inputs["step_idx"]
    stop_seqs = inputs["stop_seqs"]
    stop_seqs_len = inputs["stop_seqs_len"]
    end_ids = inputs["end_ids"]
    min_tokens = inputs["min_tokens"]

    bs = inputs["real_bsz"]
    stop_seqs_bs = inputs["stop_seqs_bs"]

    for bid in range(bs):
        for tid in range(stop_seqs_bs):
            stop_seq_len = stop_seqs_len[bid, tid]
            if stop_seq_len <= 0:
                continue

            stop_seq_now = stop_seqs[bid, tid, :]
            pre_ids_now_offset = int(prompt_lens[bid, 0]) if prompt_lens.ndim > 1 else int(prompt_lens[bid])
            pre_ids_now = token_ids_all[bid, pre_ids_now_offset:]
            accept_tokens_now = accept_tokens[bid, :]
            an = int(accept_num[bid])
            step_idx_now = int(step_idx[bid])
            min_token_limit = int(min_tokens[bid])

            can_stop = step_idx_now + an >= min_token_limit
            if not can_stop:
                continue
            if stop_flags[bid]:
                continue

            # CUDA kernel: accept_idx 从 -1 开始，检查 pre_ids 末尾
            accept_idx = -1
            is_end = False

            # loop_end = accept_num > 0 ? accept_num - 2 : -1
            loop_end = an - 2 if an > 0 else -1
            while accept_idx <= loop_end and not is_end:
                if step_idx_now + accept_idx + 1 < stop_seq_len:
                    accept_idx += 1
                    continue

                # 从后向前匹配 stop_seq 的每个 token
                for i in range(stop_seq_len - 1, -1, -1):
                    offset = stop_seq_len - 1 - i
                    accept_tokens_idx = accept_idx - offset
                    cur_token_idx = -1

                    if accept_tokens_idx >= 0:
                        cur_token_idx = accept_tokens_now[accept_tokens_idx]
                    else:
                        # 新语义: pre_ids_idx = step_idx_now + accept_tokens_idx
                        # pre_ids_now[0] 是第 1 个 output token
                        pre_ids_idx = step_idx_now + accept_tokens_idx
                        if pre_ids_idx < 0:
                            break
                        cur_token_idx = pre_ids_now[pre_ids_idx]

                    if cur_token_idx != stop_seq_now[i]:
                        break

                    if i == 0:
                        is_end = True

                accept_idx += 1

            if is_end:
                # accept_idx 已递增，指向 stop_seq 最后 token 的下一个位置
                # 保留 stop_seq 所有 token，在其后追加 eos
                accept_num[bid] = accept_idx + 1
                accept_tokens[bid, accept_idx] = end_ids[0]

    return {
        "accept_tokens": accept_tokens,
        "accept_num": accept_num,
    }


# ============================================================
# Layer 4a: TEST_CONFIGS
# ============================================================

TEST_CONFIGS = [
    # --- basic coverage ---
    {"name": "default_no_match", "real_bsz": 2, "seed": 42},
    {"name": "single_batch", "real_bsz": 1, "seed": 100},
    {"name": "large_batch", "real_bsz": 16, "seed": 200},
    # --- stop_seqs_bs variants ---
    {"name": "single_stop_seq", "real_bsz": 4, "stop_seqs_bs": 1, "seed": 300},
    {"name": "many_stop_seqs", "real_bsz": 4, "stop_seqs_bs": 8, "seed": 400},
    # --- edge: short accept ---
    {"name": "accept_1_token", "real_bsz": 4, "accept_num_range": (1, 1), "seed": 500},
    # --- edge: long stop seq ---
    {"name": "long_stop_seq", "real_bsz": 2, "stop_seqs_max_len": 8, "seed": 600},
]


# ============================================================
# Layer 4b: Test suite
# ============================================================


class TestSpeculateSetStopValueMultiSeqs(unittest.TestCase):

    # ------ shared helpers ------

    def _run_and_get(self, inputs):
        paddle_inputs = to_paddle_inputs(inputs)
        run_kernel(paddle_inputs)
        return get_outputs(paddle_inputs)

    def _check_all_outputs(self, inputs, outputs):
        """Compare ALL output tensors against reference."""
        ref = reference_spec_set_stop_value_multi_seqs(inputs)
        for key in ["accept_tokens", "accept_num"]:
            np.testing.assert_array_equal(outputs[key], ref[key], err_msg=f"{key} mismatch")

    def _run_full_test(self, config):
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

    def test_match_in_accept_tokens_only(self):
        """Stop seq found entirely within accept_tokens. Eos appended after stop_seq last token."""
        inputs = gen_inputs(real_bsz=1, accept_tokens_len=5, stop_seqs_bs=1, stop_seqs_max_len=3, seed=10)
        # Place stop seq [A, B, C] at accept_tokens positions [0,1,2]
        inputs["accept_num"][:] = 4
        inputs["accept_tokens"][0, :4] = [10, 20, 30, 40]
        inputs["stop_seqs"][0, 0, :3] = [10, 20, 30]
        inputs["stop_seqs_len"][0, 0] = 3
        inputs["step_idx"][:] = 10
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # stop_seq [10, 20, 30] matches at accept_idx=2 (window ends at accept_tokens[2]=30)
        # After loop increment, accept_idx=3, accept_num=4, eos appended at accept_tokens[3]
        self.assertEqual(outputs["accept_num"][0], 4)
        self.assertEqual(outputs["accept_tokens"][0, 3], -1)  # eos appended after stop_seq

    def test_match_spanning_pre_ids_and_accept(self):
        """Stop seq spans token_ids_all (pre_ids) and accept_tokens. Eos appended after stop_seq last token."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=4,
            seed=20,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 6
        inputs["accept_num"][:] = 3
        # stop_seq = [99, 11, 22] (len=3)
        # 新索引公式: pre_ids_idx = step_idx_now + accept_tokens_idx
        # pre_ids_now[k] = 第 k 个 output token (k >= 0)
        # step_idx = 6 表示有 6 个历史 output token，在 pre_ids_now[0..5]
        # At accept_idx=1 (window ends at accept_tokens[1]=22):
        #   i=2: offset=0, accept_tokens_idx=1 -> accept_tokens[1]=22 vs stop_seq[2]=22 ✓
        #   i=1: offset=1, accept_tokens_idx=0 -> accept_tokens[0]=11 vs stop_seq[1]=11 ✓
        #   i=0: offset=2, accept_tokens_idx=-1 -> pre_ids_idx=6+(-1)=5 -> pre_ids[5]=99 vs stop_seq[0]=99 ✓
        inputs["token_ids_all"][0, 5] = 99  # pre_ids_now[5] = 第 6 个 output token (0-indexed)
        inputs["accept_tokens"][0, :3] = [11, 22, 33]
        inputs["stop_seqs"][0, 0, :3] = [99, 11, 22]
        inputs["stop_seqs_len"][0, 0] = 3
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # Match at accept_idx=1, loop increments to 2 -> accept_num=3, eos at accept_tokens[2]
        self.assertEqual(outputs["accept_num"][0], 3)
        self.assertEqual(outputs["accept_tokens"][0, 2], -1)  # eos appended after stop_seq

    def test_match_in_pre_ids_only_not_detected(self):
        """Stop seq ending purely in pre_ids history but NOT at the end position.
        The kernel only detects stop_seq at the very end of pre_ids via accept_idx=-1 check.
        Stop seq placed earlier in pre_ids should not be detected."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=3,
            seed=30,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 8
        inputs["accept_num"][:] = 3
        # 新语义: pre_ids_now[k] = 第 k 个 output token (k >= 0)
        # step_idx = 8 表示有 8 个历史 output token，在 pre_ids_now[0..7]
        # accept_idx=-1 会检查 pre_ids_now[7] 开始的 stop_seq
        # 把 stop_seq 放在 pre_ids_now[2,3,4] - 不会被检测到
        inputs["token_ids_all"][0, 2] = 50
        inputs["token_ids_all"][0, 3] = 60
        inputs["token_ids_all"][0, 4] = 70
        inputs["accept_tokens"][0, :3] = [1, 2, 3]
        inputs["stop_seqs"][0, 0, :3] = [50, 60, 70]
        inputs["stop_seqs_len"][0, 0] = 3
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # No match: stop_seq is in pre_ids but not at the end, accept_num unchanged
        self.assertEqual(outputs["accept_num"][0], 3)

    def test_already_stopped(self):
        """Kernel skips sequences with stop_flags=True."""
        inputs = gen_inputs(real_bsz=2, stop_seqs_bs=1, stop_seqs_max_len=3, seed=40)
        inputs["stop_flags"][:] = True
        # Even if stop_seqs would match, nothing should change
        inputs["stop_seqs_len"][:] = 2
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # accept_tokens and accept_num should be unchanged
        np.testing.assert_array_equal(outputs["accept_tokens"], inputs["accept_tokens"])
        np.testing.assert_array_equal(outputs["accept_num"], inputs["accept_num"])

    def test_min_tokens_blocks_stop(self):
        """Kernel skips stop check when step_idx + accept_num < min_tokens."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=3,
            seed=50,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 8
        inputs["accept_num"][:] = 3
        # Place stop_seq in pre_ids at end position (would be detected by accept_idx=-1)
        # pre_ids_now[0..7] = 8 个历史 output token
        # accept_idx=-1 检查 pre_ids_now[5,6,7] 对应 stop_seq[0,1,2]
        inputs["token_ids_all"][0, 5] = 50
        inputs["token_ids_all"][0, 6] = 60
        inputs["token_ids_all"][0, 7] = 70
        inputs["accept_tokens"][0, :3] = [1, 2, 3]
        inputs["stop_seqs"][0, 0, :3] = [50, 60, 70]
        inputs["stop_seqs_len"][0, 0] = 3
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 100  # step_idx+accept_num=11 < 100, should NOT stop
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # min_tokens prevents stop, accept_num unchanged
        self.assertEqual(outputs["accept_num"][0], 3)

    def test_min_tokens_allows_stop(self):
        """Kernel allows stop when step_idx + accept_num >= min_tokens."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=3,
            seed=60,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 8
        inputs["accept_num"][:] = 3
        # stop_seq [X, 50] spans pre_ids and accept_tokens[0].
        # 新索引公式: pre_ids_idx = step_idx_now + accept_tokens_idx
        # At accept_idx=0 (window ends at accept_tokens[0]=50):
        #   i=1: offset=0, accept_tokens_idx=0 -> accept_tokens[0]=50 vs stop_seq[1]=50 ✓
        #   i=0: offset=1, accept_tokens_idx=-1 -> pre_ids_idx=8+(-1)=7 -> pre_ids[7]
        pre_val = int(inputs["token_ids_all"][0, 7])  # pre_ids_now[7]
        inputs["accept_tokens"][0, :3] = [50, 60, 70]
        inputs["stop_seqs"][0, 0, :2] = [pre_val, 50]
        inputs["stop_seqs_len"][0, 0] = 2
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 5  # step_idx+accept_num=11 >= 5, should stop
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)

    def test_multiple_stop_seqs_second_matches(self):
        """Second stop sequence in stop_seqs_bs dimension matches."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=2,
            stop_seqs_max_len=3,
            seed=70,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 8
        inputs["accept_num"][:] = 3
        # accept_tokens: [20, 30, 40]
        # Second stop seq [20, 30] matches at accept_idx=1 (window ends at accept_tokens[1]=30):
        #   i=1: offset=0, accept_tokens_idx=1 -> accept_tokens[1]=30 vs stop_seq[1]=30 ✓
        #   i=0: offset=1, accept_tokens_idx=0 -> accept_tokens[0]=20 vs stop_seq[0]=20 ✓
        inputs["accept_tokens"][0, :3] = [20, 30, 40]
        # First stop seq doesn't match
        inputs["stop_seqs"][0, 0, :3] = [99, 98, 97]
        inputs["stop_seqs_len"][0, 0] = 3
        # Second stop seq [20, 30] matches
        inputs["stop_seqs"][0, 1, :2] = [20, 30]
        inputs["stop_seqs_len"][0, 1] = 2
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # Match at accept_idx=1 -> accept_num=3, eos at accept_tokens[2]
        self.assertEqual(outputs["accept_num"][0], 3)
        self.assertEqual(outputs["accept_tokens"][0, 2], -1)  # eos appended after stop_seq

    def test_nonzero_prompt_lens(self):
        """Verify prompt_lens offset is applied correctly."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=2,
            seed=80,
        )
        prompt_len = 10
        inputs["prompt_lens"][:] = prompt_len
        inputs["step_idx"][:] = 5
        inputs["accept_num"][:] = 2
        inputs["accept_tokens"][0, :2] = [55, 66]
        # pre_ids_now starts at token_ids_all[0, prompt_len:]
        # pre_ids_now[k] = 第 k 个 output token (k >= 0)
        # 新索引公式: pre_ids_idx = step_idx_now + accept_tokens_idx
        # stop_seq = [X, 55] where X = pre_ids_now[5 + (-1)] = pre_ids_now[4]
        # At accept_idx=0 (window ends at accept_tokens[0]=55):
        #   i=1: offset=0, accept_tokens_idx=0 -> accept_tokens[0]=55 vs stop_seq[1]=55 ✓
        #   i=0: offset=1, accept_tokens_idx=-1 -> pre_ids_idx=5+(-1)=4 -> pre_ids[4]=token_ids_all[0, prompt_len+4]
        target_val = int(inputs["token_ids_all"][0, prompt_len + 4])
        inputs["stop_seqs"][0, 0, :2] = [target_val, 55]
        inputs["stop_seqs_len"][0, 0] = 2
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # Match at accept_idx=0 -> accept_num=2, eos at accept_tokens[1]
        self.assertEqual(outputs["accept_num"][0], 2)
        self.assertEqual(outputs["accept_tokens"][0, 1], -1)  # eos appended after stop_seq

    def test_single_token_stop_seq_preserved(self):
        """Single token stop_seq (like <|im_end|>) with eos appended after it."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=1,
            seed=90,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 10
        inputs["accept_num"][:] = 4
        # accept_tokens: [a, b, <|im_end|>, d] where <|im_end|> has token id 999
        inputs["accept_tokens"][0, :4] = [100, 200, 999, 300]
        # stop_seq = [<|im_end|>] (single token)
        inputs["stop_seqs"][0, 0, 0] = 999
        inputs["stop_seqs_len"][0, 0] = 1
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # Match at accept_idx=2 (window ends at accept_tokens[2]=999)
        # After loop increment, accept_idx=3, accept_num=4, eos at accept_tokens[3]
        self.assertEqual(outputs["accept_num"][0], 4)
        self.assertEqual(outputs["accept_tokens"][0, 3], -1)  # eos appended after stop_seq

    def test_stop_seq_at_last_position_not_detected(self):
        """Stop seq at the last position of accept_tokens is NOT detected (deferred to next round)."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=1,
            seed=100,
        )
        inputs["prompt_lens"][:] = 0
        inputs["step_idx"][:] = 10
        inputs["accept_num"][:] = 4
        # stop_seq [999] is at accept_tokens[3] (last valid position)
        # Since we only check up to accept_num - 2 = 2, this won't be detected
        inputs["accept_tokens"][0, :4] = [100, 200, 300, 999]
        inputs["stop_seqs"][0, 0, 0] = 999
        inputs["stop_seqs_len"][0, 0] = 1
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # No match because accept_idx only goes up to 2, and 999 is at position 3
        # accept_num unchanged
        self.assertEqual(outputs["accept_num"][0], 4)

    def test_stop_seq_detected_from_previous_round(self):
        """Stop seq at the end of pre_ids (from previous round) is detected via accept_idx=-1."""
        inputs = gen_inputs(
            real_bsz=1,
            accept_tokens_len=5,
            max_model_len=32,
            stop_seqs_bs=1,
            stop_seqs_max_len=1,
            seed=110,
        )
        inputs["prompt_lens"][:] = 0
        # 新语义: pre_ids_now[k] = 第 k 个 output token (k >= 0)
        # step_idx = 10 表示有 10 个历史 output token，在 pre_ids_now[0..9]
        # accept_idx=-1 检查 pre_ids_now[9] (最后一个历史 token)
        inputs["step_idx"][:] = 10
        inputs["token_ids_all"][0, 9] = 999  # pre_ids_now[9] = 第 10 个 output token (0-indexed)
        inputs["accept_num"][:] = 3
        inputs["accept_tokens"][0, :3] = [100, 200, 300]
        inputs["stop_seqs"][0, 0, 0] = 999
        inputs["stop_seqs_len"][0, 0] = 1
        inputs["stop_flags"][:] = False
        inputs["min_tokens"][:] = 0
        outputs = self._run_and_get(inputs)
        self._check_all_outputs(inputs, outputs)
        # stop_seq [999] was in pre_ids at end, accept_idx=-1 matches
        # After loop increment, accept_idx=0, accept_num=1, eos at accept_tokens[0]
        self.assertEqual(outputs["accept_num"][0], 1)
        self.assertEqual(outputs["accept_tokens"][0, 0], -1)  # replaced with eos


if __name__ == "__main__":
    unittest.main()
