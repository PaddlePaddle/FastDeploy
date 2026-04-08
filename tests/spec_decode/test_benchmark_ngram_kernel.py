#!/usr/bin/env python3
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
Multi-dimension benchmark for ngram_match GPU kernel vs CPU copy path.

Matches NKNaN's profiling methodology (5 experiment groups) using
FastDeploy's native ngram_match op interface.

Groups:
  1. seq_len     — [1024, 4096, 16384, 65536, 131072]
  2. batch_size  — [1, 8, 32, 128, 512]
  3. ngram hit   — [high_input, high_pre, low_input, low_pre, none]
  4. threshold   — [16, 32, 64, 128, 256]
  5. threshold × batch (batch=128)

Run:
    cd FastDeploy && python tests/spec_decode/test_benchmark_ngram_kernel.py
"""
import os
import sys
import time
import unittest

import numpy as np
import paddle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

MAX_NGRAM_SIZE = 3
MAX_DRAFT_TOKENS = 10
NUM_ITERS = 1
WARMUP = 1


def _build_data(batch_size, seq_len, hit_type="low_input", seed=42):
    """
    Build test tensors with controlled ngram hit placement.

    hit_type controls where the ngram match is found:
      - high_input: match near start of input_ids (fast find)
      - high_pre:   match near start of token_ids_all gen tokens
      - low_input:  match near end of input_ids (worst-case scan)
      - low_pre:    match near end of token_ids_all gen tokens
      - none:       no planted match (full scan, no hit)
    """
    rng = np.random.RandomState(seed)
    step_idx_val = max(MAX_NGRAM_SIZE + 2, 20)
    pre_len = step_idx_val + 1
    max_model_len = max(seq_len + 64, pre_len + 64)

    input_ids = rng.randint(10, 500, (batch_size, seq_len)).astype(np.int64)
    token_ids_all = rng.randint(10, 500, (batch_size, max_model_len)).astype(np.int64)
    pattern = np.arange(1001, 1001 + MAX_NGRAM_SIZE, dtype=np.int64)

    for b in range(batch_size):
        # Plant pattern in token_ids_all at step_idx alignment (the ngram to search for)
        ng_start = step_idx_val + 1 - MAX_NGRAM_SIZE
        token_ids_all[b, ng_start : step_idx_val + 1] = pattern

        if hit_type == "high_input":
            pos = 5
            if pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS <= seq_len:
                input_ids[b, pos : pos + MAX_NGRAM_SIZE] = pattern
                input_ids[b, pos + MAX_NGRAM_SIZE : pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS] = np.arange(
                    2001, 2001 + MAX_DRAFT_TOKENS, dtype=np.int64
                )

        elif hit_type == "high_pre":
            pos = 5
            if pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS < ng_start:
                token_ids_all[b, pos : pos + MAX_NGRAM_SIZE] = pattern
                token_ids_all[b, pos + MAX_NGRAM_SIZE : pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS] = np.arange(
                    2001, 2001 + MAX_DRAFT_TOKENS, dtype=np.int64
                )

        elif hit_type == "low_input":
            pos = seq_len - MAX_NGRAM_SIZE - MAX_DRAFT_TOKENS - 5
            if pos > 0:
                input_ids[b, pos : pos + MAX_NGRAM_SIZE] = pattern
                input_ids[b, pos + MAX_NGRAM_SIZE : pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS] = np.arange(
                    2001, 2001 + MAX_DRAFT_TOKENS, dtype=np.int64
                )

        elif hit_type == "low_pre":
            pos = step_idx_val - MAX_NGRAM_SIZE - MAX_DRAFT_TOKENS - 5
            if pos > 0 and pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS < ng_start:
                token_ids_all[b, pos : pos + MAX_NGRAM_SIZE] = pattern
                token_ids_all[b, pos + MAX_NGRAM_SIZE : pos + MAX_NGRAM_SIZE + MAX_DRAFT_TOKENS] = np.arange(
                    2001, 2001 + MAX_DRAFT_TOKENS, dtype=np.int64
                )

        elif hit_type == "none":
            pass  # No match planted — random data only

    input_ids_len = np.full((batch_size, 1), seq_len, dtype=np.int64)
    prompt_lens = np.zeros((batch_size, 1), dtype=np.int64)
    step_idx = np.full((batch_size, 1), step_idx_val, dtype=np.int64)
    draft_token_num = np.full((batch_size, 1), MAX_DRAFT_TOKENS, dtype=np.int32)
    draft_tokens = np.zeros((batch_size, MAX_DRAFT_TOKENS + 1), dtype=np.int64)
    seq_lens_this_time = np.ones(batch_size, dtype=np.int32)
    seq_lens_encoder = np.zeros(batch_size, dtype=np.int32)
    seq_lens_decoder = np.ones(batch_size, dtype=np.int32)
    max_dec_len = np.full((batch_size, 1), 1048576, dtype=np.int64)

    return {
        "input_ids": input_ids,
        "input_ids_len": input_ids_len,
        "token_ids_all": token_ids_all,
        "prompt_lens": prompt_lens,
        "step_idx": step_idx,
        "draft_token_num": draft_token_num,
        "draft_tokens": draft_tokens,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_encoder": seq_lens_encoder,
        "seq_lens_decoder": seq_lens_decoder,
        "max_dec_len": max_dec_len,
    }


def _to_gpu(np_dict):
    out = {}
    for k, v in np_dict.items():
        out[k] = paddle.to_tensor(v, place=paddle.CUDAPlace(0))
    return out


def _run_gpu(ngram_match_fn, gpu_data):
    """Run GPU kernel (tensors already on GPU)."""
    ngram_match_fn(
        gpu_data["input_ids"],
        gpu_data["input_ids_len"],
        gpu_data["token_ids_all"],
        gpu_data["prompt_lens"],
        gpu_data["step_idx"],
        gpu_data["draft_token_num"],
        gpu_data["draft_tokens"],
        gpu_data["seq_lens_this_time"],
        gpu_data["seq_lens_encoder"],
        gpu_data["seq_lens_decoder"],
        gpu_data["max_dec_len"],
        MAX_NGRAM_SIZE,
        MAX_DRAFT_TOKENS,
    )


def _time_gpu(ngram_match_fn, batch_size, seq_len, hit_type, n_runs):
    """Time GPU kernel with pre-created tensors (no data creation in loop)."""
    gpu_data = _to_gpu(_build_data(batch_size, seq_len, hit_type))
    # Pre-allocate mutable output buffers once — avoids per-iteration
    # paddle.zeros/ones which add ~20-40µs allocation + fill overhead.
    draft_buf = paddle.zeros([batch_size, MAX_DRAFT_TOKENS + 1], dtype="int64").cuda()
    seqlens_buf = paddle.ones([batch_size], dtype="int32").cuda()
    # Warmup
    for _ in range(WARMUP):
        seqlens_buf.fill_(1)
        gpu_data["draft_tokens"] = draft_buf
        gpu_data["seq_lens_this_time"] = seqlens_buf
        _run_gpu(ngram_match_fn, gpu_data)
    paddle.device.synchronize()

    paddle.device.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        seqlens_buf.fill_(1)
        gpu_data["draft_tokens"] = draft_buf
        gpu_data["seq_lens_this_time"] = seqlens_buf
        _run_gpu(ngram_match_fn, gpu_data)
        paddle.device.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1e6  # microseconds


def _time_cpu_copy(batch_size, seq_len, hit_type, n_runs):
    """Time the old CPU-copy path: GPU→CPU transfer + CPU→GPU transfer back."""
    gpu_data = _to_gpu(_build_data(batch_size, seq_len, hit_type))
    # Warmup
    for _ in range(WARMUP):
        _ = {k: v.cpu() for k, v in gpu_data.items()}
    paddle.device.synchronize()

    paddle.device.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        cpu_copy = {k: v.cpu() for k, v in gpu_data.items()}
        _ = cpu_copy["draft_tokens"].cuda()
        _ = cpu_copy["seq_lens_this_time"].cuda()
        paddle.device.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1e6  # microseconds


def _print_table(title, header, rows):
    """Print formatted benchmark table."""
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'─' * 80}")
    print(header)
    print(f"{'─' * 80}")
    for row in rows:
        print(row)
    print(f"{'=' * 80}")


class TestNgramBenchmarkGroups(unittest.TestCase):
    """Multi-dimension benchmark matching NKNaN's 5-group methodology."""

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA not available")
        paddle.set_device("gpu")
        try:
            from fastdeploy.model_executor.ops.gpu import ngram_match

            cls.ngram_match = staticmethod(ngram_match)
        except Exception as e:
            raise unittest.SkipTest(f"Cannot import ngram_match op: {e}")

    def test_group1_seq_len(self):
        """Group 1: Vary seq_len with fixed batch=16, threshold=512, hit=low_input."""
        seq_lens = [1024, 4096, 16384, 65536, 131072]
        batch_size = 16
        hit_type = "low_input"
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "512"
        try:
            rows = []
            for sl in seq_lens:
                gpu_us = _time_gpu(self.ngram_match, batch_size, sl, hit_type, NUM_ITERS)
                cpu_us = _time_cpu_copy(batch_size, sl, hit_type, NUM_ITERS)
                speedup = cpu_us / gpu_us if gpu_us > 0 else 0
                rows.append(f"{sl:>8}  {gpu_us:>12.1f}  {cpu_us:>12.1f}  {speedup:>8.2f}x")
            _print_table(
                f"Group 1: seq_len (batch={batch_size}, threshold=512, hit={hit_type}, {NUM_ITERS} runs)",
                f"{'seq_len':>8}  {'GPU (µs)':>12}  {'CPU copy (µs)':>12}  {'Speedup':>8}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group2_batch_size(self):
        """Group 2: Vary batch_size with fixed seq_len=16384, threshold=8192, hit=low_input."""
        batch_sizes = [1, 8, 32, 128, 512]
        seq_len = 16384
        hit_type = "low_input"
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "8192"
        try:
            rows = []
            for bsz in batch_sizes:
                gpu_us = _time_gpu(self.ngram_match, bsz, seq_len, hit_type, NUM_ITERS)
                cpu_us = _time_cpu_copy(bsz, seq_len, hit_type, NUM_ITERS)
                speedup = cpu_us / gpu_us if gpu_us > 0 else 0
                rows.append(f"{bsz:>8}  {gpu_us:>12.1f}  {cpu_us:>12.1f}  {speedup:>8.2f}x")
            _print_table(
                f"Group 2: batch_size (seq_len={seq_len}, threshold=8192, hit={hit_type}, {NUM_ITERS} runs)",
                f"{'batch':>8}  {'GPU (µs)':>12}  {'CPU copy (µs)':>12}  {'Speedup':>8}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group3_ngram_hit(self):
        """Group 3: Vary hit pattern with fixed batch=16, seq_len=32768, threshold=512."""
        hit_types = ["high_input", "high_pre", "low_input", "low_pre", "none"]
        batch_size = 16
        seq_len = 32768
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "512"
        try:
            rows = []
            for ht in hit_types:
                gpu_us = _time_gpu(self.ngram_match, batch_size, seq_len, ht, NUM_ITERS)
                cpu_us = _time_cpu_copy(batch_size, seq_len, ht, NUM_ITERS)
                speedup = cpu_us / gpu_us if gpu_us > 0 else 0
                rows.append(f"{ht:>12}  {gpu_us:>12.1f}  {cpu_us:>12.1f}  {speedup:>8.2f}x")
            _print_table(
                f"Group 3: ngram hit (batch={batch_size}, seq_len={seq_len}, threshold=512, {NUM_ITERS} runs)",
                f"{'hit_type':>12}  {'GPU (µs)':>12}  {'CPU copy (µs)':>12}  {'Speedup':>8}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group4_threshold(self):
        """Group 4: Vary threshold with fixed batch=8, seq_len=32768, hit=low_input."""
        thresholds = [16, 32, 64, 128, 256]
        batch_size = 8
        seq_len = 32768
        hit_type = "low_input"
        rows = []
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        try:
            for thr in thresholds:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(thr)
                gpu_us = _time_gpu(self.ngram_match, batch_size, seq_len, hit_type, NUM_ITERS)
                cpu_us = _time_cpu_copy(batch_size, seq_len, hit_type, NUM_ITERS)
                speedup = cpu_us / gpu_us if gpu_us > 0 else 0
                rows.append(f"{thr:>8}  {gpu_us:>12.1f}  {cpu_us:>12.1f}  {speedup:>8.2f}x")
            _print_table(
                f"Group 4: threshold (batch={batch_size}, seq_len={seq_len}, hit={hit_type}, {NUM_ITERS} runs)",
                f"{'thresh':>8}  {'GPU (µs)':>12}  {'CPU copy (µs)':>12}  {'Speedup':>8}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group5_threshold_x_batch(self):
        """Group 5: Vary threshold with large batch=128 to expose truncation effects."""
        thresholds = [16, 32, 64, 128, 256]
        batch_size = 128
        seq_len = 32768
        hit_type = "low_input"
        rows = []
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        try:
            for thr in thresholds:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(thr)
                gpu_us = _time_gpu(self.ngram_match, batch_size, seq_len, hit_type, NUM_ITERS)
                cpu_us = _time_cpu_copy(batch_size, seq_len, hit_type, NUM_ITERS)
                speedup = cpu_us / gpu_us if gpu_us > 0 else 0
                rows.append(f"{thr:>8}  {gpu_us:>12.1f}  {cpu_us:>12.1f}  {speedup:>8.2f}x")
            _print_table(
                f"Group 5: threshold×batch (batch={batch_size}, seq_len={seq_len}, hit={hit_type}, {NUM_ITERS} runs)",
                f"{'thresh':>8}  {'GPU (µs)':>12}  {'CPU copy (µs)':>12}  {'Speedup':>8}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env


if __name__ == "__main__":
    unittest.main()
