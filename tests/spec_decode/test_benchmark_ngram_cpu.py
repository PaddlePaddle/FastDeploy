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
CPU baseline benchmark for ngram_match — production C++ kernel.

Measures the actual CPU computation time of the upstream ngram_match C++
kernel (ngram_match.cc / find_candidate_pred_tokens).  Uses the same
5-group experiment dimensions as the GPU benchmark so results can be
directly compared column-by-column.

This file intentionally lives on `develop` where ngram_match.cc exists.
It is NOT for merge — it provides the missing "CPU compute" column that
the GPU PR's benchmark omitted (which only measured D2H/H2D copy time).

Groups (matching GPU benchmark):
  1. seq_len     — [1024, 4096, 16384, 65536, 131072]
  2. batch_size  — [1, 8, 32, 128, 512]
  3. ngram hit   — [high_input, high_pre, low_input, low_pre, none]
  4. threshold   — [16, 32, 64, 128, 256]
  5. threshold × batch (batch=128)
  6. latency     — batch=32, seq=512
  7. latency_ext — batch=256, seq=131072

Run:
    cd FastDeploy && python tests/spec_decode/test_benchmark_ngram_cpu.py
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
WARMUP = 5


def _build_data(batch_size, seq_len, hit_type="low_input", seed=42):
    """Build test tensors with controlled ngram hit placement."""
    rng = np.random.RandomState(seed)
    step_idx_val = max(MAX_NGRAM_SIZE + 2, 20)
    pre_len = step_idx_val + 1
    max_model_len = max(seq_len + 64, pre_len + 64)

    input_ids = rng.randint(10, 500, (batch_size, seq_len)).astype(np.int64)
    token_ids_all = rng.randint(10, 500, (batch_size, max_model_len)).astype(np.int64)
    pattern = np.arange(1001, 1001 + MAX_NGRAM_SIZE, dtype=np.int64)

    for b in range(batch_size):
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
            pass

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


def _to_cpu(np_dict):
    """Convert numpy arrays to CPU paddle tensors."""
    out = {}
    for k, v in np_dict.items():
        out[k] = paddle.to_tensor(v, place=paddle.CPUPlace())
    return out


def _run_cpu(ngram_match_fn, cpu_data):
    """Call ngram_match with CPU tensors → dispatches to .cc kernel."""
    ngram_match_fn(
        cpu_data["input_ids"],
        cpu_data["input_ids_len"],
        cpu_data["token_ids_all"],
        cpu_data["prompt_lens"],
        cpu_data["step_idx"],
        cpu_data["draft_token_num"],
        cpu_data["draft_tokens"],
        cpu_data["seq_lens_this_time"],
        cpu_data["seq_lens_encoder"],
        cpu_data["seq_lens_decoder"],
        cpu_data["max_dec_len"],
        MAX_NGRAM_SIZE,
        MAX_DRAFT_TOKENS,
    )


def _time_cpu(ngram_match_fn, batch_size, seq_len, hit_type, n_runs):
    """Time CPU C++ kernel with pre-created tensors."""
    cpu_data = _to_cpu(_build_data(batch_size, seq_len, hit_type))

    # Warmup
    for _ in range(WARMUP):
        cpu_data["draft_tokens"] = paddle.zeros([batch_size, MAX_DRAFT_TOKENS + 1], dtype="int64")
        cpu_data["seq_lens_this_time"] = paddle.ones([batch_size], dtype="int32")
        _run_cpu(ngram_match_fn, cpu_data)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        cpu_data["draft_tokens"] = paddle.zeros([batch_size, MAX_DRAFT_TOKENS + 1], dtype="int64")
        cpu_data["seq_lens_this_time"] = paddle.ones([batch_size], dtype="int32")
        _run_cpu(ngram_match_fn, cpu_data)
    elapsed = time.perf_counter() - t0
    return (elapsed / n_runs) * 1e6  # microseconds


def _print_table(title, header, rows):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'─' * 80}")
    print(header)
    print(f"{'─' * 80}")
    for row in rows:
        print(row)
    print(f"{'=' * 80}")


class TestNgramCpuBenchmark(unittest.TestCase):
    """CPU C++ kernel benchmark — 5 groups matching GPU benchmark dimensions."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("cpu")
        try:
            from fastdeploy.model_executor.ops.gpu import ngram_match

            cls.ngram_match = staticmethod(ngram_match)
        except Exception as e:
            raise unittest.SkipTest(f"Cannot import ngram_match op: {e}")

    def test_group1_seq_len(self):
        """Group 1: Vary seq_len, fixed batch=16, threshold=512, hit=low_input."""
        seq_lens = [1024, 4096, 16384, 65536, 131072]
        runs = [1000, 1000, 500, 200, 100]
        batch_size = 16
        hit_type = "low_input"

        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "512"
        try:
            rows = []
            for sl, n in zip(seq_lens, runs):
                cpu_us = _time_cpu(self.ngram_match, batch_size, sl, hit_type, n)
                rows.append(f"  seq={sl:<8d} batch={batch_size:<4d}  " f"CPU: {cpu_us:>10.1f} µs  (n={n})")
            _print_table(
                "Group 1: seq_len sweep  (batch=16, threshold=512, hit=low_input)",
                f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group2_batch_size(self):
        """Group 2: Vary batch_size, fixed seq=16384, threshold=8192, hit=low_input."""
        batch_sizes = [1, 8, 32, 128, 512]
        runs = [1000, 1000, 500, 200, 100]
        seq_len = 16384
        hit_type = "low_input"

        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "8192"
        try:
            rows = []
            for bs, n in zip(batch_sizes, runs):
                cpu_us = _time_cpu(self.ngram_match, bs, seq_len, hit_type, n)
                rows.append(f"  batch={bs:<4d} seq={seq_len:<8d}  " f"CPU: {cpu_us:>10.1f} µs  (n={n})")
            _print_table(
                "Group 2: batch_size sweep  (seq=16384, threshold=8192, hit=low_input)",
                f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group3_hit_type(self):
        """Group 3: Vary hit type, fixed batch=16, seq=16384, threshold=512."""
        hit_types = ["high_input", "high_pre", "low_input", "low_pre", "none"]
        n_runs = 1000
        batch_size = 16
        seq_len = 16384

        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "512"
        try:
            rows = []
            for ht in hit_types:
                cpu_us = _time_cpu(self.ngram_match, batch_size, seq_len, ht, n_runs)
                rows.append(f"  hit={ht:<12s} batch={batch_size:<4d}  " f"CPU: {cpu_us:>10.1f} µs  (n={n_runs})")
            _print_table(
                "Group 3: hit type sweep  (batch=16, seq=16384, threshold=512)",
                f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
                rows,
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_group4_threshold(self):
        """Group 4: Vary threshold, fixed batch=8, seq=32768, hit=low_input."""
        thresholds = [16, 32, 64, 128, 256]
        n_runs = 500
        batch_size = 8
        seq_len = 32768
        hit_type = "low_input"

        rows = []
        for thr in thresholds:
            os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(thr)
            cpu_us = _time_cpu(self.ngram_match, batch_size, seq_len, hit_type, n_runs)
            rows.append(f"  threshold={thr:<4d} batch={batch_size:<4d}  " f"CPU: {cpu_us:>10.1f} µs  (n={n_runs})")
        _print_table(
            "Group 4: threshold sweep  (batch=8, seq=32768, hit=low_input)",
            f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
            rows,
        )

    def test_group5_threshold_x_batch(self):
        """Group 5: Vary threshold with large batch=128, seq=32768, hit=low_input."""
        thresholds = [16, 32, 64, 128, 256]
        n_runs = 100
        batch_size = 128
        seq_len = 32768
        hit_type = "low_input"

        rows = []
        for thr in thresholds:
            os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(thr)
            cpu_us = _time_cpu(self.ngram_match, batch_size, seq_len, hit_type, n_runs)
            rows.append(f"  threshold={thr:<4d} batch={batch_size:<4d}  " f"CPU: {cpu_us:>10.1f} µs  (n={n_runs})")
        _print_table(
            "Group 5: threshold × batch  (batch=128, seq=32768, hit=low_input)",
            f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
            rows,
        )

    def test_latency(self):
        """Latency: batch=32, seq=512 — matches GPU benchmark test_latency."""
        batch_size = 32
        seq_len = 512
        n_runs = 1000
        hit_type = "low_input"

        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = "128"
        try:
            cpu_us = _time_cpu(self.ngram_match, batch_size, seq_len, hit_type, n_runs)
            _print_table(
                "Latency: batch=32, seq=512, threshold=128",
                f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
                [f"  batch={batch_size} seq={seq_len:<8d}  CPU: {cpu_us:>10.1f} µs  (n={n_runs})"],
            )
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

    def test_latency_extreme(self):
        """Latency extreme: batch=256, seq=131072 — matches GPU benchmark."""
        batch_size = 256
        seq_len = 131072
        hit_type = "low_input"
        n_runs = 100

        configs = [
            ("threshold=8192", "8192"),
            ("threshold=16384", "16384"),
        ]
        rows = []
        for label, thr in configs:
            os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = thr
            cpu_us = _time_cpu(self.ngram_match, batch_size, seq_len, hit_type, n_runs)
            rows.append(f"  {label:<20s} batch={batch_size:<4d}  " f"CPU: {cpu_us:>10.1f} µs  (n={n_runs})")
        _print_table(
            "Latency extreme: batch=256, seq=131072",
            f"  {'Config':<30s} {'CPU C++ kernel':>15s}",
            rows,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
