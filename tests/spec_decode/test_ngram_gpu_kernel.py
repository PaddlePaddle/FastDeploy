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
Correctness + latency test for GPU ngram_match & hybrid_mtp_ngram kernels.

Run on AI Studio V100:
    cd FastDeploy && pip install -e . && python tests/spec_decode/test_ngram_gpu_kernel.py

Or standalone (compile custom ops first):
    bash build.sh 0 && python tests/spec_decode/test_ngram_gpu_kernel.py
"""
import os
import sys
import time
import unittest

import numpy as np
import paddle

# Ensure FastDeploy ops are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def _cpu_ngram_match(
    input_ids,
    input_ids_len,
    token_ids_all,
    prompt_lens,
    step_idx,
    draft_token_num,
    draft_tokens,
    seq_lens_this_time,
    seq_lens_encoder,
    seq_lens_decoder,
    max_dec_len,
    max_ngram_size,
    max_draft_tokens_param,
    threshold=128,
):
    """Pure NumPy reference matching the original ngram_match.cc logic."""
    # Flatten (N,1) shaped arrays to 1D for scalar indexing
    max_dec_len = max_dec_len.ravel()
    step_idx = step_idx.ravel()
    draft_token_num = draft_token_num.ravel()
    prompt_lens = prompt_lens.ravel()
    input_ids_len = input_ids_len.ravel()
    max_batch_size = seq_lens_this_time.shape[0]

    unprocessed = sum(1 for b in range(max_batch_size) if seq_lens_encoder[b] > 0 or seq_lens_decoder[b] > 0)

    for batch_idx in range(max_batch_size):
        remaining = int(max_dec_len[batch_idx] - step_idx[batch_idx] - 1)
        mdt = min(int(draft_token_num[batch_idx]), remaining)

        if seq_lens_encoder[batch_idx] > 0:
            continue
        elif seq_lens_decoder[batch_idx] == 0:
            seq_lens_this_time[batch_idx] = 0
            continue

        cur_input_ids = input_ids[batch_idx]
        cur_draft = draft_tokens[batch_idx]
        prompt_len = int(prompt_lens[batch_idx])
        cur_pre_ids = token_ids_all[batch_idx, prompt_len:]
        cur_step = int(step_idx[batch_idx])
        cur_ids_len = int(input_ids_len[batch_idx])
        seq_lens_this_time[batch_idx] = 1
        unprocessed -= 1

        sum_tok = sum(int(seq_lens_this_time[i]) for i in range(batch_idx + 1))
        left_min = unprocessed

        if sum_tok + mdt + left_min > threshold:
            mdt = min(mdt, threshold - sum_tok - left_min)
        if sum_tok + left_min >= threshold - 1:
            continue

        for ngram_size in range(max_ngram_size, 0, -1):
            if cur_step < ngram_size:
                continue
            ngram = cur_pre_ids[cur_step + 1 - ngram_size : cur_step + 1]

            # Search in input_ids
            match_input = False
            for i in range(cur_ids_len - ngram_size + 1):
                if np.array_equal(cur_input_ids[i : i + ngram_size], ngram):
                    start = i + ngram_size
                    end = min(start + mdt, cur_ids_len)
                    if start >= end:
                        continue
                    n = end - start
                    seq_lens_this_time[batch_idx] = n + 1
                    cur_draft[1 : 1 + n] = cur_input_ids[start : start + n]
                    match_input = True
                    break
            if match_input:
                break

            # Search in pre_ids
            found = False
            for i in range(cur_step - ngram_size + 1):
                if np.array_equal(cur_pre_ids[i : i + ngram_size], ngram):
                    start = i + ngram_size
                    end = min(start + mdt, cur_step)
                    if start >= end:
                        continue
                    n = end - start
                    seq_lens_this_time[batch_idx] = n + 1
                    cur_draft[1 : 1 + n] = cur_pre_ids[start : start + n]
                    found = True
                    break
            if found:
                break


def _cpu_hybrid_mtp_ngram(
    input_ids,
    input_ids_len,
    pre_ids,
    step_idx,
    draft_token_num,
    draft_tokens,
    seq_lens_this_time,
    seq_lens_decoder,
    max_dec_len,
    max_ngram_size,
    min_ngram_size,
    max_draft_tokens_param,
    threshold=1024,
):
    """Pure NumPy reference matching the original ngram_match_mixed.cu CPU logic."""
    # Flatten (N,1) shaped arrays to 1D for scalar indexing
    max_dec_len = max_dec_len.ravel()
    step_idx = step_idx.ravel()
    draft_token_num = draft_token_num.ravel()
    input_ids_len = input_ids_len.ravel()
    max_batch_size = seq_lens_this_time.shape[0]

    unprocessed = sum(1 for b in range(max_batch_size) if seq_lens_decoder[b] > 0)

    for batch_idx in range(max_batch_size):
        ori_slt = int(seq_lens_this_time[batch_idx])
        remaining = int(max_dec_len[batch_idx] - step_idx[batch_idx] - 1)
        max_q = min(max_draft_tokens_param - ori_slt + 1, remaining)

        if ori_slt == 0 or max_q <= 0:
            continue

        cur_input_ids = input_ids[batch_idx]
        cur_draft = draft_tokens[batch_idx]
        cur_pre = pre_ids[batch_idx]
        cur_step = int(step_idx[batch_idx])
        cur_ids_len = int(input_ids_len[batch_idx])
        unprocessed -= 1

        sum_tok = sum(int(seq_lens_this_time[i]) for i in range(batch_idx + 1))
        left_min = unprocessed

        if sum_tok + max_q + left_min > threshold:
            max_q = min(max_q, threshold - sum_tok - left_min)
        if sum_tok + left_min >= threshold - 1:
            continue

        match_global = False
        for ngram_size in range(max_ngram_size, min_ngram_size - 1, -1):
            if match_global:
                break
            if cur_step < ngram_size:
                continue
            ngram = cur_pre[cur_step + 1 - ngram_size : cur_step + 1]

            # Search in input_ids
            for i in range(cur_ids_len - ngram_size + 1):
                if match_global:
                    break
                if np.array_equal(cur_input_ids[i : i + ngram_size], ngram):
                    start = i + ngram_size
                    end = min(start + max_q, cur_ids_len)
                    if start >= end:
                        continue
                    n = end - start
                    seq_lens_this_time[batch_idx] = ori_slt + n
                    cur_draft[ori_slt : ori_slt + n] = cur_input_ids[start : start + n]
                    match_global = True

            # Search in pre_ids
            if not match_global:
                for i in range(cur_step - ngram_size + 1):
                    if match_global:
                        break
                    if np.array_equal(cur_pre[i : i + ngram_size], ngram):
                        start = i + ngram_size
                        end = min(start + max_q, cur_step)
                        if start >= end:
                            continue
                        n = end - start
                        seq_lens_this_time[batch_idx] = ori_slt + n
                        cur_draft[ori_slt : ori_slt + n] = cur_pre[start : start + n]
                        match_global = True


def _make_ngram_test_data(batch_size=4, input_len=64, max_model_len=256, max_draft=10, seed=42):
    """Create realistic test tensors for ngram_match op."""
    rng = np.random.RandomState(seed)
    vocab_size = 1000
    # Ensure max_model_len can hold prompt + generated tokens
    max_model_len = max(max_model_len, input_len + 64)

    # Create prompt tokens with repeating patterns to ensure ngram matches
    input_ids = rng.randint(0, vocab_size, (batch_size, input_len)).astype(np.int64)
    input_ids_len = np.full((batch_size, 1), input_len, dtype=np.int64)

    # token_ids_all: [batch, max_model_len] — prompt + generated
    token_ids_all = np.zeros((batch_size, max_model_len), dtype=np.int64)
    prompt_lens = np.full((batch_size, 1), input_len, dtype=np.int64)
    step_idx = np.zeros((batch_size, 1), dtype=np.int64)
    draft_token_num = np.full((batch_size, 1), max_draft, dtype=np.int32)
    draft_tokens = np.zeros((batch_size, max_draft + 1), dtype=np.int64)
    seq_lens_this_time = np.ones(batch_size, dtype=np.int32)
    seq_lens_encoder = np.zeros(batch_size, dtype=np.int32)
    seq_lens_decoder = np.ones(batch_size, dtype=np.int32)
    max_dec_len = np.full((batch_size, 1), 200, dtype=np.int64)

    for b in range(batch_size):
        # Copy prompt into token_ids_all
        token_ids_all[b, :input_len] = input_ids[b]
        # Simulate generated tokens: copy contiguous blocks from prompt
        # to guarantee ngram matches exist
        gen_len = 20
        src = rng.randint(0, max(1, input_len - gen_len))
        token_ids_all[b, input_len : input_len + gen_len] = input_ids[b, src : src + gen_len]
        # step_idx = last valid position (0-based index)
        step_idx[b] = gen_len - 1

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


def _make_mixed_test_data(batch_size=4, input_len=64, pre_ids_len=256, max_draft=10, seed=42):
    """Create realistic test tensors for hybrid_mtp_ngram op."""
    rng = np.random.RandomState(seed)
    vocab_size = 1000

    input_ids = rng.randint(0, vocab_size, (batch_size, input_len)).astype(np.int64)
    input_ids_len = np.full((batch_size, 1), input_len, dtype=np.int64)

    pre_ids = np.zeros((batch_size, pre_ids_len), dtype=np.int64)
    step_idx = np.zeros((batch_size, 1), dtype=np.int64)
    draft_token_num = np.full((batch_size, 1), max_draft, dtype=np.int32)
    draft_tokens = np.zeros((batch_size, max_draft + 1), dtype=np.int64)
    # For mixed: seq_lens_this_time starts at 1 (already has 1 draft token)
    seq_lens_this_time = np.ones(batch_size, dtype=np.int32)
    seq_lens_decoder = np.ones(batch_size, dtype=np.int32)
    max_dec_len = np.full((batch_size, 1), 200, dtype=np.int64)

    for b in range(batch_size):
        # Copy contiguous blocks from prompt to guarantee ngram matches
        gen_len = 20
        src = rng.randint(0, max(1, input_len - gen_len))
        pre_ids[b, :gen_len] = input_ids[b, src : src + gen_len]
        # step_idx = last valid position (0-based index)
        step_idx[b] = gen_len - 1

    return {
        "input_ids": input_ids,
        "input_ids_len": input_ids_len,
        "pre_ids": pre_ids,
        "step_idx": step_idx,
        "draft_token_num": draft_token_num,
        "draft_tokens": draft_tokens,
        "seq_lens_this_time": seq_lens_this_time,
        "seq_lens_decoder": seq_lens_decoder,
        "max_dec_len": max_dec_len,
    }


def _to_gpu(np_dict):
    """Convert numpy dict to GPU paddle tensors."""
    out = {}
    for k, v in np_dict.items():
        out[k] = paddle.to_tensor(v, place=paddle.CUDAPlace(0))
    return out


class TestNgramMatchKernel(unittest.TestCase):
    """Test ngram_match GPU kernel correctness against CPU reference."""

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA not available")
        paddle.set_device("gpu")
        # Import GPU ops (requires FastDeploy build)
        try:
            from fastdeploy.model_executor.ops.gpu import ngram_match

            cls.ngram_match = staticmethod(ngram_match)
        except Exception as e:
            raise unittest.SkipTest(f"Cannot import ngram_match op: {e}")

    def test_correctness_basic(self):
        """Basic correctness: GPU output matches CPU reference."""
        data = _make_ngram_test_data(batch_size=4, seed=42)
        max_ngram_size = 3
        max_draft_tokens = 10

        # CPU reference
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_ngram_match(
            data["input_ids"],
            data["input_ids_len"],
            data["token_ids_all"],
            data["prompt_lens"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_encoder"],
            data["seq_lens_decoder"],
            data["max_dec_len"],
            max_ngram_size,
            max_draft_tokens,
        )

        # GPU kernel
        gpu_data = _to_gpu(data)
        self.ngram_match(
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
            max_ngram_size,
            max_draft_tokens,
        )
        paddle.device.synchronize()

        gpu_draft = gpu_data["draft_tokens"].numpy()
        gpu_slt = gpu_data["seq_lens_this_time"].numpy()

        np.testing.assert_array_equal(gpu_slt, cpu_slt, err_msg="seq_lens_this_time mismatch")
        np.testing.assert_array_equal(gpu_draft, cpu_draft, err_msg="draft_tokens mismatch")

    def test_correctness_varied_seeds(self):
        """Test across multiple random seeds."""
        for seed in [0, 7, 123, 999]:
            with self.subTest(seed=seed):
                data = _make_ngram_test_data(batch_size=8, seed=seed)
                cpu_draft = data["draft_tokens"].copy()
                cpu_slt = data["seq_lens_this_time"].copy()
                _cpu_ngram_match(
                    data["input_ids"],
                    data["input_ids_len"],
                    data["token_ids_all"],
                    data["prompt_lens"],
                    data["step_idx"],
                    data["draft_token_num"],
                    cpu_draft,
                    cpu_slt,
                    data["seq_lens_encoder"],
                    data["seq_lens_decoder"],
                    data["max_dec_len"],
                    3,
                    10,
                )
                gpu_data = _to_gpu(data)
                self.ngram_match(
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
                    3,
                    10,
                )
                paddle.device.synchronize()
                np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
                np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_large_batch_long_seq(self):
        """bsz=256, seq_len=128k — scale the reviewer demanded.

        Uses high threshold to ensure all batches exercise the parallel search
        path (default threshold=128 would skip all batches at bsz=256).
        """
        high_threshold = 100000
        data = _make_ngram_test_data(batch_size=256, input_len=131072, max_model_len=131072 + 64, seed=77)
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_ngram_match(
            data["input_ids"],
            data["input_ids_len"],
            data["token_ids_all"],
            data["prompt_lens"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_encoder"],
            data["seq_lens_decoder"],
            data["max_dec_len"],
            3,
            10,
            threshold=high_threshold,
        )
        gpu_data = _to_gpu(data)
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(high_threshold)
        try:
            self.ngram_match(
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
                3,
                10,
            )
            paddle.device.synchronize()
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env
        np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_single_batch_long_seq(self):
        """bsz=1, seq_len=128k — single long sequence."""
        data = _make_ngram_test_data(batch_size=1, input_len=131072, max_model_len=131072 + 64, seed=88)
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_ngram_match(
            data["input_ids"],
            data["input_ids_len"],
            data["token_ids_all"],
            data["prompt_lens"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_encoder"],
            data["seq_lens_decoder"],
            data["max_dec_len"],
            3,
            10,
        )
        gpu_data = _to_gpu(data)
        self.ngram_match(
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
            3,
            10,
        )
        paddle.device.synchronize()
        np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_many_short_seqs(self):
        """bsz=256, seq_len=1k — many short sequences."""
        high_threshold = 100000
        data = _make_ngram_test_data(batch_size=256, input_len=1024, seed=55)
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_ngram_match(
            data["input_ids"],
            data["input_ids_len"],
            data["token_ids_all"],
            data["prompt_lens"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_encoder"],
            data["seq_lens_decoder"],
            data["max_dec_len"],
            3,
            10,
            threshold=high_threshold,
        )
        gpu_data = _to_gpu(data)
        old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
        os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(high_threshold)
        try:
            self.ngram_match(
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
                3,
                10,
            )
            paddle.device.synchronize()
        finally:
            if old_env is None:
                os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
            else:
                os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env
        np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_latency(self):
        """Benchmark: GPU kernel latency vs CPU transfer overhead."""
        # Warmup
        for _ in range(5):
            d = _to_gpu(_make_ngram_test_data(batch_size=32, input_len=512, seed=42))
            self.ngram_match(
                d["input_ids"],
                d["input_ids_len"],
                d["token_ids_all"],
                d["prompt_lens"],
                d["step_idx"],
                d["draft_token_num"],
                d["draft_tokens"],
                d["seq_lens_this_time"],
                d["seq_lens_encoder"],
                d["seq_lens_decoder"],
                d["max_dec_len"],
                3,
                10,
            )
        paddle.device.synchronize()

        # GPU path: kernel execution only (pre-created tensors, no data transfer)
        gpu_data = _to_gpu(_make_ngram_test_data(batch_size=32, input_len=512, seed=42))
        cpu_data = _make_ngram_test_data(batch_size=32, input_len=512, seed=42)
        n_runs = 100
        paddle.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            self.ngram_match(
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
                3,
                10,
            )
            paddle.device.synchronize()
        t1 = time.perf_counter()
        gpu_time_ms = (t1 - t0) / n_runs * 1000

        # CPU path: simulate the old copy-to-CPU-and-back pattern
        paddle.device.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            # Simulate old path: copy all tensors to CPU then back
            cpu_tensors = {k: paddle.to_tensor(v, place=paddle.CPUPlace()) for k, v in cpu_data.items()}
            _ = cpu_tensors["draft_tokens"].cuda()
            _ = cpu_tensors["seq_lens_this_time"].cuda()
            paddle.device.synchronize()
        t1 = time.perf_counter()
        cpu_copy_time_ms = (t1 - t0) / n_runs * 1000

        print(f"\n{'='*60}")
        print(f"LATENCY BENCHMARK (batch=32, input_len=512, {n_runs} runs)")
        print(f"  GPU kernel (zero-copy):   {gpu_time_ms:.3f} ms/call")
        print(f"  CPU path (copy overhead): {cpu_copy_time_ms:.3f} ms/call")
        print(f"  Speedup: {cpu_copy_time_ms / gpu_time_ms:.2f}x")
        print(f"{'='*60}")

    def test_latency_scaling(self):
        """Benchmark GPU kernel across batch sizes to show Phase 2 scales."""
        batch_sizes = [32, 128, 256, 512, 1024]
        input_len = 512
        n_runs = 50
        results = []

        for bsz in batch_sizes:
            # Pre-create tensors once per batch size (excluded from timing)
            gpu_data = _to_gpu(_make_ngram_test_data(batch_size=bsz, input_len=input_len, seed=42))
            cpu_data = _make_ngram_test_data(batch_size=bsz, input_len=input_len, seed=42)

            # Warmup
            for _ in range(3):
                self.ngram_match(
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
                    3,
                    10,
                )
            paddle.device.synchronize()

            # GPU kernel (pure kernel time — no data creation/transfer)
            paddle.device.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                self.ngram_match(
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
                    3,
                    10,
                )
                paddle.device.synchronize()
            gpu_ms = (time.perf_counter() - t0) / n_runs * 1000

            # CPU path: simulate the old copy-to-CPU-and-back pattern
            paddle.device.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_runs):
                cpu_tensors = {k: paddle.to_tensor(v, place=paddle.CPUPlace()) for k, v in cpu_data.items()}
                _ = cpu_tensors["draft_tokens"].cuda()
                _ = cpu_tensors["seq_lens_this_time"].cuda()
                paddle.device.synchronize()
            cpu_ms = (time.perf_counter() - t0) / n_runs * 1000

            results.append((bsz, gpu_ms, cpu_ms))

        print(f"\n{'='*72}")
        print(f"SCALING BENCHMARK (input_len={input_len}, {n_runs} runs per config)")
        print(f"{'─'*72}")
        print(f"{'batch':>6}  {'GPU (ms)':>10}  {'CPU (ms)':>10}  {'Speedup':>8}  {'GPU/batch(µs)':>14}")
        print(f"{'─'*72}")
        for bsz, gpu_ms, cpu_ms in results:
            speedup = cpu_ms / gpu_ms
            per_batch_us = gpu_ms / bsz * 1000
            print(f"{bsz:>6}  {gpu_ms:>10.3f}  {cpu_ms:>10.3f}  {speedup:>7.2f}x  {per_batch_us:>14.3f}")
        print(f"{'='*72}")

    def test_latency_extreme(self):
        """Benchmark: GPU kernel at extreme scale (bsz=256, seq_len=128k).

        Addresses the NCU profiler worst-case scenario (bsz=256 + 128k)
        raised in review.  Tests with production-realistic thresholds
        (8192, 16384) rather than the unlimited threshold used in
        correctness tests.
        """
        configs = [
            {"threshold": 8192, "label": "threshold=8192"},
            {"threshold": 16384, "label": "threshold=16384"},
        ]
        batch_size = 256
        input_len = 131072  # 128k
        n_runs = 1000

        # Pre-create tensors once (excluded from timing)
        gpu_data = _to_gpu(
            _make_ngram_test_data(
                batch_size=batch_size,
                input_len=input_len,
                max_model_len=input_len + 64,
                seed=77,
            )
        )
        cpu_data = _make_ngram_test_data(
            batch_size=batch_size,
            input_len=input_len,
            max_model_len=input_len + 64,
            seed=77,
        )

        print(f"\n{'='*72}")
        print(f"EXTREME BENCHMARK (batch={batch_size}, seq_len={input_len}, {n_runs} runs)")
        print(f"{'─'*72}")

        for cfg in configs:
            threshold = cfg["threshold"]
            old_env = os.environ.get("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD")
            os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = str(threshold)
            try:
                # Warmup
                for _ in range(3):
                    self.ngram_match(
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
                        3,
                        10,
                    )
                paddle.device.synchronize()

                # GPU kernel timing
                paddle.device.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    self.ngram_match(
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
                        3,
                        10,
                    )
                    paddle.device.synchronize()
                t1 = time.perf_counter()
                gpu_ms = (t1 - t0) / n_runs * 1000
            finally:
                if old_env is None:
                    os.environ.pop("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD", None)
                else:
                    os.environ["INFER_WITH_REFERENCE_TOKENUM_THRESHOLD"] = old_env

            # CPU path: simulate copy-to-CPU-and-back overhead at extreme scale
            cpu_runs = 50  # fewer runs — CPU copy of 256x128k is slow
            paddle.device.synchronize()
            t0 = time.perf_counter()
            for _ in range(cpu_runs):
                cpu_tensors = {k: paddle.to_tensor(v, place=paddle.CPUPlace()) for k, v in cpu_data.items()}
                _ = cpu_tensors["draft_tokens"].cuda()
                _ = cpu_tensors["seq_lens_this_time"].cuda()
                paddle.device.synchronize()
            t1 = time.perf_counter()
            cpu_ms = (t1 - t0) / cpu_runs * 1000

            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("inf")
            print(f"  [{cfg['label']}]")
            print(f"    GPU kernel:   {gpu_ms:.3f} ms/call  ({gpu_ms * 1000:.1f} us)")
            print(f"    CPU path:     {cpu_ms:.3f} ms/call")
            print(f"    Speedup:      {speedup:.1f}x")
            print()

        print(f"{'='*72}")


class TestHybridMtpNgramKernel(unittest.TestCase):
    """Test hybrid_mtp_ngram GPU kernel correctness against CPU reference."""

    @classmethod
    def setUpClass(cls):
        if not paddle.is_compiled_with_cuda():
            raise unittest.SkipTest("CUDA not available")
        paddle.set_device("gpu")
        try:
            from fastdeploy.model_executor.ops.gpu import hybrid_mtp_ngram

            cls.hybrid_mtp_ngram = staticmethod(hybrid_mtp_ngram)
        except Exception as e:
            raise unittest.SkipTest(f"Cannot import hybrid_mtp_ngram op: {e}")

    def test_correctness_basic(self):
        """Basic correctness: GPU output matches CPU reference."""
        data = _make_mixed_test_data(batch_size=4, seed=42)
        max_ngram_size = 3
        min_ngram_size = 1
        max_draft_tokens = 10

        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_hybrid_mtp_ngram(
            data["input_ids"],
            data["input_ids_len"],
            data["pre_ids"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_decoder"],
            data["max_dec_len"],
            max_ngram_size,
            min_ngram_size,
            max_draft_tokens,
        )

        gpu_data = _to_gpu(data)
        self.hybrid_mtp_ngram(
            gpu_data["input_ids"],
            gpu_data["input_ids_len"],
            gpu_data["pre_ids"],
            gpu_data["step_idx"],
            gpu_data["draft_token_num"],
            gpu_data["draft_tokens"],
            gpu_data["seq_lens_this_time"],
            gpu_data["seq_lens_decoder"],
            gpu_data["max_dec_len"],
            max_ngram_size,
            min_ngram_size,
            max_draft_tokens,
        )
        paddle.device.synchronize()

        np.testing.assert_array_equal(
            gpu_data["seq_lens_this_time"].numpy(), cpu_slt, err_msg="seq_lens_this_time mismatch"
        )
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft, err_msg="draft_tokens mismatch")

    def test_correctness_varied_seeds(self):
        """Test across multiple random seeds."""
        for seed in [0, 7, 123, 999]:
            with self.subTest(seed=seed):
                data = _make_mixed_test_data(batch_size=8, seed=seed)
                cpu_draft = data["draft_tokens"].copy()
                cpu_slt = data["seq_lens_this_time"].copy()
                _cpu_hybrid_mtp_ngram(
                    data["input_ids"],
                    data["input_ids_len"],
                    data["pre_ids"],
                    data["step_idx"],
                    data["draft_token_num"],
                    cpu_draft,
                    cpu_slt,
                    data["seq_lens_decoder"],
                    data["max_dec_len"],
                    3,
                    1,
                    10,
                )
                gpu_data = _to_gpu(data)
                self.hybrid_mtp_ngram(
                    gpu_data["input_ids"],
                    gpu_data["input_ids_len"],
                    gpu_data["pre_ids"],
                    gpu_data["step_idx"],
                    gpu_data["draft_token_num"],
                    gpu_data["draft_tokens"],
                    gpu_data["seq_lens_this_time"],
                    gpu_data["seq_lens_decoder"],
                    gpu_data["max_dec_len"],
                    3,
                    1,
                    10,
                )
                paddle.device.synchronize()
                np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
                np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_large_batch_long_seq(self):
        """bsz=256, seq_len=128k — scale the reviewer demanded.

        Uses high threshold to ensure all batches exercise the parallel search
        path (default threshold=1024 would skip many batches at bsz=256).
        """
        high_threshold = 100000
        data = _make_mixed_test_data(batch_size=256, input_len=131072, pre_ids_len=131072 + 64, seed=77)
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_hybrid_mtp_ngram(
            data["input_ids"],
            data["input_ids_len"],
            data["pre_ids"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_decoder"],
            data["max_dec_len"],
            3,
            1,
            10,
            threshold=high_threshold,
        )
        gpu_data = _to_gpu(data)
        old_env = os.environ.get("SPEC_TOKENUM_THRESHOLD")
        os.environ["SPEC_TOKENUM_THRESHOLD"] = str(high_threshold)
        try:
            self.hybrid_mtp_ngram(
                gpu_data["input_ids"],
                gpu_data["input_ids_len"],
                gpu_data["pre_ids"],
                gpu_data["step_idx"],
                gpu_data["draft_token_num"],
                gpu_data["draft_tokens"],
                gpu_data["seq_lens_this_time"],
                gpu_data["seq_lens_decoder"],
                gpu_data["max_dec_len"],
                3,
                1,
                10,
            )
            paddle.device.synchronize()
        finally:
            if old_env is None:
                os.environ.pop("SPEC_TOKENUM_THRESHOLD", None)
            else:
                os.environ["SPEC_TOKENUM_THRESHOLD"] = old_env
        np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_single_batch_long_seq(self):
        """bsz=1, seq_len=128k — single long sequence."""
        data = _make_mixed_test_data(batch_size=1, input_len=131072, pre_ids_len=131072 + 64, seed=88)
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_hybrid_mtp_ngram(
            data["input_ids"],
            data["input_ids_len"],
            data["pre_ids"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_decoder"],
            data["max_dec_len"],
            3,
            1,
            10,
        )
        gpu_data = _to_gpu(data)
        self.hybrid_mtp_ngram(
            gpu_data["input_ids"],
            gpu_data["input_ids_len"],
            gpu_data["pre_ids"],
            gpu_data["step_idx"],
            gpu_data["draft_token_num"],
            gpu_data["draft_tokens"],
            gpu_data["seq_lens_this_time"],
            gpu_data["seq_lens_decoder"],
            gpu_data["max_dec_len"],
            3,
            1,
            10,
        )
        paddle.device.synchronize()
        np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)

    def test_many_short_seqs(self):
        """bsz=256, seq_len=1k — many short sequences."""
        high_threshold = 100000
        data = _make_mixed_test_data(batch_size=256, input_len=1024, seed=55)
        cpu_draft = data["draft_tokens"].copy()
        cpu_slt = data["seq_lens_this_time"].copy()
        _cpu_hybrid_mtp_ngram(
            data["input_ids"],
            data["input_ids_len"],
            data["pre_ids"],
            data["step_idx"],
            data["draft_token_num"],
            cpu_draft,
            cpu_slt,
            data["seq_lens_decoder"],
            data["max_dec_len"],
            3,
            1,
            10,
            threshold=high_threshold,
        )
        gpu_data = _to_gpu(data)
        old_env = os.environ.get("SPEC_TOKENUM_THRESHOLD")
        os.environ["SPEC_TOKENUM_THRESHOLD"] = str(high_threshold)
        try:
            self.hybrid_mtp_ngram(
                gpu_data["input_ids"],
                gpu_data["input_ids_len"],
                gpu_data["pre_ids"],
                gpu_data["step_idx"],
                gpu_data["draft_token_num"],
                gpu_data["draft_tokens"],
                gpu_data["seq_lens_this_time"],
                gpu_data["seq_lens_decoder"],
                gpu_data["max_dec_len"],
                3,
                1,
                10,
            )
            paddle.device.synchronize()
        finally:
            if old_env is None:
                os.environ.pop("SPEC_TOKENUM_THRESHOLD", None)
            else:
                os.environ["SPEC_TOKENUM_THRESHOLD"] = old_env
        np.testing.assert_array_equal(gpu_data["seq_lens_this_time"].numpy(), cpu_slt)
        np.testing.assert_array_equal(gpu_data["draft_tokens"].numpy(), cpu_draft)


if __name__ == "__main__":
    unittest.main(verbosity=2)
