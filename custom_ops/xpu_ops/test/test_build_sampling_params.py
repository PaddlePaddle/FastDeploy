# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
Unit tests for build_sampling_params XPU op.

Verifies that the XPU kernel produces the same output as the Python reference
implementation (padding_sampling_params) for all cases:
  - pure decoder batches (seq_lens_encoder == 0)
  - pure encoder batches (seq_lens_encoder > 0)
  - mixed encoder/decoder batches
  - single-item batch (bs=1)
  - seed wrap-around near MAX_INFER_SEED
"""

import unittest

import numpy as np
import paddle

DEVICE_PLACE = paddle.XPUPlace(0) if paddle.is_compiled_with_xpu() else paddle.CPUPlace()
MAX_INFER_SEED = 2147483646


# ---------------------------------------------------------------------------
# Python reference implementation (mirrors sampler.py padding_sampling_params)
# ---------------------------------------------------------------------------


def ref_build_sampling_params(top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder, increment_value):
    """
    Pure-Python reference that mirrors the cpu_wrapper logic in
    build_sampling_params.cpp.

    Returns (top_p_padding, top_k_padding, topp_seed) as numpy arrays of
    shape [token_num, 1], and infer_seed updated in-place.
    """
    bs = len(seq_lens_this_time)
    infer_seed = infer_seed.copy()  # don't mutate the input

    top_p_out, top_k_out, seed_out = [], [], []
    for bi in range(bs):
        is_decoder = seq_lens_encoder[bi] == 0
        repeat = int(seq_lens_this_time[bi]) if is_decoder else 1
        bi_seed = int(infer_seed[bi])
        for local_pos in range(repeat):
            offset = local_pos * 4 if is_decoder else 0
            top_p_out.append([top_p[bi]])
            top_k_out.append([top_k[bi]])
            seed_out.append([(bi_seed + offset) % MAX_INFER_SEED])
        infer_seed[bi] = (bi_seed + increment_value) % MAX_INFER_SEED

    top_p_out = np.array(top_p_out, dtype=np.float32)
    top_k_out = np.array(top_k_out, dtype=np.int64)
    seed_out = np.array(seed_out, dtype=np.int64)
    return top_p_out, top_k_out, seed_out, infer_seed


# ---------------------------------------------------------------------------
# Helper: run the XPU op and return numpy results
# ---------------------------------------------------------------------------


def run_op(top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder, increment_value):
    from fastdeploy.model_executor.ops.xpu import build_sampling_params

    token_num = int(
        sum(seq_lens_this_time[i] if seq_lens_encoder[i] == 0 else 1 for i in range(len(seq_lens_this_time)))
    )

    tp = paddle.to_tensor(top_p, place=DEVICE_PLACE)
    tk = paddle.to_tensor(top_k, place=DEVICE_PLACE)
    seed = paddle.to_tensor(infer_seed.copy(), place=DEVICE_PLACE)
    slt = paddle.to_tensor(seq_lens_this_time, place=DEVICE_PLACE)
    sle = paddle.to_tensor(seq_lens_encoder, place=DEVICE_PLACE)

    tp_pad, tk_pad, seed_pad = build_sampling_params(
        tp,
        tk,
        seed,
        slt,
        sle,
        token_num_output_cpu=token_num,
        increment_value=increment_value,
    )
    return (tp_pad.numpy(), tk_pad.numpy(), seed_pad.numpy(), seed.numpy())  # seed was updated in-place inside the op


# ---------------------------------------------------------------------------
# Assertion helper
# ---------------------------------------------------------------------------


def assert_close(ref, got, name, rtol=1e-5, atol=1e-5):
    assert ref.shape == got.shape, f"[{name}] shape mismatch: ref={ref.shape} got={got.shape}"
    if ref.dtype in (np.float32, np.float64):
        ok = np.allclose(ref, got, rtol=rtol, atol=atol)
    else:
        ok = np.array_equal(ref, got)
    assert ok, f"[{name}] value mismatch.\n" f"ref=\n{ref}\ngot=\n{got}\ndiff=\n{ref - got}"


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestBuildSamplingParams(unittest.TestCase):

    def _run_and_compare(self, top_p, top_k, infer_seed, seq_lens_this_time, seq_lens_encoder, increment_value):
        try:
            tp_xpu, tk_xpu, sd_xpu, seed_xpu = run_op(
                top_p,
                top_k,
                infer_seed,
                seq_lens_this_time,
                seq_lens_encoder,
                increment_value,
            )
        except ImportError as e:
            self.skipTest(f"XPU op not available: {e}")

        tp_ref, tk_ref, sd_ref, seed_ref = ref_build_sampling_params(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )

        assert_close(tp_ref, tp_xpu, "top_p_padding")
        assert_close(tk_ref, tk_xpu, "top_k_padding")
        assert_close(sd_ref, sd_xpu, "topp_seed")
        assert_close(seed_ref, seed_xpu, "infer_seed")

    # ------------------------------------------------------------------
    # Test 1: pure decoder batch (all seq_lens_encoder == 0)
    # ------------------------------------------------------------------
    def test_pure_decoder(self):
        top_p = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        top_k = np.array([50, 40, 30], dtype=np.int64)
        infer_seed = np.array([100, 200, 300], dtype=np.int64)
        seq_lens_this_time = np.array([4, 3, 2], dtype=np.int32)
        seq_lens_encoder = np.array([0, 0, 0], dtype=np.int32)
        increment_value = 16  # token_num * 4 = (4+3+2)*4 is typical

        self._run_and_compare(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )

    # ------------------------------------------------------------------
    # Test 2: pure encoder batch (all seq_lens_encoder > 0)
    #   -> each batch contributes exactly 1 output token, no seed offset
    # ------------------------------------------------------------------
    def test_pure_encoder(self):
        top_p = np.array([0.95, 0.85], dtype=np.float32)
        top_k = np.array([10, 20], dtype=np.int64)
        infer_seed = np.array([1000, 2000], dtype=np.int64)
        seq_lens_this_time = np.array([5, 7], dtype=np.int32)
        seq_lens_encoder = np.array([5, 7], dtype=np.int32)  # all encoder
        increment_value = 8

        self._run_and_compare(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )

    # ------------------------------------------------------------------
    # Test 3: mixed encoder/decoder
    # ------------------------------------------------------------------
    def test_mixed(self):
        top_p = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
        top_k = np.array([50, 40, 30, 20], dtype=np.int64)
        infer_seed = np.array([10, 20, 30, 40], dtype=np.int64)
        seq_lens_this_time = np.array([3, 4, 2, 5], dtype=np.int32)
        # batch 0,2 are decoder; batch 1,3 are encoder
        seq_lens_encoder = np.array([0, 4, 0, 5], dtype=np.int32)
        increment_value = 20

        self._run_and_compare(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )

    # ------------------------------------------------------------------
    # Test 4: bs=1 (single item)
    # ------------------------------------------------------------------
    def test_single_item(self):
        top_p = np.array([0.5], dtype=np.float32)
        top_k = np.array([5], dtype=np.int64)
        infer_seed = np.array([42], dtype=np.int64)
        seq_lens_this_time = np.array([6], dtype=np.int32)
        seq_lens_encoder = np.array([0], dtype=np.int32)
        increment_value = 24

        self._run_and_compare(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )

    # ------------------------------------------------------------------
    # Test 5: seed near wrap-around boundary
    # ------------------------------------------------------------------
    def test_seed_wraparound(self):
        # Seeds close to MAX_INFER_SEED to trigger modulo wrap
        near_max = MAX_INFER_SEED - 8
        top_p = np.array([0.9, 0.9], dtype=np.float32)
        top_k = np.array([50, 50], dtype=np.int64)
        infer_seed = np.array([near_max, near_max - 1], dtype=np.int64)
        seq_lens_this_time = np.array([4, 4], dtype=np.int32)
        seq_lens_encoder = np.array([0, 0], dtype=np.int32)
        increment_value = 16

        self._run_and_compare(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )

    # ------------------------------------------------------------------
    # Test 6: seq_lens_this_time == 1 for all decoder batches
    #   (degenerate case: each decoder produces exactly one token)
    # ------------------------------------------------------------------
    def test_single_token_per_batch(self):
        top_p = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        top_k = np.array([50, 40, 30], dtype=np.int64)
        infer_seed = np.array([1, 2, 3], dtype=np.int64)
        seq_lens_this_time = np.array([1, 1, 1], dtype=np.int32)
        seq_lens_encoder = np.array([0, 0, 0], dtype=np.int32)
        increment_value = 4

        self._run_and_compare(
            top_p,
            top_k,
            infer_seed,
            seq_lens_this_time,
            seq_lens_encoder,
            increment_value,
        )


if __name__ == "__main__":
    unittest.main()
