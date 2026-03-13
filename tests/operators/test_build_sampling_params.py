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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.ops.gpu import build_sampling_params

MAX_INFER_SEED = 9223372036854775806
BLOCK_DIM = 64


def build_sampling_params_ref(
    top_p,
    top_k,
    infer_seed,
    cu_seq_lens_q_output,
    token_num_output_cpu,
    increment_value,
):
    """
    Python reference implementation for BuildSamplingParamsKernel.

    Returns:
        top_p_padding:  float32[token_num_output_cpu, 1]
        top_k_padding:  int64[token_num_output_cpu, 1]
        topp_seed:      int64[token_num_output_cpu, 1]
        infer_seed:     int64[real_bsz] (updated in-place)
    """
    real_bsz = len(top_p)
    top_p_padding = np.zeros((token_num_output_cpu, 1), dtype=np.float32)
    top_k_padding = np.zeros((token_num_output_cpu, 1), dtype=np.int64)
    topp_seed = np.zeros((token_num_output_cpu, 1), dtype=np.int64)
    infer_seed = infer_seed.copy()

    for bi in range(real_bsz):
        cur_start = cu_seq_lens_q_output[bi]
        cur_end = cu_seq_lens_q_output[bi + 1]
        bi_top_p = top_p[bi]
        bi_top_k = top_k[bi]

        for tid in range(BLOCK_DIM):
            bi_infer_seed = (infer_seed[bi] + tid * 4) % MAX_INFER_SEED
            i = tid
            while i < cur_end - cur_start:
                pad_idx = cur_start + i
                top_p_padding[pad_idx, 0] = bi_top_p
                top_k_padding[pad_idx, 0] = bi_top_k
                topp_seed[pad_idx, 0] = bi_infer_seed
                bi_infer_seed = (bi_infer_seed + BLOCK_DIM * 4) % MAX_INFER_SEED
                i += BLOCK_DIM

        infer_seed[bi] = (infer_seed[bi] + increment_value) % MAX_INFER_SEED

    return top_p_padding, top_k_padding, topp_seed, infer_seed


def build_inputs(real_bsz, seq_lens_this_time_list, seq_lens_encoder_list, seed=42):
    """
    Helper to build test inputs.

    For prefill requests (seq_lens_encoder > 0), the output length is 1.
    For decode requests (seq_lens_encoder == 0), the output length equals seq_lens_this_time.
    seq_lens_this_time == 0 means the slot is empty, output length is 0.
    """
    rng = np.random.default_rng(seed)

    top_p = rng.uniform(0.0, 1.0, size=(real_bsz,)).astype(np.float32)
    top_k = rng.integers(1, 100, size=(real_bsz,)).astype(np.int64)
    infer_seed = rng.integers(0, MAX_INFER_SEED, size=(real_bsz,)).astype(np.int64)

    seq_lens_this_time = np.array(seq_lens_this_time_list, dtype=np.int32)
    seq_lens_encoder = np.array(seq_lens_encoder_list, dtype=np.int32)

    seq_lens_output = np.zeros(real_bsz, dtype=np.int32)
    for bid in range(real_bsz):
        if seq_lens_this_time[bid] == 0:
            seq_lens_output[bid] = 0
        elif seq_lens_encoder[bid] > 0:
            seq_lens_output[bid] = 1
        else:
            seq_lens_output[bid] = seq_lens_this_time[bid]

    cu_seq_lens_q_output = np.zeros(real_bsz + 1, dtype=np.int32)
    for i in range(real_bsz):
        cu_seq_lens_q_output[i + 1] = cu_seq_lens_q_output[i] + seq_lens_output[i]

    token_num_output_cpu = int(cu_seq_lens_q_output[-1])

    return {
        "top_p": top_p,
        "top_k": top_k,
        "infer_seed": infer_seed,
        "seq_lens_this_time": seq_lens_this_time,
        "cu_seq_lens_q_output": cu_seq_lens_q_output,
        "token_num_output_cpu": token_num_output_cpu,
    }


def run_and_compare(tc, inputs, increment_value):
    """
    Call GPU op and Python reference, compare all outputs.
    """
    t_top_p = paddle.to_tensor(inputs["top_p"], dtype="float32")
    t_top_k = paddle.to_tensor(inputs["top_k"], dtype="int64")
    t_infer_seed = paddle.to_tensor(inputs["infer_seed"], dtype="int64")
    t_seq_lens_this_time = paddle.to_tensor(inputs["seq_lens_this_time"], dtype="int32")
    t_cu_seq_lens_q_output = paddle.to_tensor(inputs["cu_seq_lens_q_output"], dtype="int32")
    token_num_output_cpu = inputs["token_num_output_cpu"]

    gpu_outs = build_sampling_params(
        t_top_p,
        t_top_k,
        t_infer_seed,
        t_seq_lens_this_time,
        t_cu_seq_lens_q_output,
        token_num_output_cpu,
        increment_value,
    )

    ref_outs = build_sampling_params_ref(
        inputs["top_p"],
        inputs["top_k"],
        inputs["infer_seed"],
        inputs["cu_seq_lens_q_output"],
        token_num_output_cpu,
        increment_value,
    )

    np.testing.assert_allclose(gpu_outs[0].numpy(), ref_outs[0], rtol=1e-6, err_msg="Mismatch in top_p_padding")
    np.testing.assert_allclose(gpu_outs[1].numpy(), ref_outs[1], err_msg="Mismatch in top_k_padding")
    np.testing.assert_allclose(gpu_outs[2].numpy(), ref_outs[2], err_msg="Mismatch in topp_seed")
    np.testing.assert_allclose(t_infer_seed.numpy(), ref_outs[3], err_msg="Mismatch in infer_seed (in-place update)")


class TestBuildSamplingParams(unittest.TestCase):
    """Unit tests for build_sampling_params custom operator."""

    # ----------------------------------------------------------------
    # Test 1: exact golden values — mixed prefill and decode
    #   bid=0: decode, seq_lens_this_time=2 => output=2
    #   bid=1: prefill, seq_lens_this_time=10 => output=1
    # ----------------------------------------------------------------
    def test_exact_golden_values(self):
        top_p = np.array([0.9, 0.5], dtype=np.float32)
        top_k = np.array([50, 10], dtype=np.int64)
        infer_seed = np.array([100, 200], dtype=np.int64)
        cu_seq_lens_q_output = np.array([0, 2, 3], dtype=np.int32)
        seq_lens_this_time = np.array([2, 10], dtype=np.int32)

        t_top_p = paddle.to_tensor(top_p, dtype="float32")
        t_top_k = paddle.to_tensor(top_k, dtype="int64")
        t_infer_seed = paddle.to_tensor(infer_seed, dtype="int64")
        t_seq_lens_this_time = paddle.to_tensor(seq_lens_this_time, dtype="int32")
        t_cu_seq_lens_q_output = paddle.to_tensor(cu_seq_lens_q_output, dtype="int32")

        gpu_outs = build_sampling_params(
            t_top_p,
            t_top_k,
            t_infer_seed,
            t_seq_lens_this_time,
            t_cu_seq_lens_q_output,
            3,
            1,
        )

        np.testing.assert_allclose(gpu_outs[0].numpy().flatten(), [0.9, 0.9, 0.5], rtol=1e-6)
        np.testing.assert_allclose(gpu_outs[1].numpy().flatten(), [50, 50, 10])
        # topp_seed: bi=0 tid=0 => 100, bi=0 tid=1 => 104; bi=1 tid=0 => 200
        np.testing.assert_allclose(gpu_outs[2].numpy().flatten(), [100, 104, 200])
        np.testing.assert_allclose(t_infer_seed.numpy(), [101, 201])

    # ----------------------------------------------------------------
    # Test 2: mixed prefill/decode batch with reference comparison
    #   bid=0: decode, seq_lens_this_time=3 => output=3
    #   bid=1: prefill, seq_lens_this_time=50 => output=1
    #   bid=2: decode, seq_lens_this_time=5 => output=5
    #   bid=3: prefill, seq_lens_this_time=100 => output=1
    #   bid=4: empty slot => output=0
    # ----------------------------------------------------------------
    def test_mixed_prefill_decode(self):
        inputs = build_inputs(
            real_bsz=5,
            seq_lens_this_time_list=[3, 50, 5, 100, 0],
            seq_lens_encoder_list=[0, 50, 0, 100, 0],
            seed=300,
        )
        self.assertEqual(inputs["token_num_output_cpu"], 10)
        run_and_compare(self, inputs, increment_value=5)

    # ----------------------------------------------------------------
    # Test 3: random stress test with mixed prefill/decode configs
    # ----------------------------------------------------------------
    def test_random_configs(self):
        configs = [
            {"real_bsz": 8, "max_seq_len": 4, "increment_value": 1, "seed": 700},
            {"real_bsz": 32, "max_seq_len": 16, "increment_value": 16, "seed": 800},
        ]
        for cfg in configs:
            with self.subTest(**cfg):
                rng = np.random.default_rng(cfg["seed"])
                real_bsz = cfg["real_bsz"]
                max_seq_len = cfg["max_seq_len"]
                seq_lens_this_time_list = rng.integers(0, max_seq_len + 1, size=real_bsz).tolist()
                seq_lens_encoder_list = []
                for s in seq_lens_this_time_list:
                    if s > 0 and rng.random() < 0.3:
                        seq_lens_encoder_list.append(s)
                    else:
                        seq_lens_encoder_list.append(0)

                inputs = build_inputs(
                    real_bsz=real_bsz,
                    seq_lens_this_time_list=seq_lens_this_time_list,
                    seq_lens_encoder_list=seq_lens_encoder_list,
                    seed=cfg["seed"],
                )
                if inputs["token_num_output_cpu"] == 0:
                    continue
                run_and_compare(self, inputs, increment_value=cfg["increment_value"])


if __name__ == "__main__":
    unittest.main()
