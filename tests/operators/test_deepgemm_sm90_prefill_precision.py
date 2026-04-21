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

import unittest

import numpy as np
import paddle

from fastdeploy.model_executor.layers.quantization.fp8_utils import deep_gemm

paddle.set_default_dtype("bfloat16")


class TestDeepGemmPrefill(unittest.TestCase):
    def setUp(self):
        pass

    def one_invoke(self, num_experts, M, N, K):
        token_num_in_eatch_batch = (paddle.zeros([num_experts], dtype="int32") + M).numpy().tolist()
        total_m = sum(token_num_in_eatch_batch)
        block_size = 128

        raw_x = paddle.randn([total_m, K], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        raw_x_scale = paddle.randn([total_m, K // block_size], dtype="float32")

        raw_w = paddle.randn([num_experts, N, K], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        raw_w_scale = paddle.randn([num_experts, N // block_size, K // block_size], dtype="float32")

        m_indices = np.zeros([total_m], dtype="int32")

        baseline_out = paddle.empty([total_m, N], dtype="bfloat16")
        for i in range(num_experts):
            start = sum(token_num_in_eatch_batch[:i])
            end = start + token_num_in_eatch_batch[i]

            this_expert_token = raw_x[start:end].contiguous().cast("float32")
            this_expert_token_scale = (
                raw_x_scale[start:end]
                .contiguous()
                .reshape([0, 0, 1])
                .tile([1, 1, block_size])
                .reshape([0, -1])
                .cast("float32")
            )
            tmp0 = this_expert_token * this_expert_token_scale

            this_expert_weight = raw_w[i].contiguous().cast("float32")
            this_expert_weight_scale = (
                raw_w_scale[i]
                .contiguous()
                .reshape([0, 1, -1, 1])
                .tile([1, block_size, 1, block_size])
                .reshape([N, K])
                .cast("float32")
            )
            tmp1 = this_expert_weight * this_expert_weight_scale

            out = paddle.matmul(tmp0, tmp1, False, True)
            baseline_out[start:end] = out

            m_indices[start:end] = i

        deepgemm_output = paddle.zeros_like(baseline_out)

        m_indices = paddle.to_tensor(m_indices, dtype="int32")

        for i in range(10):
            a = paddle.zeros([1024, 1024, 1024]) + 1
            del a
            deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt_contiguous(
                (raw_x, raw_x_scale.transpose([1, 0]).contiguous().transpose([1, 0])),
                (raw_w, raw_w_scale),
                deepgemm_output,
                m_indices,
            )

        print(baseline_out - deepgemm_output)

    def test_main(self):
        # import paddle.profiler as profiler
        # p = profiler.Profiler(
        #     targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        #     on_trace_ready=profiler.export_chrome_tracing("./profile_log"),
        # )
        # p.start()
        # p.step()

        self.one_invoke(48, 128 * 20, 2048, 4096)
        self.one_invoke(96, 128 * 20, 2048, 2048)

        # p.stop()


if __name__ == "__main__":
    unittest.main()
