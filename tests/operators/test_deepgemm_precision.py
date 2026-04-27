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

import paddle

paddle.enable_compat(scope={"deep_gemm"})


paddle.set_default_dtype("bfloat16")


class TestDeepDenseGemm(unittest.TestCase):
    def setUp(self):
        pass

    def one_invoke(self, M, N, K):
        prop = paddle.device.cuda.get_device_properties()
        if prop.major != 10:
            return

        import deep_gemm

        block_size = 128

        raw_x = paddle.randn([M, K], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        raw_x_scale = paddle.randn([M, K // block_size], dtype="float32")

        raw_x_scale = paddle.randn([M, K // block_size], dtype="float32") * 10 + 127
        raw_x_scale = paddle.clip(raw_x_scale, 0, 127)
        raw_x_scale = raw_x_scale.cast("int32")
        raw_x_scale = raw_x_scale.cast("uint8").view("int32")

        float32_x_scale = raw_x_scale.view("uint8").cast("int32").flatten().numpy().tolist()
        for i in range(len(float32_x_scale)):
            float32_x_scale[i] = 2.0 ** (float32_x_scale[i] - 127)
        float32_x_scale = (
            paddle.to_tensor(float32_x_scale, dtype="float32")
            .reshape([M, K // block_size, 1])
            .tile([1, 1, block_size])
            .reshape([M, K])
        )

        raw_w = paddle.randn([N, K], dtype="bfloat16").cast(paddle.float8_e4m3fn)
        raw_w_scale = paddle.randn([N // block_size, K // block_size], dtype="float32")

        raw_w_scale = paddle.zeros([N, K // block_size], dtype="int32") + 128
        raw_w_scale = raw_w_scale.cast("uint8").view("int32")

        baseline_out = paddle.empty([M, N], dtype="bfloat16")
        tmp0 = raw_x.cast("float32") * float32_x_scale

        tmp1 = raw_w.cast("float32") * 2

        baseline_out = paddle.matmul(tmp0, tmp1, False, True)

        deepgemm_output = paddle.zeros_like(baseline_out)
        for i in range(10):
            a = paddle.zeros([1024, 1024, 1024]) + 1
            del a

            a = raw_x_scale.transpose([1, 0]).contiguous().transpose([1, 0])
            b = raw_w_scale.transpose([1, 0]).contiguous().transpose([1, 0])

            deep_gemm.fp8_gemm_nt(
                (raw_x, a),
                (raw_w, b),
                deepgemm_output,
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

        self.one_invoke(128 * 20, 2048, 4096)
        self.one_invoke(128 * 20, 2048, 2048)

        # p.stop()


if __name__ == "__main__":
    unittest.main()
