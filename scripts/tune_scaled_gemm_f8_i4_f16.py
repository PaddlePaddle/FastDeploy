# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
"""tune_cutlass_fp8int4_gemm"""

import os

import paddle
from tqdm import tqdm

from fastdeploy.model_executor.ops.gpu import scaled_gemm_f8_i4_f16


def tune_scaled_gemm_f8_i4_f16(ns: list, ks: list, dtype="int8", is_test=True, is_read_from_file=False):
    """
    Tune fp8 int4 gemm.
    """
    assert len(ns) == len(ks), "list[n] and list[k] should have the same length!"
    os.environ["FLAGS_fastdeploy_op_configs"] = "tune"
    mm_tmp = []

    for m in range(1, 4, 1):
        mm_tmp.append(m)

    for m in range(4, 16, 4):
        mm_tmp.append(m)

    for m in range(16, 64, 16):
        mm_tmp.append(m)

    for m in range(64, 256, 32):
        mm_tmp.append(m)

    for m in range(256, 512, 64):
        mm_tmp.append(m)

    for m in range(512, 1024, 128):
        mm_tmp.append(m)

    for m in range(1024, 8192, 1024):
        mm_tmp.append(m)

    # Note the end value is 32769 to include 32768
    for m in range(8192, 32769, 4096):
        mm_tmp.append(m)

    for m in tqdm(mm_tmp):
        for idx in range(0, len(ns)):
            n = ns[idx]
            k = ks[idx]

            A = paddle.cast(paddle.ones((m, k)), "float8_e4m3fn")
            B = paddle.cast(paddle.ones((n // 2, k)), "int8")
            w_scale = paddle.ones(n)
            scaled_gemm_f8_i4_f16(
                x=A.cuda(),
                y=B.cuda(),
                scale=paddle.cast(w_scale, dtype).cuda(),
                zero_points=None,
                bias=None,
                out_scale=1.0,
                groupsize=-1,
                out_dtype=dtype,
            )
