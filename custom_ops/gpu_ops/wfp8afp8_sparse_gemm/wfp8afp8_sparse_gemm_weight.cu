// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#include "helper.h"
#include "paddle/extension.h"


void pack_E(const uint8_t *E_src, int32_t *E_dst, const int M, const int K, const int Batch) {
    // 选择的下标   对应的16进制
    //    01          4
    //    02          8
    //    03          12
    //    12          9
    //    13          13
    //    23          14
    const int ld1 = K / 4;
    const int ld2 = K / 4 / 8;
    const uint8_t select_idx[6] = {14, 13, 9, 12, 8, 4};
    for (int b = 0; b < Batch; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int k = 0; k < ld1; k+=8) {
                uint32_t dst = 0;
                for (int k2 = 7; k2 > 0; --k2) {
                    dst |= select_idx[E_src[b * M * ld1 + m * ld1 + k + k2]];
                    dst <<= 4;
                }
                dst |= select_idx[E_src[b * M * ld1 + m * ld1 + k]];
                E_dst[b * M * ld2 + m * ld2 + k / 8] = dst;
            }
        }
    }
}

void peruate_E(const int32_t *E_src, int32_t *E_dst, const int M, const int K, const int Batch) {
    const int m_nums = M / 128;
    const int k_nums = K / 128;
    for (int b = 0; b < Batch; ++b) {
        for (int m = 0; m < m_nums; ++m) {
            for (int k = 0; k < k_nums; ++k) {
                const int dst_idx = b * m_nums * k_nums * 512 + m * k_nums * 512 + k * 512;
                for (int i = 0; i < 8; ++i) {
                    for (int j = 0; j < 8; ++j) {
                        E_dst[dst_idx + 0 + j * 32 + 4 * i] = E_src[dst_idx + 0 + j * 64 + 4 * i];
                        E_dst[dst_idx + 2 + j * 32 + 4 * i] = E_src[dst_idx + 1 + j * 64 + 4 * i];
                        E_dst[dst_idx + 1 + j * 32 + 4 * i] = E_src[dst_idx + 32 + j * 64 + 4 * i];
                        E_dst[dst_idx + 3 + j * 32 + 4 * i] = E_src[dst_idx + 33 + j * 64 + 4 * i];
                    }
                    for (int j = 0; j < 8; ++j) {
                        E_dst[dst_idx + 256 + j * 32 + 4 * i] = E_src[dst_idx + 2 + j * 64 + 4 * i];
                        E_dst[dst_idx + 258 + j * 32 + 4 * i] = E_src[dst_idx + 3 + j * 64 + 4 * i];
                        E_dst[dst_idx + 257 + j * 32 + 4 * i] = E_src[dst_idx + 34 + j * 64 + 4 * i];
                        E_dst[dst_idx + 259 + j * 32 + 4 * i] = E_src[dst_idx + 35 + j * 64 + 4 * i];
                    }
                }
            }
        }
    }
}

std::vector<paddle::Tensor> WFp8AFp8GemmSparseIdxConvert(
        const paddle::Tensor& weight,
        const int batch_size,
        const int M,
        const int K) {

    paddle::Tensor weight_temp = paddle::empty({batch_size, M, K / 32}, paddle::DataType::INT32, weight.place());
    paddle::Tensor weight_new = paddle::empty({batch_size, M, K / 32}, paddle::DataType::INT32, weight.place());
    pack_E(weight.data<uint8_t>(), weight_temp.data<int32_t>(), M, K, batch_size);
    peruate_E(weight_temp.data<int32_t>(), weight_new.data<int32_t>(), M, K, batch_size);
    return {weight_new};
}



PD_BUILD_STATIC_OP(wfp8afp8_gemm_sparse_idx_convert)
    .Inputs({"weight"})
    .Outputs({"converted_weight"})
    .Attrs({"batch: int",
            "M: int",
            "K: int"})
    .SetKernelFn(PD_KERNEL(WFp8AFp8GemmSparseIdxConvert));
