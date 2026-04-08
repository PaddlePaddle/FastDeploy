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
#include "helper.h"
#include "paddle/extension.h"

void weight_convert(
    const uint8_t* weight, uint8_t* weight_new, int experts, int M, int K) {
  assert(K % 64 == 0);
  for (int b = 0; b < experts; ++b) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; k += 64) {
        for (int k_inner = 0; k_inner < 32; ++k_inner) {
          uint8_t temp = 0;
          uint8_t left = weight[b * M * K + m * K + k + k_inner];
          uint8_t right = weight[b * M * K + m * K + k + k_inner + 32];
          temp |= left << 4;
          temp |= right;
          weight_new[b * M * K / 2 + m * K / 2 + k / 2 + k_inner] =
              *reinterpret_cast<uint8_t*>(&temp);
        }
      }
    }
  }
}

__global__ void weight_permute_interleave_kernelw4afp8(const int8_t* input_data,
                                                       int8_t* output_data,
                                                       const int original_k,
                                                       const int original_n) {
  const int numel = original_k * original_n / 4;
  const int pack_group_size = 64;
  const int thread_group_size = pack_group_size / 4;  // 16
  const int thread_k_stride = original_k / 4;

  const int linear_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (linear_idx >= numel) return;

  const int n_id = linear_idx / thread_k_stride;
  const int k_id = linear_idx % thread_k_stride;
  const int k_group_idx = k_id / thread_group_size;
  const int k_idx_in_group = k_id % thread_group_size;

  const int8_t* src = input_data +
                      k_group_idx * pack_group_size / 2 * original_n +
                      k_idx_in_group * original_n + n_id;

  int8_t tmp0 = src[0];
  int8_t tmp1 = src[pack_group_size / 4 * original_n];

  int8_t tmp00 = (tmp0 & 0xF0) + 112;
  int8_t tmp01 = ((tmp0 << 4) & 0xF0) + 112;
  int8_t tmp10 = (tmp1 & 0xF0) + 112;
  int8_t tmp11 = ((tmp1 << 4) & 0xF0) + 112;

  uint8_t utmp00 = *(reinterpret_cast<uint8_t*>(&tmp00));
  uint8_t utmp01 = *(reinterpret_cast<uint8_t*>(&tmp01));
  uint8_t utmp10 = *(reinterpret_cast<uint8_t*>(&tmp10));
  uint8_t utmp11 = *(reinterpret_cast<uint8_t*>(&tmp11));

  utmp00 = (utmp00 & 0xF0) >> 4;
  utmp01 = (utmp01 & 0xF0) >> 4;
  utmp10 = (utmp10 & 0xF0) >> 4;
  utmp11 = (utmp11 & 0xF0) >> 4;

  tmp00 = *(reinterpret_cast<int8_t*>(&utmp00)) - 7;
  tmp01 = *(reinterpret_cast<int8_t*>(&utmp01)) - 7;
  tmp10 = *(reinterpret_cast<int8_t*>(&utmp10)) - 7;
  tmp11 = *(reinterpret_cast<int8_t*>(&utmp11)) - 7;

  if (tmp00 <= 0) {
    tmp00 = 8 - tmp00;
  }
  if (tmp01 <= 0) {
    tmp01 = 8 - tmp01;
  }
  if (tmp10 <= 0) {
    tmp10 = 8 - tmp10;
  }
  if (tmp11 <= 0) {
    tmp11 = 8 - tmp11;
  }

  int8_t dst0 = (tmp01 << 4) | tmp11;
  int8_t dst1 = (tmp00 << 4) | tmp10;

  int8_t* dst = output_data + n_id * original_k / 2 +
                (k_group_idx * pack_group_size / 2) + k_idx_in_group * 2;
  dst[0] = dst0;
  dst[1] = dst1;
}

std::vector<paddle::Tensor> W4AFp8GemmWeightConvert(
    const paddle::Tensor& weight) {
  if (weight.place() == paddle::CPUPlace()) {
    const int experts = weight.dims()[0];
    const int M = weight.dims()[1];
    const int K = weight.dims()[2];
    paddle::Tensor weight_new = paddle::empty(
        {experts, M, K / 2}, paddle::DataType::UINT8, weight.place());
    weight_convert(
        weight.data<uint8_t>(), weight_new.data<uint8_t>(), experts, M, K);
    return {weight_new};
  } else {
    const int original_k = weight.dims()[0] * 2;
    const int original_n = weight.dims()[1];
    paddle::Tensor weight_new =
        paddle::empty(weight.shape(), paddle::DataType::INT8, weight.place());
    const int block_dim = 256;
    const int original_numel = original_k * original_n;
    const int grid_size = (original_numel + block_dim - 1) / block_dim;

    weight_permute_interleave_kernelw4afp8<<<grid_size, block_dim>>>(
        weight.data<int8_t>(),
        weight_new.data<int8_t>(),
        original_k,
        original_n);
    return {weight_new};
  }
}
