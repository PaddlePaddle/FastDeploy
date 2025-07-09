
/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "fused_moe_op.h"

namespace phi {

template <typename T, int VecSize>
__global__ void moe_token_type_ids_kernel(T *gating_output,
                                          const int *moe_token_type_ids_out,
                                          const int num_rows,
                                          const int num_experts,
                                          const int k) {
  const int moe_token_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (moe_token_index >= num_rows) {
    return;
  }

  gating_output[moe_token_index * 2] =
      gating_output[moe_token_index * 2] +
      (moe_token_type_ids_out[moe_token_index]) * -1e10;
  gating_output[moe_token_index * 2 + 1] =
      gating_output[moe_token_index * 2 + 1] +
      (1 - moe_token_type_ids_out[moe_token_index]) * -1e10;
}

template <typename T>
void moe_token_type_ids_kernelLauncher(T *gating_output,
                                       const int *moe_token_type_ids_out,
                                       const int num_rows,
                                       const int num_experts,
                                       const int k,
                                       cudaStream_t stream) {
  const int blocks = num_rows * k / 512 + 1;
  const int threads = 512;
  moe_token_type_ids_kernel<T, 1><<<blocks, 512, 0, stream>>>(
      gating_output, moe_token_type_ids_out, num_rows, num_experts, k);
}

}  // namespace phi
