// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

constexpr float epsilon = 1e-10;

template <typename T>
__global__ void masked_quant_per_token_per_block(
    const T* __restrict__ input,
    const int* __restrict__ recv_expert_count,
    phi::dtype::float8_e4m3fn* __restrict__ quanted_res,
    float* __restrict__ quanted_scale,
    const int token_num,
    const int hidden_size,
    const int hidden_size_scale,
    const int num_max_tokens_per_expert,
    const bool use_finegrained_range) {
  constexpr int BLOCK = 128;
  constexpr float FP8_MAX = 448.f;

  int bid = blockIdx.x;
  int tid = threadIdx.x;
  int warp = tid / 32;
  int lane = tid % 32;

  int num_warps = blockDim.x / 32;
  int num_iters = hidden_size / BLOCK;

  for (int token = bid; token < token_num; token += gridDim.x) {
    int token_in_expert = token % num_max_tokens_per_expert;
    int expert = token / num_max_tokens_per_expert;

    if (token_in_expert >= recv_expert_count[expert]) continue;

    const T* in = input + token * hidden_size;
    auto* out = quanted_res + token * hidden_size;

    for (int iter = warp; iter < num_iters; iter += num_warps) {
      int base = iter * BLOCK + lane * 4;
      float v[4];

#pragma unroll
      for (int i = 0; i < 4; ++i) v[i] = static_cast<float>(in[base + i]);

      // ---------------- amax reduction ----------------
      float amax = fabsf(v[0]);
#pragma unroll
      for (int i = 1; i < 4; ++i) amax = fmaxf(amax, fabsf(v[i]));

#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));

      amax = __shfl_sync(0xffffffff, amax, 0);
      amax = fmaxf(amax, epsilon);

      if (use_finegrained_range) amax *= 7.f;

      float scale = amax / FP8_MAX;

      // ---------------- quantize ----------------
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float q = v[i] * FP8_MAX / amax;
        q = fminf(fmaxf(q, -FP8_MAX), FP8_MAX);
        out[base + i] = static_cast<phi::dtype::float8_e4m3fn>(q);
      }

      // ---------------- store scale ----------------
      if (lane == 0) {
        quanted_scale[expert * hidden_size_scale * num_max_tokens_per_expert +
                      iter * num_max_tokens_per_expert + token_in_expert] =
            scale;
      }
    }
  }
}

std::vector<paddle::Tensor> MaskedPerTokenQuant(
    paddle::Tensor& input,
    paddle::Tensor& recv_expert_count,
    const int block_size) {
  auto input_dim = input.dims();
  const int num_local_expert = input_dim[0];
  const int num_max_tokens_per_expert = input_dim[1];
  const int hidden_size = input_dim[2];
  const int hidden_size_scale = hidden_size / block_size;
  const int token_num = num_local_expert * num_max_tokens_per_expert;
  auto quanted_x =
      GetEmptyTensor({num_local_expert, num_max_tokens_per_expert, hidden_size},
                     paddle::DataType::FLOAT8_E4M3FN,
                     input.place());
  auto quanted_scale = GetEmptyTensor(
      {num_local_expert, num_max_tokens_per_expert, hidden_size_scale},
      {hidden_size_scale * num_max_tokens_per_expert,
       1,
       num_max_tokens_per_expert},
      paddle::DataType::FLOAT32,
      input.place());
  // const int gridx = min(132 * 2, token_num);
  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

  constexpr int BLOCKS_PER_SM = 2;

  int gridx = std::min(sm_count * BLOCKS_PER_SM, token_num);
  const int blockx = min(1024, hidden_size / 128 * 32);

  bool use_finegrained_range = false;
  char* env_var = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE");
  if (env_var) {
    use_finegrained_range = static_cast<bool>(std::stoi(env_var));
  }

  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      masked_quant_per_token_per_block<<<gridx, blockx, 0, input.stream()>>>(
          input.data<paddle::bfloat16>(),
          recv_expert_count.data<int>(),
          quanted_x.data<phi::dtype::float8_e4m3fn>(),
          quanted_scale.data<float>(),
          token_num,
          hidden_size,
          hidden_size_scale,
          num_max_tokens_per_expert,
          use_finegrained_range);
      break;
    case paddle::DataType::FLOAT16:
      masked_quant_per_token_per_block<<<gridx, blockx, 0, input.stream()>>>(
          input.data<paddle::float16>(),
          recv_expert_count.data<int>(),
          quanted_x.data<phi::dtype::float8_e4m3fn>(),
          quanted_scale.data<float>(),
          token_num,
          hidden_size,
          hidden_size_scale,
          num_max_tokens_per_expert,
          use_finegrained_range);
      break;
    default:
      PD_THROW("Unsupported data type for PerTokenQuant");
  }
  return {quanted_x, quanted_scale};
}

PD_BUILD_STATIC_OP(masked_per_token_quant)
    .Inputs({"input", "recv_expert_count"})
    .Outputs({"output", "output_scale"})
    .Attrs({"block_size: int"})
    .SetKernelFn(PD_KERNEL(MaskedPerTokenQuant));
