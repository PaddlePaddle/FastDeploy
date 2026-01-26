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

constexpr float kEpsilon = 1e-10f;
constexpr float kFP8Max = 448.f;

__device__ __forceinline__ float fp32_to_bf16_to_fp32(float x) {
  uint32_t bits = reinterpret_cast<uint32_t&>(x);
  bits += 0x00008000;  // round
  bits &= 0xFFFF0000;  // truncate
  return reinterpret_cast<float&>(bits);
}

template <typename T, typename index_t>
__global__ void fused_swiglu_fp8_quant_kernel(
    const T* __restrict__ input,  // [group, max_tokens, hidden*2]
    const index_t* __restrict__ token_nums_per_expert,
    phi::dtype::float8_e4m3fn* __restrict__ out_fp8,
    float* __restrict__ out_scale,
    int group_num,
    int group_size,
    int hidden_size,
    int hidden_size_scale,
    int num_max_tokens_per_expert,
    bool use_finegrained_range) {
  constexpr int BLOCK = 128;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int num_warps = blockDim.x >> 5;

  int block_id = blockIdx.x;

  // ================= token mapping =================
  int expert = -1;
  int token_in_expert = -1;

  if (lane == 0) {
    int cumsum = 0;
    for (int i = 0; i < group_num; ++i) {
      int cnt = token_nums_per_expert[i];
      if (block_id >= cumsum && block_id < cumsum + cnt) {
        expert = i;
        token_in_expert = block_id - cumsum;
        break;
      }
      cumsum += cnt;
    }
  }

  expert = __shfl_sync(0xffffffff, expert, 0);
  token_in_expert = __shfl_sync(0xffffffff, token_in_expert, 0);

  if (expert < 0 || token_in_expert >= group_size) return;

  // ================= base pointers =================
  int token = expert * num_max_tokens_per_expert + token_in_expert;

  const T* in =
      input + (expert * group_size + token_in_expert) * hidden_size * 2;

  auto* out = out_fp8 + token * hidden_size;

  int num_iters = hidden_size / BLOCK;

  // ================= main loop =================
  for (int iter = warp; iter < num_iters; iter += num_warps) {
    int base = iter * BLOCK + lane * 4;

    float v[4];
    float amax = 0.f;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float x1 = static_cast<float>(in[base + i]);
      float x2 = static_cast<float>(in[base + i + hidden_size]);

      float y = x2 * x1 / (1.f + expf(-x1));
      float y_r = fp32_to_bf16_to_fp32(
          y);  // To simulate the data transformation before the fusion of
               // swiglu and quant operators
      v[i] = y_r;
      amax = fmaxf(amax, fabsf(y_r));
    }

    // ---------- warp reduce amax ----------
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      amax = fmaxf(amax, __shfl_down_sync(0xffffffff, amax, offset));

    amax = __shfl_sync(0xffffffff, amax, 0);
    amax = fmaxf(amax, kEpsilon);

    if (use_finegrained_range) amax *= 7.f;

    float scale = amax / kFP8Max;

    // ---------- quantize ----------
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float q = v[i] * kFP8Max / amax;
      q = fminf(fmaxf(q, -kFP8Max), kFP8Max);
      out[base + i] = static_cast<phi::dtype::float8_e4m3fn>(q);
    }

    // ---------- store scale ----------
    if (lane == 0) {
      out_scale[expert * hidden_size_scale * num_max_tokens_per_expert +
                iter * num_max_tokens_per_expert + token_in_expert] = scale;
    }
  }
}

std::vector<paddle::Tensor> FusedMaskSwigluFP8Quant(
    paddle::Tensor& input,
    paddle::Tensor& token_nums_per_expert,
    const int block_size) {
  auto dim = input.dims();
  const int group_num = token_nums_per_expert.shape()[0];
  const int group_size = dim[1];
  const int hidden_size = dim[2] / 2;
  const int hidden_size_scale = hidden_size / block_size;
  const int num_max_tokens_per_expert = group_size;
  const int token_num = group_num * num_max_tokens_per_expert;

  auto out_fp8 =
      GetEmptyTensor({group_num, num_max_tokens_per_expert, hidden_size},
                     paddle::DataType::FLOAT8_E4M3FN,
                     input.place());

  auto out_scale =
      GetEmptyTensor({group_num, num_max_tokens_per_expert, hidden_size_scale},
                     {hidden_size_scale * num_max_tokens_per_expert,
                      1,
                      num_max_tokens_per_expert},
                     paddle::DataType::FLOAT32,
                     input.place());

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

  constexpr int BLOCKS_PER_SM = 2;
  int gridx = std::min(sm_count * BLOCKS_PER_SM, token_num);
  int blockx = std::min(1024, hidden_size / 128 * 32);

  bool use_finegrained_range = false;
  if (auto* env = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE"))
    use_finegrained_range = static_cast<bool>(std::stoi(env));

  if (input.dtype() == paddle::DataType::BFLOAT16) {
    fused_swiglu_fp8_quant_kernel<paddle::bfloat16, int>
        <<<gridx, blockx, 0, input.stream()>>>(
            input.data<paddle::bfloat16>(),
            token_nums_per_expert.data<int>(),
            out_fp8.data<phi::dtype::float8_e4m3fn>(),
            out_scale.data<float>(),
            group_num,
            group_size,
            hidden_size,
            hidden_size_scale,
            num_max_tokens_per_expert,
            use_finegrained_range);
  } else {
    PD_THROW("Only BF16 supported");
  }

  return {out_fp8, out_scale};
}

PD_BUILD_STATIC_OP(fused_mask_swiglu_fp8_quant)
    .Inputs({"input", "token_nums_per_expert"})
    .Outputs({"out_fp8", "output_scale"})
    .Attrs({"block_size: int"})
    .SetKernelFn(PD_KERNEL(FusedMaskSwigluFP8Quant));
