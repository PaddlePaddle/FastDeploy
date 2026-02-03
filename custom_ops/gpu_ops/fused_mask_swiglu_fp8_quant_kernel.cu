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

constexpr float kEpsilon = 1e-10;
constexpr float kFP8Max = 448.f;
__host__ __device__ __forceinline__ int ceil_div(int x, int y) {
  return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int align(int x, int y) {
  return ceil_div(x, y) * y;
}

#ifndef BOOL_SWITCH
#define BOOL_SWITCH(cond, name, ...) \
  if (cond) {                        \
    constexpr bool name = true;      \
    __VA_ARGS__();                   \
  } else {                           \
    constexpr bool name = false;     \
    __VA_ARGS__();                   \
  }
#endif

template <typename T, typename index_t, typename ScaleT, bool UseUE8M0>
__global__ void fused_swiglu_fp8_quant_kernel(
    const T* __restrict__ input,  // [group, max_tokens, hidden*2]
    const index_t* __restrict__ token_nums_per_expert,
    phi::dtype::float8_e4m3fn* __restrict__ out_fp8,
    ScaleT* __restrict__ out_scale,
    int group_num,
    int group_size,
    int hidden_size,
    int hidden_size_scale,
    bool use_finegrained_range) {
  constexpr int BLOCK = 128;

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int num_warps = blockDim.x >> 5;

  int block_id = static_cast<int64_t>(blockIdx.x);

  using VecBF16 = AlignedVector<T, 4>;
  VecBF16 x1_vec, x2_vec;
  using VecFP8 = AlignedVector<phi::dtype::float8_e4m3fn, 4>;
  VecFP8 q_vec;

  while (true) {
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

    if (expert < 0 || token_in_expert >= group_size) break;

    // ================= base pointers =================
    int token = expert * group_size + token_in_expert;

    const T* in = input + token * hidden_size * 2;

    auto* out = out_fp8 + token * hidden_size;

    int num_iters = hidden_size / BLOCK;

    // ================= main loop =================
    for (int iter = warp; iter < num_iters; iter += num_warps) {
      int base = iter * BLOCK + lane * 4;

      // vec load
      Load(in + base, &x1_vec);
      Load(in + base + hidden_size, &x2_vec);

      float v[4];
      float amax = -5e4;

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        float x1 = static_cast<float>(x1_vec[i]);
        float x2 = static_cast<float>(x2_vec[i]);

        float y = x2 * x1 / (1.f + expf(-x1));
        float y_r = static_cast<float>(
            static_cast<T>(y));  // To simulate the data transformation before
                                 // the fusion of swiglu and quant operators
        v[i] = y_r;
        amax = max(amax, abs(y_r));
      }

      // ---------- warp reduce amax ----------
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        amax = max(amax, __shfl_down_sync(0xffffffff, amax, offset));

      amax = __shfl_sync(0xffffffff, amax, 0);
      amax = max(amax, kEpsilon);

      if (use_finegrained_range) amax *= 7.f;

      float scale = amax / kFP8Max;
      // ---------- quantize ----------
      if constexpr (UseUE8M0) {
        scale = exp2f(ceilf(log2f(fmaxf(scale, kEpsilon))));
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          float q = v[i] / scale;
          q_vec[i] = static_cast<phi::dtype::float8_e4m3fn>(q);
        }
        // ---------- store scale ----------
        if (lane == 0) {
          // 1. extract exponent
          const int exp = (__float_as_int(scale) >> 23) & 0xFF;

          // 2. pack information
          const int pack_idx = iter >> 2;  // iter / 4
          const int byte_idx = iter & 3;   // iter % 4

          // 3. layout parameters
          const int pack_num = ceil_div(hidden_size_scale, 4);
          const int token_stride = align(group_size, 4);

          // 4. base pointer (int32 pack)
          auto* scale_pack = reinterpret_cast<int32_t*>(out_scale);

          // 5. column-major offset:
          //    [expert][pack][token]
          const int base_idx = expert * pack_num * token_stride +
                               pack_idx * token_stride + token_in_expert;
          // 6. write one byte into pack
          reinterpret_cast<uint8_t*>(&scale_pack[base_idx])[byte_idx] =
              static_cast<uint8_t>(exp);
        }
      } else {
#pragma unroll
        for (int i = 0; i < 4; i++) {
          float q = v[i] * kFP8Max / amax;
          q_vec[i] = static_cast<phi::dtype::float8_e4m3fn>(q);
        }
        // ---------- store scale ----------
        if (lane == 0) {
          out_scale[expert * hidden_size_scale * group_size +
                    iter * group_size + token_in_expert] = scale;
        }
      }

      Store(q_vec, out + base);
    }
    block_id += gridDim.x;
  }
}

std::vector<paddle::Tensor> FusedMaskSwigluFP8Quant(
    paddle::Tensor& input,
    paddle::Tensor& token_nums_per_expert,
    const int block_size,
    const bool use_ue8m0) {
  auto dim = input.dims();
  const int group_num = token_nums_per_expert.shape()[0];
  const int group_size = dim[1];
  const int hidden_size = dim[2] / 2;
  const int hidden_size_scale = hidden_size / block_size;
  const int token_num = group_num * group_size;

  auto out_fp8 = GetEmptyTensor({group_num, group_size, hidden_size},
                                paddle::DataType::FLOAT8_E4M3FN,
                                input.place());

  auto out_scale =
      GetEmptyTensor({group_num, group_size, hidden_size_scale},
                     {hidden_size_scale * group_size, 1, group_size},
                     paddle::DataType::FLOAT32,
                     input.place());
  if (use_ue8m0) {
    int hidden_size_scale_pack = ceil_div(hidden_size_scale, 4);
    out_scale = GetEmptyTensor({group_num, group_size, hidden_size_scale_pack},
                               {hidden_size_scale_pack * align(group_size, 4),
                                1,
                                align(group_size, 4)},
                               paddle::DataType::INT32,
                               input.place());
  }

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

  constexpr int BLOCKS_PER_SM = 2;
  int gridx = std::min(sm_count * BLOCKS_PER_SM, token_num);
  int blockx = std::min(1024, hidden_size / 128 * 32);

  bool use_finegrained_range = false;
  if (auto* env = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE"))
    use_finegrained_range = static_cast<bool>(std::stoi(env));

  if (input.dtype() == paddle::DataType::BFLOAT16) {
    BOOL_SWITCH(use_ue8m0, UseUE8M0, [&] {
      using ScaleT = std::conditional_t<UseUE8M0, int, float>;
      fused_swiglu_fp8_quant_kernel<paddle::bfloat16, int, ScaleT, UseUE8M0>
          <<<gridx, blockx, 0, input.stream()>>>(
              input.data<paddle::bfloat16>(),
              token_nums_per_expert.data<int>(),
              out_fp8.data<phi::dtype::float8_e4m3fn>(),
              out_scale.data<ScaleT>(),
              group_num,
              group_size,
              hidden_size,
              hidden_size_scale,
              use_finegrained_range);
    });
  } else {
    PD_THROW("Only BF16 supported");
  }
  return {out_fp8, out_scale};
}

PD_BUILD_STATIC_OP(fused_mask_swiglu_fp8_quant)
    .Inputs({"input", "token_nums_per_expert"})
    .Outputs({"out_fp8", "output_scale"})
    .Attrs({"block_size: int", "use_ue8m0: bool"})
    .SetKernelFn(PD_KERNEL(FusedMaskSwigluFP8Quant));
