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

__host__ __device__ __forceinline__ int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

__host__ __device__ __forceinline__ int align(int x, int y) {
  return ceil_div(x, y) * y;
}

__host__ __device__ __forceinline__ int64_t align(int64_t x, int64_t y) {
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
    int64_t group_num,
    int64_t group_size,
    int64_t hidden_size,
    int64_t hidden_size_scale,
    bool use_finegrained_range) {
  constexpr int BLOCK = 128;
  constexpr int VEC_SIZE = 8;  // 8 × bf16 = 16 bytes = 128-bit load

  int tid = threadIdx.x;
  int lane = tid & 31;
  int warp = tid >> 5;
  int num_warps = blockDim.x >> 5;

  // Build prefix-sum + per-expert token offset lookup in shared memory.
  // Layout: [0..group_num] = cumsum, [group_num+1..2*group_num] =
  // expert_of[token_range]
  extern __shared__ int smem[];
  int* smem_cumsum = smem;
  // Build a flat lookup table: for each cumsum bucket, store the expert index.
  // Since group_num is small (typically 20-64), this is very compact.
  int* smem_expert_lut = smem + group_num + 1;

  if (tid == 0) {
    smem_cumsum[0] = 0;
    for (int i = 0; i < group_num; ++i) {
      smem_cumsum[i + 1] =
          smem_cumsum[i] + static_cast<int>(token_nums_per_expert[i]);
    }
  }
  __syncthreads();

  int total_tokens = smem_cumsum[group_num];

  using VecBF16 = AlignedVector<T, VEC_SIZE>;
  VecBF16 x1_vec, x2_vec;
  using VecFP8 = AlignedVector<phi::dtype::float8_e4m3fn, VEC_SIZE>;
  VecFP8 q_vec;

  // Pre-compute scale constants outside loop
  const float inv_fp8_max = 1.f / kFP8Max;

  // Each warp tracks its current expert to avoid repeated binary search.
  // When block_id moves to the next token, we check if it's still in the
  // same expert range (which is the common case for sequential iteration).
  int cached_expert = -1;
  int cached_cumsum_lo = 0;
  int cached_cumsum_hi = 0;

  for (int64_t block_id = static_cast<int64_t>(blockIdx.x);
       block_id < total_tokens;
       block_id += gridDim.x) {
    // ================= token mapping with cached expert =============
    int64_t expert, token_in_expert;
    if (lane == 0) {
      int bid = static_cast<int>(block_id);
      // Fast path: check if still in same expert range
      if (bid >= cached_cumsum_lo && bid < cached_cumsum_hi) {
        expert = cached_expert;
        token_in_expert = bid - cached_cumsum_lo;
      } else {
        // Binary search fallback
        int lo = 0, hi = static_cast<int>(group_num) + 1;
        while (lo < hi) {
          int mid = (lo + hi) >> 1;
          if (smem_cumsum[mid] <= bid)
            lo = mid + 1;
          else
            hi = mid;
        }
        expert = static_cast<int64_t>(lo - 1);
        token_in_expert = bid - static_cast<int64_t>(smem_cumsum[lo - 1]);
        // Cache for next iteration
        cached_expert = static_cast<int>(expert);
        cached_cumsum_lo = smem_cumsum[lo - 1];
        cached_cumsum_hi = smem_cumsum[lo];  // lo is already the upper bound
      }
    }
    expert = __shfl_sync(0xffffffff, expert, 0);
    token_in_expert = __shfl_sync(0xffffffff, token_in_expert, 0);

    // Also broadcast cache values so all lanes in the warp have them
    // (only lane 0 updates, but we need consistency for the next iteration
    // check)
    cached_expert = __shfl_sync(0xffffffff, cached_expert, 0);
    cached_cumsum_lo = __shfl_sync(0xffffffff, cached_cumsum_lo, 0);
    cached_cumsum_hi = __shfl_sync(0xffffffff, cached_cumsum_hi, 0);

    // ================= base pointers =================
    int64_t token = expert * group_size + token_in_expert;

    const T* in = input + token * hidden_size * 2;

    auto* out = out_fp8 + token * hidden_size;

    // With VEC_SIZE=8, each lane processes 8 elements, 32 lanes process 256
    // elements. We need to process BLOCK=128 elements per scale group. Each
    // warp iteration: 32 lanes × 8 elements = 256 elements = 2 scale groups. So
    // we process 2 scale groups per warp iteration.
    int64_t num_iters = hidden_size / BLOCK;

    // ================= main loop =================
    // Process 2 scale groups (2 × 128 = 256 elements) per warp iteration
    for (int64_t iter_pair = warp; iter_pair < num_iters / 2;
         iter_pair += num_warps) {
      int64_t iter0 = iter_pair * 2;
      int64_t base = iter0 * BLOCK + lane * VEC_SIZE;

      // 128-bit vectorized load: 8 × bf16 = 16 bytes
      Load(in + base, &x1_vec);
      Load(in + base + hidden_size, &x2_vec);

      float v[VEC_SIZE];
      float amax0 = 0.f;
      float amax1 = 0.f;

#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        float x1 = static_cast<float>(x1_vec[i]);
        float x2 = static_cast<float>(x2_vec[i]);

        // SwiGLU: x2 * silu(x1) = x2 * x1 / (1 + exp(-x1))
        float y = x2 * x1 / (1.f + expf(-x1));
        float y_r = static_cast<float>(
            static_cast<T>(y));  // bf16 round-trip to match reference
        v[i] = y_r;
        // Split amax for two scale groups:
        // Elements 0..3 belong to scale group iter0, elements 4..7 to iter0+1
        if (i < 4) {
          amax0 = fmaxf(amax0, fabsf(y_r));
        } else {
          amax1 = fmaxf(amax1, fabsf(y_r));
        }
      }

      // ---------- warp reduce amax for group 0 (lanes 0-15 contribute lower
      // half) ---------- All lanes have amax0 from elements [0..3], but we need
      // to split by 128-element boundary. lane * 8 + [0..3] maps to elements in
      // range [lane*8 .. lane*8+3] A 128-element group covers lanes where
      // (lane*8)/128 is the same. 128/8 = 16 lanes per group. So lanes 0-15 →
      // group iter0, lanes 16-31 → group iter0+1. Merge: lanes 0-15 have group0
      // in amax0 and group1 doesn't exist for them (amax1 from elements 4-7 =
      // group0 still since 16*8=128 > lane*8+7 for lane<16... wait, let me
      // reconsider).

      // Actually: lane L processes elements at offset base + [0..7] = iter0*128
      // + L*8 + [0..7] For L in [0..15]: offsets are in [iter0*128 .. iter0*128
      // + 127] → scale group iter0 For L in [16..31]: offsets are in [iter0*128
      // + 128 .. iter0*128 + 255] → scale group iter0+1 So: lanes 0-15 all
      // contribute to amax of group iter0, lanes 16-31 to group iter0+1.

      // Combine amax0 and amax1 per lane (both belong to same group for that
      // lane)
      float my_amax = fmaxf(amax0, amax1);

      // Half-warp reduce for each group
      // Lanes 0-15 reduce among themselves, lanes 16-31 reduce among themselves
#pragma unroll
      for (int offset = 8; offset > 0; offset >>= 1)
        my_amax = fmaxf(my_amax, __shfl_xor_sync(0xffffffff, my_amax, offset));

      // Now lane 0 has amax for group iter0, lane 16 has amax for group iter0+1
      float group0_amax = __shfl_sync(0xffffffff, my_amax, 0);
      float group1_amax = __shfl_sync(0xffffffff, my_amax, 16);

      // Select the correct amax for this lane's group
      float amax = (lane < 16) ? group0_amax : group1_amax;
      amax = fmaxf(amax, kEpsilon);

      if (use_finegrained_range) amax *= 7.f;

      float scale = amax * inv_fp8_max;
      int64_t my_iter = iter0 + (lane >= 16 ? 1 : 0);

      // ---------- quantize ----------
      if constexpr (UseUE8M0) {
        float ue8m0_scale = exp2f(ceilf(log2f(fmaxf(scale, kEpsilon))));
        float inv_scale = __frcp_rn(ue8m0_scale);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          q_vec[i] = static_cast<phi::dtype::float8_e4m3fn>(v[i] * inv_scale);
        }
        // ---------- store scale (lane 0 writes both groups to avoid race)
        // ----------
        if (lane == 0) {
          const int64_t pack_num = ceil_div(hidden_size_scale, (int64_t)4);
          const int64_t token_stride = align(group_size, (int64_t)4);
          auto* scale_pack = reinterpret_cast<int32_t*>(out_scale);

          // Group 0 scale (from lane 0's own value)
          float s0 =
              exp2f(ceilf(log2f(fmaxf(group0_amax * inv_fp8_max, kEpsilon))));
          const int exp0 = (__float_as_int(s0) >> 23) & 0xFF;
          const int64_t pack_idx0 = iter0 >> 2;
          const int64_t byte_idx0 = iter0 & 3;
          const int64_t base_idx0 = expert * pack_num * token_stride +
                                    pack_idx0 * token_stride + token_in_expert;
          reinterpret_cast<uint8_t*>(&scale_pack[base_idx0])[byte_idx0] =
              static_cast<uint8_t>(exp0);

          // Group 1 scale (from lane 16's value, broadcast earlier)
          int64_t iter1 = iter0 + 1;
          float s1 =
              exp2f(ceilf(log2f(fmaxf(group1_amax * inv_fp8_max, kEpsilon))));
          const int exp1 = (__float_as_int(s1) >> 23) & 0xFF;
          const int64_t pack_idx1 = iter1 >> 2;
          const int64_t byte_idx1 = iter1 & 3;
          const int64_t base_idx1 = expert * pack_num * token_stride +
                                    pack_idx1 * token_stride + token_in_expert;
          reinterpret_cast<uint8_t*>(&scale_pack[base_idx1])[byte_idx1] =
              static_cast<uint8_t>(exp1);
        }
      } else {
        float inv_amax = __frcp_rn(amax);
#pragma unroll
        for (int i = 0; i < VEC_SIZE; i++) {
          float q = v[i] * kFP8Max * inv_amax;
          q_vec[i] = static_cast<phi::dtype::float8_e4m3fn>(q);
        }
        // ---------- store scale ----------
        if (lane == 0 || lane == 16) {
          out_scale[expert * hidden_size_scale * group_size +
                    my_iter * group_size + token_in_expert] = scale;
        }
      }

      Store(q_vec, out + base);
    }

    // Handle remainder if num_iters is odd
    if (num_iters & 1) {
      int64_t iter = num_iters - 1;
      // Only the last warp handles this
      if (warp == (num_iters / 2) % num_warps ||
          num_iters / 2 < static_cast<int64_t>(num_warps)) {
        // Fall back to vec4 for the remainder
        using VecBF16_4 = AlignedVector<T, 4>;
        using VecFP8_4 = AlignedVector<phi::dtype::float8_e4m3fn, 4>;
        VecBF16_4 rx1, rx2;
        VecFP8_4 rq;

        int64_t rbase = iter * BLOCK + lane * 4;
        if (rbase < hidden_size) {
          Load(in + rbase, &rx1);
          Load(in + rbase + hidden_size, &rx2);

          float rv[4];
          float ramax = 0.f;
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            float x1 = static_cast<float>(rx1[i]);
            float x2 = static_cast<float>(rx2[i]);
            float y = x2 * x1 / (1.f + expf(-x1));
            float y_r = static_cast<float>(
                static_cast<T>(y));  // bf16 round-trip to match reference
            rv[i] = y_r;
            ramax = fmaxf(ramax, fabsf(y_r));
          }

#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1)
            ramax = fmaxf(ramax, __shfl_down_sync(0xffffffff, ramax, offset));
          ramax = __shfl_sync(0xffffffff, ramax, 0);
          ramax = fmaxf(ramax, kEpsilon);

          if (use_finegrained_range) ramax *= 7.f;
          float rscale = ramax * inv_fp8_max;

          if constexpr (UseUE8M0) {
            float s = exp2f(ceilf(log2f(fmaxf(rscale, kEpsilon))));
            float inv_s = __frcp_rn(s);
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              rq[i] = static_cast<phi::dtype::float8_e4m3fn>(rv[i] * inv_s);
            }
            if (lane == 0) {
              const int exp = (__float_as_int(s) >> 23) & 0xFF;
              const int64_t pack_idx = iter >> 2;
              const int64_t byte_idx = iter & 3;
              const int64_t pack_num = ceil_div(hidden_size_scale, (int64_t)4);
              const int64_t token_stride = align(group_size, (int64_t)4);
              auto* scale_pack = reinterpret_cast<int32_t*>(out_scale);
              const int64_t base_idx = expert * pack_num * token_stride +
                                       pack_idx * token_stride +
                                       token_in_expert;
              reinterpret_cast<uint8_t*>(&scale_pack[base_idx])[byte_idx] =
                  static_cast<uint8_t>(exp);
            }
          } else {
            float inv_ramax = __frcp_rn(ramax);
#pragma unroll
            for (int i = 0; i < 4; i++) {
              rq[i] = static_cast<phi::dtype::float8_e4m3fn>(rv[i] * kFP8Max *
                                                             inv_ramax);
            }
            if (lane == 0) {
              out_scale[expert * hidden_size_scale * group_size +
                        iter * group_size + token_in_expert] = rscale;
            }
          }
          Store(rq, out + rbase);
        }
      }
    }
  }
}

std::vector<paddle::Tensor> FusedMaskSwigluFP8Quant(
    paddle::Tensor& input,
    paddle::Tensor& token_nums_per_expert,
    const int block_size,
    const bool use_ue8m0) {
  auto dim = input.dims();
  const int64_t group_num = token_nums_per_expert.shape()[0];
  const int64_t group_size = dim[1];
  const int64_t hidden_size = dim[2] / 2;
  const int64_t hidden_size_scale = hidden_size / block_size;
  const int64_t token_num = group_num * group_size;

  auto out_fp8 = GetEmptyTensor({group_num, group_size, hidden_size},
                                paddle::DataType::FLOAT8_E4M3FN,
                                input.place());

  auto out_scale =
      GetEmptyTensor({group_num, group_size, hidden_size_scale},
                     {hidden_size_scale * group_size, 1, group_size},
                     paddle::DataType::FLOAT32,
                     input.place());
  if (use_ue8m0) {
    int64_t hidden_size_scale_pack = ceil_div(hidden_size_scale, (int64_t)4);
    int64_t group_size_aligned = align(group_size, (int64_t)4);
    out_scale = GetEmptyTensor(
        {group_num, group_size, hidden_size_scale_pack},
        {hidden_size_scale_pack * group_size_aligned, 1, group_size_aligned},
        paddle::DataType::INT32,
        input.place());
  }

  int sm_count = 0;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0);

  constexpr int BLOCKS_PER_SM = 3;
  int blockx = std::min(512L, hidden_size / 128 * 32);
  int gridx =
      std::min(static_cast<int64_t>(sm_count * BLOCKS_PER_SM), token_num);
  int smem_bytes = (group_num + 1) * sizeof(int);

  bool use_finegrained_range = false;
  if (auto* env = getenv("PER_TOKEN_QUANT_FP8_USE_FINEGRAINED_RANGE"))
    use_finegrained_range = static_cast<bool>(std::stoi(env));

  if (input.dtype() == paddle::DataType::BFLOAT16) {
    BOOL_SWITCH(use_ue8m0, UseUE8M0, [&] {
      using ScaleT = std::conditional_t<UseUE8M0, int, float>;
      fused_swiglu_fp8_quant_kernel<paddle::bfloat16, int, ScaleT, UseUE8M0>
          <<<gridx, blockx, smem_bytes, input.stream()>>>(
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
