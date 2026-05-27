// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"
#include "helper.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace {

constexpr float kFP8E4M3Max = 448.0f;
constexpr uint32_t kVecElems = 8;

template <uint32_t kNumThreads>
__device__ __forceinline__ float WarpReduceMax(float value) {
  static_assert(kNumThreads >= 1 && kNumThreads <= WARP_SIZE,
                "kNumThreads must be in [1, 32]");
  static_assert((kNumThreads & (kNumThreads - 1)) == 0,
                "kNumThreads must be a power of 2");
#pragma unroll
  for (int mask = kNumThreads / 2; mask > 0; mask >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask, WARP_SIZE));
  }
  return value;
}

__device__ __forceinline__ uint32_t CastToUE8M0(float value) {
  value = fabsf(value);
  uint32_t bits = __float_as_uint(value);
  uint32_t exp = (bits >> 23) & 0xffu;
  const uint32_t mantissa = bits & 0x7fffffu;
  exp += mantissa != 0;
  exp = min(max(exp, 1u), 254u);
  return exp;
}

struct MegaMoEPreDispatchParams {
  const __nv_bfloat16* __restrict__ x;
  const int64_t* __restrict__ topk_idx;
  const float* __restrict__ topk_weights;

  phi::dtype::float8_e4m3fn* __restrict__ buf_x;
  int32_t* __restrict__ buf_x_sf;
  int64_t* __restrict__ buf_topk_idx;
  float* __restrict__ buf_topk_weights;

  uint32_t num_tokens;
  uint32_t padded_max;
  uint32_t hidden;
  uint32_t num_groups;
  uint32_t top_k;
};

template <uint32_t kGroupSize>
__global__ __launch_bounds__(1024, 2) void MegaMoEPreDispatchKernel(
    const MegaMoEPreDispatchParams params) {
  static_assert(kGroupSize == 32 || kGroupSize == 64 || kGroupSize == 128,
                "unsupported group_size");
  static_assert(kGroupSize % kVecElems == 0,
                "group_size must be a multiple of 8");
  constexpr uint32_t kThreadsPerGroup = kGroupSize / kVecElems;

  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;

  if (bid < params.num_tokens) {
    const uint32_t token_id = bid;
    const __nv_bfloat16* token_in =
        params.x + static_cast<uint64_t>(token_id) * params.hidden;
    phi::dtype::float8_e4m3fn* token_out =
        params.buf_x + static_cast<uint64_t>(token_id) * params.hidden;

    const uint32_t base = tid * kVecElems;
    float vals[kVecElems];
    float local_max = 0.0f;

#pragma unroll
    for (uint32_t i = 0; i < kVecElems; ++i) {
      const float v = __bfloat162float(token_in[base + i]);
      vals[i] = v;
      local_max = fmaxf(local_max, fabsf(v));
    }

    local_max = WarpReduceMax<kThreadsPerGroup>(local_max);

    const float absmax = fmaxf(local_max, 1e-10f);
    const float raw_scale = absmax / kFP8E4M3Max;
    const uint32_t ue8m0_exp = CastToUE8M0(raw_scale);
    const float inv_scale = __uint_as_float((127u + 127u - ue8m0_exp) << 23);

#pragma unroll
    for (uint32_t i = 0; i < kVecElems; ++i) {
      token_out[base + i] = phi::dtype::float8_e4m3fn(vals[i] * inv_scale);
    }

    const uint32_t group_id = tid / kThreadsPerGroup;
    const uint32_t within_group_id = tid % kThreadsPerGroup;
    if (within_group_id == 0 && group_id < params.num_groups) {
      const uint32_t byte_off = token_id * params.num_groups + group_id;
      reinterpret_cast<uint8_t*>(params.buf_x_sf)[byte_off] =
          static_cast<uint8_t>(ue8m0_exp);
    }

    if (tid < params.top_k) {
      const uint32_t off = token_id * params.top_k + tid;
      params.buf_topk_idx[off] = static_cast<int64_t>(params.topk_idx[off]);
      params.buf_topk_weights[off] = params.topk_weights[off];
    }
  }
}

void CheckShape2D(const paddle::Tensor& tensor, const char* name) {
  PD_CHECK(tensor.shape().size() == 2, name, " must be a 2D tensor");
}

void CheckSameShape(const paddle::Tensor& lhs,
                    const paddle::Tensor& rhs,
                    const char* lhs_name,
                    const char* rhs_name) {
  PD_CHECK(lhs.shape() == rhs.shape(), lhs_name, " shape must equal ", rhs_name,
           " shape");
}

template <uint32_t kGroupSize>
void LaunchMegaMoEPreDispatch(const MegaMoEPreDispatchParams& params,
                              uint32_t num_total_blocks,
                              uint32_t num_threads,
                              cudaStream_t stream) {
  MegaMoEPreDispatchKernel<kGroupSize>
      <<<num_total_blocks, num_threads, 0, stream>>>(params);
}

}  // namespace

void MegaMoePreDispatch(
    const paddle::Tensor& x,
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& buf_x,
    const paddle::Tensor& buf_x_sf,
    const paddle::Tensor& buf_topk_idx,
    const paddle::Tensor& buf_topk_weights,
    int64_t num_max_tokens_per_rank,
    int64_t group_size) {
  CheckShape2D(x, "x");
  CheckShape2D(topk_idx, "topk_idx");
  CheckShape2D(topk_weights, "topk_weights");
  CheckShape2D(buf_x, "buf_x");
  CheckShape2D(buf_x_sf, "buf_x_sf");
  CheckShape2D(buf_topk_idx, "buf_topk_idx");
  CheckShape2D(buf_topk_weights, "buf_topk_weights");
  CheckSameShape(topk_idx, topk_weights, "topk_idx", "topk_weights");
  CheckSameShape(buf_topk_idx, buf_topk_weights, "buf_topk_idx",
                 "buf_topk_weights");

  PD_CHECK(x.dtype() == paddle::DataType::BFLOAT16,
           "x must be bfloat16, but got ", x.dtype());
  PD_CHECK(topk_idx.dtype() == paddle::DataType::INT64,
           "topk_idx must be int64, but got ", topk_idx.dtype());
  PD_CHECK(topk_weights.dtype() == paddle::DataType::FLOAT32,
           "topk_weights must be float32, but got ", topk_weights.dtype());
  PD_CHECK(buf_x.dtype() == paddle::DataType::FLOAT8_E4M3FN,
           "buf_x must be float8_e4m3fn, but got ", buf_x.dtype());
  PD_CHECK(buf_x_sf.dtype() == paddle::DataType::INT32,
           "buf_x_sf must be int32, but got ", buf_x_sf.dtype());
  PD_CHECK(buf_topk_idx.dtype() == paddle::DataType::INT64,
           "buf_topk_idx must be int64, but got ", buf_topk_idx.dtype());
  PD_CHECK(buf_topk_weights.dtype() == paddle::DataType::FLOAT32,
           "buf_topk_weights must be float32, but got ",
           buf_topk_weights.dtype());

  const int64_t num_tokens_i64 = x.shape()[0];
  const int64_t hidden_i64 = x.shape()[1];
  const int64_t top_k_i64 = topk_idx.shape()[1];
  const int64_t padded_max_i64 = buf_x.shape()[0];

  PD_CHECK(num_max_tokens_per_rank <= padded_max_i64,
           "num_max_tokens_per_rank must not exceed buf_x.shape[0], but got ",
           num_max_tokens_per_rank, " vs ", padded_max_i64);
  PD_CHECK(num_tokens_i64 == topk_idx.shape()[0],
           "x.shape[0] must equal topk_idx.shape[0]");
  PD_CHECK(buf_x.shape()[1] == hidden_i64,
           "buf_x.shape[1] must equal hidden, but got ", buf_x.shape()[1],
           " vs ", hidden_i64);
  PD_CHECK(buf_topk_idx.shape()[0] == padded_max_i64,
           "buf_topk_idx.shape[0] must equal padded_max");
  PD_CHECK(buf_topk_idx.shape()[1] == top_k_i64,
           "buf_topk_idx.shape[1] must equal top_k");

  PD_CHECK(group_size == 32 || group_size == 64 || group_size == 128,
           "unsupported group_size: ", group_size);
  PD_CHECK(num_tokens_i64 <= num_max_tokens_per_rank,
           "num_tokens must not exceed padded_max");
  PD_CHECK(hidden_i64 % group_size == 0,
           "hidden must be a multiple of group_size");
  const int64_t num_groups_i64 = hidden_i64 / group_size;
  PD_CHECK(num_groups_i64 % 4 == 0, "num_groups must be a multiple of 4");
  PD_CHECK(buf_x_sf.shape()[0] == padded_max_i64,
           "buf_x_sf.shape[0] must equal padded_max");
  PD_CHECK(buf_x_sf.shape()[1] == num_groups_i64 / 4,
           "buf_x_sf.shape[1] must equal hidden/group_size/4, but got ",
           buf_x_sf.shape()[1], " vs ", num_groups_i64 / 4);
  PD_CHECK(hidden_i64 % static_cast<int64_t>(kVecElems) == 0,
           "hidden must be a multiple of 8 (16B bf16 loads)");
  const int64_t num_threads_i64 = hidden_i64 / static_cast<int64_t>(kVecElems);
  PD_CHECK(num_threads_i64 <= 1024,
           "hidden too large for single-block-per-row quant");
  PD_CHECK(num_threads_i64 >= top_k_i64, "top_k must fit into one quant CTA");

  const uint32_t num_tokens = static_cast<uint32_t>(num_tokens_i64);
  const uint32_t padded_max = static_cast<uint32_t>(padded_max_i64);
  const uint32_t hidden = static_cast<uint32_t>(hidden_i64);
  const uint32_t num_groups = static_cast<uint32_t>(num_groups_i64);
  const uint32_t top_k = static_cast<uint32_t>(top_k_i64);
  const uint32_t num_threads = static_cast<uint32_t>(num_threads_i64);
  const uint32_t num_total_blocks = num_tokens;

  const MegaMoEPreDispatchParams params{
      reinterpret_cast<const __nv_bfloat16*>(x.data<paddle::bfloat16>()),
      topk_idx.data<int64_t>(),
      topk_weights.data<float>(),
      const_cast<phi::dtype::float8_e4m3fn*>(
          buf_x.data<phi::dtype::float8_e4m3fn>()),
      const_cast<int32_t*>(buf_x_sf.data<int32_t>()),
      const_cast<int64_t*>(buf_topk_idx.data<int64_t>()),
      const_cast<float*>(buf_topk_weights.data<float>()),
      num_tokens,
      padded_max,
      hidden,
      num_groups,
      top_k,
  };

  if (num_total_blocks > 0) {
    auto stream = x.stream();
    switch (group_size) {
      case 32:
        LaunchMegaMoEPreDispatch<32>(params, num_total_blocks, num_threads,
                                     stream);
        break;
      case 64:
        LaunchMegaMoEPreDispatch<64>(params, num_total_blocks, num_threads,
                                     stream);
        break;
      case 128:
        LaunchMegaMoEPreDispatch<128>(params, num_total_blocks, num_threads,
                                      stream);
        break;
      default:
        PD_THROW("unsupported group_size: ", group_size);
    }
  }

  // return {buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights};
}

std::vector<paddle::DataType> MegaMoePreDispatchInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& topk_idx_dtype,
    const paddle::DataType& topk_weights_dtype,
    const paddle::DataType& buf_x_dtype,
    const paddle::DataType& buf_x_sf_dtype,
    const paddle::DataType& buf_topk_idx_dtype,
    const paddle::DataType& buf_topk_weights_dtype) {
  return {buf_x_dtype, buf_x_sf_dtype, buf_topk_idx_dtype, buf_topk_weights_dtype};
}

std::vector<std::vector<int64_t>> MegaMoePreDispatchInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& topk_idx_shape,
    const std::vector<int64_t>& topk_weights_shape,
    const std::vector<int64_t>& buf_x_shape,
    const std::vector<int64_t>& buf_x_sf_shape,
    const std::vector<int64_t>& buf_topk_idx_shape,
    const std::vector<int64_t>& buf_topk_weights_shape) {
  return {buf_x_shape, buf_x_sf_shape, buf_topk_idx_shape, buf_topk_weights_shape};
}


PD_BUILD_STATIC_OP(mega_moe_pre_dispatch)
    .Inputs({"x",
             "topk_idx",
             "topk_weights",
             "buf_x",
             "buf_x_sf",
             "buf_topk_idx",
             "buf_topk_weights"})
    .Outputs({"buf_x_out",
              "buf_x_sf_out",
              "buf_topk_idx_out",
              "buf_topk_weights_out"})
    .Attrs({"num_max_tokens_per_rank: int64_t", "group_size: int64_t"})
    .SetInplaceMap({{"buf_x", "buf_x_out"},
                    {"buf_x_sf", "buf_x_sf_out"},
                    {"buf_topk_idx", "buf_topk_idx_out"},
                    {"buf_topk_weights", "buf_topk_weights_out"}})
    .SetKernelFn(PD_KERNEL(MegaMoePreDispatch))
    .SetInferShapeFn(PD_INFER_SHAPE(MegaMoePreDispatchInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MegaMoePreDispatchInferDtype));
