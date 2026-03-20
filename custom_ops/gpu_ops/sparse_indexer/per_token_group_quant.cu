/*
 * Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "paddle/extension.h"
#include "vec_dtypes.cuh"

// ========================= Helper Device Functions =========================

__device__ __forceinline__ float GroupReduceMax(float val) {
  unsigned mask = threadIdx.x % 32 >= 16 ? 0xffff0000 : 0x0000ffff;

  val = fmaxf(val, __shfl_xor_sync(mask, val, 8));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 4));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 2));
  val = fmaxf(val, __shfl_xor_sync(mask, val, 1));
  return val;
}

// Vectorized load/store helpers
template <typename T, int vec_size>
__device__ __forceinline__ void vec_load(const T* src, T* dst) {
  using vec_t = typename std::conditional<
      vec_size == 1,
      T,
      typename std::conditional<
          vec_size == 2,
          typename std::conditional<sizeof(T) == 2, uint32_t, uint64_t>::type,
          typename std::conditional<
              vec_size == 4,
              typename std::conditional<sizeof(T) == 2, uint64_t, int4>::type,
              int4>::type>::type>::type;

  if constexpr (vec_size == 1) {
    dst[0] = src[0];
  } else {
    *reinterpret_cast<vec_t*>(dst) = *reinterpret_cast<const vec_t*>(src);
  }
}

template <typename T, int vec_size>
__device__ __forceinline__ void vec_store(T* dst, const T* src) {
  using vec_t = typename std::conditional<
      vec_size == 1,
      T,
      typename std::conditional<
          vec_size == 2,
          typename std::conditional<sizeof(T) == 2, uint32_t, uint64_t>::type,
          typename std::conditional<
              vec_size == 4,
              typename std::conditional<sizeof(T) == 2, uint64_t, int4>::type,
              int4>::type>::type>::type;

  if constexpr (vec_size == 1) {
    dst[0] = src[0];
  } else {
    *reinterpret_cast<vec_t*>(dst) = *reinterpret_cast<const vec_t*>(src);
  }
}

template <typename T, bool SCALE_UE8M0>
__device__ __forceinline__ float ComputeGroupScale(
    const T* __restrict__ group_input,
    T* __restrict__ smem_group,
    const int group_size,
    const int lane_id,
    const int threads_per_group,
    const float eps,
    const float max_8bit) {
  float local_absmax = eps;

  // Copy from global to shared memory and compute absmax
  for (int i = lane_id; i < group_size; i += threads_per_group) {
    T val = group_input[i];
    smem_group[i] = val;
    float abs_v = fabsf(static_cast<float>(val));
    local_absmax = fmaxf(local_absmax, abs_v);
  }

  local_absmax = GroupReduceMax(local_absmax);

  float y_s = local_absmax / max_8bit;
  if constexpr (SCALE_UE8M0) {
    y_s = exp2f(ceilf(log2f(fmaxf(fabsf(y_s), 1e-10f))));
  }

  return y_s;
}

template <typename T, typename DST_DTYPE>
__device__ __forceinline__ void QuantizeGroup(
    const T* __restrict__ smem_group,
    DST_DTYPE* __restrict__ group_output,
    const int group_size,
    const int lane_id,
    const int threads_per_group,
    const float y_s,
    const float min_8bit,
    const float max_8bit) {
  // Quantize from shared memory to global memory
  for (int i = lane_id; i < group_size; i += threads_per_group) {
    float val = static_cast<float>(smem_group[i]);
    float q = fminf(fmaxf(val / y_s, min_8bit), max_8bit);
    group_output[i] = DST_DTYPE(q);
  }
}

// ========================= Main Kernels =========================

template <typename T,
          typename DST_DTYPE,
          bool IS_COLUMN_MAJOR = false,
          bool SCALE_UE8M0 = false,
          typename scale_packed_t = float>
__global__ void per_token_group_quant_8bit_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    scale_packed_t* __restrict__ output_s,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const float eps,
    const float min_8bit,
    const float max_8bit,
    const int scale_num_rows = 0,
    const int scale_stride = 0) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;

  if (global_group_id >= num_groups) return;

  const int64_t block_group_offset = global_group_id * group_size;

  using scale_element_t = float;

  const T* group_input = input + block_group_offset;
  DST_DTYPE* group_output =
      static_cast<DST_DTYPE*>(output_q) + block_group_offset;
  scale_element_t* scale_output;

  if constexpr (IS_COLUMN_MAJOR) {
    const int num_elems_per_pack =
        static_cast<int>(sizeof(scale_packed_t) / sizeof(scale_element_t));
    const int scale_num_rows_element = scale_num_rows * num_elems_per_pack;
    const int row_idx = global_group_id / scale_num_rows_element;
    const int col_idx_raw = global_group_id % scale_num_rows_element;
    const int col_idx = col_idx_raw / num_elems_per_pack;
    const int pack_idx = col_idx_raw % num_elems_per_pack;
    scale_output = reinterpret_cast<scale_element_t*>(output_s) +
                   (col_idx * scale_stride * num_elems_per_pack +
                    row_idx * num_elems_per_pack + pack_idx);
  } else {
    scale_output =
        reinterpret_cast<scale_element_t*>(output_s) + global_group_id;
  }

  // Shared memory to cache each group's data
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  const float y_s = ComputeGroupScale<T, SCALE_UE8M0>(group_input,
                                                      smem_group,
                                                      group_size,
                                                      lane_id,
                                                      threads_per_group,
                                                      eps,
                                                      max_8bit);

  if (lane_id == 0) {
    *scale_output = y_s;
  }

  __syncthreads();

  QuantizeGroup<T, DST_DTYPE>(smem_group,
                              group_output,
                              group_size,
                              lane_id,
                              threads_per_group,
                              y_s,
                              min_8bit,
                              max_8bit);
}

template <typename T, typename DST_DTYPE>
__global__ void per_token_group_quant_8bit_packed_kernel(
    const T* __restrict__ input,
    void* __restrict__ output_q,
    unsigned int* __restrict__ output_s_packed,
    const int group_size,
    const int num_groups,
    const int groups_per_block,
    const int groups_per_row,
    const int mn,
    const int tma_aligned_mn,
    const float eps,
    const float min_8bit,
    const float max_8bit) {
  const int threads_per_group = 16;
  const int64_t local_group_id = threadIdx.x / threads_per_group;
  const int lane_id = threadIdx.x % threads_per_group;

  const int64_t block_group_id = blockIdx.x * groups_per_block;
  const int64_t global_group_id = block_group_id + local_group_id;

  // Check if this group is valid (don't return early to avoid __syncthreads
  // deadlock)
  const bool valid_group = (global_group_id < num_groups);

  const int64_t block_group_offset = global_group_id * group_size;

  // Use safe default pointers for invalid groups
  const T* group_input = valid_group ? (input + block_group_offset) : input;
  DST_DTYPE* group_output =
      valid_group ? (static_cast<DST_DTYPE*>(output_q) + block_group_offset)
                  : static_cast<DST_DTYPE*>(output_q);

  // Shared memory to cache each group's data
  extern __shared__ __align__(16) char smem_raw[];
  T* smem = reinterpret_cast<T*>(smem_raw);
  T* smem_group = smem + local_group_id * group_size;

  float y_s = 1.0f;  // Default scale for invalid groups
  if (valid_group) {
    y_s = ComputeGroupScale<T, true>(group_input,
                                     smem_group,
                                     group_size,
                                     lane_id,
                                     threads_per_group,
                                     eps,
                                     max_8bit);

    // Pack 4 scales into a uint32
    if (lane_id == 0) {
      const int sf_k_idx = static_cast<int>(global_group_id % groups_per_row);
      const int mn_idx = static_cast<int>(global_group_id / groups_per_row);

      if (mn_idx < mn) {
        const int sf_k_pack_idx = sf_k_idx / 4;
        const int pos = sf_k_idx % 4;

        const unsigned int bits = __float_as_uint(y_s);
        const unsigned int exponent = (bits >> 23u) & 0xffu;
        const unsigned int contrib = exponent << (pos * 8u);

        const int out_idx = sf_k_pack_idx * tma_aligned_mn + mn_idx;
        atomicOr(output_s_packed + out_idx, contrib);
      }
    }
  }

  __syncthreads();

  if (valid_group) {
    QuantizeGroup<T, DST_DTYPE>(smem_group,
                                group_output,
                                group_size,
                                lane_id,
                                threads_per_group,
                                y_s,
                                min_8bit,
                                max_8bit);
  }
}

// ========================= Host Functions =========================

inline int GetGroupsPerBlock(int64_t num_groups) {
  if (num_groups % 16 == 0) {
    return 16;
  }
  if (num_groups % 8 == 0) {
    return 8;
  }
  if (num_groups % 4 == 0) {
    return 4;
  }
  if (num_groups % 2 == 0) {
    return 2;
  }
  return 1;
}

template <typename T, typename DST_DTYPE>
void launch_per_token_group_quant_8bit(const paddle::Tensor& input,
                                       paddle::Tensor& output_q,
                                       paddle::Tensor& output_s,
                                       int64_t group_size,
                                       float eps,
                                       float min_8bit,
                                       float max_8bit,
                                       bool scale_ue8m0,
                                       cudaStream_t stream) {
  const int num_groups = input.numel() / group_size;
  constexpr int THREADS_PER_GROUP = 16;
  const int groups_per_block = GetGroupsPerBlock(num_groups);
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  const bool is_column_major = output_s.strides()[0] < output_s.strides()[1];
  const int scale_num_rows = output_s.dims()[1];
  const int scale_stride = output_s.strides()[1];

  dim3 grid(num_blocks);
  dim3 block(num_threads);
  size_t smem_bytes =
      static_cast<size_t>(groups_per_block) * group_size * sizeof(T);

  // Use data() to get void* pointer since Paddle doesn't instantiate data<T>()
  // for all types
  const T* input_ptr = static_cast<const T*>(input.data());
  void* output_q_ptr = output_q.data();
  float* output_s_ptr = static_cast<float*>(output_s.data());

  if (is_column_major) {
    if (scale_ue8m0) {
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, true>
          <<<grid, block, smem_bytes, stream>>>(input_ptr,
                                                output_q_ptr,
                                                output_s_ptr,
                                                group_size,
                                                num_groups,
                                                groups_per_block,
                                                eps,
                                                min_8bit,
                                                max_8bit,
                                                scale_num_rows,
                                                scale_stride);
    } else {
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, true, false>
          <<<grid, block, smem_bytes, stream>>>(input_ptr,
                                                output_q_ptr,
                                                output_s_ptr,
                                                group_size,
                                                num_groups,
                                                groups_per_block,
                                                eps,
                                                min_8bit,
                                                max_8bit,
                                                scale_num_rows,
                                                scale_stride);
    }
  } else {
    if (scale_ue8m0) {
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, true>
          <<<grid, block, smem_bytes, stream>>>(input_ptr,
                                                output_q_ptr,
                                                output_s_ptr,
                                                group_size,
                                                num_groups,
                                                groups_per_block,
                                                eps,
                                                min_8bit,
                                                max_8bit);
    } else {
      per_token_group_quant_8bit_kernel<T, DST_DTYPE, false, false>
          <<<grid, block, smem_bytes, stream>>>(input_ptr,
                                                output_q_ptr,
                                                output_s_ptr,
                                                group_size,
                                                num_groups,
                                                groups_per_block,
                                                eps,
                                                min_8bit,
                                                max_8bit);
    }
  }
}

template <typename T, typename DST_DTYPE>
void launch_per_token_group_quant_8bit_packed(const paddle::Tensor& input,
                                              paddle::Tensor& output_q,
                                              paddle::Tensor& output_s_packed,
                                              int64_t group_size,
                                              float eps,
                                              float min_8bit,
                                              float max_8bit,
                                              cudaStream_t stream) {
  const int64_t k = input.dims()[input.dims().size() - 1];
  const int64_t mn = input.numel() / k;
  const int64_t groups_per_row = k / group_size;
  const int64_t num_groups = mn * groups_per_row;
  const int64_t tma_aligned_mn = ((mn + 3) / 4) * 4;

  constexpr int THREADS_PER_GROUP = 16;
  const int groups_per_block = GetGroupsPerBlock(num_groups);
  const int num_blocks = (num_groups + groups_per_block - 1) / groups_per_block;
  const int num_threads = groups_per_block * THREADS_PER_GROUP;

  dim3 grid(num_blocks);
  dim3 block(num_threads);
  size_t smem_bytes =
      static_cast<size_t>(groups_per_block) * group_size * sizeof(T);

  // Use data() to get void* pointer since Paddle doesn't instantiate data<T>()
  // for all types
  const T* input_ptr = static_cast<const T*>(input.data());
  void* output_q_ptr = output_q.data();
  unsigned int* output_s_ptr =
      static_cast<unsigned int*>(output_s_packed.data());

  per_token_group_quant_8bit_packed_kernel<T, DST_DTYPE>
      <<<grid, block, smem_bytes, stream>>>(input_ptr,
                                            output_q_ptr,
                                            output_s_ptr,
                                            static_cast<int>(group_size),
                                            static_cast<int>(num_groups),
                                            groups_per_block,
                                            static_cast<int>(groups_per_row),
                                            static_cast<int>(mn),
                                            static_cast<int>(tma_aligned_mn),
                                            eps,
                                            min_8bit,
                                            max_8bit);
}

// ========================= API Functions =========================

void PerTokenGroupQuantFp8(const paddle::Tensor& input,
                           paddle::Tensor& output_q,
                           paddle::Tensor& output_s,
                           int64_t group_size,
                           double eps,
                           double fp8_min,
                           double fp8_max,
                           bool scale_ue8m0) {
  PD_CHECK(input.is_gpu(), "Input tensor must be on GPU");
  PD_CHECK(output_q.is_gpu(), "Output Q tensor must be on GPU");
  PD_CHECK(output_s.is_gpu(), "Output S tensor must be on GPU");
  PD_CHECK(input.numel() % group_size == 0,
           "Input numel must be divisible by group_size");
  PD_CHECK(output_s.dims().size() == 2, "Output S tensor must be 2D");

  cudaStream_t stream = input.stream();

  auto input_dtype = input.dtype();
  auto output_dtype = output_q.dtype();

  if (input_dtype == paddle::DataType::BFLOAT16) {
    if (output_dtype == paddle::DataType::FLOAT8_E4M3FN) {
      launch_per_token_group_quant_8bit<nv_bfloat16, __nv_fp8_e4m3>(
          input,
          output_q,
          output_s,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          scale_ue8m0,
          stream);
    } else if (output_dtype == paddle::DataType::INT8) {
      launch_per_token_group_quant_8bit<nv_bfloat16, int8_t>(
          input,
          output_q,
          output_s,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          scale_ue8m0,
          stream);
    } else {
      PD_CHECK(false,
               "PerTokenGroupQuantFp8 only supports output_q dtypes "
               "FLOAT8_E4M3FN and INT8 for BFLOAT16 input.");
    }
  } else if (input_dtype == paddle::DataType::FLOAT16) {
    if (output_dtype == paddle::DataType::FLOAT8_E4M3FN) {
      launch_per_token_group_quant_8bit<half, __nv_fp8_e4m3>(
          input,
          output_q,
          output_s,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          scale_ue8m0,
          stream);
    } else if (output_dtype == paddle::DataType::INT8) {
      launch_per_token_group_quant_8bit<half, int8_t>(
          input,
          output_q,
          output_s,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          scale_ue8m0,
          stream);
    } else {
      PD_CHECK(false,
               "PerTokenGroupQuantFp8 only supports output_q dtypes "
               "FLOAT8_E4M3FN and INT8 for FLOAT16 input.");
    }
  } else if (input_dtype == paddle::DataType::FLOAT32) {
    if (output_dtype == paddle::DataType::FLOAT8_E4M3FN) {
      launch_per_token_group_quant_8bit<float, __nv_fp8_e4m3>(
          input,
          output_q,
          output_s,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          scale_ue8m0,
          stream);
    } else if (output_dtype == paddle::DataType::INT8) {
      launch_per_token_group_quant_8bit<float, int8_t>(
          input,
          output_q,
          output_s,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          scale_ue8m0,
          stream);
    } else {
      PD_CHECK(false,
               "PerTokenGroupQuantFp8 only supports output_q dtypes "
               "FLOAT8_E4M3FN and INT8 for FLOAT32 input.");
    }
  } else {
    PD_CHECK(false,
             "PerTokenGroupQuantFp8 only supports input dtypes BFLOAT16, "
             "FLOAT16 and FLOAT32.");
  }
}

void PerTokenGroupQuantFp8Packed(const paddle::Tensor& input,
                                 paddle::Tensor& output_q,
                                 paddle::Tensor& output_s_packed,
                                 int64_t group_size,
                                 double eps,
                                 double fp8_min,
                                 double fp8_max) {
  PD_CHECK(input.is_gpu(), "Input tensor must be on GPU");
  PD_CHECK(output_q.is_gpu(), "Output Q tensor must be on GPU");
  PD_CHECK(output_s_packed.is_gpu(), "Output S packed tensor must be on GPU");

  const int64_t k = input.dims()[input.dims().size() - 1];
  PD_CHECK(k % group_size == 0,
           "Last dimension must be divisible by group_size");
  PD_CHECK(output_s_packed.dims().size() == 2, "output_s_packed must be 2D");
  PD_CHECK(output_s_packed.dtype() == paddle::DataType::INT32,
           "output_s_packed must have dtype int32");

  // Zero-initialize packed scales
  cudaStream_t stream = input.stream();
  cudaMemsetAsync(output_s_packed.data(),
                  0,
                  output_s_packed.numel() * sizeof(int32_t),
                  stream);

  auto input_dtype = input.dtype();
  auto output_dtype = output_q.dtype();

  if (input_dtype == paddle::DataType::BFLOAT16) {
    if (output_dtype == paddle::DataType::FLOAT8_E4M3FN) {
      launch_per_token_group_quant_8bit_packed<nv_bfloat16, __nv_fp8_e4m3>(
          input,
          output_q,
          output_s_packed,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          stream);
    } else if (output_dtype == paddle::DataType::INT8) {
      launch_per_token_group_quant_8bit_packed<nv_bfloat16, int8_t>(
          input,
          output_q,
          output_s_packed,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          stream);
    } else {
      PD_CHECK(false,
               "PerTokenGroupQuantFp8 only supports output_q dtypes "
               "FLOAT8_E4M3FN and INT8 for BFLOAT16 input.");
    }
  } else if (input_dtype == paddle::DataType::FLOAT16) {
    if (output_dtype == paddle::DataType::FLOAT8_E4M3FN) {
      launch_per_token_group_quant_8bit_packed<half, __nv_fp8_e4m3>(
          input,
          output_q,
          output_s_packed,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          stream);
    } else if (output_dtype == paddle::DataType::INT8) {
      launch_per_token_group_quant_8bit_packed<half, int8_t>(
          input,
          output_q,
          output_s_packed,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          stream);
    } else {
      PD_CHECK(false,
               "PerTokenGroupQuantFp8 only supports output_q dtypes "
               "FLOAT8_E4M3FN and INT8 for FLOAT16 input.");
    }
  } else if (input_dtype == paddle::DataType::FLOAT32) {
    if (output_dtype == paddle::DataType::FLOAT8_E4M3FN) {
      launch_per_token_group_quant_8bit_packed<float, __nv_fp8_e4m3>(
          input,
          output_q,
          output_s_packed,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          stream);
    } else if (output_dtype == paddle::DataType::INT8) {
      launch_per_token_group_quant_8bit_packed<float, int8_t>(
          input,
          output_q,
          output_s_packed,
          group_size,
          static_cast<float>(eps),
          static_cast<float>(fp8_min),
          static_cast<float>(fp8_max),
          stream);
    } else {
      PD_CHECK(false,
               "PerTokenGroupQuantFp8 only supports output_q dtypes "
               "FLOAT8_E4M3FN and INT8 for FLOAT32 input.");
    }
  } else {
    PD_CHECK(false,
             "PerTokenGroupQuantFp8 only supports input dtypes BFLOAT16, "
             "FLOAT16 and FLOAT32.");
  }
}

// ========================= Paddle Custom Op Registration
// =========================

PD_BUILD_OP(per_token_group_quant_fp8)
    .Inputs({"input", "output_q", "output_s"})
    .Outputs({"output_q_out", "output_s_out"})
    .Attrs({"group_size: int64_t",
            "eps: double",
            "fp8_min: double",
            "fp8_max: double",
            "scale_ue8m0: bool"})
    .SetInplaceMap({{"output_q", "output_q_out"}, {"output_s", "output_s_out"}})
    .SetKernelFn(PD_KERNEL(PerTokenGroupQuantFp8));

PD_BUILD_OP(per_token_group_quant_fp8_packed)
    .Inputs({"input", "output_q", "output_s_packed"})
    .Outputs({"output_q_out", "output_s_packed_out"})
    .Attrs({"group_size: int64_t",
            "eps: double",
            "fp8_min: double",
            "fp8_max: double"})
    .SetInplaceMap({{"output_q", "output_q_out"},
                    {"output_s_packed", "output_s_packed_out"}})
    .SetKernelFn(PD_KERNEL(PerTokenGroupQuantFp8Packed));
