// /*
//  * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION &
//  * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
//  *
//  * Licensed under the Apache License, Version 2.0 (the "License");
//  * you may not use this file except in compliance with the License.
//  * You may obtain a copy of the License at
//  *
//  * http://www.apache.org/licenses/LICENSE-2.0
//  *
//  * Unless required by applicable law or agreed to in writing, software
//  * distributed under the License is distributed on an "AS IS" BASIS,
//  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  * See the License for the specific language governing permissions and
//  * limitations under the License.
//  */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include "cutlass/numeric_conversion.h"
#include "moe/fused_moe_helper.h"
#include "moe/fused_moe_imp_op.h"
// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

// #include "paddle/phi/backends/gpu/gpu_info.h"
#pragma GCC diagnostic pop

#include "helper.h"

#define WARP_SIZE 32

namespace phi {

struct GpuLaunchConfig {
  dim3 block_per_grid;
  dim3 thread_per_block;
};

inline GpuLaunchConfig Get1DBlocksAnd2DGridsMoe(const int64_t cols) {
  int blocks_x = cols;
  int blocks_y = 1;
  int blocks_z = 1;
  if (blocks_x > 1024) {
    blocks_y = 256;
    blocks_x = (blocks_x + blocks_y - 1) / blocks_y;
  }

  GpuLaunchConfig config;
  config.block_per_grid.x = blocks_x;
  config.block_per_grid.y = blocks_y;
  config.block_per_grid.z = blocks_z;
  return config;
}

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;
template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  using Type = typename U::Element;
  static_assert(T::kElements == U::kElements);
  U u;
#pragma unroll
  for (int i = 0; i < U::kElements; i++) {
    u[i] = static_cast<Type>(input[i]);
  }
  return u;
}

struct uint8 {
  uint4 u;
  uint4 v;
};

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
  typedef cub::BlockReduce<T, block_size> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T result_broadcast;
  T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
  if (threadIdx.x == 0) {
    result_broadcast = result;
  }
  __syncthreads();
  return result_broadcast;
}

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) {
    return x + y;
  }
};

template <typename T>
struct MaxOp {
  __device__ inline T operator()(T const& x, T const& y) {
    return x > y ? x : y;
  }
};

template <>
struct MaxOp<float> {
  __device__ inline float operator()(float const& x, float const& y) {
    return fmax(x, y);
  }
};

template <typename InType, typename OutType>
__forceinline__ __device__ OutType QuantHelperFunc(const InType input,
                                                   const float scale,
                                                   const float max_bound,
                                                   const float min_bound) {
  float quant_value = max_bound * scale * static_cast<float>(input);
  return static_cast<OutType>(
      ClipFunc<float>(quant_value, min_bound, max_bound));
}

template <typename T, typename OutT, int VecSize, int Kthread>
__global__ void masked_quantize_moe_input_kernel(
    const T* permuted_inputs,
    const int64_t* expert_idx_per_token,
    const int64_t token_num,
    const int64_t dim,
    float* input_dequant_scale,
    const int64_t* recv_expert_count,
    const int num_max_tokens_per_expert,
    OutT* out) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadOutT = AlignedVector<OutT, VecSize>;
  LoadT input_vec;
  LoadOutT output_vec;
  using vec_t = typename BytesToType<sizeof(OutT) * VecSize>::Type;
  extern __shared__ char smem_[];
  for (int token_idx = blockIdx.x; token_idx < token_num;
       token_idx += gridDim.x) {
    const auto token_idx_in_expert = token_idx % num_max_tokens_per_expert;
    const auto expert_id = token_idx / num_max_tokens_per_expert;
    if (token_idx_in_expert >= recv_expert_count[expert_id]) {
      auto next_expert_start_idx = (expert_id + 1) * num_max_tokens_per_expert;
      auto num_iters_to_next_expert =
          (next_expert_start_idx - token_idx - 1) / gridDim.x;
      token_idx += num_iters_to_next_expert * gridDim.x;
      continue;
    }
    int64_t expert_idx = expert_idx_per_token[token_idx];
    float abs_max = 0.0f;
    for (int idx = threadIdx.x; idx < dim / VecSize; idx += blockDim.x) {
      int64_t offset = token_idx * dim + idx * VecSize;
      Load<T, VecSize>(&permuted_inputs[offset], &input_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        float res = static_cast<float>(input_vec[i]);
        abs_max = fmax(abs_max, fabs(res));
      }
      Store<T, VecSize>(input_vec, reinterpret_cast<T*>(smem_) + idx * VecSize);
    }
    abs_max = BlockAllReduce<MaxOp, float, Kthread>(abs_max);
    input_dequant_scale[token_idx] = abs_max;
    float quant_scale = 440.0f / abs_max;
    for (int idx = threadIdx.x; idx < dim / VecSize; idx += blockDim.x) {
      int64_t offset = token_idx * dim + idx * VecSize;
      Load<T, VecSize>(reinterpret_cast<T*>(smem_) + idx * VecSize, &input_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        float res = static_cast<float>(input_vec[i]);
        output_vec[i] = static_cast<OutT>(res * quant_scale);
      }
      *(reinterpret_cast<vec_t*>(&out[offset])) =
          *(reinterpret_cast<const vec_t*>(&output_vec));
    }
  }
}

template <typename T, typename OutT, int VecSize, int Kthread>
__global__ void quantize_moe_input_kernel(const T* permuted_inputs,
                                          const int64_t* expert_idx_per_token,
                                          const int64_t token_num,
                                          const int64_t dim,
                                          float* input_dequant_scale,
                                          const int64_t* recv_expert_count,
                                          const int num_max_tokens_per_expert,
                                          OutT* out) {
  using LoadT = AlignedVector<T, VecSize>;
  using LoadOutT = AlignedVector<OutT, VecSize>;
  LoadT input_vec;
  LoadOutT output_vec;
  using vec_t = typename BytesToType<sizeof(OutT) * VecSize>::Type;

  extern __shared__ char smem_[];

  for (int token_idx = blockIdx.x; token_idx < token_num;
       token_idx += gridDim.x) {
    int64_t expert_idx = expert_idx_per_token[token_idx];
    float abs_max = 0.0f;
    for (int idx = threadIdx.x; idx < dim / VecSize; idx += blockDim.x) {
      int64_t offset = token_idx * dim + idx * VecSize;
      Load<T, VecSize>(&permuted_inputs[offset], &input_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        float res = static_cast<float>(input_vec[i]);
        abs_max = fmax(abs_max, fabs(res));
      }
      Store<T, VecSize>(input_vec, reinterpret_cast<T*>(smem_) + idx * VecSize);
    }
    abs_max = BlockAllReduce<MaxOp, float, Kthread>(abs_max);
    input_dequant_scale[token_idx] = abs_max;
    float quant_scale = 440.0f / abs_max;

    for (int idx = threadIdx.x; idx < dim / VecSize; idx += blockDim.x) {
      int64_t offset = token_idx * dim + idx * VecSize;
      Load<T, VecSize>(reinterpret_cast<T*>(smem_) + idx * VecSize, &input_vec);
#pragma unroll
      for (int i = 0; i < VecSize; i++) {
        float res = static_cast<float>(input_vec[i]);
        output_vec[i] = static_cast<OutT>(res * quant_scale);
      }
      *(reinterpret_cast<vec_t*>(&out[offset])) =
          *(reinterpret_cast<const vec_t*>(&output_vec));
    }
  }
}

template <typename T, typename OutT>
void quantize_moe_input(const T* permuted_inputs,
                        const int64_t* expert_idx_per_token,
                        const int64_t token_num,
                        const int64_t dim,
                        float* input_quant_scale,
                        const int64_t* recv_expert_count,
                        const int num_max_tokens_per_expert,
                        bool used_in_ep_low_latency,
                        OutT* out,
                        cudaStream_t stream) {
  constexpr int VecSize = 16 / sizeof(T);
  constexpr int threads_per_block = 128;
  const int dev_id = 0;
  int sm_count;
  int act_blocks_per_sm;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  assert(dim % VecSize == 0);
  auto kernel =
      used_in_ep_low_latency
          ? masked_quantize_moe_input_kernel<T,
                                             OutT,
                                             VecSize,
                                             threads_per_block>
          : quantize_moe_input_kernel<T, OutT, VecSize, threads_per_block>;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &act_blocks_per_sm, kernel, threads_per_block, 0);
  const int num_blocks_per_wave = sm_count * act_blocks_per_sm;
  dim3 grid;
  grid.x = min(static_cast<int64_t>(num_blocks_per_wave), token_num);
  const int smem_size = dim * sizeof(T);
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  kernel<<<grid, threads_per_block, smem_size, stream>>>(
      permuted_inputs,
      expert_idx_per_token,
      token_num,
      dim,
      input_quant_scale,
      recv_expert_count,
      num_max_tokens_per_expert,
      out);
}

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing
// the output in the softmax kernel when we extend this module to support
// expert-choice routing.
template <typename T, int TPB>
__launch_bounds__(TPB) __global__
    void group_moe_softmax(const T* input,
                           T* output,
                           T* softmax_max_prob,
                           const int64_t num_cols,
                           const int64_t softmax_num_rows) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;
  __shared__ float max_out;

  int globalIdx = blockIdx.x + blockIdx.y * gridDim.x;
  if (globalIdx >= softmax_num_rows) {
    return;
  }
  const int64_t thread_row_offset = globalIdx * num_cols;

  cub::Sum sum;
  float threadData(-FLT_MAX);

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = max(static_cast<float>(input[idx]), threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((static_cast<float>(input[idx]) - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val =
        exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = T(val);
    threadData = max(static_cast<float>(T(val)), threadData);
  }

  const float maxOut = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    // group max probs
    max_out = 1.f / maxOut;
    softmax_max_prob[globalIdx] = T(max_out);
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    // group softmax normalization
    output[idx] = output[idx] * static_cast<T>(max_out);
  }
}

template <typename T, int TPB>
__launch_bounds__(TPB) __global__ void moe_softmax(const T* input,
                                                   T* output,
                                                   const int64_t num_cols,
                                                   const int64_t num_rows) {
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  int globalIdx = blockIdx.x + blockIdx.y * gridDim.x;
  if (globalIdx >= num_rows) {
    return;
  }
  const int64_t thread_row_offset = globalIdx * num_cols;

  cub::Sum sum;
  float threadData(-FLT_MAX);

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData = max(static_cast<float>(input[idx]), threadData);
  }

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  threadData = 0;

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    threadData += exp((static_cast<float>(input[idx]) - float_max));
  }

  const auto Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }
  __syncthreads();

  for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
    const int idx = thread_row_offset + ii;
    const float val =
        exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
    output[idx] = T(val);
  }
}

template <typename T, int TPB, typename IdxT = int>
__launch_bounds__(TPB) __global__
    void group_moe_top_k(const T* inputs_after_softmax,
                         T* output,
                         IdxT* indices,
                         int* source_rows,
                         T* softmax_max_prob,
                         const int64_t num_experts,
                         const int64_t k,
                         const int64_t num_rows) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x + blockIdx.y * gridDim.x;
  if (block_row >= num_rows) {
    return;
  }

  const bool should_process_row = true;
  const int thread_read_offset = block_row * num_experts;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const IdxT prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      // restore normalized probes
      output[idx] = result_kvp.value / T(softmax_max_prob[idx]);
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();
  }
}

template <typename T, int TPB, bool NormWeights = false, typename IdxT = int>
__launch_bounds__(TPB) __global__ void moe_top_k(const T* inputs_after_softmax,
                                                 const T* bias,
                                                 T* output,
                                                 IdxT* indices,
                                                 int* source_rows,
                                                 const int64_t num_experts,
                                                 const int64_t k,
                                                 const int64_t num_rows) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x + blockIdx.y * gridDim.x;
  if (block_row >= num_rows) {
    return;
  }

  const bool should_process_row = true;
  const int thread_read_offset = block_row * num_experts;
  T weight_sum = static_cast<T>(0);
  T* row_outputs = nullptr;

  if constexpr (NormWeights) {
    extern __shared__ char smem[];
    row_outputs = reinterpret_cast<T*>(smem);
  }

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = bias ? inputs_after_softmax[idx] + bias[expert]
                           : inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;

      if constexpr (NormWeights) {
        T row_out =
            bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]
                 : result_kvp.value;
        row_outputs[k_idx] = row_out;
        weight_sum += row_out;
      } else {
        output[idx] =
            bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]
                 : result_kvp.value;
      }
    }
    __syncthreads();
  }
  if constexpr (NormWeights) {
    if (threadIdx.x < WARP_SIZE) {
      weight_sum = __shfl_sync(0xffffffff, weight_sum, 0);
    }
    if (threadIdx.x < k) {
      output[k * block_row + threadIdx.x] =
          row_outputs[threadIdx.x] / weight_sum;
    }
  }
}

template <typename T, int TPB, bool NormWeights = false, typename IdxT = int>
__launch_bounds__(TPB) __global__
    void moe_softmax_top_k_fused(const T* input,
                                 const T* bias,
                                 T* output,
                                 IdxT* indices,
                                 int* source_rows,
                                 const int64_t num_experts,
                                 const int64_t k,
                                 const int64_t num_rows) {
  // softmax
  using BlockReduce = cub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  __shared__ float normalizing_factor;
  __shared__ float float_max;

  int globalIdx = blockIdx.x + blockIdx.y * gridDim.x;
  if (globalIdx >= num_rows) {
    return;
  }
  const int64_t thread_row_offset = globalIdx * num_experts;
  const int64_t idx = thread_row_offset + threadIdx.x;

  cub::Sum sum;

  float threadData =
      (threadIdx.x < num_experts) ? static_cast<float>(input[idx]) : (-FLT_MAX);

  const float maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
  if (threadIdx.x == 0) {
    float_max = maxElem;
  }
  __syncthreads();

  float threadDataSub = threadData - float_max;
  float threadDataExp = exp(threadDataSub);

  const auto Z = BlockReduce(tmpStorage).Reduce(threadDataExp, sum);

  if (threadIdx.x == 0) {
    normalizing_factor = 1.f / Z;
  }

  __syncthreads();

  T val = T(threadDataExp * normalizing_factor);

  // top_k
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduceP = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduceP::TempStorage tmpStorageP;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  T weight_sum = static_cast<T>(0);
  T* row_outputs = nullptr;
  if constexpr (NormWeights) {
    extern __shared__ char smem[];
    row_outputs = reinterpret_cast<T*>(smem);
  }

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    if (threadIdx.x < num_experts) {
      cub_kvp inp_kvp;
      int expert = threadIdx.x;
      inp_kvp.key = expert;
      inp_kvp.value = bias ? val + bias[expert] : val;

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const IdxT prior_winning_expert = indices[k * globalIdx + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }
      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduceP(tmpStorageP).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int cur_idx = k * globalIdx + k_idx;

      indices[cur_idx] = result_kvp.key;
      source_rows[cur_idx] = k_idx * num_rows + globalIdx;

      if constexpr (NormWeights) {
        T row_out =
            bias ? (result_kvp.value - bias[result_kvp.key]) : result_kvp.value;
        row_outputs[k_idx] = row_out;
        weight_sum += row_out;
      } else {
        output[cur_idx] =
            bias ? (result_kvp.value - bias[result_kvp.key]) : result_kvp.value;
      }
    }
    __syncthreads();
  }
  if constexpr (NormWeights) {
    if (threadIdx.x < WARP_SIZE) {
      weight_sum = __shfl_sync(0xffffffff, weight_sum, 0);
    }

    if (threadIdx.x < k) {
      output[k * globalIdx + threadIdx.x] =
          row_outputs[threadIdx.x] / weight_sum;
    }
  }
}

inline __device__ unsigned int xorwow_moe(unsigned int& state) {
  state ^= state >> 7;
  state ^= state << 9;
  state ^= state >> 13;
  return state;
}

template <typename T, int TPB, typename IdxT = int>
__launch_bounds__(TPB) __global__
    void moe_redundant_top_k_normed(const T* inputs_after_softmax,
                                    const T* bias,
                                    const int* expert_id_to_ep_rank_array,
                                    const int* expert_in_rank_num_list,
                                    int* tokens_per_expert_stats_list,
                                    T* output,
                                    IdxT* indices,
                                    IdxT* indices_tmp,
                                    int* source_rows,
                                    const int64_t num_experts,
                                    const int64_t k,
                                    const int64_t num_rows,
                                    const int redundant_ep_rank_num_plus_one) {
  using cub_kvp = cub::KeyValuePair<int, T>;
  using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
  __shared__ typename BlockReduce::TempStorage tmpStorage;

  cub_kvp thread_kvp;
  cub::ArgMax arg_max;

  const int block_row = blockIdx.x + blockIdx.y * gridDim.x;
  // unsigned int state = block_row + blockIdx.x * blockDim.x +
  // *kernel_call_num;
  unsigned int state = block_row + blockIdx.x * blockDim.x;

  if (block_row >= num_rows) {
    return;
  }

  const bool should_process_row = true;
  const int thread_read_offset = block_row * num_experts;
  T weight_sum = static_cast<T>(0);

  extern __shared__ char smem[];

  T* row_outputs = reinterpret_cast<T*>(smem);

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    thread_kvp.key = 0;
    thread_kvp.value = T(-1.f);  // This is OK because inputs are probabilities

    cub_kvp inp_kvp;
    for (int expert = threadIdx.x; expert < num_experts; expert += TPB) {
      const int idx = thread_read_offset + expert;
      inp_kvp.key = expert;
      inp_kvp.value = bias ? inputs_after_softmax[idx] + bias[expert]
                           : inputs_after_softmax[idx];

      for (int prior_k = 0; prior_k < k_idx; ++prior_k) {
        const int prior_winning_expert = indices_tmp[k * block_row + prior_k];

        if (prior_winning_expert == expert) {
          inp_kvp = thread_kvp;
        }
      }

      thread_kvp = arg_max(inp_kvp, thread_kvp);
    }

    const cub_kvp result_kvp =
        BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
    if (threadIdx.x == 0) {
      const int idx = k * block_row + k_idx;
      // output[idx] = bias ? inputs_after_softmax[thread_read_offset +
      // result_kvp.key]: result_kvp.value;
      source_rows[idx] = k_idx * num_rows + block_row;
      int expert_topk = should_process_row ? result_kvp.key : num_experts;

      // runduncy
      int len = expert_in_rank_num_list[expert_topk];
      int select = (int)xorwow_moe(state) % len;
      int selected_rank =
          expert_id_to_ep_rank_array[expert_topk *
                                         redundant_ep_rank_num_plus_one +
                                     select];

      indices[idx] = (IdxT)selected_rank;
      indices_tmp[idx] = result_kvp.key;
      atomicAdd(&tokens_per_expert_stats_list[result_kvp.key], 1);

      T row_out =
          bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]
               : result_kvp.value;
      row_outputs[k_idx] = row_out;
      weight_sum += row_out;
    }
    __syncthreads();
  }
  if (threadIdx.x < WARP_SIZE) {
    weight_sum = __shfl_sync(0xffffffff, weight_sum, 0);
  }

  if (threadIdx.x < k) {
    output[k * block_row + threadIdx.x] = row_outputs[threadIdx.x] / weight_sum;
  }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the
  MoE layers are a small power of 2. This allows us to cleanly share the rows
  among the threads in a single warp and eliminate communication between warps
  (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small
  power of 2. 2) This implementation assumes k is small, but will work for any
  k.
*/

template <typename T,
          int VPT,
          int NUM_EXPERTS,
          int WARPS_PER_CTA,
          int BYTES_PER_LDG,
          bool Norm_Weights = false,
          typename IdxT = int>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__
    void topk_gating_softmax(const T* input,
                             const T* bias,
                             T* output,
                             const int64_t num_rows,
                             IdxT* indices,
                             int* source_rows,
                             const int64_t k) {
  // We begin by enforcing compile time assertions and setting up compile time
  // constants.
  static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
  static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS),
                "NUM_EXPERTS must be power of 2");
  static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG),
                "BYTES_PER_LDG must be power of 2");
  static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

  // Number of bytes each thread pulls in per load
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
  static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
  static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

  // Restrictions based on previous section.
  static_assert(
      VPT % ELTS_PER_LDG == 0,
      "The elements per thread must be a multiple of the elements per ldg");
  static_assert(WARP_SIZE % THREADS_PER_ROW == 0,
                "The threads per row must cleanly divide the threads per warp");
  static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW),
                "THREADS_PER_ROW must be power of 2");
  static_assert(THREADS_PER_ROW <= WARP_SIZE,
                "THREADS_PER_ROW can be at most warp size");

  // We have NUM_EXPERTS elements per row. We specialize for small #experts
  static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
  static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
  static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

  // Restrictions for previous section.
  static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0,
                "The elts per row must cleanly divide the total elt per warp");

  // ===================== From this point, we finally start computing run-time
  // variables. ========================

  // Compute CTA and warp rows. We pack multiple rows into a single warp, and a
  // block contains WARPS_PER_CTA warps. This, each block processes a chunk of
  // rows. We start by computing the start row for each block.
  const int cta_base_row = blockIdx.x * ROWS_PER_CTA;

  // Now, using the base row per thread block, we compute the base row per warp.
  const int warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

  // The threads in a warp are split into sub-groups that will work on a row.
  // We compute row offset for each thread sub-group
  const int thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
  const int thread_row = warp_base_row + thread_row_in_warp;
  const int thread_row_in_cta = thread_row - cta_base_row;

  // Threads with indices out of bounds should early exit here.
  if (thread_row >= num_rows) return;
  const bool should_process_row = true;

  // We finally start setting up the read pointers for each thread. First, each
  // thread jumps to the start of the row it will read.
  const T* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

  // Now, we compute the group each thread belong to in order to determine the
  // first column to start loads.
  const int thread_group_idx = threadIdx.x % THREADS_PER_ROW;
  const int first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
  const T* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

  T weight_sum = static_cast<T>(0);
  extern __shared__ T row_output[];

  // Determine the pointer type to use to read in the data depending on the
  // BYTES_PER_LDG template param. In theory, this can support all powers of 2
  // up to 16.
  using AccessType = cutlass::AlignedArray<T, ELTS_PER_LDG>;

  // Finally, we pull in the data from global mem
  cutlass::Array<T, VPT> row_chunk_input;
  AccessType* row_chunk_vec_ptr =
      reinterpret_cast<AccessType*>(&row_chunk_input);
  const AccessType* vec_thread_read_ptr =
      reinterpret_cast<const AccessType*>(thread_read_ptr);
#pragma unroll
  for (int ii = 0; ii < LDG_PER_THREAD; ++ii) {
    row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
  }

  using ComputeType = float;
  using Converter = cutlass::NumericArrayConverter<ComputeType, T, VPT>;
  Converter compute_type_converter;
  cutlass::Array<ComputeType, VPT> row_chunk =
      compute_type_converter(row_chunk_input);

  // First, we perform a max reduce within the thread. We can do the max in fp16
  // safely (I think) and just convert to float afterwards for the exp + sum
  // reduction.
  ComputeType thread_max = row_chunk[0];
#pragma unroll
  for (int ii = 1; ii < VPT; ++ii) {
    thread_max = max(thread_max, row_chunk[ii]);
  }

// Now, we find the max within the thread group and distribute among the
// threads. We use a butterfly reduce.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    thread_max =
        max(thread_max,
            __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
  }

  // From this point, thread max in all the threads have the max within the row.
  // Now, we subtract the max from each element in the thread and take the exp.
  // We also compute the thread local sum.
  float row_sum = 0;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = expf(row_chunk[ii] - thread_max);
    row_sum += row_chunk[ii];
  }

// Now, we perform the sum reduce within each thread group. Similar to the max
// reduce, we use a bufferfly pattern.
#pragma unroll
  for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
    row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
  }

  // From this point, all threads have the max and the sum for their rows in the
  // thread_max and thread_sum variables respectively. Finally, we can scale the
  // rows for the softmax. Technically, for top-k gating we don't need to
  // compute the entire softmax row. We can likely look at the maxes and only
  // compute for the top-k values in the row. However, this kernel will likely
  // not be a bottle neck and it seems better to closer match torch and find the
  // argmax after computing the softmax.
  const float reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    row_chunk[ii] = bias ? row_chunk[ii] * reciprocal_row_sum +
                               bias[first_elt_read_by_thread + ii]
                         : row_chunk[ii] * reciprocal_row_sum;
  }

  // Now, softmax_res contains the softmax of the row chunk. Now, I want to find
  // the topk elements in each row, along with the max index.​
  int start_col = first_elt_read_by_thread;
  static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

  for (int k_idx = 0; k_idx < k; ++k_idx) {
    // First, each thread does the local argmax
    float max_val = row_chunk[0];
    int expert = start_col;
#pragma unroll
    for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD;
         ++ldg, col += COLS_PER_GROUP_LDG) {
#pragma unroll
      for (int ii = 0; ii < ELTS_PER_LDG; ++ii) {
        float val = row_chunk[ldg * ELTS_PER_LDG + ii];

        // No check on the experts here since columns with the smallest index
        // are processed first and only updated if > (not >=)
        if (val > max_val) {
          max_val = val;
          expert = col + ii;
        }
      }
    }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads
// reach consensus about the max. This will be useful for K > 1 so that the
// threads can agree on "who" had the max value. That thread can then blank out
// their max with -inf and the warp can run more iterations...
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2) {
      float other_max =
          __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
      int other_expert =
          __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

      // We want lower indices to "win" in every thread so we break ties this
      // way
      if (other_max > max_val ||
          (other_max == max_val && other_expert < expert)) {
        max_val = other_max;
        expert = other_expert;
      }
    }

    // Write the max for this k iteration to global memory.
    T final_val = bias ? T(max_val) - bias[expert] : T(max_val);
    if (thread_group_idx == 0) {
      // The lead thread from each sub-group will write out the final results to
      // global memory. (This will be a single) thread per row of the
      // input/output matrices.
      const int idx = k * thread_row + k_idx;
      if constexpr (Norm_Weights) {
        const int idx_in_cta = k * thread_row_in_cta + k_idx;
        row_output[idx_in_cta] = final_val;
        weight_sum += final_val;
      } else {
        output[idx] = final_val;
      }
      indices[idx] = should_process_row ? expert : NUM_EXPERTS;
      source_rows[idx] = k_idx * num_rows + thread_row;
    }

    // Finally, we clear the value in the thread with the current max if there
    // is another iteration to run.
    if (k_idx + 1 < k) {
      const int ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
      const int thread_to_clear_in_group =
          (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

      // Only the thread in the group which produced the max will reset the
      // "winning" value to -inf.
      if (thread_group_idx == thread_to_clear_in_group) {
        const int offset_for_expert = expert % ELTS_PER_LDG;
        // Safe to set to any negative value since row_chunk values must be
        // between 0 and 1.
        row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] =
            ComputeType(-10000.f);
      }
    }
  }
  if constexpr (Norm_Weights) {
#pragma unroll
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      if (thread_group_idx == 0) {
        const int idx = k * thread_row + k_idx;
        const int idx_in_cta = k * thread_row_in_cta + k_idx;
        output[idx] = row_output[idx_in_cta] / weight_sum;
      }
    }
  }
}

namespace detail {
// Constructs some constants needed to partition the work across threads at
// compile time.
template <typename T, int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
  static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(T);
  static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 ||
                    EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0,
                "");
  static constexpr int VECs_PER_THREAD =
      std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
  static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
  static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
  static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
}  // namespace detail

template <typename T,
          int EXPERTS,
          int WARPS_PER_TB,
          bool Norm_Weights = false,
          typename IdxT = int>
void topk_gating_softmax_launcher_helper(const T* input,
                                         const T* bias,
                                         T* output,
                                         IdxT* indices,
                                         int* source_row,
                                         const int64_t num_rows,
                                         const int64_t num_experts,
                                         const int64_t k,
                                         cudaStream_t stream) {
  static constexpr uint64_t MAX_BYTES_PER_LDG = 16;
  static constexpr int BYTES_PER_LDG =
      std::min(MAX_BYTES_PER_LDG, sizeof(T) * EXPERTS);
  using Constants = detail::TopkConstants<T, EXPERTS, BYTES_PER_LDG>;
  static constexpr int VPT = Constants::VPT;
  static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
  const int num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
  const int num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

  dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
  static constexpr int ROWS_PER_CTA = WARPS_PER_TB * ROWS_PER_WARP;
  topk_gating_softmax<T,
                      VPT,
                      EXPERTS,
                      WARPS_PER_TB,
                      BYTES_PER_LDG,
                      Norm_Weights>
      <<<num_blocks, block_dim, ROWS_PER_CTA * k * sizeof(T), stream>>>(
          input, bias, output, num_rows, indices, source_row, k);
}

template <typename T, typename IdxT = int>
void topk_gating_softmax_kernelLauncher(const T* input,
                                        const T* gating_correction_bias,
                                        T* output,
                                        T* softmax,
                                        IdxT* indices,
                                        int* source_row,
                                        T* softmax_max_prob,
                                        const int64_t num_rows,
                                        const int64_t num_experts,
                                        const int64_t k,
                                        const bool group_moe,
                                        cudaStream_t stream,
                                        const bool topk_only_mode = false) {
  if (topk_only_mode) {
    static constexpr int TPB = 256;
    const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
    moe_top_k<T, TPB>
        <<<config_topk.block_per_grid, TPB, 0, stream>>>(input,
                                                         gating_correction_bias,
                                                         output,
                                                         indices,
                                                         source_row,
                                                         num_experts,
                                                         k,
                                                         num_rows);
    return;
  }
  static constexpr int WARPS_PER_TB = 4;

#define LAUNCH_TOPK_GATING_SOFTMAX_HELPER(N)                 \
  case N: {                                                  \
    topk_gating_softmax_launcher_helper<T, N, WARPS_PER_TB>( \
        input,                                               \
        gating_correction_bias,                              \
        output,                                              \
        indices,                                             \
        source_row,                                          \
        num_rows,                                            \
        num_experts,                                         \
        k,                                                   \
        stream);                                             \
    break;                                                   \
  }
  int64_t tem_num_experts = num_experts;
  if (gating_correction_bias != nullptr) tem_num_experts = 0;
  switch (tem_num_experts) {
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(2)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(4)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(8)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(16)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(32)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(64)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(128)
    LAUNCH_TOPK_GATING_SOFTMAX_HELPER(256)

    default: {
      static constexpr int TPB = 256;
      if (group_moe) {
        const int group_experts = num_experts / k;
        const int softmax_num_rows = num_rows * k;
        const auto config_softmax = Get1DBlocksAnd2DGridsMoe(softmax_num_rows);
        group_moe_softmax<T, TPB>
            <<<config_softmax.block_per_grid, TPB, 0, stream>>>(
                input,
                softmax,
                softmax_max_prob,
                group_experts,
                softmax_num_rows);
        const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
        group_moe_top_k<T, TPB>
            <<<config_topk.block_per_grid, TPB, 0, stream>>>(softmax,
                                                             output,
                                                             indices,
                                                             source_row,
                                                             softmax_max_prob,
                                                             num_experts,
                                                             k,
                                                             num_rows);
      } else {
        const auto config_topk = Get1DBlocksAnd2DGridsMoe(num_rows);
        moe_softmax<T, TPB><<<config_topk.block_per_grid, TPB, 0, stream>>>(
            input, softmax, num_experts, num_rows);
        moe_top_k<T, TPB><<<config_topk.block_per_grid, TPB, 0, stream>>>(
            softmax,
            gating_correction_bias,
            output,
            indices,
            source_row,
            num_experts,
            k,
            num_rows);
      }
    }
  }
}

// ========================== Permutation things
// =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation
// map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It
// is "expanded" since we will have to duplicate some rows in the input matrix
// to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here
// has indices in the range (0, k*rows_in_input - 1). However, it is set up so
// that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input all map
// to row 0 in the original matrix. Thus, to know where to read in the source
// matrix, we simply take the modulus of the expanded index.

template <typename T, int VecSize, int Kthread, typename OutT = T>
__global__ void initialize_moe_routing_kernel(
    const T* unpermuted_input,
    OutT* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    const int* expert_idx_per_token,
    const float* w4a8_in_scale,
    int* expanded_source_row_to_expanded_dest_row,
    float* dequant_scale,
    const int64_t num_rows,
    const int64_t active_rows,
    const int64_t cols,
    const int64_t num_rows_k) {
  using LoadT = AlignedVector<T, VecSize>;
  LoadT src_vec;

  // Reverse permutation map.
  // I do this so that later, we can use the source -> dest map to do the k-way
  // reduction and unpermuting. I need the reverse map for that reduction to
  // allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
  // thread block will be responsible for all k summations.
  const int expanded_dest_row = blockIdx.x + blockIdx.y * gridDim.x;
  if (expanded_dest_row >= num_rows_k) return;
  const int expanded_source_row =
      expanded_dest_row_to_expanded_source_row[expanded_dest_row];
  if (threadIdx.x == 0) {
    expanded_source_row_to_expanded_dest_row[expanded_source_row] =
        expanded_dest_row;
  }

  extern __shared__ char smem_[];

  T* data_smem = reinterpret_cast<T*>(smem_);

  if (expanded_dest_row < active_rows) {
    const int expert_idx = expert_idx_per_token[expanded_dest_row];
    float scale;
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    OutT* dest_row_ptr = permuted_output + expanded_dest_row * cols;

    if constexpr (std::is_same<OutT, phi::dtype::float8_e4m3fn>::value) {
      if (dequant_scale != nullptr) {
        float abs_max = 0.f;
        for (int tid = threadIdx.x * VecSize; tid < cols;
             tid += blockDim.x * VecSize) {
          Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
          Store<T, VecSize>(src_vec, &data_smem[tid]);
#pragma unroll
          for (int j = 0; j < VecSize; j++) {
            abs_max = fmaxf(abs_max, fabsf(static_cast<float>(src_vec[j])));
          }
        }
        abs_max = BlockAllReduce<MaxOp, float, Kthread>(abs_max);
        scale = 440.0f / abs_max;
        dequant_scale[expanded_dest_row] = abs_max;
        for (int tid = threadIdx.x * VecSize; tid < cols;
             tid += blockDim.x * VecSize) {
          Load<T, VecSize>(&data_smem[tid], &src_vec);
          using StoreT = AlignedVector<OutT, VecSize>;
          StoreT dest_vec;
          for (int j = 0; j < VecSize; j++) {
            float quant_value = scale * static_cast<float>(src_vec[j]);
            dest_vec[j] = static_cast<OutT>(quant_value);
          }
          Store<OutT, VecSize>(dest_vec, &dest_row_ptr[tid]);
        }
        return;
      } else {
        scale = w4a8_in_scale ? w4a8_in_scale[expert_idx] : -1;
      }
    } else {
      scale = w4a8_in_scale ? w4a8_in_scale[expert_idx] : -1;
    }
    for (int tid = threadIdx.x * VecSize; tid < cols;
         tid += blockDim.x * VecSize) {
      // dest_row_ptr[tid] = source_row_ptr[tid];
      Load<T, VecSize>(&source_row_ptr[tid], &src_vec);

      if constexpr (std::is_same<OutT, int8_t>::value) {
        using StoreT = AlignedVector<OutT, VecSize>;
        StoreT dest_vec;
        const float max_bound = 127.f;
        const float min_bound = -127.f;
        for (int j = 0; j < VecSize; j++) {
          float quant_value =
              max_bound * scale * static_cast<float>(src_vec[j]);
          quant_value = quant_value > max_bound ? max_bound : quant_value;
          quant_value = quant_value < min_bound ? min_bound : quant_value;
          dest_vec[j] = static_cast<int8_t>(round(quant_value));
        }
        Store<OutT, VecSize>(dest_vec, &dest_row_ptr[tid]);
      } else if constexpr (std::is_same<OutT,
                                        phi::dtype::float8_e4m3fn>::value) {
        using StoreT = AlignedVector<OutT, VecSize>;
        StoreT dest_vec;
        const float max_bound = 448.f;
        const float min_bound = -448.f;
        for (int j = 0; j < VecSize; j++) {
          float quant_value =
              max_bound * scale * static_cast<float>(src_vec[j]);
          quant_value = quant_value > max_bound ? max_bound : quant_value;
          quant_value = quant_value < min_bound ? min_bound : quant_value;
          dest_vec[j] = static_cast<phi::dtype::float8_e4m3fn>(quant_value);
        }
        Store<phi::dtype::float8_e4m3fn, VecSize>(dest_vec, &dest_row_ptr[tid]);
      } else {
        Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
      }
    }
  }
}

template <typename T, typename OutT = T>
void initialize_moe_routing_kernelLauncher(
    const T* unpermuted_input,
    OutT* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    const int* expert_idx_per_token,
    const float* w4a8_in_scale,
    int* expanded_source_row_to_expanded_dest_row,
    float* dequant_scale,
    const int64_t num_rows,
    const int64_t active_rows,
    const int64_t cols,
    const int64_t k,
    cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int max_pack_size = 16 / sizeof(T);
  const auto config_initialize = Get1DBlocksAnd2DGridsMoe(num_rows * k);
  const int smem_size = cols * sizeof(float);
  auto kernel = &initialize_moe_routing_kernel<T, max_pack_size, threads, OutT>;
  if (cols % max_pack_size != 0) {
    kernel = &initialize_moe_routing_kernel<T, 1, threads, OutT>;
  }
  if (smem_size >= 48 * 1024) {
    cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  }
  kernel<<<config_initialize.block_per_grid, threads, smem_size, stream>>>(
      unpermuted_input,
      permuted_output,
      expanded_dest_row_to_expanded_source_row,
      expert_idx_per_token,
      w4a8_in_scale,
      expanded_source_row_to_expanded_dest_row,
      dequant_scale,
      num_rows,
      k * active_rows,
      cols,
      num_rows * k);
}

// ============================== Infer GEMM sizes
// =================================
__device__ inline int find_total_elts_leq_target(int* sorted_indices,
                                                 const int64_t arr_length,
                                                 const int64_t target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] > target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

void compute_total_rows_before_expert(int* sorted_indices,
                                      const int64_t total_indices,
                                      const int64_t num_experts,
                                      int64_t* total_rows_before_expert,
                                      cudaStream_t stream);

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and
// performs the final skip connection.
template <typename T, int RESIDUAL_NUM>
__global__ void finalize_moe_routing_kernel(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* bias,
    const float* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row,
    const int64_t cols,
    const int64_t k,
    const int64_t compute_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor,
    const int64_t num_rows) {
  const int original_row = blockIdx.x;
  auto const offset = original_row * cols;

  T* reduced_row_ptr = reduced_unpermuted_output + offset;
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / cutlass::sizeof_bits<T>::value;
  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_col = cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  using SharedOutputElem = cutlass::Array<T, FINALIZE_ELEM_PER_THREAD>;

  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v =
      reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_col;
       elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    float row_rescale{0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      int64_t const expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];
      int64_t const k_offset = original_row * k + k_idx;
      const float row_scale = scales[k_offset];
      row_rescale = row_rescale + row_scale;

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

      int const expert_idx = expert_for_source_row[k_offset];
      auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;

      ComputeElem bias_value;
      if (bias) {
        bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      } else {
        bias_value.fill(0);
      }

      ComputeElem expert_result = arrayConvert<InputElem, ComputeElem>(
          expanded_permuted_rows_row_ptr[elem_index]);

      thread_output = thread_output + row_scale * (expert_result + bias_value);
    }
    for (auto& elem : thread_output) {
      elem =
          elem / (norm_topk_prob ? row_rescale : 1.0f) * routed_scaling_factor;
    }
    OutputElem output_elem =
        arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
}

template <typename T>
void finalize_moe_routing_kernelLauncher(
    const T* expanded_permuted_rows,
    T* reduced_unpermuted_output,
    const T* bias,
    const float* scales,
    const int* expanded_source_row_to_expanded_dest_row,
    const int* expert_for_source_row,
    const int64_t num_rows,
    const int64_t cols,
    const int64_t k,
    const int64_t compute_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor,
    cudaStream_t stream) {
  const int blocks = num_rows;
  const int threads = FINALIZE_THREADS_PER_BLOCK;

  finalize_moe_routing_kernel<T, 1>
      <<<blocks, threads, 0, stream>>>(expanded_permuted_rows,
                                       reduced_unpermuted_output,
                                       bias,
                                       scales,
                                       expanded_source_row_to_expanded_dest_row,
                                       expert_for_source_row,
                                       cols,
                                       k,
                                       compute_bias,
                                       norm_topk_prob,
                                       routed_scaling_factor,
                                       num_rows);
}
}  // namespace phi
