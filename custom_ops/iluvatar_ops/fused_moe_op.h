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
#include "fused_moe_helper.h"
#include "fused_moe_imp_op.h"
// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wunused-function"

// #include "paddle/phi/backends/gpu/gpu_info.h"
#pragma GCC diagnostic pop

#include "helper.h"

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

template <typename T, int TPB, typename IdxT = int>
__launch_bounds__(TPB) __global__ void moe_top_k(const T* inputs_after_softmax,
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
      output[idx] =
          bias ? inputs_after_softmax[thread_read_offset + result_kvp.key]
               : result_kvp.value;
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;
    }
    __syncthreads();
  }
}

template <typename T, int TPB, typename IdxT = int>
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
      output[cur_idx] =
          bias ? (result_kvp.value - bias[result_kvp.key]) : result_kvp.value;
      indices[cur_idx] = result_kvp.key;
      source_rows[cur_idx] = k_idx * num_rows + globalIdx;
    }
    __syncthreads();
  }
}

template <typename T, int TPB, typename IdxT = int>
__launch_bounds__(TPB) __global__
    void moe_top_k_normed(const T* inputs_after_softmax,
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
      // output[idx] = bias ? inputs_after_softmax[thread_read_offset +
      // result_kvp.key]: result_kvp.value;
      indices[idx] = should_process_row ? result_kvp.key : num_experts;
      source_rows[idx] = k_idx * num_rows + block_row;

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

template <typename T, int TPB, typename IdxT = int>
__launch_bounds__(TPB) __global__
    void moe_softmax_top_k_normed_fused(const T* input,
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
  extern __shared__ char smem[];
  T* row_outputs = reinterpret_cast<T*>(smem);

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

      T row_out =
          bias ? (result_kvp.value - bias[result_kvp.key]) : result_kvp.value;
      row_outputs[k_idx] = row_out;
      weight_sum += row_out;

      indices[cur_idx] = result_kvp.key;
      source_rows[cur_idx] = k_idx * num_rows + globalIdx;
    }
    __syncthreads();
  }

  if (threadIdx.x < WARP_SIZE) {
    weight_sum = __shfl_sync(0xffffffff, weight_sum, 0);
  }

  if (threadIdx.x < k) {
    output[k * globalIdx + threadIdx.x] = row_outputs[threadIdx.x] / weight_sum;
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

#define LAUNCH_TOPK_GATING_SOFTMAX_HELPER(N)                                   \
  case N: {                                                                    \
    topk_gating_softmax_launcher_helper<T, N, WARPS_PER_TB>(                   \
        input, output, indices, source_row, num_rows, num_experts, k, stream); \
    break;                                                                     \
  }
  int64_t tem_num_experts = num_experts;
  if (gating_correction_bias != nullptr) tem_num_experts = 0;
  switch (tem_num_experts) {
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(2)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(4)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(8)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(16)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(32)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(64)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(128)
      // LAUNCH_TOPK_GATING_SOFTMAX_HELPER(256)

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
        moe_top_k<T, TPB>
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

template <typename T, int VecSize>
__global__ void initialize_moe_routing_kernel(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
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

  if ((blockIdx.x + blockIdx.y * gridDim.x) < active_rows) {
    // Duplicate and permute rows
    const int source_row = expanded_source_row % num_rows;

    const T* source_row_ptr = unpermuted_input + source_row * cols;
    T* dest_row_ptr = permuted_output + expanded_dest_row * cols;

    for (int tid = threadIdx.x * VecSize; tid < cols;
         tid += blockDim.x * VecSize) {
      // dest_row_ptr[tid] = source_row_ptr[tid];
      Load<T, VecSize>(&source_row_ptr[tid], &src_vec);
      Store<T, VecSize>(src_vec, &dest_row_ptr[tid]);
    }
  }
}

template <typename T>
void initialize_moe_routing_kernelLauncher(
    const T* unpermuted_input,
    T* permuted_output,
    const int* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row,
    const int64_t num_rows,
    const int64_t active_rows,
    const int64_t cols,
    const int64_t k,
    cudaStream_t stream) {
  const int threads = std::min(cols, int64_t(1024));
  constexpr int max_pack_size = 16 / sizeof(T);
  const auto config_initialize = Get1DBlocksAnd2DGridsMoe(num_rows * k);
  if (cols % max_pack_size == 0) {
    initialize_moe_routing_kernel<T, max_pack_size>
        <<<config_initialize.block_per_grid, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            num_rows,
            k * active_rows,
            cols,
            num_rows * k);
  } else {
    initialize_moe_routing_kernel<T, 1>
        <<<config_initialize.block_per_grid, threads, 0, stream>>>(
            unpermuted_input,
            permuted_output,
            expanded_dest_row_to_expanded_source_row,
            expanded_source_row_to_expanded_dest_row,
            num_rows,
            k * active_rows,
            cols,
            num_rows * k);
  }
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
  const int original_row = blockIdx.x + blockIdx.y * gridDim.x;
  // const int original_row = blockIdx.x;
  // const int num_rows = gridDim.x;
  if (original_row >= num_rows) return;
  T* reduced_row_ptr = reduced_unpermuted_output + original_row * cols;

  for (int tid = threadIdx.x; tid < cols; tid += blockDim.x) {
    T thread_output{0.f};
    float row_rescale{0.f};
    for (int k_idx = 0; k_idx < k; ++k_idx) {
      const int expanded_original_row = original_row + k_idx * num_rows;
      const int expanded_permuted_row =
          expanded_source_row_to_expanded_dest_row[expanded_original_row];

      const int64_t k_offset = original_row * k + k_idx;
      const float row_scale = scales[k_offset];
      row_rescale = row_rescale + row_scale;

      const T* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows + expanded_permuted_row * cols;

      const int expert_idx = expert_for_source_row[k_offset];
      const T* bias_ptr = bias ? bias + expert_idx * cols : nullptr;
      const T bias_value = bias_ptr ? bias_ptr[tid] : T{0.f};

      thread_output =
          static_cast<float>(thread_output) +
          row_scale * static_cast<float>(
                          expanded_permuted_rows_row_ptr[tid] +
                          bias_value *
                              static_cast<T>(static_cast<float>(compute_bias)));
    }

    thread_output = static_cast<float>(thread_output) /
                    (norm_topk_prob ? row_rescale : 1.0f) *
                    routed_scaling_factor;
    reduced_row_ptr[tid] = thread_output;
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
  const int threads = std::min(cols, int64_t(1024));
  const auto config_final = Get1DBlocksAnd2DGridsMoe(num_rows);

  finalize_moe_routing_kernel<T, 1>
      <<<config_final.block_per_grid, threads, 0, stream>>>(
          expanded_permuted_rows,
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

// ========================= TopK Softmax specializations
// ===========================
template void topk_gating_softmax_kernelLauncher(const float*,
                                                 const float*,
                                                 float*,
                                                 float*,
                                                 int*,
                                                 int*,
                                                 float*,
                                                 const int64_t,
                                                 const int64_t,
                                                 const int64_t,
                                                 const bool,
                                                 cudaStream_t,
                                                 const bool);
template void topk_gating_softmax_kernelLauncher(const half*,
                                                 const half*,
                                                 half*,
                                                 half*,
                                                 int*,
                                                 int*,
                                                 half*,
                                                 const int64_t,
                                                 const int64_t,
                                                 const int64_t,
                                                 const bool,
                                                 cudaStream_t,
                                                 const bool);
#ifdef PADDLE_CUDA_BF16
template void topk_gating_softmax_kernelLauncher(const __nv_bfloat16*,
                                                 const __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 __nv_bfloat16*,
                                                 int*,
                                                 int*,
                                                 __nv_bfloat16*,
                                                 const int64_t,
                                                 const int64_t,
                                                 const int64_t,
                                                 const bool,
                                                 cudaStream_t,
                                                 const bool);
#endif
// ===================== Specializations for init routing
// =========================
template void initialize_moe_routing_kernelLauncher(const float*,
                                                    float*,
                                                    const int*,
                                                    int*,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    cudaStream_t);
template void initialize_moe_routing_kernelLauncher(const half*,
                                                    half*,
                                                    const int*,
                                                    int*,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    cudaStream_t);
#ifdef PADDLE_CUDA_BF16
template void initialize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                    __nv_bfloat16*,
                                                    const int*,
                                                    int*,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    const int64_t,
                                                    cudaStream_t);
#endif
// ==================== Specializations for final routing
// ===================================
template void finalize_moe_routing_kernelLauncher(const float*,
                                                  float*,
                                                  const float*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const bool,
                                                  const float,
                                                  cudaStream_t);
template void finalize_moe_routing_kernelLauncher(const half*,
                                                  half*,
                                                  const half*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const bool,
                                                  const float,
                                                  cudaStream_t);
#ifdef PADDLE_CUDA_BF16
template void finalize_moe_routing_kernelLauncher(const __nv_bfloat16*,
                                                  __nv_bfloat16*,
                                                  const __nv_bfloat16*,
                                                  const float*,
                                                  const int*,
                                                  const int*,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const int64_t,
                                                  const bool,
                                                  const float,
                                                  cudaStream_t);
#endif

}  // namespace phi
