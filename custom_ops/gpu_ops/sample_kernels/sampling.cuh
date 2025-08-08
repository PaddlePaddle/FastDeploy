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

// This code is partially inspired by and references the implementation found
// in FlashInfer.Specifically, the implementation of Top-p Sampling
// functionality in this code is inspired by the logic of FlashInfer’s
// flashinfer.sampling.top_p_sampling_from_probs . For more details on
// FlashInfer’s documentation, please refer to:
// https://docs.flashinfer.ai/generated/flashinfer.sampling.top_p_sampling_from_probs.html

#pragma once

#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <numeric>

#include "sample_kernels/utils.cuh"

namespace sampling {

using namespace cub;

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

#if (__CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 >= 120100)
#define SAMPLING_CUB_SUBTRACTLEFT_DEFINED
#endif

template <typename T> struct Pair {
  T value;
  int count;

  __device__ Pair operator+(const Pair &other) const {
    return {value + other.value, count + other.count};
  }
  __device__ Pair &operator+=(const Pair &other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

template <typename T>
struct ValueCount {
  T value;
  int count;

  __device__ ValueCount operator+(const ValueCount& other) const {
    return {value + other.value, count + other.count};
  }
  __device__ ValueCount& operator+=(const ValueCount& other) {
    value += other.value;
    count += other.count;
    return *this;
  }
};

struct BoolDiffOp {
  __device__ __forceinline__ bool operator()(const bool &lhs,
                                             const bool &rhs) const {
    return lhs != rhs;
  }
};

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM>
struct SamplingTempStorage {
  union {
    float deterministic_scan[BLOCK_THREADS / 32];
    typename BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>::TempStorage scan;
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_value_count;
    typename BlockAdjacentDifference<bool, BLOCK_THREADS>::TempStorage adj_diff;
  } block_prim;
  struct {
    int32_t sampled_id;
    int32_t last_valid_id;
    float max_val;
    union {
      float value;
      ValueCount<float> pair;
    } block_aggregate;
  };
};

/*!
 * \brief Deterministic inclusive scan implementation, use Belloch scan
 * algorithm. \note This implementation is slower than the cub::BlockScan, but
 * it is deterministic.
 */
template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, typename T>
__device__ __forceinline__ void
DeterministicInclusiveSum(const T *in_data, T *out_data,
                          SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM,
                                              REDUCE_ALGORITHM> *temp_storage) {
  T *smem_prefix_sum = temp_storage->block_prim.deterministic_scan;
  T thread_data[VEC_SIZE];
  T thread_sum = 0;
#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    thread_sum += in_data[i];
    thread_data[i] = thread_sum;
  }

  T thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < 32; offset *= 2) {
    T tmp = __shfl_up_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }

  T warp_sum = __shfl_sync(0xffffffff, thread_exclusive_prefix_sum,
                           threadIdx.x | 0xffffffff);
  if (threadIdx.x % 32 == 31) {
    thread_exclusive_prefix_sum = 0;
  }

#pragma unroll
  for (uint32_t offset = 16; offset >= 1; offset /= 2) {
    T tmp = __shfl_xor_sync(0xffffffff, thread_exclusive_prefix_sum, offset);
    if ((threadIdx.x + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum = tmp + thread_exclusive_prefix_sum;
    }
    if ((threadIdx.x + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[threadIdx.x / 32] = warp_sum;
  __syncthreads();

  if (threadIdx.x < 32) {
    T warp_exclusive_prefix_sum =
        (threadIdx.x < BLOCK_THREADS / 32) ? smem_prefix_sum[threadIdx.x] : 0;

#pragma unroll
    for (uint32_t offset = 1; offset < 32; offset *= 2) {
      T tmp = __shfl_up_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }

    if (threadIdx.x % 32 == 31) {
      warp_exclusive_prefix_sum = 0;
    }

#pragma unroll
    for (uint32_t offset = 16; offset >= 1; offset /= 2) {
      T tmp = __shfl_xor_sync(0xffffffff, warp_exclusive_prefix_sum, offset);
      if ((threadIdx.x + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum = tmp + warp_exclusive_prefix_sum;
      }
      if ((threadIdx.x + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (threadIdx.x < BLOCK_THREADS / 32) {
      smem_prefix_sum[threadIdx.x] = warp_exclusive_prefix_sum;
    }
  }
  __syncthreads();

#pragma unroll
  for (uint32_t i = 0; i < VEC_SIZE; ++i) {
    out_data[i] = smem_prefix_sum[threadIdx.x / 32] +
                  thread_exclusive_prefix_sum + thread_data[i];
  }
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, bool DETERMINISTIC, typename Predicate>
__device__ __forceinline__ void DeviceSamplingFromProb(
    uint32_t i, uint32_t d, Predicate pred, float u, vec_t<float, VEC_SIZE> prob_vec,
    float& aggregate,
    SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>* temp_storage) {
  const uint32_t tx = threadIdx.x;
  float prob_greater_than_threshold[VEC_SIZE];
  float inclusive_cdf[VEC_SIZE];
  bool greater_than_u[VEC_SIZE], valid[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    prob_greater_than_threshold[j] = pred(prob_vec[j]) ? prob_vec[j] : 0;
    valid[j] = pred(prob_vec[j]) && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d;
  }
#ifdef PADDLE_WITH_COREX
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .Sum(prob_greater_than_threshold);
#else
  float aggregate_local =
      BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce)
          .Sum<VEC_SIZE>(prob_greater_than_threshold);
#endif
  if (tx == 0) {
    temp_storage->block_aggregate.value = aggregate_local;
  }
  __syncthreads();
  aggregate_local = temp_storage->block_aggregate.value;

  if (aggregate + aggregate_local > u) {
    if constexpr (DETERMINISTIC) {
      DeterministicInclusiveSum<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>(
          prob_greater_than_threshold, inclusive_cdf, temp_storage);
    } else {
#ifdef PADDLE_WITH_COREX
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .InclusiveSum(prob_greater_than_threshold, inclusive_cdf);
#else
      BlockScan<float, BLOCK_THREADS, SCAN_ALGORITHM>(temp_storage->block_prim.scan)
          .InclusiveSum<VEC_SIZE>(prob_greater_than_threshold, inclusive_cdf);
#endif

      __syncthreads();
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      greater_than_u[j] = (inclusive_cdf[j] + aggregate > u) && valid[j];
    }

    bool greater_than_u_diff[VEC_SIZE];
#ifdef SAMPLING_CUB_SUBTRACTLEFT_DEFINED
    #ifdef PADDLE_WITH_COREX
      BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
          .SubtractLeft(greater_than_u, greater_than_u_diff, BoolDiffOp());
    #else
      BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
          .SubtractLeft<VEC_SIZE>(greater_than_u, greater_than_u_diff, BoolDiffOp());
    #endif
#else
    #ifdef PADDLE_WITH_COREX
      BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
          .FlagHeads(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
    #else
      BlockAdjacentDifference<bool, BLOCK_THREADS>(temp_storage->block_prim.adj_diff)
          .FlagHeads<VEC_SIZE>(greater_than_u_diff, greater_than_u, BoolDiffOp(), 0);
    #endif
#endif
    __syncthreads();

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (greater_than_u_diff[j]) {
        atomicMin(&(temp_storage->sampled_id), (i * BLOCK_THREADS + tx) * VEC_SIZE + j);
      }
    }
    __syncthreads();
  }

  // update the last valid index
  int valid_index[VEC_SIZE];
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    if (valid[j]) {
      valid_index[j] = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    } else {
      valid_index[j] = -1;
    }
  }
  int max_valid_index =
      BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage->block_prim.reduce_int)
          .Reduce(valid_index, cub::Max());
  if (tx == 0 && max_valid_index != -1) {
    temp_storage->last_valid_id = max_valid_index;
  }
  __syncthreads();
  aggregate += aggregate_local;
}




template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType, typename IdType>
__global__ void TopKTopPSamplingFromProbKernel(DType* probs, IdType* output,
                                               float* top_p_arr, IdType* top_k_arr,
                                               uint32_t d, uint64_t philox_seed,
                                               uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = bx;
  const uint32_t k = top_k_arr[row_idx] == 0 ? d : top_k_arr[row_idx];
  const float p = top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      ValueCount<float> probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = {
            (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
        probs_gt_pivot_1[j] = {
            (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
            (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
      }

#ifdef PADDLE_WITH_COREX
      aggregate_gt_pivot_0 +=
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum(probs_gt_pivot_0);
#else
      aggregate_gt_pivot_0 +=
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum<VEC_SIZE>(probs_gt_pivot_0);
#endif
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot_0;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.pair;

#ifdef PADDLE_WITH_COREX
      aggregate_gt_pivot_1 +=
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum(probs_gt_pivot_1);
#else
      aggregate_gt_pivot_1 +=
          BlockReduce<ValueCount<float>, BLOCK_THREADS>(temp_storage.block_prim.reduce_value_count)
              .Sum<VEC_SIZE>(probs_gt_pivot_1);
#endif
      if (tx == 0) {
        temp_storage.block_aggregate.pair = aggregate_gt_pivot_1;
      }
      __syncthreads();
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.pair;
    }
    if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1.count < k && aggregate_gt_pivot_1.value < p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0.value;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1.value;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}



template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          bool DETERMINISTIC, typename DType, typename IdType>
__global__ void TopPSamplingFromProbKernel(DType* probs, IdType* output,
                                           float* top_p_arr, uint32_t d,
                                           uint64_t philox_seed, uint64_t philox_offset) {
  const uint32_t batch_size = gridDim.x;
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  curandStatePhilox4_32_10_t state;
  curand_init(philox_seed, bx, philox_offset, &state);
  const uint32_t row_idx = bx;
  float top_p = top_p_arr[row_idx];

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  vec_t<float, VEC_SIZE> probs_vec;
  float aggregate;
  float q = 1;
  double low = 0, high = 1.f;
  int sampled_id;
  do {
    temp_storage.sampled_id = d;
    __syncthreads();
    float u = curand_uniform(&state) * q;
    aggregate = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM,
                             DETERMINISTIC>(
          i, d, [&](float x) { return x > low; }, u, probs_vec, aggregate, &temp_storage);
      if (aggregate > u) {
        break;
      }
    }
    __syncthreads();
    sampled_id = temp_storage.sampled_id;
    if (sampled_id == d) {
      // NOTE(Zihao): this would happen when u is very close to 1
      // and the sum of probabilities is smaller than u
      // In this case, we use the last valid index as the sampled id
      sampled_id = temp_storage.last_valid_id;
    }
    double pivot_0 = probs[row_idx * d + sampled_id];
    double pivot_1 = (pivot_0 + high) / 2;

    float aggregate_gt_pivot_0 = 0, aggregate_gt_pivot_1 = 0;
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
      probs_vec.fill(0);
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
      }

      float probs_gt_pivot_0[VEC_SIZE], probs_gt_pivot_1[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        probs_gt_pivot_0[j] = (probs_vec[j] > pivot_0) ? probs_vec[j] : 0;
        probs_gt_pivot_1[j] = (probs_vec[j] > pivot_1) ? probs_vec[j] : 0;
      }

#ifdef PADDLE_WITH_COREX
      aggregate_gt_pivot_0 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                  .Sum(probs_gt_pivot_0);
#else
      aggregate_gt_pivot_0 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                  .Sum<VEC_SIZE>(probs_gt_pivot_0);
#endif
      if (tx == 0) {
        temp_storage.block_aggregate.value = aggregate_gt_pivot_0;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.value;

#ifdef PADDLE_WITH_COREX
      aggregate_gt_pivot_1 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                  .Sum(probs_gt_pivot_1);
#else
      aggregate_gt_pivot_1 += BlockReduce<float, BLOCK_THREADS>(temp_storage.block_prim.reduce)
                                  .Sum<VEC_SIZE>(probs_gt_pivot_1);
#endif
      if (tx == 0) {
        temp_storage.block_aggregate.value = aggregate_gt_pivot_1;
      }
      __syncthreads();
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.value;
    }
    if (aggregate_gt_pivot_0 < top_p) {
      // case 1: pivot_0 accepted
      break;
    }
    if (aggregate_gt_pivot_1 < top_p) {
      // case 2: pivot_0 rejected, pivot_1 accepted
      low = pivot_0;
      high = pivot_1;
      q = aggregate_gt_pivot_0;
    } else {
      // case 3: pivot_0 rejected, pivot_1 rejected
      low = pivot_1;
      q = aggregate_gt_pivot_1;
    }
  } while (low < high);
  __syncthreads();
  if (tx == 0) {
    output[bx] = sampled_id;
  }
}

template <uint32_t VEC_SIZE, uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM,
          typename TempStorage>
__device__ __forceinline__ float GetMaxValue(float* in_data, uint32_t row_idx, uint32_t d,
                                             TempStorage& temp_storage) {
  const uint32_t tx = threadIdx.x;
  vec_t<float, VEC_SIZE> in_data_vec;

  float max_val = 0;
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    in_data_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      in_data_vec.cast_load(in_data + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }
    float in_data_[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      in_data_[j] = in_data_vec[j];
    }
#ifdef PADDLE_WITH_COREX
    max_val = max(
        max_val, BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce(in_data_, cub::Max()));
#else
    max_val = max(
        max_val, BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                     .Reduce<VEC_SIZE>(in_data_, cub::Max()));
#endif
    __syncthreads();
  }
  if (tx == 0) {
    temp_storage.max_val = max_val;
  }
  __syncthreads();
  return temp_storage.max_val;
}

template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM>
struct RenormTempStorage {
  union {
    typename BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce;
    typename BlockReduce<int, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage reduce_int;
    typename BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>::TempStorage
        reduce_value_count;
  } block_prim;
  struct {
    float max_val;
    float min_val;
    union {
      struct {
        float values[2];
      };
      struct {
        int counts[2];
      };
      struct {
        ValueCount<float> pairs[2];
      };
    } block_aggregate;
  };
};

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE, bool DETERMINISTIC,
          typename DType,typename IdType>
__global__ void MinPSamplingFromProbKernel(DType* probs, const float* min_p_arr,
                                            DType* renormed_prob,uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  float p = (min_p_arr == nullptr) ? 0 : min_p_arr[bx];
  const uint32_t row_idx = bx;

  extern __shared__ __align__(
      alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
      uint8_t smem_sampling[];
  auto& temp_storage =
      reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(
          smem_sampling);

  float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                              SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>>(
      probs, row_idx, d, temp_storage);
  float pivot = max_val * p;

  vec_t<float, VEC_SIZE> probs_vec;
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE);
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = (probs_vec[j] >= pivot) ? probs_vec[j] : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }

  }
}


template <uint32_t BLOCK_THREADS, BlockReduceAlgorithm REDUCE_ALGORITHM, uint32_t VEC_SIZE,
          typename DType, typename IdType>
__global__ void TopKRenormProbKernel(DType* probs, DType* renormed_prob, IdType* top_k_arr, uint32_t d) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = bx;
  const uint32_t k = top_k_arr[row_idx] == 0 ? d : top_k_arr[row_idx];
#ifdef PADDLE_WITH_COREX
  double pivot = std::numeric_limits<float>::infinity(), normalizer = 1;
#else
  double pivot = -cuda::std::numeric_limits<float>::infinity(), normalizer = 1;
#endif
  vec_t<float, VEC_SIZE> probs_vec;
  if (k < d) {
    extern __shared__ __align__(alignof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>))
        uint8_t smem_renorm[];
    auto& temp_storage =
        reinterpret_cast<RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>&>(smem_renorm);
    temp_storage.max_val = 0;

    float max_val = GetMaxValue<VEC_SIZE, BLOCK_THREADS, REDUCE_ALGORITHM,
                                RenormTempStorage<BLOCK_THREADS, REDUCE_ALGORITHM>>(
        probs, row_idx, d, temp_storage);

    double low = 0, high = max_val;
    float min_gt_low, max_le_high;
    float sum_low = 1;
    // f(x) = len(nonzero(probs > x)), f(x) is non-increasing
    // min_gt_low = min{p \in probs | p > low}, max_le_high = max{p \in probs | p <= high}
    // loop invariant:
    // - f(low) >= k, f(high) < k
    // - f(low) > f(min_gt_low) >= f(max_le_high) == f(high)
    // stopping condition: min_gt_low == max_le_high
    // - f(low) >= k, f(min_gt_low) == f(max_le_high) == f(high) < k
    do {
      double pivot_0 = (high + 2 * low) / 3;
      double pivot_1 = (2 * high + low) / 3;

      ValueCount<float> aggregate_gt_pivot_0{0, 0}, aggregate_gt_pivot_1{0, 0};
      min_gt_low = high;
      max_le_high = low;
#pragma unroll 2
      for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
        probs_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
          probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
        }
        ValueCount<float> probs_gt_pivot_0_pair[VEC_SIZE], probs_gt_pivot_1_pair[VEC_SIZE];
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j) {
          probs_gt_pivot_0_pair[j] = {
              (probs_vec[j] > pivot_0) ? probs_vec[j] : 0,
              (probs_vec[j] > pivot_0 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};
          probs_gt_pivot_1_pair[j] = {
              (probs_vec[j] > pivot_1) ? probs_vec[j] : 0,
              (probs_vec[j] > pivot_1 && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d)};

          if (probs_vec[j] > low && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
            min_gt_low = min(min_gt_low, probs_vec[j]);
          }
          if (probs_vec[j] <= high && (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d) {
            max_le_high = max(max_le_high, probs_vec[j]);
          }
        }

#ifdef PADDLE_WITH_COREX
        aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .Sum(probs_gt_pivot_0_pair);
#else
        aggregate_gt_pivot_0 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .Sum<VEC_SIZE>(probs_gt_pivot_0_pair);
#endif
        __syncthreads();

#ifdef PADDLE_WITH_COREX
        aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .Sum(probs_gt_pivot_1_pair);
#else
        aggregate_gt_pivot_1 += BlockReduce<ValueCount<float>, BLOCK_THREADS, REDUCE_ALGORITHM>(
                                    temp_storage.block_prim.reduce_value_count)
                                    .Sum<VEC_SIZE>(probs_gt_pivot_1_pair);
#endif
        __syncthreads();
      }
      min_gt_low =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Reduce(min_gt_low, cub::Min());
      __syncthreads();
      max_le_high =
          BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
              .Reduce(max_le_high, cub::Max());
      if (tx == 0) {
        temp_storage.block_aggregate.pairs[0] = aggregate_gt_pivot_0;
        temp_storage.block_aggregate.pairs[1] = aggregate_gt_pivot_1;
        temp_storage.min_val = min_gt_low;
        temp_storage.max_val = max_le_high;
      }
      __syncthreads();
      aggregate_gt_pivot_0 = temp_storage.block_aggregate.pairs[0];
      aggregate_gt_pivot_1 = temp_storage.block_aggregate.pairs[1];
      min_gt_low = temp_storage.min_val;
      max_le_high = temp_storage.max_val;

      if (aggregate_gt_pivot_1.count >= k) {
        low = pivot_1;
        sum_low = float(aggregate_gt_pivot_1.value);
      } else if (aggregate_gt_pivot_0.count >= k) {
        low = pivot_0;
        high = min(pivot_1, max_le_high);
        sum_low = float(aggregate_gt_pivot_0.value);
      } else {
        high = min(pivot_0, max_le_high);
      }
    } while (min_gt_low != max_le_high);

    normalizer = ptx_rcp(max(sum_low, 1e-8));
    pivot = low;
  }

  // normalize
#pragma unroll 2
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    probs_vec.fill(0);
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.cast_load(probs + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      probs_vec[j] = (probs_vec[j] > pivot) ? probs_vec[j] * normalizer : 0;
    }
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      probs_vec.store(renormed_prob + row_idx * d + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
    }
  }
}

template <typename T, typename IdType>
cudaError_t TopPSamplingFromProb(T *probs, IdType *output,
                                 uint32_t batch_size, const T *top_p_val,
                                 uint32_t d, bool deterministic,
                                 uint64_t philox_seed, uint64_t philox_offset,
                                 cudaStream_t stream = 0) {
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size =
      sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs,     &output,       &top_p_val,
                  &d,         &philox_seed,  &philox_offset};

  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE,
      {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        auto kernel =
            TopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                       VEC_SIZE, DETERMINISTIC, T, IdType>;
        CUDA_CALL(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args,
                                   smem_size, stream));
      })});
  return cudaSuccess;
}

template <typename T,typename IdType>
cudaError_t MinPSamplingFromProb(T *probs, const T* min_p_arr,T *renormed_prob,
                                 uint32_t batch_size,
                                 uint32_t d, bool deterministic,
                                 cudaStream_t stream = 0){
  constexpr uint32_t BLOCK_THREADS = 1024;
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
  dim3 nblks(batch_size);
  dim3 nthrs(BLOCK_THREADS);
  void* args[] = {&probs, &min_p_arr,&renormed_prob,&d};
  DISPATCH_ALIGNED_VEC_SIZE(
      vec_size, VEC_SIZE,
      {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
        auto kernel =
            MinPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                       VEC_SIZE, DETERMINISTIC, T,IdType>;
        CUDA_CALL(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        CUDA_CALL(cudaLaunchKernel((void *)kernel, nblks, nthrs, args,
                                   smem_size, stream));
      })});
  return cudaSuccess;
}


template <typename T, typename IdType>
cudaError_t TopKTopPSamplingFromProb(T *probs, IdType *output,
                                     uint32_t batch_size, const T *top_p_val, const IdType *top_k_val,
                                     uint32_t d, bool deterministic,
                                     uint64_t philox_seed, uint64_t philox_offset,
                                     cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs,     &output,       &top_p_val, &top_k_val,
                    &d,         &philox_seed,  &philox_offset};

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = TopKTopPSamplingFromProbKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO,
                                                       VEC_SIZE, DETERMINISTIC, T, IdType>;
          CUDA_CALL(
              cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
          CUDA_CALL(
              cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
        })});
    return cudaSuccess;
  });
}

template <typename DType, typename IdType>
cudaError_t TopKRenormProb(DType* probs, DType* renormed_prob, IdType* top_k_arr,
                           uint32_t batch_size, uint32_t d,
                           cudaStream_t stream = 0) {
  const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    const uint32_t smem_size = sizeof(RenormTempStorage<BLOCK_THREADS, REDUCE_ALGO>);
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&probs, &renormed_prob, &top_k_arr, &d};
    DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {
      auto kernel = TopKRenormProbKernel<BLOCK_THREADS, REDUCE_ALGO, VEC_SIZE, DType, IdType>;
      CUDA_CALL(
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
      CUDA_CALL(cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    });
    return cudaSuccess;
  });
}

} // namespace sampling
