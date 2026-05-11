
//  Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//  Reference
//  https://raw.githubusercontent.com/sgl-project/sglang/refs/heads/main/sgl-kernel/csrc/moe/moe_align_kernel.cu
//  Licensed under Apache License 2.0
//  with further performance optimizations applied.

#include <cooperative_groups.h>

#include "helper.h"
#include "paddle/extension.h"

#define VEC_SIZE 4
using Vec = int4;

template <typename scalar_t>
__global__ void count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel) {
  const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t stride = blockDim.x * gridDim.x;

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
    sorted_token_ids[rank_post_pad] = i;
  }
}

#ifdef __CUDA_ARCH__
__device__ __forceinline__ int warp_exclusive_scan(
    int v, unsigned mask = 0xffffffffu) {
  int original = v;
#pragma unroll
  for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
    int n = __shfl_up_sync(mask, v, offset);
    if ((threadIdx.x & (WARP_SIZE - 1)) >= offset) v += n;
  }
  return v - original;
}
#endif

template <typename scalar_t>
__global__ void moe_align_block_size_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum,
    bool pad_sorted_token_ids,
    const int32_t scan_size,
    int32_t max_num_tokens_padded) {
  // Use a separate thread block to populate sorted_token_ids
  if (blockIdx.x == 1) {
    if (pad_sorted_token_ids) {
      Vec fill_vec;
      fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w = numel;
      int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
      Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
      for (int32_t i = threadIdx.x; i < total_vecs; i += blockDim.x) {
        out_ptr[i] = fill_vec;
      }
    }
    return;
  }

  extern __shared__ int32_t smem[];
  int32_t* shared_counts = smem;                  // [num_experts]
  int32_t* prefix = shared_counts + num_experts;  // [num_experts + 1]
  int32_t* scan_buf = prefix + num_experts + 1;   // [scan_size]
  __shared__ int32_t s_total_tokens_post_pad;

  const size_t tid = threadIdx.x;
  const size_t stride = blockDim.x;

  if (tid < num_experts) {
    shared_counts[tid] = 0;
  }

  __syncthreads();

  for (size_t i = tid; i < numel; i += stride) {
    int expert_id = topk_ids[i] + 1;
    atomicAdd(&shared_counts[expert_id], 1);
  }

  __syncthreads();

  int32_t padded_count = 0;
  if (tid < num_experts) {
    int32_t count = shared_counts[tid];
    padded_count = (count + block_size - 1) / block_size * block_size;
    scan_buf[tid] = padded_count;
  }

#ifndef __CUDA_ARCH__  // HIP

  if (tid >= num_experts && tid < scan_size) {
    scan_buf[tid] = 0;
  }

  __syncthreads();

  // Blelloch scan
  int offset = 1;
#pragma unroll
  for (int d = scan_size >> 1; d > 0; d >>= 1) {
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      scan_buf[bi] += scan_buf[ai];
    }
    offset <<= 1;
    __syncthreads();
  }

  // down-sweep
  if (tid == 0) {
    prefix[num_experts] = scan_buf[scan_size - 1];
    scan_buf[scan_size - 1] = 0;
  }
  __syncthreads();

#pragma unroll
  for (int d = 1; d < scan_size; d <<= 1) {
    offset >>= 1;
    if (tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      if (bi < scan_size) {
        int temp = scan_buf[ai];
        scan_buf[ai] = scan_buf[bi];
        scan_buf[bi] += temp;
      }
    }
    __syncthreads();
  }

  if (tid < num_experts) {
    prefix[tid] = scan_buf[tid];
  }

  if (tid == 0) {
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

#else  // CUDA

  // Intra warp prefix sum
  int32_t* warp_sums = scan_buf + scan_size;  // [<= 32]
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid & (WARP_SIZE - 1);
  const int num_warps_for_scan = (scan_size + WARP_SIZE - 1) / WARP_SIZE;
  const int warp_sum = warp_exclusive_scan(padded_count) + padded_count;
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = warp_sum;
  __syncthreads();

  // warp0 accumulate all the block's prefix sum
  if (tid < WARP_SIZE) {
    int val = (tid < num_warps_for_scan) ? warp_sums[tid] : 0;
    int incl = warp_exclusive_scan(val) + val;
    warp_sums[tid] = incl;
  }
  __syncthreads();

  // Every thread obtains the whole block's sum
  if (tid == 0) {
    prefix[num_experts] = warp_sums[num_warps_for_scan - 1];
    s_total_tokens_post_pad = prefix[num_experts];
    *total_tokens_post_pad = s_total_tokens_post_pad;
  }
  __syncthreads();

  // Fill 0 to scan_buf extended area (tid >= num_expert)
  if (tid >= num_experts && tid < scan_size) scan_buf[tid] = 0;
  __syncthreads();

  // Perform 2 level exclusive-prefix-sum to scan_buf
  int v = (tid < scan_size) ? scan_buf[tid] : 0;
  int pre = warp_exclusive_scan(v);
  if (lane_id == WARP_SIZE - 1) warp_sums[warp_id] = pre + v;
  __syncthreads();

  if (warp_id == 0) {
    int val = (lane_id < num_warps_for_scan) ? warp_sums[lane_id] : 0;
    warp_sums[lane_id] = warp_exclusive_scan(val);
  }
  __syncthreads();

  int offset = warp_sums[warp_id];
  if (tid < scan_size) scan_buf[tid] = pre + offset;
  __syncthreads();

  // Write prefix[0..num_experts - 1] and cumsum
  if (tid < num_experts) prefix[tid] = scan_buf[tid];
#endif

  if (tid <= num_experts) {
    cumsum[tid] = prefix[tid];
  }
  // fill expert_ids
  const int32_t num_blocks = s_total_tokens_post_pad / block_size;
  for (int32_t i = tid; i < num_blocks; i += stride) {
    int32_t block_start = i * block_size;
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (prefix[mid] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 2;
  }
}

// ===== Cooperative fused kernel for large batch (single launch, grid.sync)

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void moe_align_block_size_cooperative_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t* __restrict__ global_counts,  // [num_experts+1], zeroed by caller
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  cg::grid_group grid = cg::this_grid();

  extern __shared__ int32_t smem[];
  // smem layout: [num_experts] local_hist + [num_experts+1] expert_starts
  int32_t* local_hist = smem;
  int32_t* expert_starts_local = smem + num_experts;

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int nblocks = gridDim.x;

  __shared__ int32_t s_total;

  // ===== Stage 0: Cooperative initialization =====
  // Fill sorted_token_ids with sentinel value (all blocks cooperate)
  if (pad_sorted_token_ids) {
    Vec fill_vec;
    fill_vec.x = fill_vec.y = fill_vec.z = fill_vec.w =
        static_cast<int32_t>(numel);
    int32_t total_vecs = (max_num_tokens_padded + VEC_SIZE - 1) / VEC_SIZE;
    Vec* out_ptr = reinterpret_cast<Vec*>(sorted_token_ids);
    for (int32_t i = bid * nthreads + tid; i < total_vecs;
         i += nblocks * nthreads) {
      out_ptr[i] = fill_vec;
    }
  }

  // Initialize local histogram to 0
  for (int i = tid; i < num_experts; i += nthreads) {
    local_hist[i] = 0;
  }
  __syncthreads();

  // ===== Stage 1: Local histogram + global atomic merge =====
  for (size_t i = (size_t)bid * nthreads + tid; i < numel;
       i += (size_t)nblocks * nthreads) {
    int expert_id = static_cast<int>(topk_ids[i]) + 1;
    atomicAdd(&local_hist[expert_id], 1);
  }
  __syncthreads();

  // Merge local counts into global via atomic fetch-and-add.
  // Return value = prefix_before (reuse local_hist to store it).
  for (int i = tid; i < num_experts; i += nthreads) {
    int32_t count = local_hist[i];
    int32_t prefix_before = atomicAdd(&global_counts[i], count);
    local_hist[i] = prefix_before;
  }

  grid.sync();  // all histograms merged, global_counts has totals

  // ===== Stage 2: Redundant prefix sum per block   =====
  if (tid == 0) {
    int32_t running_sum = 0;
    for (int i = 0; i < num_experts; i++) {
      int32_t count = global_counts[i];
      int32_t padded = (count + block_size - 1) / block_size * block_size;
      expert_starts_local[i] = running_sum;
      running_sum += padded;
    }
    expert_starts_local[num_experts] = running_sum;  // total
    s_total = running_sum;
  }

  grid.sync();

  // Block 0 writes total_tokens_post_pad and cumsum (global_counts)
  if (bid == 0) {
    // Write expert starts to global_counts for external consumers
    if (tid <= num_experts) {
      global_counts[tid] = expert_starts_local[tid];
    }
    if (tid == 0) {
      *total_tokens_post_pad = s_total;
    }
  }

  // ===== Stage 3: Fill expert_ids (all blocks cooperate) =====
  const int32_t num_blocks_out = s_total / block_size;
  for (int32_t i = bid * nthreads + tid; i < num_blocks_out;
       i += nblocks * nthreads) {
    int32_t block_start = i * block_size;
    // Binary search: find the expert whose start <= block_start < next start
    int left = 0, right = num_experts;
    while (left < right) {
      int mid = (left + right) >> 1;
      if (expert_starts_local[mid + 1] <= block_start) {
        left = mid + 1;
      } else {
        right = mid;
      }
    }
    expert_ids[i] = left - 1;  // expert indexing: topk_ids uses +1 offset
  }

  // ===== Stage 4: Scatter tokens using shared memory atomics =====
  // local_hist[i] currently holds prefix_before for this block.
  // We do atomic_add on local_hist to get each token's rank within the expert,
  // then add expert_starts_local to get the final position.
  for (size_t i = (size_t)bid * nthreads + tid; i < numel;
       i += (size_t)nblocks * nthreads) {
    int expert_id = static_cast<int>(topk_ids[i]) + 1;
    int32_t rank = atomicAdd(&local_hist[expert_id], 1);
    int32_t pos = rank + expert_starts_local[expert_id];
    sorted_token_ids[pos] = i;
  }
}

template <typename scalar_t, int32_t fill_threads>
__global__ void moe_align_block_size_small_batch_expert_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel,
    bool pad_sorted_token_ids,
    int32_t max_num_tokens_padded) {
  // Adapted from
  // https://github.com/vllm-project/vllm/pull/29642/files#diff-5647b1413f4ae9aacba904eca8f8a8aee9079321eadff4c10101a2c6962dcc53R226
  // Use an additional group of threads to fill sorted_token_ids.
  // Since the kernel will use sorted_token_ids afterward,
  // we fill sorted_token_ids within the same threadblock to make
  // synchronization easier.
  if (threadIdx.x < fill_threads) {
    // Initialize sorted_token_ids with numel
    if (pad_sorted_token_ids) {
      for (int32_t it = threadIdx.x; it < max_num_tokens_padded;
           it += fill_threads) {
        sorted_token_ids[it] = numel;
      }
    }
    // Three __syncthreads() corresponding to the other threads
    __syncthreads();
    __syncthreads();
    __syncthreads();
    return;
  }

  const size_t tid = threadIdx.x - fill_threads;
  const size_t stride = blockDim.x - fill_threads;

  extern __shared__ int32_t shared_mem[];
  int32_t* cumsum = shared_mem;
  int32_t* tokens_cnts = (int32_t*)(shared_mem + num_experts + 1);

  for (int i = 0; i < num_experts; ++i) {
    tokens_cnts[(tid + 1) * num_experts + i] = 0;
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    ++tokens_cnts[(tid + 1) * num_experts + expert_id];
  }

  __syncthreads();

  if (tid < num_experts) {
    tokens_cnts[tid] = 0;
    for (int i = 1; i <= stride; ++i) {
      tokens_cnts[i * num_experts + tid] +=
          tokens_cnts[(i - 1) * num_experts + tid];
    }
  }

  __syncthreads();

  if (tid == 0) {
    cumsum[0] = 0;
    for (int i = 1; i <= num_experts; ++i) {
      cumsum[i] =
          cumsum[i - 1] +
          CEILDIV(tokens_cnts[stride * num_experts + i - 1], block_size) *
              block_size;
    }
    *total_tokens_post_pad = static_cast<int32_t>(cumsum[num_experts]);
  }

  __syncthreads();

  if (tid < num_experts) {
    for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) {
      expert_ids[i / block_size] = tid - 1;
    }
  }

  for (size_t i = tid; i < numel; i += stride) {
    int32_t expert_id = topk_ids[i] + 1;
    int32_t rank_post_pad =
        tokens_cnts[tid * num_experts + expert_id] + cumsum[expert_id];
    sorted_token_ids[rank_post_pad] = i;
    ++tokens_cnts[tid * num_experts + expert_id];
  }
}

template <typename scalar_t>
void moe_align_block_size(const paddle::Tensor& topk_ids,
                          int64_t num_experts,
                          int64_t block_size,
                          paddle::Tensor& sorted_token_ids,
                          paddle::Tensor& experts_ids,
                          paddle::Tensor& num_tokens_post_pad,
                          paddle::Tensor& cumsum_buffer,
                          bool pad_sorted_token_ids) {
  int threads = 1024;
  threads = ((threads + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  auto stream = topk_ids.stream();

  const size_t numel = topk_ids.numel();
  const int64_t max_num_tokens_padded = sorted_token_ids.shape()[0];

  bool small_batch_expert_mode = (numel < 1024) && (num_experts <= 64);

  if (small_batch_expert_mode) {
    const int32_t expert_threads = max((int32_t)num_experts, WARP_SIZE);
    constexpr int32_t fill_threads = 256;
    const int32_t shared_mem_size =
        ((expert_threads + 1) * num_experts + (num_experts + 1)) *
        sizeof(int32_t);

    auto small_batch_expert_kernel =
        moe_align_block_size_small_batch_expert_kernel<scalar_t, fill_threads>;
    small_batch_expert_kernel<<<1,
                                fill_threads + expert_threads,
                                shared_mem_size,
                                stream>>>(topk_ids.data<scalar_t>(),
                                          sorted_token_ids.data<int32_t>(),
                                          experts_ids.data<int32_t>(),
                                          num_tokens_post_pad.data<int32_t>(),
                                          num_experts,
                                          block_size,
                                          numel,
                                          pad_sorted_token_ids,
                                          max_num_tokens_padded);
  } else {
    // Use cooperative fused kernel for large inputs where multi-block
    // parallelism outweighs cooperative launch overhead
    if (numel >= 16384) {
      const int coop_threads = 256;
      const size_t coop_smem = (2 * num_experts + 1) * sizeof(int32_t);

      auto coop_kernel = moe_align_block_size_cooperative_kernel<scalar_t>;

      static int cached_max_blocks_per_sm = 0;
      static int cached_num_sms = 0;
      if (cached_num_sms == 0) {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&cached_max_blocks_per_sm,
                                                      (void*)coop_kernel,
                                                      coop_threads,
                                                      coop_smem);
        int device_id;
        cudaGetDevice(&device_id);
        cudaDeviceGetAttribute(
            &cached_num_sms, cudaDevAttrMultiProcessorCount, device_id);
      }

      int max_coop_blocks = cached_max_blocks_per_sm * cached_num_sms;
      int desired_blocks = std::max(
          1, std::min(256, static_cast<int>(numel / (coop_threads * 4))));
      int coop_blocks = std::min(desired_blocks, max_coop_blocks);
      if (coop_blocks < 1) coop_blocks = 1;

      const scalar_t* topk_ids_ptr = topk_ids.data<scalar_t>();
      int32_t* sorted_token_ids_ptr = sorted_token_ids.data<int32_t>();
      int32_t* experts_ids_ptr = experts_ids.data<int32_t>();
      int32_t* num_tokens_post_pad_ptr = num_tokens_post_pad.data<int32_t>();
      int32_t* cumsum_ptr = cumsum_buffer.data<int32_t>();
      int32_t num_experts_i32 = static_cast<int32_t>(num_experts);
      int32_t block_size_i32 = static_cast<int32_t>(block_size);
      size_t numel_val = numel;
      bool pad_val = pad_sorted_token_ids;
      int32_t max_padded_i32 = static_cast<int32_t>(max_num_tokens_padded);

      void* args[] = {&topk_ids_ptr,
                      &sorted_token_ids_ptr,
                      &experts_ids_ptr,
                      &num_tokens_post_pad_ptr,
                      &cumsum_ptr,
                      &num_experts_i32,
                      &block_size_i32,
                      &numel_val,
                      &pad_val,
                      &max_padded_i32};

      cudaError_t err = cudaLaunchCooperativeKernel((void*)coop_kernel,
                                                    dim3(coop_blocks),
                                                    dim3(coop_threads),
                                                    args,
                                                    coop_smem,
                                                    stream);

      if (err == cudaSuccess) {
        return;
      }
      // Fall through to original path if cooperative launch failed
    }

    // Original 2-kernel approach (for medium inputs or cooperative fallback)
    auto align_kernel = moe_align_block_size_kernel<scalar_t>;

    const size_t scan_size = next_pow_2(num_experts);
    const size_t shared_mem_size =
        (num_experts + (num_experts + 1) + scan_size + WARP_SIZE) *
        sizeof(int32_t);
    align_kernel<<<2, threads, shared_mem_size, stream>>>(
        topk_ids.data<scalar_t>(),
        sorted_token_ids.data<int32_t>(),
        experts_ids.data<int32_t>(),
        num_tokens_post_pad.data<int32_t>(),
        num_experts,
        block_size,
        numel,
        cumsum_buffer.data<int32_t>(),
        pad_sorted_token_ids,
        scan_size,
        max_num_tokens_padded);

    const int block_threads = std::min(256, (int)threads);
    const int num_blocks = ((int)numel + block_threads - 1) / block_threads;
    const int max_blocks = 65535;
    const int actual_blocks = std::min(num_blocks, max_blocks);

    auto sort_kernel = count_and_sort_expert_tokens_kernel<scalar_t>;
    sort_kernel<<<actual_blocks, block_threads, 0, stream>>>(
        topk_ids.data<scalar_t>(),
        sorted_token_ids.data<int32_t>(),
        cumsum_buffer.data<int32_t>(),
        numel);
  }
}

// Explicit instantiations for use from other translation units (e.g.
// tritonmoe_preprocess.cu)
template void moe_align_block_size<int32_t>(const paddle::Tensor&,
                                            int64_t,
                                            int64_t,
                                            paddle::Tensor&,
                                            paddle::Tensor&,
                                            paddle::Tensor&,
                                            paddle::Tensor&,
                                            bool);
template void moe_align_block_size<int64_t>(const paddle::Tensor&,
                                            int64_t,
                                            int64_t,
                                            paddle::Tensor&,
                                            paddle::Tensor&,
                                            paddle::Tensor&,
                                            paddle::Tensor&,
                                            bool);
