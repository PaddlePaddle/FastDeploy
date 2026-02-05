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

#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace ngram_match_gpu {

// Constants for kernel configuration
constexpr int kMaxNgramSize = 16;
constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

// Structure to hold match result information
struct MatchResult {
  int64_t match_pos;      // Position where match was found (-1 if no match)
  int ngram_size;         // The ngram size that matched
  bool is_in_input;       // true if match is in input_ids, false if in pre_ids
};

// Device function: Check if ngram matches at a specific position
__device__ __forceinline__ bool check_ngram_match(
    const int64_t* __restrict__ sequence,
    const int64_t* __restrict__ ngram,
    int ngram_size) {
  for (int j = 0; j < ngram_size; ++j) {
    if (sequence[j] != ngram[j]) {
      return false;
    }
  }
  return true;
}

// Device function: Parallel search for ngram match in a sequence
// Each thread checks different positions, returns the first matching position
// Returns -1 if no match found
template <int BLOCK_SIZE>
__device__ int64_t parallel_ngram_search(
    const int64_t* __restrict__ sequence,
    int64_t sequence_len,
    const int64_t* __restrict__ ngram,
    int ngram_size,
    int max_draft_tokens) {
  typedef cub::BlockReduce<int64_t, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  const int tid = threadIdx.x;
  const int64_t search_range = sequence_len - ngram_size + 1;

  // Each thread finds the minimum matching position (or INT64_MAX if no match)
  int64_t local_min_pos = INT64_MAX;

  // Parallel search through all positions
  for (int64_t i = tid; i < search_range; i += BLOCK_SIZE) {
    if (check_ngram_match(sequence + i, ngram, ngram_size)) {
      // Check if we can get at least 1 draft token
      int64_t start_idx = i + ngram_size;
      if (start_idx < sequence_len) {
        local_min_pos = min(local_min_pos, i);
      }
    }
  }

  __syncthreads();

  // Find the global minimum position across all threads
  int64_t min_pos = BlockReduce(temp_storage).Reduce(local_min_pos, cub::Min());

  return min_pos;
}

// Device function: Copy tokens from source to destination
__device__ __forceinline__ void copy_tokens_device(
    int64_t* __restrict__ dst,
    const int64_t* __restrict__ src,
    int64_t count) {
  const int tid = threadIdx.x;
  for (int64_t i = tid; i < count; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// Kernel for ngram_match operation
// Each block processes one batch sample
template <int BLOCK_SIZE>
__global__ void ngram_match_kernel(
    const int64_t* __restrict__ input_ids,
    const int64_t* __restrict__ input_ids_len,
    const int64_t* __restrict__ pre_ids,
    const int64_t* __restrict__ step_idx,
    const int* __restrict__ draft_token_num,
    int64_t* __restrict__ draft_tokens,
    int* __restrict__ seq_lens_this_time,
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    const int64_t* __restrict__ max_dec_len,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int64_t draft_tokens_stride,
    int max_batch_size,
    int max_ngram_size,
    int max_draft_tokens,
    int threshold,
    const int* __restrict__ unprocessed_counts) {

  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduceInt;
  typedef cub::BlockReduce<int64_t, BLOCK_SIZE> BlockReduceInt64;
  __shared__ typename BlockReduceInt::TempStorage temp_storage_int;
  __shared__ typename BlockReduceInt64::TempStorage temp_storage_int64;

  // Shared memory for ngram tokens and results
  __shared__ int64_t s_ngram[kMaxNgramSize];
  __shared__ int64_t s_match_pos;
  __shared__ int s_match_ngram_size;
  __shared__ bool s_match_in_input;
  __shared__ int s_seq_lens_this_time[1];
  __shared__ int s_left_min_token_num;

  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (batch_idx >= max_batch_size) return;

  // Skip encoder requests
  if (seq_lens_encoder[batch_idx] > 0) {
    return;
  }

  // Skip if decoder is not active
  if (seq_lens_decoder[batch_idx] == 0) {
    if (tid == 0) {
      seq_lens_this_time[batch_idx] = 0;
    }
    return;
  }

  // Calculate pointers for this batch
  const int64_t* cur_input_ids = input_ids + batch_idx * input_ids_stride;
  int64_t* cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
  const int64_t* cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
  const int64_t cur_step_idx = step_idx[batch_idx];
  const int64_t cur_input_ids_len = input_ids_len[batch_idx];

  // Calculate dynamic max_draft_tokens
  int local_max_draft_tokens = min(
      static_cast<int64_t>(draft_token_num[batch_idx]),
      max_dec_len[batch_idx] - cur_step_idx - 1);

  // Initialize seq_lens_this_time
  if (tid == 0) {
    seq_lens_this_time[batch_idx] = 1;
    s_seq_lens_this_time[0] = 1;
    s_match_pos = -1;
    s_match_ngram_size = 0;
    s_match_in_input = false;
    s_left_min_token_num = unprocessed_counts[batch_idx];
  }
  __syncthreads();

  // Calculate sum of previous seq_lens_this_time
  int sum_token_num = 0;
  for (int i = tid; i < batch_idx; i += BLOCK_SIZE) {
    sum_token_num += seq_lens_this_time[i];
  }
  sum_token_num = BlockReduceInt(temp_storage_int).Sum(sum_token_num);
  if (tid == 0) {
    sum_token_num += 1;  // Add 1 for current batch
  }
  __syncthreads();

  // Adjust max_draft_tokens based on threshold
  int left_min_token_num = s_left_min_token_num;
  if (sum_token_num + local_max_draft_tokens + left_min_token_num > threshold) {
    int tmp = threshold - sum_token_num - left_min_token_num;
    local_max_draft_tokens = min(local_max_draft_tokens, tmp);
  }

  if (sum_token_num + left_min_token_num >= threshold - 1) {
    return;
  }

  // Try different ngram sizes from max to 1
  for (int ngram_size = max_ngram_size; ngram_size > 0; --ngram_size) {
    if (cur_step_idx < ngram_size) {
      continue;
    }

    // Load ngram into shared memory
    if (tid < ngram_size) {
      s_ngram[tid] = cur_pre_ids[cur_step_idx + 1 - ngram_size + tid];
    }
    __syncthreads();

    // Search in input_ids first
    int64_t search_range_input = cur_input_ids_len - ngram_size + 1;
    int64_t local_min_pos = INT64_MAX;

    for (int64_t i = tid; i < search_range_input; i += BLOCK_SIZE) {
      bool match = true;
      for (int j = 0; j < ngram_size; ++j) {
        if (cur_input_ids[i + j] != s_ngram[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        int64_t start_idx = i + ngram_size;
        int64_t end_idx = min(start_idx + local_max_draft_tokens, cur_input_ids_len);
        if (start_idx < end_idx) {
          local_min_pos = min(local_min_pos, i);
        }
      }
    }

    __syncthreads();
    int64_t min_pos = BlockReduceInt64(temp_storage_int64).Reduce(local_min_pos, cub::Min());

    if (tid == 0 && min_pos != INT64_MAX) {
      s_match_pos = min_pos;
      s_match_ngram_size = ngram_size;
      s_match_in_input = true;
    }
    __syncthreads();

    if (s_match_pos != -1) {
      // Found a match in input_ids
      int64_t start_idx = s_match_pos + s_match_ngram_size;
      int64_t end_idx = min(start_idx + local_max_draft_tokens, cur_input_ids_len);
      int64_t cur_draft_token_num = end_idx - start_idx;

      if (tid == 0) {
        seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
      }

      // Copy draft tokens in parallel
      copy_tokens_device(cur_draft_tokens + 1, cur_input_ids + start_idx, cur_draft_token_num);
      return;
    }

    // Search in pre_ids if not found in input_ids
    int64_t search_range_pre = cur_step_idx - ngram_size + 1;
    local_min_pos = INT64_MAX;

    for (int64_t i = tid; i < search_range_pre; i += BLOCK_SIZE) {
      bool match = true;
      for (int j = 0; j < ngram_size; ++j) {
        if (cur_pre_ids[i + j] != s_ngram[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        int64_t start_idx = i + ngram_size;
        int64_t end_idx = min(start_idx + local_max_draft_tokens, cur_step_idx);
        if (start_idx < end_idx) {
          local_min_pos = min(local_min_pos, i);
        }
      }
    }

    __syncthreads();
    min_pos = BlockReduceInt64(temp_storage_int64).Reduce(local_min_pos, cub::Min());

    if (tid == 0 && min_pos != INT64_MAX) {
      s_match_pos = min_pos;
      s_match_ngram_size = ngram_size;
      s_match_in_input = false;
    }
    __syncthreads();

    if (s_match_pos != -1) {
      // Found a match in pre_ids
      int64_t start_idx = s_match_pos + s_match_ngram_size;
      int64_t end_idx = min(start_idx + local_max_draft_tokens, cur_step_idx);
      int64_t cur_draft_token_num = end_idx - start_idx;

      if (tid == 0) {
        seq_lens_this_time[batch_idx] = cur_draft_token_num + 1;
      }

      // Copy draft tokens in parallel
      copy_tokens_device(cur_draft_tokens + 1, cur_pre_ids + start_idx, cur_draft_token_num);
      return;
    }
  }
}

// Kernel for hybrid_mtp_ngram operation (mixed mode)
// Each block processes one batch sample
template <int BLOCK_SIZE>
__global__ void hybrid_mtp_ngram_kernel(
    const int64_t* __restrict__ input_ids,
    const int64_t* __restrict__ input_ids_len,
    const int64_t* __restrict__ pre_ids,
    const int64_t* __restrict__ step_idx,
    const int* __restrict__ draft_token_num,
    int64_t* __restrict__ draft_tokens,
    int* __restrict__ seq_lens_this_time,
    const int* __restrict__ seq_lens_decoder,
    const int64_t* __restrict__ max_dec_len,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int64_t draft_tokens_stride,
    int max_batch_size,
    int max_ngram_size,
    int min_ngram_size,
    int max_draft_tokens,
    int threshold,
    const int* __restrict__ unprocessed_counts) {

  typedef cub::BlockReduce<int, BLOCK_SIZE> BlockReduceInt;
  typedef cub::BlockReduce<int64_t, BLOCK_SIZE> BlockReduceInt64;
  __shared__ typename BlockReduceInt::TempStorage temp_storage_int;
  __shared__ typename BlockReduceInt64::TempStorage temp_storage_int64;

  // Shared memory for ngram tokens and results
  __shared__ int64_t s_ngram[kMaxNgramSize];
  __shared__ int64_t s_match_pos;
  __shared__ int s_match_ngram_size;
  __shared__ bool s_match_in_input;

  const int batch_idx = blockIdx.x;
  const int tid = threadIdx.x;

  if (batch_idx >= max_batch_size) return;

  // Get original seq_len_this_time
  const int ori_seq_len_this_time = seq_lens_this_time[batch_idx];

  // Calculate max draft tokens for query
  int max_draft_tokens_query = min(
      static_cast<int64_t>(max_draft_tokens - ori_seq_len_this_time + 1),
      max_dec_len[batch_idx] - step_idx[batch_idx] - 1);

  // Skip if no work to do
  if (ori_seq_len_this_time == 0 || max_draft_tokens_query <= 0) {
    return;
  }

  // Calculate pointers for this batch
  const int64_t* cur_input_ids = input_ids + batch_idx * input_ids_stride;
  int64_t* cur_draft_tokens = draft_tokens + batch_idx * draft_tokens_stride;
  const int64_t* cur_pre_ids = pre_ids + batch_idx * pre_ids_stride;
  const int64_t cur_step_idx = step_idx[batch_idx];
  const int64_t cur_input_ids_len = input_ids_len[batch_idx];

  // Initialize shared memory
  if (tid == 0) {
    s_match_pos = -1;
    s_match_ngram_size = 0;
    s_match_in_input = false;
  }
  __syncthreads();

  // Calculate sum of previous seq_lens_this_time for threshold check
  int sum_token_num = 0;
  for (int i = tid; i < batch_idx; i += BLOCK_SIZE) {
    sum_token_num += seq_lens_this_time[i];
  }
  sum_token_num = BlockReduceInt(temp_storage_int).Sum(sum_token_num);
  __syncthreads();

  // Get unprocessed batch count
  int left_min_token_num = unprocessed_counts ? unprocessed_counts[batch_idx] : 0;

  // Adjust max_draft_tokens based on threshold
  if (sum_token_num + max_draft_tokens_query + left_min_token_num > threshold) {
    int tmp = threshold - sum_token_num - left_min_token_num;
    max_draft_tokens_query = min(max_draft_tokens_query, tmp);
  }

  if (sum_token_num + left_min_token_num >= threshold - 1) {
    return;
  }

  // Try different ngram sizes from max to min
  for (int ngram_size = max_ngram_size; ngram_size >= min_ngram_size; --ngram_size) {
    if (s_match_pos != -1) break;  // Already found a match

    if (cur_step_idx < ngram_size) {
      continue;
    }

    // Load ngram into shared memory
    if (tid < ngram_size) {
      s_ngram[tid] = cur_pre_ids[cur_step_idx + 1 - ngram_size + tid];
    }
    __syncthreads();

    // Search in input_ids first
    int64_t search_range_input = cur_input_ids_len - ngram_size + 1;
    int64_t local_min_pos = INT64_MAX;

    for (int64_t i = tid; i < search_range_input; i += BLOCK_SIZE) {
      if (s_match_pos != -1) break;  // Early exit if match found

      bool match = true;
      for (int j = 0; j < ngram_size; ++j) {
        if (cur_input_ids[i + j] != s_ngram[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        int64_t start_idx = i + ngram_size;
        int64_t end_idx = min(start_idx + max_draft_tokens_query, cur_input_ids_len);
        if (start_idx < end_idx) {
          local_min_pos = min(local_min_pos, i);
        }
      }
    }

    __syncthreads();
    int64_t min_pos = BlockReduceInt64(temp_storage_int64).Reduce(local_min_pos, cub::Min());

    if (tid == 0 && min_pos != INT64_MAX) {
      s_match_pos = min_pos;
      s_match_ngram_size = ngram_size;
      s_match_in_input = true;
    }
    __syncthreads();

    if (s_match_pos != -1) {
      // Found a match in input_ids
      int64_t start_idx = s_match_pos + s_match_ngram_size;
      int64_t end_idx = min(start_idx + max_draft_tokens_query, cur_input_ids_len);
      int64_t cur_draft_token_num = end_idx - start_idx;

      if (tid == 0) {
        seq_lens_this_time[batch_idx] = ori_seq_len_this_time + cur_draft_token_num;
      }

      // Copy draft tokens in parallel
      copy_tokens_device(cur_draft_tokens + ori_seq_len_this_time, 
                         cur_input_ids + start_idx, cur_draft_token_num);
      return;
    }

    // Search in pre_ids if not found in input_ids
    int64_t search_range_pre = cur_step_idx - ngram_size + 1;
    local_min_pos = INT64_MAX;

    for (int64_t i = tid; i < search_range_pre; i += BLOCK_SIZE) {
      if (s_match_pos != -1) break;

      bool match = true;
      for (int j = 0; j < ngram_size; ++j) {
        if (cur_pre_ids[i + j] != s_ngram[j]) {
          match = false;
          break;
        }
      }
      if (match) {
        int64_t start_idx = i + ngram_size;
        int64_t end_idx = min(start_idx + max_draft_tokens_query, cur_step_idx);
        if (start_idx < end_idx) {
          local_min_pos = min(local_min_pos, i);
        }
      }
    }

    __syncthreads();
    min_pos = BlockReduceInt64(temp_storage_int64).Reduce(local_min_pos, cub::Min());

    if (tid == 0 && min_pos != INT64_MAX) {
      s_match_pos = min_pos;
      s_match_ngram_size = ngram_size;
      s_match_in_input = false;
    }
    __syncthreads();

    if (s_match_pos != -1) {
      // Found a match in pre_ids
      int64_t start_idx = s_match_pos + s_match_ngram_size;
      int64_t end_idx = min(start_idx + max_draft_tokens_query, cur_step_idx);
      int64_t cur_draft_token_num = end_idx - start_idx;

      if (tid == 0) {
        seq_lens_this_time[batch_idx] = ori_seq_len_this_time + cur_draft_token_num;
      }

      // Copy draft tokens in parallel
      copy_tokens_device(cur_draft_tokens + ori_seq_len_this_time, 
                         cur_pre_ids + start_idx, cur_draft_token_num);
      return;
    }
  }
}

// Helper kernel to calculate unprocessed batch counts
// Using template to avoid multiple definition errors when included in multiple TUs
template <int DUMMY = 0>
__global__ void calc_unprocessed_counts_kernel_impl(
    const int* __restrict__ seq_lens_encoder,
    const int* __restrict__ seq_lens_decoder,
    int* __restrict__ unprocessed_counts,
    int max_batch_size) {
  
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid >= max_batch_size) return;
  
  // Count unprocessed batches after current batch
  int count = 0;
  for (int i = tid + 1; i < max_batch_size; ++i) {
    if (seq_lens_encoder[i] > 0 || seq_lens_decoder[i] > 0) {
      count++;
    }
  }
  unprocessed_counts[tid] = count;
}

// Wrapper for easier calling
inline void launch_calc_unprocessed_counts_kernel(
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
    int* unprocessed_counts,
    int max_batch_size,
    int threads_per_block,
    int num_blocks,
    cudaStream_t stream) {
  calc_unprocessed_counts_kernel_impl<0><<<num_blocks, threads_per_block, 0, stream>>>(
      seq_lens_encoder, seq_lens_decoder, unprocessed_counts, max_batch_size);
}

// Helper kernel for hybrid mode
// Using template to avoid multiple definition errors when included in multiple TUs
template <int DUMMY = 0>
__global__ void calc_unprocessed_counts_mixed_kernel_impl(
    const int* __restrict__ seq_lens_decoder,
    int* __restrict__ unprocessed_counts,
    int max_batch_size) {
  
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (tid >= max_batch_size) return;
  
  // Count unprocessed batches after current batch
  int count = 0;
  for (int i = tid + 1; i < max_batch_size; ++i) {
    if (seq_lens_decoder[i] > 0) {
      count++;
    }
  }
  unprocessed_counts[tid] = count;
}

// Wrapper for easier calling
inline void launch_calc_unprocessed_counts_mixed_kernel(
    const int* seq_lens_decoder,
    int* unprocessed_counts,
    int max_batch_size,
    int threads_per_block,
    int num_blocks,
    cudaStream_t stream) {
  calc_unprocessed_counts_mixed_kernel_impl<0><<<num_blocks, threads_per_block, 0, stream>>>(
      seq_lens_decoder, unprocessed_counts, max_batch_size);
}

}  // namespace ngram_match_gpu
