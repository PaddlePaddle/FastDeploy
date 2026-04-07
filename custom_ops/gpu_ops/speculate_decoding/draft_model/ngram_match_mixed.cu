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

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "helper.h"
#include "paddle/extension.h"
#include "../ngram_match_core.cuh"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

static __device__ int d_mixed_cutoff;

// Phase 0: Finds cutoff batch id so Phase 1 only launches blocks for active
// batches (decoder-only). Handles seq_lens bookkeeping for skipped batches.
template <int NUM_THREADS>
__global__ void mixed_compute_active_prefix(int32_t *seq_lens_decoder,
                                            int32_t *seq_lens_this_time,
                                            int32_t *seq_lens_this_time_copy,
                                            int64_t max_batch_size,
                                            int threshold,
                                            int *cutoff_out) {
  int tid = threadIdx.x;

  int is_active = 0;
  if (tid < (int)max_batch_size) {
    if (seq_lens_decoder[tid] > 0) is_active = 1;
  }

  typedef cub::BlockScan<int, NUM_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage scan_storage;
  int exclusive_sum;
  BlockScan(scan_storage).ExclusiveSum(is_active, exclusive_sum);

  __shared__ int s_cutoff;
  if (tid == 0) s_cutoff = (int)max_batch_size;
  __syncthreads();

  if (tid < (int)max_batch_size && exclusive_sum >= threshold - 1)
    atomicMin(&s_cutoff, tid);
  __syncthreads();

  // Bookkeeping for batches >= cutoff: preserve seq_lens_this_time as-is.
  if (tid >= s_cutoff && tid < (int)max_batch_size) {
    seq_lens_this_time_copy[tid] = seq_lens_this_time[tid];
  }

  if (tid == 0) *cutoff_out = s_cutoff;
}

// Phase 1: Block 0 counts unprocessed batches (seq_lens_decoder > 0).
//          Blocks 1..N find ngram candidates for each batch in parallel.
__global__ void mixed_count_and_find_candidate_kernel(
    const int64_t *input_ids,
    const int64_t *input_ids_len,
    const int64_t *pre_ids,
    const int64_t *step_idx,
    const int *draft_token_num,
    int64_t *draft_tokens,
    int64_t *draft_tokens_copy,
    int32_t *seq_lens_this_time,
    int32_t *seq_lens_this_time_copy,
    int32_t *seq_lens_decoder,
    int64_t *max_dec_len,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int64_t draft_tokens_stride,
    int max_ngram_size,
    int min_ngram_size,
    int max_draft_tokens,
    int64_t max_batch_size) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  if (bid >= (int)max_batch_size) return;

  __shared__ int32_t s_ori_seq_len;
  __shared__ bool skip;
  __shared__ int s_max_draft_tokens_query;
  __shared__ int64_t shared_match_idx;

  if (tid == 0) {
    s_ori_seq_len = seq_lens_this_time[bid];
    int mdtq = max_draft_tokens - s_ori_seq_len + 1;
    int64_t remaining = max_dec_len[bid] - step_idx[bid] - 1;
    if (static_cast<int64_t>(mdtq) > remaining)
      mdtq = static_cast<int>(remaining);
    s_max_draft_tokens_query = mdtq;

    // Initialize copy with original value
    seq_lens_this_time_copy[bid] = s_ori_seq_len;

    skip = (s_ori_seq_len == 0 || mdtq <= 0);
    shared_match_idx = 0x7FFFFFFFFFFFFFFF;
  }
  __syncthreads();

  if (skip) return;

  const int64_t *cur_input_ids = input_ids + bid * input_ids_stride;
  int64_t *cur_draft_tokens_copy =
      draft_tokens_copy + bid * draft_tokens_stride;
  const int64_t *cur_pre_ids = pre_ids + bid * pre_ids_stride;
  const int64_t cur_step_idx = step_idx[bid];
  const int64_t cur_input_ids_len = input_ids_len[bid];
  const int ori_seq_len = s_ori_seq_len;
  const int max_draft_q = s_max_draft_tokens_query;

  for (int ngram_size = max_ngram_size; ngram_size >= min_ngram_size;
       --ngram_size) {
    if (cur_step_idx < ngram_size) continue;

    const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

    // Search in input_ids
    sliding_window_search(cur_input_ids,
                          ngram,
                          cur_input_ids_len - ngram_size,
                          &shared_match_idx,
                          tid,
                          ngram_size);

    if (shared_match_idx < 0x7FFFFFFFFFFFFFFF) {
      if (tid == 0) {
        int64_t start_idx = shared_match_idx + ngram_size;
        int64_t end_idx = start_idx + max_draft_q;
        if (end_idx > cur_input_ids_len) end_idx = cur_input_ids_len;
        if (start_idx < end_idx) {
          int64_t count = end_idx - start_idx;
          seq_lens_this_time_copy[bid] =
              ori_seq_len + static_cast<int32_t>(count);
          memcpy(cur_draft_tokens_copy + ori_seq_len,
                 cur_input_ids + start_idx,
                 sizeof(int64_t) * count);
        }
      }
      break;
    }

    // Search in generated tokens (pre_ids)
    sliding_window_search(cur_pre_ids,
                          ngram,
                          cur_step_idx - ngram_size,
                          &shared_match_idx,
                          tid,
                          ngram_size);

    if (shared_match_idx < 0x7FFFFFFFFFFFFFFF) {
      if (tid == 0) {
        int64_t start_idx = shared_match_idx + ngram_size;
        int64_t end_idx = start_idx + max_draft_q;
        if (end_idx > cur_step_idx) end_idx = cur_step_idx;
        if (start_idx < end_idx) {
          int64_t count = end_idx - start_idx;
          seq_lens_this_time_copy[bid] =
              ori_seq_len + static_cast<int32_t>(count);
          memcpy(cur_draft_tokens_copy + ori_seq_len,
                 cur_pre_ids + start_idx,
                 sizeof(int64_t) * count);
        }
      }
      break;
    }
  }
}

// Phase 2: Single block truncation with threshold.
template <int NUM_THREADS>
__global__ void mixed_truncate_candidate(const int64_t *step_idx,
                                         const int *draft_token_num,
                                         int64_t *max_dec_len,
                                         int32_t *seq_lens_this_time,
                                         int32_t *seq_lens_this_time_copy,
                                         int64_t *draft_tokens,
                                         int64_t *draft_tokens_copy,
                                         int64_t draft_tokens_stride,
                                         int64_t max_batch_size,
                                         int max_draft_tokens,
                                         int threshold) {
  int tid = threadIdx.x;

  int is_active_here =
      (tid < (int)max_batch_size && seq_lens_this_time[tid] > 0) ? 1 : 0;
  typedef cub::BlockReduce<int, NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  int total_active = BlockReduce(reduce_storage).Sum(is_active_here);
  __shared__ int s_unprocessed;
  if (tid == 0) s_unprocessed = total_active;
  __syncthreads();

  int is_processed = 0;
  int allocating_token_num = 0;
  int ori_seq_len = 0;
  int max_draft_tokens_query = 0;

  if (tid < max_batch_size) {
    ori_seq_len = seq_lens_this_time[tid];
    max_draft_tokens_query = max_draft_tokens - ori_seq_len + 1;
    int64_t remaining = max_dec_len[tid] - step_idx[tid] - 1;
    if (static_cast<int64_t>(max_draft_tokens_query) > remaining)
      max_draft_tokens_query = static_cast<int>(remaining);

    if (ori_seq_len > 0 && max_draft_tokens_query > 0) {
      is_processed = 1;
      allocating_token_num = seq_lens_this_time_copy[tid];
    } else {
      allocating_token_num = ori_seq_len;
    }
  }

  typedef cub::BlockScan<int, NUM_THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage_batch;
  int processed_batch_size;
  BlockScan(temp_storage_batch)
      .InclusiveSum(is_processed, processed_batch_size);
  __syncthreads();

  __shared__ typename BlockScan::TempStorage temp_storage_token;
  int sum_token_num;
  BlockScan(temp_storage_token)
      .InclusiveSum(allocating_token_num, sum_token_num);

  if (is_processed && tid < max_batch_size) {
    // Sum before this batch: prefix_sum - this_allocation + ori_seq_len
    int sum_before = sum_token_num - allocating_token_num + ori_seq_len;
    int unprocessed_tid = s_unprocessed - processed_batch_size;

    if (sum_before + unprocessed_tid < threshold - 1) {
      int64_t *cur_draft_tokens = draft_tokens + tid * draft_tokens_stride;
      int64_t *cur_draft_tokens_copy_ptr =
          draft_tokens_copy + tid * draft_tokens_stride;

      int found_count = seq_lens_this_time_copy[tid] - ori_seq_len;

      if (sum_before + max_draft_tokens_query + unprocessed_tid > threshold) {
        max_draft_tokens_query = threshold - sum_before - unprocessed_tid;
        int actual_count = found_count < max_draft_tokens_query
                               ? found_count
                               : max_draft_tokens_query;
        if (actual_count > 0) {
          memcpy(cur_draft_tokens + ori_seq_len,
                 cur_draft_tokens_copy_ptr + ori_seq_len,
                 sizeof(int64_t) * actual_count);
          seq_lens_this_time[tid] = ori_seq_len + actual_count;
        }
      } else {
        if (found_count > 0) {
          memcpy(cur_draft_tokens + ori_seq_len,
                 cur_draft_tokens_copy_ptr + ori_seq_len,
                 sizeof(int64_t) * found_count);
          seq_lens_this_time[tid] = seq_lens_this_time_copy[tid];
        }
      }
    }
  }
}

void HybridMtpNgram(const paddle::Tensor &input_ids,
                    const paddle::Tensor &input_ids_len,
                    const paddle::Tensor &pre_ids,
                    const paddle::Tensor &step_idx,
                    const paddle::Tensor &draft_token_num,
                    const paddle::Tensor &draft_tokens,
                    const paddle::Tensor &draft_tokens_copy,
                    const paddle::Tensor &seq_lens_this_time,
                    const paddle::Tensor &seq_lens_this_time_copy,
                    const paddle::Tensor &seq_lens_decoder,
                    const paddle::Tensor &max_dec_len,
                    const int max_ngram_size,
                    const int min_ngram_size,
                    const int max_draft_tokens) {
  auto input_ids_shape = input_ids.shape();
  const int64_t input_ids_stride = input_ids_shape[1];

  auto pre_ids_shape = pre_ids.shape();
  const int64_t pre_ids_stride = pre_ids_shape[1];

  auto draft_tokens_shape = draft_tokens.shape();
  const int64_t draft_tokens_stride = draft_tokens_shape[1];

  const int64_t max_batch_size = seq_lens_this_time.shape()[0];
  assert(max_batch_size <= MAXBATCHSIZE);

  auto cu_stream = input_ids.stream();

  int tokennum_threshold = 1024;
  char *env_var = getenv("SPEC_TOKENUM_THRESHOLD");
  if (env_var) {
    tokennum_threshold = std::stoi(env_var);
  }

  static int one_wave_capacity = []() {
    int dev = 0, sm_count = 0, tpm = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
    return sm_count * tpm / NGRAM_SEARCH_THREADS;
  }();

  int launch_size = static_cast<int>(max_batch_size);

  if (tokennum_threshold < static_cast<int>(max_batch_size) &&
      static_cast<int>(max_batch_size) > one_wave_capacity) {
    int *d_cutoff_ptr;
    cudaGetSymbolAddress(reinterpret_cast<void **>(&d_cutoff_ptr),
                         d_mixed_cutoff);
    mixed_compute_active_prefix<MAXBATCHSIZE>
        <<<1, MAXBATCHSIZE, 0, cu_stream>>>(
            const_cast<int32_t *>(seq_lens_decoder.data<int32_t>()),
            const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
            const_cast<int32_t *>(seq_lens_this_time_copy.data<int32_t>()),
            max_batch_size,
            tokennum_threshold,
            d_cutoff_ptr);
    int h_cutoff = static_cast<int>(max_batch_size);
    cudaMemcpyAsync(&h_cutoff,
                    d_cutoff_ptr,
                    sizeof(int),
                    cudaMemcpyDeviceToHost,
                    cu_stream);
    cudaStreamSynchronize(cu_stream);
    launch_size = h_cutoff;
  }

  mixed_count_and_find_candidate_kernel<<<launch_size,
                                          NGRAM_SEARCH_THREADS,
                                          0,
                                          cu_stream>>>(

      input_ids.data<int64_t>(),
      input_ids_len.data<int64_t>(),
      pre_ids.data<int64_t>(),
      step_idx.data<int64_t>(),
      draft_token_num.data<int>(),
      const_cast<int64_t *>(draft_tokens.data<int64_t>()),
      const_cast<int64_t *>(draft_tokens_copy.data<int64_t>()),
      const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
      const_cast<int32_t *>(seq_lens_this_time_copy.data<int32_t>()),
      const_cast<int32_t *>(seq_lens_decoder.data<int32_t>()),
      const_cast<int64_t *>(max_dec_len.data<int64_t>()),
      input_ids_stride,
      pre_ids_stride,
      draft_tokens_stride,
      max_ngram_size,
      min_ngram_size,
      max_draft_tokens,
      max_batch_size);

  mixed_truncate_candidate<NGRAM_TRUNCATION_THREADS>
      <<<1, NGRAM_TRUNCATION_THREADS, 0, cu_stream>>>(
          step_idx.data<int64_t>(),
          draft_token_num.data<int>(),
          const_cast<int64_t *>(max_dec_len.data<int64_t>()),
          const_cast<int32_t *>(seq_lens_this_time.data<int32_t>()),
          const_cast<int32_t *>(seq_lens_this_time_copy.data<int32_t>()),
          const_cast<int64_t *>(draft_tokens.data<int64_t>()),
          const_cast<int64_t *>(draft_tokens_copy.data<int64_t>()),
          draft_tokens_stride,
          max_batch_size,
          max_draft_tokens,
          tokennum_threshold);
}

PD_BUILD_STATIC_OP(hybrid_mtp_ngram)
    .Inputs({"input_ids",
             "input_ids_len",
             "pre_ids",
             "step_idx",
             "draft_token_num",
             "draft_tokens",
             "draft_tokens_copy",
             "seq_lens_this_time",
             "seq_lens_this_time_copy",
             "seq_lens_decoder",
             "max_dec_len"})
    .Attrs({"max_ngram_size: int",
            "min_ngram_size: int",
            "max_draft_tokens: int"})
    .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
    .SetKernelFn(PD_KERNEL(HybridMtpNgram))
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"}});
