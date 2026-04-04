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

#pragma once

#include <climits>

// Shared ngram matching logic used by both ngram_match_kernel and
// ngram_match_mixed_kernel.  Extracted per upstream requirement:
// "两个Kernel逻辑有较为相似部分，Kernel 形式为提取共用的匹配逻辑，外加业务逻辑"
//
// Two-phase parallel architecture:
//   Phase 1 — <<<bsz, NGRAM_BLOCK_THREADS>>>: parallel sliding-window
//             search + tentative token copy (one block per batch item).
//   Phase 2 — <<<1, NGRAM_GATHER_THREADS>>>: parallel threshold truncation
//             via CUB BlockScan prefix-sum, then copy winners to output

#define NGRAM_BLOCK_THREADS 1024
#define NGRAM_GATHER_THREADS 1024

// ------------------------------------------------------------
// atomicMin for int64_t via CAS loop.  CUDA has no native
// int64 atomicMin.  All values are non-negative positions or
// INT64_MAX, so unsigned reinterpretation is safe.
// ------------------------------------------------------------
__device__ __forceinline__ void atomicMin64(int64_t *addr, int64_t val) {
  unsigned long long *addr_ull = reinterpret_cast<unsigned long long *>(addr);
  unsigned long long val_ull = static_cast<unsigned long long>(val);
  // Non-atomic initial read is intentional: the CAS loop below detects and
  // retries on any stale value, so a torn read here is harmless.
  unsigned long long old = *addr_ull;
  while (val_ull < old) {
    unsigned long long assumed = old;
    old = atomicCAS(addr_ull, assumed, val_ull);
    if (old == assumed) break;
  }
}

// ------------------------------------------------------------
// parallel_ngram_search — Block-cooperative haystack search.
//
// Template-specialized for common ngram sizes (1-3) to enable:
//   - Register caching of ngram tokens (avoid repeated global loads)
//   - Full compile-time unrolling of inner comparison loop
//   - __restrict__ hints for pointer non-aliasing optimization
//
// Runtime dispatcher preserves the original call signature so both
// ngram_match.cu and ngram_match_mixed.cu work transparently.
//
// Early-exit (A2): once a match is found (s_min_pos < INT64_MAX),
// threads that are past the current best skip remaining work.
//
// Returns the leftmost match position, or INT64_MAX if no match.
// Caller must provide __shared__ int64_t s_min_pos.
// ------------------------------------------------------------
template <int NGRAM_SIZE>
__device__ __forceinline__ int64_t
parallel_ngram_search_specialized(const int64_t *__restrict__ haystack,
                                  int64_t haystack_len,
                                  const int64_t *__restrict__ ngram,
                                  int64_t *s_min_pos) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  if (tid == 0) *s_min_pos = INT64_MAX;
  __syncthreads();

  int64_t search_len = haystack_len - NGRAM_SIZE + 1;
  if (search_len <= 0) {
    __syncthreads();
    return *s_min_pos;
  }

  // Cache ngram tokens in registers — eliminates repeated global reads.
  int64_t ng[NGRAM_SIZE];
#pragma unroll
  for (int j = 0; j < NGRAM_SIZE; j++) ng[j] = ngram[j];

  for (int64_t i = tid; i < search_len; i += nthreads) {
    // A2: Early-exit — skip positions beyond current best match.
    if (i > *s_min_pos) break;

    bool match = true;
#pragma unroll
    for (int j = 0; j < NGRAM_SIZE; j++) {
      if (ng[j] != haystack[i + j]) {
        match = false;
        break;
      }
    }
    if (match) atomicMin64(s_min_pos, i);
  }
  __syncthreads();
  return *s_min_pos;
}

// Runtime dispatcher — same signature as original, transparent to callers.
__device__ __forceinline__ int64_t
parallel_ngram_search(const int64_t *__restrict__ haystack,
                      int64_t haystack_len,
                      const int64_t *__restrict__ ngram,
                      int ngram_size,
                      int64_t *s_min_pos) {
  switch (ngram_size) {
    case 1:
      return parallel_ngram_search_specialized<1>(
          haystack, haystack_len, ngram, s_min_pos);
    case 2:
      return parallel_ngram_search_specialized<2>(
          haystack, haystack_len, ngram, s_min_pos);
    case 3:
      return parallel_ngram_search_specialized<3>(
          haystack, haystack_len, ngram, s_min_pos);
    default:
      break;
  }
  // Fallback for ngram_size > 3 — runtime loop, no unrolling.
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  if (tid == 0) *s_min_pos = INT64_MAX;
  __syncthreads();
  int64_t search_len = haystack_len - ngram_size + 1;
  if (search_len <= 0) {
    __syncthreads();
    return *s_min_pos;
  }
  for (int64_t i = tid; i < search_len; i += nthreads) {
    if (i > *s_min_pos) break;
    bool match = true;
    for (int j = 0; j < ngram_size; j++) {
      if (ngram[j] != haystack[i + j]) {
        match = false;
        break;
      }
    }
    if (match) atomicMin64(s_min_pos, i);
  }
  __syncthreads();
  return *s_min_pos;
}
