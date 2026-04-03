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
//   Phase 1 — <<<bsz, NGRAM_BLOCK_THREADS>>>: parallel sliding-window search
//   Phase 2 — <<<1, 1>>>: serial threshold + token copy (inter-batch dep)

#define NGRAM_BLOCK_THREADS 256

// Intermediate result for one batch item produced by Phase 1 (parallel search)
// and consumed by Phase 2 (serial threshold + copy).
struct NgramMatchResult {
  int64_t match_pos;  // first (leftmost) match position in haystack (-1=none)
  int ngram_size;     // which ngram_size produced this match
  int haystack_type;  // 0 = input_ids, 1 = pre_ids
};

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
// Called by NGRAM_BLOCK_THREADS threads within a single block.
// Searches for ngram[0..ngram_size-1] in haystack[0..haystack_len-1].
// Uses shared-memory s_min_pos to reduce to the FIRST (leftmost)
// match position.
//
// Returns the leftmost match position, or INT64_MAX if no match.
// Caller must provide __shared__ int64_t s_min_pos.
// ------------------------------------------------------------
__device__ __forceinline__ int64_t
parallel_ngram_search(const int64_t *haystack,
                      int64_t haystack_len,
                      const int64_t *ngram,
                      int ngram_size,
                      int64_t *s_min_pos) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  if (tid == 0) {
    *s_min_pos = INT64_MAX;
  }
  __syncthreads();

  int64_t search_len = haystack_len - ngram_size + 1;
  if (search_len <= 0) {
    __syncthreads();
    return *s_min_pos;
  }

  for (int64_t i = tid; i < search_len; i += nthreads) {
    bool match = true;
    for (int j = 0; j < ngram_size; j++) {
      if (ngram[j] != haystack[i + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      atomicMin64(s_min_pos, i);
    }
  }
  __syncthreads();

  return *s_min_pos;
}
