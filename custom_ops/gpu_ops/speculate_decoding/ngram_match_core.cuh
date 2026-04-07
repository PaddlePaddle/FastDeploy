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

#ifndef NGRAM_SEARCH_THREADS
#define NGRAM_SEARCH_THREADS 1024
#endif
#ifndef NGRAM_TRUNCATION_THREADS
#define NGRAM_TRUNCATION_THREADS 1024
#endif
#ifndef MAXBATCHSIZE
#define MAXBATCHSIZE 1024
#endif

// CAS-based atomicMin for int64_t
__device__ __forceinline__ void atomicMin_int64(int64_t* addr, int64_t val) {
  unsigned long long* ull_addr = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long val_ull = static_cast<unsigned long long>(val);
  unsigned long long old_ull = *ull_addr;
  while (static_cast<int64_t>(old_ull) > val) {
    unsigned long long prev = atomicCAS(ull_addr, old_ull, val_ull);
    if (prev == old_ull) break;
    old_ull = prev;
  }
}

__device__ __forceinline__ void sliding_window_search(
    const int64_t* cur_input_ids,
    const int64_t* ngram,
    const int64_t search_len,
    int64_t* shared_start_idx,
    int tid,
    int ngram_size) {
  for (int64_t i = tid; i <= search_len; i += blockDim.x) {
    if (i > *shared_start_idx) break;
    bool match = true;
    for (int j = 0; j < ngram_size; ++j) {
      if (ngram[j] != cur_input_ids[i + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      atomicMin_int64(shared_start_idx, i);
      break;
    }
  }
  __syncthreads();
}
