// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

// Shared device function: parallel sliding window ngram search.
// Searches for the first occurrence of ngram[0..ngram_size-1] within
// cur_input_ids[0..search_len+ngram_size-1] using multiple threads.
// The minimum matching index is atomically written to shared_start_idx.
__device__ __forceinline__ void sliding_window_search(
    const int64_t* cur_input_ids,
    const int64_t* ngram,
    const int64_t search_len,
    int64_t* shared_start_idx,
    int tid,
    int ngram_size) {
  for (int64_t i = tid; i <= search_len; i += blockDim.x) {
    bool match = true;
    for (int j = 0; j < ngram_size; ++j) {
      if (ngram[j] != cur_input_ids[i + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      atomicMin(reinterpret_cast<unsigned long long*>(shared_start_idx),
                static_cast<unsigned long long>(i));
      break;
    }
  }
}
