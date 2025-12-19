// Copyright © 2024 PaddlePaddle Name. All Rights Reserved.
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

// Ignore CUTLASS warnings about type punning
#pragma once
#pragma once
#include <cuda_runtime.h>

__device__ inline bool d_match_ngram(const int64_t* ngram, int n,
                                     const int64_t* window, int pos) {
  for (int i = 0; i < n; ++i) if (ngram[i] != window[pos + i]) return false;
  return true;
}

/*
 * 1 thread / batch ，无 __syncthreads
 * 输出：
 *   d_match_pos[batch] : 匹配窗口起始偏移（-1 未命中）
 *   d_draft_cnt[batch] : 实际可拷贝 token 数
 */
__global__ void ngram_match_kernel_single_thread(
    const int64_t* __restrict__ d_input_ids,
    const int64_t* __restrict__ d_input_ids_len,
    const int64_t* __restrict__ d_pre_ids,
    const int64_t* __restrict__ d_step_idx,
    int max_ngram_size,
    int min_ngram_size,
    int max_draft_tokens_query,
    int64_t input_ids_stride,
    int64_t pre_ids_stride,
    int* __restrict__ d_match_pos,
    int* __restrict__ d_draft_cnt)
{
    const int b = blockIdx.x;
    const int64_t step        = d_step_idx[b];
    const int64_t input_len   = d_input_ids_len[b];
    const int64_t* cur_inp    = d_input_ids + b * input_ids_stride;
    const int64_t* cur_pre    = d_pre_ids   + b * pre_ids_stride;

    d_match_pos[b] = -1;
    d_draft_cnt[b] = 0;

    for (int n = max_ngram_size; n >= min_ngram_size; --n) {
        if (step < n) continue;
        const int64_t* ngram = cur_pre + (step + 1 - n);

        // 1. 先在 input_ids 里搜
        for (int64_t i = 0; i <= input_len - n; ++i) {
            if (d_match_ngram(ngram, n, cur_inp, i)) {
                int64_t start = i + n;
                int64_t end   = min(start + max_draft_tokens_query, input_len);
                if (start < end) {
                    d_match_pos[b] = start;
                    d_draft_cnt[b] = end - start;
                    return;
                }
            }
        }
        // 2. 再去 pre_ids 里搜
        for (int64_t i = 0; i <= step - n; ++i) {
            if (d_match_ngram(ngram, n, cur_pre, i)) {
                int64_t start = i + n;
                int64_t end   = min(start + max_draft_tokens_query, step);
                if (start < end) {
                    d_match_pos[b] = start;
                    d_draft_cnt[b] = end - start;
                    return;
                }
            }
        }
    }
}