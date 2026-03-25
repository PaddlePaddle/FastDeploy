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

// Shared ngram matching logic used by both ngram_match_kernel and
// ngram_match_mixed_kernel.  Extracted per upstream requirement:
// "两个Kernel逻辑有较为相似部分，Kernel 形式为提取共用的匹配逻辑，外加业务逻辑"

// ------------------------------------------------------------
// ngram_search_and_copy — Core sliding-window ngram match.
//
// Searches for `ngram[0..ngram_size-1]` in `haystack[0..haystack_len-1]`.
// On first match at position i, copies tokens from haystack[i+ngram_size ..]
// into draft_tokens[write_offset ..], capped by max_draft_tokens and
// haystack_len.  Updates seq_lens_this_time to write_offset + n_copied.
//
// Returns true if a match was found and tokens were written.
// ------------------------------------------------------------
__device__ __forceinline__ bool ngram_search_and_copy(
    const int64_t *haystack,
    int64_t haystack_len,
    const int64_t *ngram,
    int ngram_size,
    int64_t *draft_tokens,
    int write_offset,
    int max_draft_tokens,
    int32_t *seq_lens_this_time_ptr) {
  for (int64_t i = 0; i <= haystack_len - ngram_size; ++i) {
    bool match = true;
    for (int j = 0; j < ngram_size; j++) {
      if (ngram[j] != haystack[i + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      int64_t start_idx = i + ngram_size;
      int64_t end_idx =
          min(start_idx + static_cast<int64_t>(max_draft_tokens), haystack_len);
      if (start_idx >= end_idx) continue;

      int64_t n = end_idx - start_idx;
      *seq_lens_this_time_ptr = static_cast<int32_t>(write_offset + n);
      for (int64_t k = 0; k < n; k++) {
        draft_tokens[write_offset + k] = haystack[start_idx + k];
      }
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------
// ngram_search_batch_item — Two-phase search for one batch item.
//
// Phase 1: search in input_ids (prompt tokens).
// Phase 2: if no match, search in pre_ids (previously generated tokens).
//
// The pre_ids search uses cur_step_idx as the haystack length
// (only tokens up to the current step are valid).
//
// write_offset controls where matched tokens are written:
//   - ngram_match:       write_offset = 1
//   - ngram_match_mixed: write_offset = ori_seq_len_this_time
// ------------------------------------------------------------
__device__ __forceinline__ bool ngram_search_batch_item(
    const int64_t *cur_input_ids,
    int64_t cur_input_ids_len,
    const int64_t *cur_pre_ids,
    int64_t cur_step_idx,
    int64_t *cur_draft_tokens,
    int32_t *seq_lens_this_time_ptr,
    int max_ngram_size,
    int min_ngram_size,
    int max_draft_tokens,
    int write_offset) {
  for (int ngram_size = max_ngram_size; ngram_size >= min_ngram_size;
       --ngram_size) {
    if (cur_step_idx < ngram_size) continue;

    const int64_t *ngram = cur_pre_ids + (cur_step_idx + 1 - ngram_size);

    // Phase 1: search in input_ids (prompt tokens)
    if (ngram_search_and_copy(cur_input_ids,
                              cur_input_ids_len,
                              ngram,
                              ngram_size,
                              cur_draft_tokens,
                              write_offset,
                              max_draft_tokens,
                              seq_lens_this_time_ptr)) {
      return true;
    }

    // Phase 2: search in pre_ids (previously generated tokens)
    if (ngram_search_and_copy(cur_pre_ids,
                              cur_step_idx,
                              ngram,
                              ngram_size,
                              cur_draft_tokens,
                              write_offset,
                              max_draft_tokens,
                              seq_lens_this_time_ptr)) {
      return true;
    }
  }
  return false;
}
