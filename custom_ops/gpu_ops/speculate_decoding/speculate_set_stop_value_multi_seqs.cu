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

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif
// #define DEBUG_SPEC_STOP_SEQS

__global__ void spec_set_value_by_stop_seqs(bool *stop_flags,
                                            int64_t *accept_tokens,
                                            int *accept_nums,
                                            const int64_t *pre_ids,
                                            const int64_t *step_idx,
                                            const int64_t *stop_seqs,
                                            const int *stop_seqs_len,
                                            const int *seq_lens,
                                            const int64_t *end_ids,
                                            const int bs,
                                            const int accept_tokens_len,
                                            const int stop_seqs_bs,
                                            const int stop_seqs_max_len,
                                            const int pre_ids_len) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  if (tid >= stop_seqs_bs) return;
  const int stop_seq_len = stop_seqs_len[bid * stop_seqs_bs + tid];
  if (stop_seq_len <= 0) return;
  if (bid < bs) {
    const int64_t *stop_seq_now = stop_seqs +
                                  bid * stop_seqs_max_len * stop_seqs_bs +
                                  tid * stop_seqs_max_len;
    const int64_t *pre_ids_now = pre_ids + bid * pre_ids_len;
    int64_t *accept_tokens_now = accept_tokens + bid * accept_tokens_len;
    const int accept_num = accept_nums[bid];
    const int64_t step_idx_now = step_idx[bid];
    if (!stop_flags[bid]) {
      int accept_idx = 0;
      bool is_end = false;
      // 遍历起始位置
      for (; accept_idx <= accept_num - 1 && !is_end; accept_idx++) {
        if (step_idx_now - accept_num + accept_idx + 1 < stop_seq_len) {
#ifdef DEBUG_SPEC_STOP_SEQS
          printf("num %d < stop_seq_len %d\n",
                 step_idx_now - accept_num + accept_idx + 1,
                 stop_seq_len);
#endif
          continue;
        }
        // 遍历一个 stop_seqs
        for (int i = stop_seq_len - 1; i >= 0; --i) {
          int64_t cur_token_idx = -1;

          // 通过当前值判断 token 是在 pre_ids 还是 accept_token 里
          if (stop_seq_len - 1 - i < accept_idx) {
#ifdef DEBUG_SPEC_STOP_SEQS
            printf(
                "AcceptTokens bid:%d. tid:%d, accept_idx:%d, "
                "accept_token_idx: "
                "%d\n",
                bid,
                tid,
                accept_idx,
                accept_idx - (stop_seq_len - 1 - i) - 1);
#endif
            cur_token_idx =
                accept_tokens_now[accept_idx - (stop_seq_len - 1 - i) - 1];
          } else {
#ifdef DEBUG_SPEC_STOP_SEQS
            printf(
                "PreIds bid:%d. tid:%d, step_idx_now:%ld. "
                "accept_idx:%d. "
                "pre_id_idx: %ld\n",
                bid,
                tid,
                step_idx_now,
                accept_idx,
                step_idx_now - accept_num + accept_idx -
                    (stop_seq_len - 1 - i));
#endif
            int pre_ids_idx =
                step_idx_now - accept_num + accept_idx - (stop_seq_len - 1 - i);
            // EC3
            // 特殊拼接会导致input_ids最后一位无特殊token，即pre_ids[0]可能为23,
            // 导致异常结束
            if (pre_ids_idx <= 0) {
              break;
            }
            cur_token_idx = pre_ids_now[pre_ids_idx];
          }
#ifdef DEBUG_SPEC_STOP_SEQS
          printf(
              "bid:%d. tid:%d, cur_token_idx: %ld. stop_seq_now "
              "%ld\n",
              bid,
              tid,
              cur_token_idx,
              stop_seq_now[i]);
#endif
          if (cur_token_idx != stop_seq_now[i]) {
            break;
          }
          if (i == 0) {
            is_end = true;
          }
        }
      }
      if (is_end) {
#ifdef DEBUG_SPEC_STOP_SEQS
        printf("bid:%d end with accept_idx %d", bid, accept_idx);
#endif

        accept_nums[bid] = accept_idx;
        accept_tokens_now[accept_idx - 1] = end_ids[0];
        stop_flags[bid] = true;
      }
    }
  }
}

void SpecGetStopFlagsMultiSeqs(const paddle::Tensor &accept_tokens,
                               const paddle::Tensor &accept_num,
                               const paddle::Tensor &pre_ids,
                               const paddle::Tensor &step_idx,
                               const paddle::Tensor &stop_flags,
                               const paddle::Tensor &seq_lens,
                               const paddle::Tensor &stop_seqs,
                               const paddle::Tensor &stop_seqs_len,
                               const paddle::Tensor &end_ids) {
  PD_CHECK(accept_tokens.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);

  auto cu_stream = accept_tokens.stream();
  std::vector<int64_t> shape = accept_tokens.shape();
  std::vector<int64_t> stop_seqs_shape = stop_seqs.shape();
  int bs_now = shape[0];
  int stop_seqs_bs = stop_seqs_shape[1];
  int stop_seqs_max_len = stop_seqs_shape[2];
  int pre_ids_len = pre_ids.shape()[1];
  int accept_tokens_len = accept_tokens.shape()[1];

  int block_size = (stop_seqs_bs + 31) / 32 * 32;
  spec_set_value_by_stop_seqs<<<bs_now, block_size, 0, cu_stream>>>(
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<int *>(accept_num.data<int>()),
      pre_ids.data<int64_t>(),
      step_idx.data<int64_t>(),
      stop_seqs.data<int64_t>(),
      stop_seqs_len.data<int>(),
      seq_lens.data<int>(),
      end_ids.data<int64_t>(),
      bs_now,
      accept_tokens_len,
      stop_seqs_bs,
      stop_seqs_max_len,
      pre_ids_len);
}

PD_BUILD_STATIC_OP(speculate_set_stop_value_multi_seqs)
    .Inputs({"accept_tokens",
             "accept_num",
             "pre_ids",
             "step_idx",
             "stop_flags",
             "seq_lens",
             "stop_seqs",
             "stop_seqs_len",
             "end_ids"})
    .Outputs({"accept_tokens_out", "stop_flags_out"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(SpecGetStopFlagsMultiSeqs));
