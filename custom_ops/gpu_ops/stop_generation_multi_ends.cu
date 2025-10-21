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

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include "helper.h"
#include "paddle/extension.h"

__global__ void set_value_by_flags(bool *stop_flags,
                                   int64_t *topk_ids,
                                   int64_t *next_tokens,
                                   const int64_t *end_ids,
                                   const int *seq_lens,
                                   const int bs,
                                   const int end_length,
                                   const int64_t *pre_ids,
                                   const int pre_ids_len,
                                   const int64_t *step_idx,
                                   const int64_t *stop_seqs,
                                   const int *stop_seqs_len,
                                   const int stop_seqs_bs,
                                   const int stop_seqs_max_len,
                                   bool beam_search,
                                   bool prefill_one_step_stop) {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if (tid >= stop_seqs_bs) return;
    if (bid < bs) {
        if(tid == 0){
            if (prefill_one_step_stop) {
                stop_flags[bid] = true;
                if (seq_lens[bid] == 0) {
                    topk_ids[bid] = -1;
                }
                next_tokens[bid] = topk_ids[bid];
            } else {
                if (stop_flags[bid]) {
                    if (seq_lens[bid] == 0) {
                        topk_ids[bid] = -1;
                    } else {
                        topk_ids[bid] = end_ids[0];
                        next_tokens[bid] = end_ids[0];
                    }
                } else {
                    next_tokens[bid] = topk_ids[bid];
                }
            }
            if (!beam_search && is_in_end(topk_ids[bid], end_ids, end_length)) {
                stop_flags[bid] = true;
                topk_ids[bid] = end_ids[0];
                next_tokens[bid] = end_ids[0];
            }
        }
        // dealing stop_seqs
        const int stop_seq_len = (stop_seqs_len + bid * stop_seqs_bs)[tid];
        if (stop_seq_len <= 0) return;
        const int64_t *stop_seq_now = stop_seqs + bid * stop_seqs_bs + tid * stop_seqs_max_len;
        const int64_t *pre_ids_now = pre_ids + bid * pre_ids_len;
        const int64_t step_idx_now = step_idx[bid];

        bool is_end = true;
        int count = 1;
        for (int i = stop_seq_len - 1; i >= 0; --i) {
            if ((step_idx_now - count) < 0 ||
                pre_ids_now[step_idx_now - count++] != stop_seq_now[i]) {
                is_end = false;
                break;
            }
        }
        if (is_end) {
            next_tokens[bid] = end_ids[0];
            stop_flags[bid] = true;
            topk_ids[bid] = end_ids[0];
        }
    }
}

void GetStopFlagsMulti(const paddle::Tensor &topk_ids,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens,
                       const paddle::Tensor &end_ids,
                       const paddle::Tensor &next_tokens,
                       const paddle::Tensor &pre_ids,
                       const paddle::Tensor &step_idx,
                       const paddle::Tensor &stop_seqs,
                       const paddle::Tensor &stop_seqs_len,
                       const bool beam_search) {
    PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
    bool prefill_one_step_stop = false;
    if (const char *env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP")) {
        // std::cout << "Your PATH is: " << env_p << '\n';
        if (env_p[0] == '1') {
            prefill_one_step_stop = true;
        }
    }

#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto dev_ctx = static_cast<const phi::CustomContext*>(paddle::experimental::DeviceContextPool::Instance().Get(topk_ids.place()));
    auto cu_stream = dev_ctx->stream();
#else
    auto cu_stream = topk_ids.stream();
#endif
    std::vector<int64_t> shape = topk_ids.shape();
    int64_t bs_now = shape[0];
    int64_t end_length = end_ids.shape()[0];
    int stop_seqs_bs = stop_seqs.shape()[1];
    int stop_seqs_max_len = stop_seqs.shape()[2];
    int block_size = (stop_seqs_bs + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
    set_value_by_flags<<<bs_now, block_size, 0, cu_stream>>>(
        const_cast<bool *>(stop_flags.data<bool>()),
        const_cast<int64_t *>(topk_ids.data<int64_t>()),
        const_cast<int64_t *>(next_tokens.data<int64_t>()),
        end_ids.data<int64_t>(),
        seq_lens.data<int>(),
        bs_now,
        end_length,
        pre_ids.data<int64_t>(),
        pre_ids.shape()[1],
        step_idx.data<int64_t>(),
        stop_seqs.data<int64_t>(),
        stop_seqs_len.data<int>(),
        stop_seqs_bs,
        stop_seqs_max_len,
        beam_search,
        prefill_one_step_stop);
}

PD_BUILD_STATIC_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "seq_lens", "end_ids", "next_tokens", "pre_ids", "step_idx", "stop_seqs", "stop_seqs_len"})
    .Attrs({"beam_search: bool"})
    .Outputs({"topk_ids_out", "stop_flags_out", "next_tokens_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti));
