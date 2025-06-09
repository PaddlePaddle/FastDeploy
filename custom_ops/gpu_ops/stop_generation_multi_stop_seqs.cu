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
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

__global__ void set_value_by_stop_seqs(bool *stop_flags,
                                       int64_t *topk_ids,
                                       const int64_t *pre_ids,
                                       const int64_t *step_idx,
                                       const int64_t *stop_seqs,
                                       const int *stop_seqs_len,
                                       const int *seq_lens,
                                       const int64_t *end_ids,
                                       const int bs,
                                       const int stop_seqs_bs,
                                       const int stop_seqs_max_len,
                                       const int pre_ids_len) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (tid >= stop_seqs_bs) return;

    const int stop_seq_len = stop_seqs_len[tid];
    if (stop_seq_len <= 0) return;
    const int64_t *stop_seq_now = stop_seqs + tid * stop_seqs_max_len;
    const int64_t *pre_ids_now = pre_ids + bid * pre_ids_len;
    const int64_t step_idx_now = step_idx[bid];
    if (bid < bs) {
        if (stop_flags[bid]) {  // 长度超限，当前位置置为2
            topk_ids[bid] = end_ids[0];
            if (seq_lens[bid] == 0) {  // 已终止，当前位置置为-1
                topk_ids[bid] = -1;
            }
            return;
        }
        bool is_end = true;
        int count = 1;
        if (topk_ids[bid] == end_ids[0]) {
            if (tid == 0) {
                stop_flags[bid] = true;
            }
            return;
        }
        for (int i = stop_seq_len - 1; i >= 0; --i) {
            if ((step_idx_now - count) < 0 ||
                pre_ids_now[step_idx_now - count++] != stop_seq_now[i]) {
                is_end = false;
                break;
            }
        }
        if (is_end) {
            topk_ids[bid] = end_ids[0];
            stop_flags[bid] = true;
        }
    }
}

void GetStopFlagsMultiSeqs(const paddle::Tensor &topk_ids,
                           const paddle::Tensor &pre_ids,
                           const paddle::Tensor &step_idx,
                           const paddle::Tensor &stop_flags,
                           const paddle::Tensor &seq_lens,
                           const paddle::Tensor &stop_seqs,
                           const paddle::Tensor &stop_seqs_len,
                           const paddle::Tensor &end_ids) {
    PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
    PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);

    auto cu_stream = topk_ids.stream();
    std::vector<int64_t> shape = topk_ids.shape();
    std::vector<int64_t> stop_seqs_shape = stop_seqs.shape();
    int bs_now = shape[0];
    int stop_seqs_bs = stop_seqs_shape[0];
    int stop_seqs_max_len = stop_seqs_shape[1];
    int pre_ids_len = pre_ids.shape()[1];

    int block_size = (stop_seqs_bs + 31) / 32 * 32;
    set_value_by_stop_seqs<<<bs_now, block_size, 0, cu_stream>>>(
        const_cast<bool *>(stop_flags.data<bool>()),
        const_cast<int64_t *>(topk_ids.data<int64_t>()),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        stop_seqs.data<int64_t>(),
        stop_seqs_len.data<int>(),
        seq_lens.data<int>(),
        end_ids.data<int64_t>(),
        bs_now,
        stop_seqs_bs,
        stop_seqs_max_len,
        pre_ids_len);
}

PD_BUILD_STATIC_OP(set_stop_value_multi_seqs)
    .Inputs({"topk_ids",
             "pre_ids",
             "step_idx",
             "stop_flags",
             "seq_lens",
             "stop_seqs",
             "stop_seqs_len",
             "end_ids"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .SetInplaceMap({{"topk_ids", "topk_ids_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMultiSeqs));
