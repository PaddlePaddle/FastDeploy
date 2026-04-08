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

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void __global__
update_split_fuse_inputs_kernel(int* split_fuse_seq_lens,
                                int* split_fuse_cur_seq_lens,
                                int64_t* split_fuse_all_input_ids,
                                int64_t* input_ids,
                                int* seq_lens_this_time,
                                int* seq_lens_encoder,
                                int* seq_lens_decoder,
                                int64_t* step_idx,
                                const int split_fuse_size,
                                const int max_seq_len) {
    const int bi = blockIdx.x;
    const int tidx = threadIdx.x;
    if (split_fuse_seq_lens[bi] <= 0) {
        return;
    }
    if (split_fuse_cur_seq_lens[bi] < split_fuse_seq_lens[bi]) {
        const int cur_add_tokens =
            min(split_fuse_seq_lens[bi] - split_fuse_cur_seq_lens[bi],
                split_fuse_size);
        int64_t* split_fuse_all_input_ids_cur_batch =
            split_fuse_all_input_ids + bi * max_seq_len +
            split_fuse_cur_seq_lens[bi];
        int64_t* input_ids_cur_batch = input_ids + bi * max_seq_len;
        for (int i = tidx; i < cur_add_tokens; i += blockDim.x) {
            input_ids_cur_batch[i] = split_fuse_all_input_ids_cur_batch[i];
        }
        if (threadIdx.x == 0) {
            seq_lens_this_time[bi] = cur_add_tokens;
            seq_lens_encoder[bi] = cur_add_tokens;
            seq_lens_decoder[bi] = split_fuse_cur_seq_lens[bi];
            step_idx[bi] = 0;
            split_fuse_cur_seq_lens[bi] += cur_add_tokens;
        }
    } else if (split_fuse_cur_seq_lens[bi] >= split_fuse_seq_lens[bi]) {
        if (threadIdx.x == 0) {
            seq_lens_decoder[bi] = split_fuse_cur_seq_lens[bi];
            seq_lens_this_time[bi] = 1;
            step_idx[bi] = 1;
            seq_lens_encoder[bi] = 0;
            split_fuse_cur_seq_lens[bi] = 0;
            split_fuse_seq_lens[bi] = 0;
        }
    }
}

void UpdateSplitFuseInputes(const paddle::Tensor& split_fuse_seq_lens,
                            const paddle::Tensor& split_fuse_cur_seq_lens,
                            const paddle::Tensor& split_fuse_all_input_ids,
                            const paddle::Tensor& input_ids,
                            const paddle::Tensor& seq_lens_this_time,
                            const paddle::Tensor& seq_lens_encoder,
                            const paddle::Tensor& seq_lens_decoder,
                            const paddle::Tensor& step_idx,
                            const int max_seq_len,
                            const int max_batch_size,
                            const int split_fuse_size) {
    dim3 grids;
    grids.x = max_batch_size;
    const int block_size = 128;
    update_split_fuse_inputs_kernel<<<grids,
                                      block_size,
                                      0,
                                      input_ids.stream()>>>(
        const_cast<int*>(split_fuse_seq_lens.data<int>()),
        const_cast<int*>(split_fuse_cur_seq_lens.data<int>()),
        const_cast<int64_t*>(split_fuse_all_input_ids.data<int64_t>()),
        const_cast<int64_t*>(input_ids.data<int64_t>()),
        const_cast<int*>(seq_lens_this_time.data<int>()),
        const_cast<int*>(seq_lens_encoder.data<int>()),
        const_cast<int*>(seq_lens_decoder.data<int>()),
        const_cast<int64_t*>(step_idx.data<int64_t>()),
        split_fuse_size,
        max_seq_len);
}

PD_BUILD_STATIC_OP(update_split_fuse_inputs)
    .Inputs(
        {"split_fuse_seq_lens",  // 当前query的长度
         "split_fuse_cur_seq_lens",  // 当前query已经计算完成的长度，是split
                                     // size的整数倍
         "split_fuse_all_input_ids",  // 当前query经过split的input
                                      // ids，长度是split size的整数倍
         "input_ids",                 // 当前query所有的input ids
         "seq_lens_this_time",        // 当前query需要计算的长度
         "seq_lens_encoder",  // 当前query encoder需要计算的长度,decoder时为0
         "seq_lens_decoder",  // 当前query decoder需要计算的长度,encoder时为0
         "step_idx"})  // 当前query的token index，首token为0，第二个token为1
    .Outputs({"input_ids_out"})
    .Attrs({"max_seq_len: int",       // 最大的seq len
            "max_batch_size: int",    // 最大的batch size
            "split_fuse_size: int"})  // 切分的长度
    .SetInplaceMap({{"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(UpdateSplitFuseInputes));
