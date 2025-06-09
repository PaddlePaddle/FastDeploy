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

#include "helper.h"

template <int THREADBLOCK_SIZE>
__global__ void CalculateKernel(int32_t* sum_draft_num,
                                int32_t* sum_accept_num,
                                const int32_t* accept_nums,
                                const int32_t* seq_lens_this_time,
                                const int32_t* seq_lens_decoder,
                                const bool* stop_flags,
                                int real_bsz) {
    int tid = threadIdx.x;
    int draft_num = 0, accept_num = 0;
    if (tid < real_bsz) {
        if (seq_lens_decoder[tid] > 0 &&
            seq_lens_this_time[tid] != seq_lens_decoder[tid]) {
            draft_num = seq_lens_this_time[tid] - 1;
            accept_num = accept_nums[tid] - 1;
        } else if (seq_lens_this_time[tid] > 0 &&
                   stop_flags[tid]) {  // last step
            draft_num = seq_lens_this_time[tid] - 1;
            accept_num = accept_nums[tid];
        }
    }
    __syncthreads();
    typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int draft_nums_sum = BlockReduce(temp_storage).Sum(draft_num);
    int accept_nums_sum = BlockReduce(temp_storage).Sum(accept_num);

    if (tid == 0 && draft_nums_sum != 0) {
        sum_draft_num[0] += draft_nums_sum;
        sum_accept_num[0] += accept_nums_sum;
    }
}

void Calculate(const paddle::Tensor& sum_draft_num,
               const paddle::Tensor& sum_accept_num,
               const paddle::Tensor& accept_nums,
               const paddle::Tensor& seq_lens_this_time,
               const paddle::Tensor& seq_lens_decoder,
               const paddle::Tensor& stop_flags) {
    int real_bsz = seq_lens_this_time.shape()[0];
    constexpr int BLOCK_SIZE = 512;

    CalculateKernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, accept_nums.stream()>>>(
        const_cast<int*>(sum_draft_num.data<int32_t>()),
        const_cast<int*>(sum_accept_num.data<int32_t>()),
        accept_nums.data<int32_t>(),
        seq_lens_this_time.data<int32_t>(),
        seq_lens_decoder.data<int32_t>(),
        stop_flags.data<bool>(),
        real_bsz);
}

PD_BUILD_STATIC_OP(speculate_calcu_accept_ratio)
    .Inputs({"sum_draft_num",
             "sum_accept_num",
             "accept_nums",
             "seq_lens_this_time",
             "seq_lens_decoder",
             "stop_flags"})
    .Outputs({"sum_draft_num_out", "sum_accept_num_out"})
    .SetInplaceMap({{"sum_draft_num", "sum_draft_num_out"},
                    {"sum_accept_num", "sum_accept_num_out"}})
    .SetKernelFn(PD_KERNEL(Calculate));
