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
#include "helper.h"
#include "group_swiglu_with_masked.h"

#pragma once

template <typename index, typename T, int VecSize>
__global__ void group_swiglu_with_masked_kernel(T* act_out,
                                 const T* input,
                                 const index *token_nums_per_expert,
                                 const int64_t group_num,
                                 const int64_t group_size,
                                 const int64_t hidden_dim) {
    int64_t global_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num = group_num * group_size * hidden_dim;
    using LoadT = AlignedVector<T, VecSize>;
    LoadT src_vec0, src_vec1;
    LoadT res_vec;

    int64_t block_id = static_cast<int64_t>(blockIdx.x);
    const int lane_idx = threadIdx.x % 32;

    while(true) {
        int dealt_group_id = -1;
        int dealt_seq_id = -1;
        if (lane_idx == 0 ) {
            int cumsum1 = 0;
            int cumsum2 = 0;
            for (int i = 0; i < group_num; i++) {
                int tmp = token_nums_per_expert[i];
                cumsum2 += tmp;
                if (block_id >= cumsum1 && block_id < cumsum2) {
                    dealt_group_id = i;
                    dealt_seq_id = block_id - cumsum1;
                    break;
                }
                cumsum1 += tmp;
            }
        }
        dealt_group_id = __shfl_sync(0xffffffff, dealt_group_id, 0);
        dealt_seq_id =   __shfl_sync(0xffffffff, dealt_seq_id, 0);
        if (dealt_group_id < 0) break;

        const int64_t r_offset = (dealt_group_id * group_size + dealt_seq_id) * hidden_dim * 2;
        const int64_t w_offset = (dealt_group_id * group_size + dealt_seq_id) * hidden_dim;

        for (int64_t col_id = threadIdx.x * VecSize; col_id < hidden_dim; col_id += blockDim.x * VecSize) {

            Load<T, VecSize>(&input[r_offset + col_id], &src_vec0);
            Load<T, VecSize>(&input[r_offset + col_id + hidden_dim], &src_vec1);

            for (int j = 0; j < VecSize; ++j) {
                float a = static_cast<float>(src_vec0[j]);
                float b = static_cast<float>(src_vec1[j]);
                float res = b * a / (1.f + exp(-a));
                res_vec[j] = static_cast<T>(res);
            }

            Store<T, VecSize>(res_vec, &act_out[w_offset + col_id]);
        }
        block_id += gridDim.x;
    }
}

paddle::Tensor GroupSwigluWithMasked(const paddle::Tensor& fc1_out_tensor,
                                                  const paddle::Tensor& token_nums_per_expert
                                                  )
{
    const int64_t group_num = token_nums_per_expert.shape()[0];
    const int64_t group_size = fc1_out_tensor.shape()[1];
    const int64_t hidden_dim = fc1_out_tensor.shape()[2] / 2;
    auto act_out_tensor = GetEmptyTensor({group_num, group_size, hidden_dim}, fc1_out_tensor.dtype(), fc1_out_tensor.place());

    constexpr int VecSize = 8;
    PD_CHECK(fc1_out_tensor.dtype() == paddle::DataType::BFLOAT16);
    PD_CHECK(hidden_dim % VecSize == 0);

    constexpr paddle::DataType D = paddle::DataType::BFLOAT16;
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int threads = 512;
    const int blocks = 256;

    #define dispatch_by_index(index) {\
    group_swiglu_with_masked_kernel<index, DataType_, VecSize><<<blocks, threads, 0, fc1_out_tensor.stream()>>>(\
        reinterpret_cast<DataType_*>(const_cast<data_t*>(act_out_tensor.data<data_t>())),\
        reinterpret_cast<const DataType_*>(fc1_out_tensor.data<data_t>()),\
        token_nums_per_expert.data<index>(),\
        group_num,\
        group_size,\
        hidden_dim\
    );} while(0)
    if (token_nums_per_expert.dtype() == paddle::DataType::INT64) {
        dispatch_by_index(int64_t);
    } else if(token_nums_per_expert.dtype() == paddle::DataType::INT32) {
        dispatch_by_index(int32_t);
    } else {
        PD_THROW("Unsupported token_nums_per_expert's data dtype.");
    }

    return act_out_tensor;
}




std::vector<paddle::Tensor> GroupSwigluWithMaskedWrapper(
    const paddle::Tensor& input,
    const paddle::Tensor& token_nums_per_expert) {
     return {GroupSwigluWithMasked(input, token_nums_per_expert)};
}

PD_BUILD_STATIC_OP(group_swiglu_with_masked)
    .Inputs({"input",
             "token_nums_per_expert"})
    .Outputs({"output_tensor"})
    .SetKernelFn(PD_KERNEL(GroupSwigluWithMaskedWrapper));
