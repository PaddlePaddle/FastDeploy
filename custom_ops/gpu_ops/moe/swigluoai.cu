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
#include "swigluoai.h"

#pragma once


// dim3 grid(256)
// dim3 block(512)
template <typename T, int VecSize>
__global__ void swigluoai_interleave_kernel(T* act_out,
                                 const T* input,
                                 const float alpha,
                                 const float limit,
                                 const int64_t seq_len,
                                 const int64_t hidden_dim) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num = seq_len * hidden_dim;
    using LoadT = AlignedVector<T, VecSize>;
    LoadT src_vec0, src_vec1;
    LoadT res_vec;

    int64_t vec_num = hidden_dim / VecSize * seq_len;
    int64_t col_size = hidden_dim / VecSize;
    int64_t times = (vec_num - 1) / (gridDim.x * blockDim.x) + 1;

    for(int i = 0; i < times; i++)
    {
        int64_t index = tid + i * gridDim.x * blockDim.x ;
        int64_t row = index / col_size;
        int64_t col = index % col_size;

        if(row < seq_len && col < col_size)
        {
            Load<T, VecSize>(&input[row*hidden_dim*2 + col*VecSize*2], &src_vec0);
            Load<T, VecSize>(&input[row*hidden_dim*2 + col*VecSize*2 + VecSize], &src_vec1);

            for (int j = 0; j < VecSize/2; ++j) {
                float a = static_cast<float>(src_vec0[2*j]);
                float b = static_cast<float>(src_vec0[2*j + 1]);
                a = fminf(a, limit);
                b = fminf(fmaxf(b,-limit), limit);
                float res = (b + 1) * a / (1.f + expf(-a * alpha));
                res_vec[j] = static_cast<T>(res);
            }
            for (int j = 0; j < VecSize/2; ++j) {
                float a = static_cast<float>(src_vec1[2*j]);
                float b = static_cast<float>(src_vec1[2*j + 1]);
                a = fminf(a, limit);
                b = fminf(fmaxf(b,-limit), limit);
                float res = (b + 1) * a / (1.f + expf(-a * alpha));
                res_vec[j + VecSize/2] = static_cast<T>(res);
            }

            Store<T, VecSize>(res_vec, &act_out[row*hidden_dim + col*VecSize]);
        }
    }
}


// dim3 grid(256)
// dim3 block(512)
template <typename T, int VecSize>
__global__ void swigluoai_norm_kernel(T* act_out,
                                 const T* input,
                                 const float alpha,
                                 const float limit,
                                 const int64_t seq_len,
                                 const int64_t hidden_dim) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t num = seq_len * hidden_dim;
    using LoadT = AlignedVector<T, VecSize>;
    LoadT src_vec0, src_vec1;
    LoadT res_vec;

    int64_t vec_num = hidden_dim / VecSize * seq_len;
    int64_t col_size = hidden_dim / VecSize;
    int64_t times = (vec_num - 1) / (gridDim.x * blockDim.x) + 1;

    for(int i = 0; i < times; i++)
    {
        int64_t index = tid + i * gridDim.x * blockDim.x ;
        int64_t row = index / col_size;
        int64_t col = index % col_size;

        if(row < seq_len && col < col_size)
        {
            Load<T, VecSize>(&input[row*hidden_dim*2 + col*VecSize], &src_vec0);
            Load<T, VecSize>(&input[row*hidden_dim*2 + hidden_dim + col*VecSize], &src_vec1);

            for (int j = 0; j < VecSize; ++j) {
                float a = static_cast<float>(src_vec0[j]);
                float b = static_cast<float>(src_vec1[j]);
                float z = fminf(fmaxf(a * alpha, -limit), limit);
                float res = b * a / (1.f + expf(-z));
                res_vec[j] = static_cast<T>(res);
            }

            Store<T, VecSize>(res_vec, &act_out[row*hidden_dim + col*VecSize]);
        }
    }
}

paddle::Tensor SwigluOAI(const paddle::Tensor &fc1_out_tensor, const float alpha, const float limit, const std::string& type)
{
    // const int64_t group_size = fc1_out_tensor.shape()[1];
    const int64_t seq_len = fc1_out_tensor.shape()[0];
    const int64_t hidden_dim = fc1_out_tensor.shape()[1] / 2;
    auto act_out_tensor = GetEmptyTensor({seq_len, hidden_dim}, fc1_out_tensor.dtype(), fc1_out_tensor.place());

    constexpr int VecSize = 8;
    PD_CHECK(fc1_out_tensor.dtype() == paddle::DataType::BFLOAT16);
    PD_CHECK(hidden_dim % VecSize == 0);

    constexpr paddle::DataType D = paddle::DataType::BFLOAT16;
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int block_size = 512;
    const int grid_size = 256;

    #define dispatch_norm() do {\
    swigluoai_norm_kernel<DataType_, VecSize><<<grid_size, block_size, 0, fc1_out_tensor.stream()>>>(\
        reinterpret_cast<DataType_*>(const_cast<data_t*>(act_out_tensor.data<data_t>())),\
        reinterpret_cast<const DataType_*>(fc1_out_tensor.data<data_t>()),\
        alpha,\
        limit,\
        seq_len,\
        hidden_dim\
    );} while(0)

    #define dispatch_interleave() do {\
    swigluoai_interleave_kernel<DataType_, VecSize><<<grid_size, block_size, 0, fc1_out_tensor.stream()>>>(\
        reinterpret_cast<DataType_*>(const_cast<data_t*>(act_out_tensor.data<data_t>())),\
        reinterpret_cast<const DataType_*>(fc1_out_tensor.data<data_t>()),\
        alpha,\
        limit,\
        seq_len,\
        hidden_dim\
    );} while(0)

    if(type == "interleave")
    {
        dispatch_interleave();
    }
    else
    {
        dispatch_norm();
    }
    // if (token_nums_per_expert.dtype() == paddle::DataType::INT64) {
    //     dispatch_by_index(int64_t);
    // } else if(token_nums_per_expert.dtype() == paddle::DataType::INT32) {
    //     dispatch_by_index(int32_t);
    // } else {
    //     PD_THROW("Unsupported token_nums_per_expert's data dtype.");
    // }

    return act_out_tensor;
}


std::vector<paddle::Tensor> SwigluOAIWrapper(
    const paddle::Tensor& fc1_out_tensor,
    const float alpha,
    const float limit,
    const std::string& type) {
     return {SwigluOAI(fc1_out_tensor, alpha, limit, type)};
}

PD_BUILD_STATIC_OP(swigluoai)
    .Inputs({"fc1_out_tensor"})
    .Attrs({"alpha: float", "limit: float", "type: std::string"})
    .Outputs({"output_tensor"})
    .SetKernelFn(PD_KERNEL(SwigluOAIWrapper));
