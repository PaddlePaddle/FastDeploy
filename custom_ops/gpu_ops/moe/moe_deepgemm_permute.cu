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

// topk warps
template<typename T, int VecSize>
__global__ void MoEDeepGEMMPermuteKernel(T* out, int* token_nums_per_expert, int* permute_indices_per_token, const T* x, const int64_t* topk_idx, const int token_num, const int topk, const int num_vecs, const int hidden, const int max_tokens_per_expert) {

    AlignedVector<T, VecSize> in_vec;

    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;
    const int tid = threadIdx.x % 32;
    for (int token_idx = bid; token_idx < token_num; token_idx += gridDim.x) {
        const int tgt_expert_id = topk_idx[token_idx * topk + wid];
        int tgt_expert_token;
        if (tid == 0) {
            tgt_expert_token = atomicAdd(token_nums_per_expert + tgt_expert_id, 1);
            permute_indices_per_token[token_idx * topk + wid] = tgt_expert_token;
        }
        tgt_expert_token = __shfl_sync(0xFFFFFFFF, tgt_expert_token, 0);


        for (int hidden_vec_id = tid; hidden_vec_id < num_vecs; hidden_vec_id += 32) {
            Load<T, VecSize>(x + token_idx * hidden + hidden_vec_id * VecSize, &in_vec);
            Store<T, VecSize>(in_vec, out + tgt_expert_id * max_tokens_per_expert * hidden + tgt_expert_token * hidden + hidden_vec_id * VecSize);
        }
    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> MoEDeepGEMMPermuteDispatch(
    const paddle::Tensor& x,
    const paddle::Tensor& topk_idx,
    const int num_experts,
    const int max_tokens_per_expert
) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int token_num = x.shape()[0];
    const int hidden = x.shape()[1];
    const int topk = topk_idx.shape()[1];

    auto place = x.place();
    auto stream = x.stream();

    auto token_nums_per_expert = GetEmptyTensor({num_experts}, paddle::DataType::INT32, place);
    auto permute_indices_per_token = GetEmptyTensor({token_num, topk}, paddle::DataType::INT32, place);

    PADDLE_ENFORCE_GPU_SUCCESS(cudaMemsetAsync(token_nums_per_expert.data<int32_t>(), 0, num_experts * sizeof(int32_t), stream));

    auto permute_output = GetEmptyTensor({num_experts, max_tokens_per_expert, hidden}, x.dtype(), place);

    auto permute_output_data = permute_output.data<data_t>();

    constexpr int VecSize = 16 / sizeof(data_t);

    int blocks = 32 * topk;
    int grids = min(132 * 4, token_num);
    int num_vecs = hidden / VecSize;

    assert(blocks <= 1024);

    MoEDeepGEMMPermuteKernel<DataType_, VecSize><<<grids, blocks, 0, stream>>>(
        reinterpret_cast<DataType_*>(permute_output_data),
        token_nums_per_expert.data<int32_t>(),
        permute_indices_per_token.data<int32_t>(),
        reinterpret_cast<const DataType_ *>(x.data<data_t>()),
        topk_idx.data<int64_t>(),
        token_num, topk, num_vecs,
        hidden, max_tokens_per_expert
    );

    return {permute_output, token_nums_per_expert, permute_indices_per_token};
}

std::vector<paddle::Tensor> MoEDeepGEMMPermute(
    const paddle::Tensor& x,
    const paddle::Tensor& topk_idx,
    const int num_experts,
    const int max_tokens_per_expert
) {
    switch (x.dtype()) {
        case paddle::DataType::BFLOAT16:
            return MoEDeepGEMMPermuteDispatch<paddle::DataType::BFLOAT16>(
                x, topk_idx, num_experts, max_tokens_per_expert
            );
        case paddle::DataType::FLOAT16:
            return MoEDeepGEMMPermuteDispatch<paddle::DataType::FLOAT16>(
                x, topk_idx, num_experts, max_tokens_per_expert
            );
        default:
            PD_THROW("Unsupported data type");
    }
}

PD_BUILD_STATIC_OP(moe_deepgemm_permute)
    .Inputs({"x", "topk_idx"})
    .Outputs({"permute_output", "token_nums_per_expert", "permute_indices_per_token"})
    .Attrs({"num_experts: int", "max_tokens_per_expert: int"})
    .SetKernelFn(PD_KERNEL(MoEDeepGEMMPermute));
