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

template<typename T, int VecSize, int TopK>
__global__ void MoEDeepGEMMDePermuteKernel(T* out, const T* ffn_out, const int* permute_indices_per_token, const int64_t* topk_idx, const float* topk_weights, const int token_num, const int num_vecs, const int hidden, const int max_tokens_per_expert) {
    AlignedVector<T, VecSize> in_vec;

    AlignedVector<T, VecSize> acc_vec[TopK];

    const int bid = blockIdx.x;
    const int wid = threadIdx.x / 32;
    const int tid = threadIdx.x % 32;
    extern __shared__ char shm[]; // TopK * hidden
    T* shm_hidden = reinterpret_cast<T*>(shm);

    for (int token_idx = bid; token_idx < token_num; token_idx += gridDim.x) {
        int src_expert_id = topk_idx[token_idx * TopK + wid];
        int src_expert_token = permute_indices_per_token[token_idx * TopK + wid];
        float weight = topk_weights[token_idx * TopK + wid];

        for (int hidden_vec_id = tid; hidden_vec_id < num_vecs; hidden_vec_id += 32) {
            Load<T, VecSize>(ffn_out + src_expert_id * max_tokens_per_expert * hidden + src_expert_token * hidden + hidden_vec_id * VecSize, &in_vec);
#pragma unroll
            for (int i = 0; i < VecSize; i++) {
                in_vec[i] *= weight;
            }
            Store<T, VecSize>(in_vec, shm_hidden + wid * hidden + hidden_vec_id * VecSize);
        }

        __syncthreads();

        for (int hidden_vec_id = threadIdx.x; hidden_vec_id < num_vecs; hidden_vec_id += blockDim.x) {
#pragma unroll
            for (int topk_id = 0; topk_id < TopK; topk_id++) {
                Load<T, VecSize>(shm_hidden + topk_id * hidden + hidden_vec_id * VecSize, &acc_vec[topk_id]);
            }
#pragma unroll
            for (int i = 0; i < VecSize; i++) {
#pragma unroll
                 for (int topk_id = 1; topk_id < TopK; topk_id++) {
                    acc_vec[0][i] += acc_vec[topk_id][i];
                 }
            }
            Store<T, VecSize>(acc_vec[0], out + token_idx * hidden + hidden_vec_id * VecSize);
        }

    }
}

template <paddle::DataType D>
std::vector<paddle::Tensor> MoEDeepGEMMDePermuteDispatch(
    const paddle::Tensor& ffn_out, // [num_experts, max_tokens_per_expert, hidden]
    const paddle::Tensor& permute_indices_per_token, // [token_num, topk}]
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& topk_weights
) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;

    const int token_num = permute_indices_per_token.shape()[0];
    const int max_tokens_per_expert = ffn_out.shape()[1];
    const int hidden = ffn_out.shape()[2];
    const int topk = permute_indices_per_token.shape()[1];

    auto place = ffn_out.place();
    auto stream = ffn_out.stream();

    auto out = GetEmptyTensor({token_num, hidden}, ffn_out.dtype(), place);

    constexpr int VecSize = 16 / sizeof(data_t);
    int blocks = 32 * topk;
    int grids = min(132 * 4, token_num);
    int num_vecs = hidden / VecSize;

    assert(blocks <= 1024);
    int dyn_smem_size = 0;

    switch (topk) {
        case 4:
        dyn_smem_size =  topk * hidden * sizeof(DataType_);
        if (dyn_smem_size >= (48 << 10)) {
            cudaFuncSetAttribute(
                MoEDeepGEMMDePermuteKernel<DataType_, VecSize, 4>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dyn_smem_size);
        }
        MoEDeepGEMMDePermuteKernel<DataType_, VecSize, 4><<<grids, blocks, dyn_smem_size, stream>>>(
            reinterpret_cast<DataType_*>(out.data<data_t>()),
            reinterpret_cast<const DataType_*>(ffn_out.data<data_t>()),
            permute_indices_per_token.data<int32_t>(),
            topk_idx.data<int64_t>(),
            topk_weights.data<float>(),
            token_num, num_vecs, hidden, max_tokens_per_expert
        );
        break;

        case 8:
        dyn_smem_size =  topk * hidden * sizeof(DataType_);
        if (dyn_smem_size >= (48 << 10)) {
            cudaFuncSetAttribute(
                MoEDeepGEMMDePermuteKernel<DataType_, VecSize, 8>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                dyn_smem_size);
        }
        MoEDeepGEMMDePermuteKernel<DataType_, VecSize, 8><<<grids, blocks, topk * hidden * sizeof(DataType_), stream>>>(
            reinterpret_cast<DataType_*>(out.data<data_t>()),
            reinterpret_cast<const DataType_*>(ffn_out.data<data_t>()),
            permute_indices_per_token.data<int32_t>(),
            topk_idx.data<int64_t>(),
            topk_weights.data<float>(),
            token_num, num_vecs, hidden, max_tokens_per_expert
        );
        break;

        default:
        PD_THROW("Unsupported topk");
    }
    return {out};
}


std::vector<paddle::Tensor> MoEDeepGEMMDePermute(
    const paddle::Tensor& ffn_out, // [num_experts, max_tokens_per_expert, hidden]
    const paddle::Tensor& permute_indices_per_token, // [token_num, topk}]
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& topk_weights
) {
    switch (ffn_out.dtype()) {
        case paddle::DataType::BFLOAT16:
            return MoEDeepGEMMDePermuteDispatch<paddle::DataType::BFLOAT16>(
                ffn_out, permute_indices_per_token, topk_idx, topk_weights
            );
        case paddle::DataType::FLOAT16:
            return MoEDeepGEMMDePermuteDispatch<paddle::DataType::FLOAT16>(
                ffn_out, permute_indices_per_token, topk_idx, topk_weights
            );
        default:
            PD_THROW("Unsupported data type");
    }
}

PD_BUILD_STATIC_OP(moe_deepgemm_depermute)
    .Inputs({"ffn_out", "permute_indices_per_token", "topk_idx", "topk_weights"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(MoEDeepGEMMDePermute));
