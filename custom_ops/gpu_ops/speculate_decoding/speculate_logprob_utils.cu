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
__global__ void get_token_num_per_batch_kernel(int* batch_token_num,
                                               int* total_token_num,
                                               const int* seq_lens_this_time,
                                               const int* seq_lens_encoder,
                                               const int real_bsz) {
    int bid = threadIdx.x;
    typedef cub::BlockReduce<int, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int token_num_now = 0;
    if (bid < real_bsz) {
        token_num_now = seq_lens_encoder[bid] > 0 ? 2 : seq_lens_this_time[bid];
        batch_token_num[bid] = token_num_now;
    }

    __syncthreads();
    int token_num_sum = BlockReduce(temp_storage).Sum(token_num_now);
    if (bid == 0) {
        total_token_num[0] = token_num_sum;
    }
}

template <int VecSize>
__global__ void speculate_get_logits_kernel(float* draft_logits,
                                            const float* logits,
                                            const float* first_token_logits,
                                            const int* cu_seqlens_q,
                                            const int* cu_batch_token_offset,
                                            const int* seq_lens_this_time,
                                            const int* seq_lens_encoder,
                                            const int vocab_size,
                                            const int real_bsz) {
    AlignedVector<float, VecSize> src_vec;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid < real_bsz) {
        auto* draft_logits_now =
            draft_logits + cu_batch_token_offset[bid] * vocab_size;
        auto* logits_now = logits + cu_seqlens_q[bid] * vocab_size;
        for (int i = tid * VecSize; i < vocab_size; i += blockDim.x * VecSize) {
            if (seq_lens_encoder[bid] > 0) {
                Load<float, VecSize>(&first_token_logits[bid * vocab_size + i],
                                     &src_vec);
                Store<float, VecSize>(src_vec, &draft_logits_now[i]);

                Load<float, VecSize>(&logits_now[i], &src_vec);
                Store<float, VecSize>(src_vec,
                                      &draft_logits_now[vocab_size + i]);
            } else {
                for (int j = 0; j < seq_lens_this_time[bid]; j++) {
                    Load<float, VecSize>(&logits_now[j * vocab_size + i],
                                         &src_vec);
                    Store<float, VecSize>(
                        src_vec, &draft_logits_now[j * vocab_size + i]);
                }
            }
        }
    }
}

std::vector<paddle::Tensor> SpeculateGetLogits(
    const paddle::Tensor& logits,
    const paddle::Tensor& first_token_logits,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder) {
    auto cu_stream = seq_lens_this_time.stream();
    const int vocab_size = logits.shape()[1];
    const int real_bsz = seq_lens_this_time.shape()[0];

    auto total_token_num = paddle::full(
        {1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());
    auto batch_token_num = paddle::full(
        {real_bsz}, 0, paddle::DataType::INT32, seq_lens_this_time.place());

    constexpr int THREADBLOCK_SIZE = 512;
    get_token_num_per_batch_kernel<THREADBLOCK_SIZE>
        <<<1, THREADBLOCK_SIZE, 0, cu_stream>>>(batch_token_num.data<int>(),
                                                total_token_num.data<int>(),
                                                seq_lens_this_time.data<int>(),
                                                seq_lens_encoder.data<int>(),
                                                real_bsz);

    auto total_token_num_cpu =
        total_token_num.copy_to(paddle::CPUPlace(), true);

    auto draft_logits =
        paddle::empty({total_token_num_cpu.data<int>()[0], vocab_size},
                      paddle::DataType::FLOAT32,
                      seq_lens_this_time.place());
    auto cu_batch_token_offset = paddle::full(
        {real_bsz + 1}, 0, paddle::DataType::INT32, seq_lens_this_time.place());

    void* temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(temp_storage,
                                  temp_storage_bytes,
                                  batch_token_num.data<int>(),
                                  &cu_batch_token_offset.data<int>()[1],
                                  real_bsz,
                                  cu_stream);
    cudaMalloc(&temp_storage, temp_storage_bytes);
    cub::DeviceScan::InclusiveSum(temp_storage,
                                  temp_storage_bytes,
                                  batch_token_num.data<int>(),
                                  &cu_batch_token_offset.data<int>()[1],
                                  real_bsz,
                                  cu_stream);

    constexpr int PackSize = VEC_16B / sizeof(float);
    dim3 grid_dim(real_bsz);
    dim3 block_dim(128);
    speculate_get_logits_kernel<PackSize>
        <<<grid_dim, block_dim, 0, cu_stream>>>(
            const_cast<float*>(draft_logits.data<float>()),
            logits.data<float>(),
            first_token_logits.data<float>(),
            cu_seqlens_q.data<int>(),
            cu_batch_token_offset.data<int>(),
            seq_lens_this_time.data<int>(),
            seq_lens_encoder.data<int>(),
            vocab_size,
            real_bsz);

    return {draft_logits, batch_token_num, cu_batch_token_offset};
}

__global__ void speculate_insert_first_token_kernel(
    int64_t* token_ids,
    const int64_t* accept_tokens,
    const int64_t* next_tokens,
    const int* cu_seqlens_q,
    const int* cu_batch_token_offset,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int max_draft_tokens,
    const int real_bsz) {
    const int bid = threadIdx.x;

    auto* token_ids_now = token_ids + cu_batch_token_offset[bid];
    auto* accept_tokens_now = accept_tokens + bid * max_draft_tokens;
    auto* next_tokens_now = next_tokens + cu_seqlens_q[bid];
    if (seq_lens_encoder[bid] != 0) {
        token_ids_now[0] = accept_tokens_now[0];
        token_ids_now[1] = next_tokens_now[0];
    } else {
        for (int i = 0; i < seq_lens_this_time[bid]; i++) {
            token_ids_now[i] = next_tokens_now[i];
        }
    }
}

void SpeculateInsertFirstToken(const paddle::Tensor& token_ids,
                               const paddle::Tensor& accept_tokens,
                               const paddle::Tensor& next_tokens,
                               const paddle::Tensor& cu_seqlens_q,
                               const paddle::Tensor& cu_batch_token_offset,
                               const paddle::Tensor& seq_lens_this_time,
                               const paddle::Tensor& seq_lens_encoder) {
    auto cu_stream = seq_lens_this_time.stream();
    const int max_draft_tokens = accept_tokens.shape()[1];
    const int real_bsz = seq_lens_this_time.shape()[0];

    speculate_insert_first_token_kernel<<<1, real_bsz, 0, cu_stream>>>(
        const_cast<int64_t*>(token_ids.data<int64_t>()),
        accept_tokens.data<int64_t>(),
        next_tokens.data<int64_t>(),
        cu_seqlens_q.data<int>(),
        cu_batch_token_offset.data<int>(),
        seq_lens_this_time.data<int>(),
        seq_lens_encoder.data<int>(),
        max_draft_tokens,
        real_bsz);
}

template <int VecSize>
__global__ void speculate_get_target_logits_kernel(
    float* target_logtis,
    const float* logits,
    const int* cu_batch_token_offset,
    const int* ori_cu_batch_token_offset,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* accept_num,
    const int vocab_size,
    const int real_bsz) {
    AlignedVector<float, VecSize> src_vec;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid < real_bsz) {
        auto* target_logtis_now =
            target_logtis + cu_batch_token_offset[bid] * vocab_size;
        auto* logits_now = logits + ori_cu_batch_token_offset[bid] * vocab_size;
        for (int i = tid * VecSize; i < vocab_size; i += blockDim.x * VecSize) {
            if (seq_lens_encoder[bid] > 0) {
                Load<float, VecSize>(&logits_now[i], &src_vec);
                Store<float, VecSize>(src_vec, &target_logtis_now[i]);
            } else {
                for (int j = 0; j < accept_num[bid]; j++) {
                    Load<float, VecSize>(&logits_now[j * vocab_size + i],
                                         &src_vec);
                    Store<float, VecSize>(
                        src_vec, &target_logtis_now[j * vocab_size + i]);
                }
            }
        }
    }
}

void SpeculateGetTargetLogits(const paddle::Tensor& target_logits,
                              const paddle::Tensor& logits,
                              const paddle::Tensor& cu_batch_token_offset,
                              const paddle::Tensor& ori_cu_batch_token_offset,
                              const paddle::Tensor& seq_lens_this_time,
                              const paddle::Tensor& seq_lens_encoder,
                              const paddle::Tensor& accept_num) {
    auto cu_stream = seq_lens_this_time.stream();
    const int vocab_size = logits.shape()[1];
    const int real_bsz = seq_lens_this_time.shape()[0];

    constexpr int PackSize = VEC_16B / sizeof(float);
    dim3 grid_dim(real_bsz);
    dim3 block_dim(128);
    speculate_get_target_logits_kernel<PackSize>
        <<<grid_dim, block_dim, 0, cu_stream>>>(
            const_cast<float*>(target_logits.data<float>()),
            logits.data<float>(),
            cu_batch_token_offset.data<int>(),
            ori_cu_batch_token_offset.data<int>(),
            seq_lens_this_time.data<int>(),
            seq_lens_encoder.data<int>(),
            accept_num.data<int>(),
            vocab_size,
            real_bsz);
}

PD_BUILD_STATIC_OP(speculate_get_logits)
    .Inputs({"logits",
             "first_token_logits",
             "cu_seqlens_q",
             "seq_lens_this_time",
             "seq_lens_encoder"})
    .Outputs({"draft_logits", "batch_token_num", "cu_batch_token_offset"})
    .SetKernelFn(PD_KERNEL(SpeculateGetLogits));

PD_BUILD_STATIC_OP(speculate_insert_first_token)
    .Inputs({"token_ids",
             "accept_tokens",
             "next_tokens",
             "cu_seqlens_q",
             "cu_batch_token_offset",
             "seq_lens_this_time",
             "seq_lens_encoder"})
    .Outputs({"token_ids_out"})
    .SetInplaceMap({{"token_ids", "token_ids_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateInsertFirstToken));

PD_BUILD_STATIC_OP(speculate_get_target_logits)
    .Inputs({"target_logits",
             "logits",
             "cu_batch_token_offset",
             "ori_cu_batch_token_offset",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "accept_num"})
    .Outputs({"target_logits_out"})
    .SetInplaceMap({{"target_logits", "target_logits_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateGetTargetLogits));
