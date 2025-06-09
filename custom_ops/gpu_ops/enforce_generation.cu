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

#include "helper.h"

template <typename T>
__global__ void apply_token_enforce_generation_scores_kernel(
    T *logits,
    bool *logit_mask,
    const int32_t *status_and_tokens,
    int logits_length,
    int logit_mask_length,
    int allowed_token_max_len) {
    int bs = blockIdx.x;
    int ti = threadIdx.x;
    int32_t cur_allowed_token_num =
        status_and_tokens[bs * allowed_token_max_len + 2];
#pragma unroll
    if (cur_allowed_token_num > 0) {
        for (int i = ti; i < logit_mask_length; i += blockDim.x) {
            logit_mask[bs * logit_mask_length + i] = true;
        }
        __syncthreads();
#pragma unroll
        for (int i = ti; i < cur_allowed_token_num; i += blockDim.x) {
            int idx = status_and_tokens[bs * allowed_token_max_len + i + 3];
            logit_mask[bs * logit_mask_length + idx] = false;
        }
        __syncthreads();

#pragma unroll
        for (int i = ti; i < logits_length; i += blockDim.x) {
            if (logit_mask[bs * logits_length + i] == true) {
                logits[bs * logits_length + i] = -1e10;
            }
        }
    }
}

template <paddle::DataType D>
void token_enforce_generation_scores_kernel(
    const paddle::Tensor &logits,
    const paddle::Tensor &logit_mask,
    const paddle::Tensor &status_and_tokens) {
    typedef PDTraits<D> traits_;
    typedef typename traits_::DataType DataType_;
    typedef typename traits_::data_t data_t;
    auto cu_stream = logits.stream();
    std::vector<int64_t> logits_shape = logits.shape();

    std::vector<int64_t> allowed_token_shape = status_and_tokens.shape();
    std::vector<int64_t> logit_mask_shape = logit_mask.shape();
    int bs = logits_shape[0];
    int logits_length = logits_shape[1];
    int logit_mask_length = logit_mask_shape[1];
    int allowed_token_max_len = allowed_token_shape[1];
    int block_size = (logits_length + 32 - 1) / 32 * 32;
    block_size = min(block_size, 512);

    // TODO(liuzichang): Reserved for multi-process
    // int32_t con_gen_flag;
    // printf("before for loop\n");
    // for (;;) {
    //   cudaMemcpy(reinterpret_cast<void*>(&con_gen_flag),
    //   status_and_tokens.data<int32_t>(), sizeof(int32_t),
    //   cudaMemcpyDeviceToHost); if (con_gen_flag == 1) {
    //     break;
    //   }
    // }
    // printf("finish for loop\n");
    // printf("bs: %d, logits_length: %d, logit_mask_length: %d,
    // allowed_token_max_len: %d, block_size: %d\n", bs, logits_length,
    // logit_mask_length, allowed_token_max_len, block_size);
    apply_token_enforce_generation_scores_kernel<<<bs,
                                                   block_size,
                                                   0,
                                                   logits.stream()>>>(
        reinterpret_cast<DataType_ *>(
            const_cast<data_t *>(logits.data<data_t>())),
        const_cast<bool *>(logit_mask.data<bool>()),
        status_and_tokens.data<int32_t>(),
        logits_length,
        logit_mask_length,
        allowed_token_max_len);
}

void TokenEnforceGenerationScores(const paddle::Tensor &logits,
                                  const paddle::Tensor &logit_mask,
                                  const paddle::Tensor &status_and_tokens) {
    switch (logits.type()) {
        case paddle::DataType::BFLOAT16: {
            return token_enforce_generation_scores_kernel<
                paddle::DataType::BFLOAT16>(
                logits, logit_mask, status_and_tokens);
        }
        case paddle::DataType::FLOAT16: {
            return token_enforce_generation_scores_kernel<
                paddle::DataType::FLOAT16>(
                logits, logit_mask, status_and_tokens);
        }
        case paddle::DataType::FLOAT32: {
            return token_enforce_generation_scores_kernel<
                paddle::DataType::FLOAT32>(
                logits, logit_mask, status_and_tokens);
        }
        default: {
            PD_THROW(
                "NOT supported data type. "
                "Only float16, bfloat16 and float32 are supported. ");
            break;
        }
    }
}

void __global__ update_enf_gen_values_kernel(int32_t *status_and_tokens,
                                             const bool *stop_flags,
                                             const int64_t *next_tokens,
                                             int32_t now_bs,
                                             int32_t status_and_tokens_max_len,
                                             int32_t next_tokens_len) {
    int tid = threadIdx.x;
    for (int32_t bs_idx = tid; bs_idx < now_bs; bs_idx += blockDim.x) {
        bool stop_flag = stop_flags[bs_idx];
        int32_t *cur_status_and_tokens =
            status_and_tokens + bs_idx * status_and_tokens_max_len;
        bool is_first = (cur_status_and_tokens[1] == 2);
        if (!stop_flag) {
            cur_status_and_tokens[1] = 1;
            cur_status_and_tokens[2] =
                static_cast<int32_t>(next_tokens[bs_idx]);
        } else if (!is_first) {  // stop_flag && not first
            cur_status_and_tokens[1] = 0;
        }
    }
    __syncthreads();
    if (tid == 0) {
        status_and_tokens[0] = 2;
    }
}

void UpdateEnfGenValues(const paddle::Tensor &status_and_tokens,
                        const paddle::Tensor &stop_flags,
                        const paddle::Tensor &next_tokens) {
    const int bsz = next_tokens.shape()[0];
    const int status_and_tokens_max_len = status_and_tokens.shape()[1];
    const int next_tokens_len = next_tokens.shape()[0];

    update_enf_gen_values_kernel<<<1, 1024, 0, next_tokens.stream()>>>(
        const_cast<int32_t *>(status_and_tokens.data<int32_t>()),
        stop_flags.data<bool>(),
        next_tokens.data<int64_t>(),
        bsz,
        status_and_tokens_max_len,
        next_tokens_len);
}

PD_BUILD_STATIC_OP(get_enf_gen_scores)
    .Inputs({"logits", "logit_mask", "status_and_tokens"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenEnforceGenerationScores));

PD_BUILD_STATIC_OP(update_enf_gen_values)
    .Inputs({"status_and_tokens", "stop_flags", "next_tokens"})
    .Outputs({"status_and_tokens_out"})
    .SetInplaceMap({{"status_and_tokens", "status_and_tokens_out"}})
    .SetKernelFn(PD_KERNEL(UpdateEnfGenValues));
