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

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdlib>
#include <algorithm>
#include "paddle/extension.h"
#include "ngram_match_common.cuh"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// Get threshold from environment variable
static int get_threshold() {
    static int threshold = -1;
    if (threshold < 0) {
        char *env_var = getenv("INFER_WITH_REFERENCE_TOKENUM_THRESHOLD");
        threshold = env_var ? std::stoi(env_var) : 128;
    }
    return threshold;
}

void NgramMatchGPU(const paddle::Tensor &input_ids,
        const paddle::Tensor &input_ids_len,
        const paddle::Tensor &pre_ids,
        const paddle::Tensor &step_idx,
        const paddle::Tensor &draft_token_num,
        const paddle::Tensor &draft_tokens,
        const paddle::Tensor &seq_lens_this_time,
        const paddle::Tensor &seq_lens_encoder,
        const paddle::Tensor &seq_lens_decoder,
        const paddle::Tensor &max_dec_len,
        const int max_ngram_size,
        const int max_draft_tokens) {
    auto input_ids_shape = input_ids.shape();
    const int64_t input_ids_stride = input_ids_shape[1];

    auto pre_ids_shape = pre_ids.shape();
    const int64_t pre_ids_stride = pre_ids_shape[1];

    auto draft_tokens_shape = draft_tokens.shape();
    const int64_t draft_tokens_stride = draft_tokens_shape[1];

    const int max_batch_size = static_cast<int>(seq_lens_this_time.shape()[0]);
    const int threshold = get_threshold();

    cudaStream_t stream = input_ids.stream();

    // Allocate temporary buffer for unprocessed counts
    auto unprocessed_counts = paddle::empty({max_batch_size}, paddle::DataType::INT32, input_ids.place());
    int* unprocessed_counts_ptr = unprocessed_counts.data<int>();

    // Calculate unprocessed counts for each batch
    int threads_per_block = std::min(256, max_batch_size);
    int num_blocks = (max_batch_size + threads_per_block - 1) / threads_per_block;
    
    ngram_match_gpu::launch_calc_unprocessed_counts_kernel(
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        unprocessed_counts_ptr,
        max_batch_size,
        threads_per_block,
        num_blocks,
        stream);

    // Launch main kernel - one block per batch sample
    constexpr int kBlockSize = 256;
    ngram_match_gpu::ngram_match_kernel<kBlockSize><<<max_batch_size, kBlockSize, 0, stream>>>(
        input_ids.data<int64_t>(),
        input_ids_len.data<int64_t>(),
        pre_ids.data<int64_t>(),
        step_idx.data<int64_t>(),
        draft_token_num.data<int>(),
        const_cast<int64_t*>(draft_tokens.data<int64_t>()),
        const_cast<int*>(seq_lens_this_time.data<int>()),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        max_dec_len.data<int64_t>(),
        input_ids_stride,
        pre_ids_stride,
        draft_tokens_stride,
        max_batch_size,
        max_ngram_size,
        max_draft_tokens,
        threshold,
        unprocessed_counts_ptr);
}

// Wrapper function to maintain API compatibility
void NgramMatch(const paddle::Tensor &input_ids,
        const paddle::Tensor &input_ids_len,
        const paddle::Tensor &pre_ids,
        const paddle::Tensor &step_idx,
        const paddle::Tensor &draft_token_num,
        const paddle::Tensor &draft_tokens,
        const paddle::Tensor &seq_lens_this_time,
        const paddle::Tensor &seq_lens_encoder,
        const paddle::Tensor &seq_lens_decoder,
        const paddle::Tensor &max_dec_len,
        const int max_ngram_size,
        const int max_draft_tokens) {
    NgramMatchGPU(input_ids, input_ids_len, pre_ids, step_idx, draft_token_num,
                  draft_tokens, seq_lens_this_time, seq_lens_encoder, 
                  seq_lens_decoder, max_dec_len, max_ngram_size, max_draft_tokens);
}

PD_BUILD_STATIC_OP(ngram_match)
        .Inputs({"input_ids",
                "input_ids_len",
                "pre_ids",
                "step_idx",
                "draft_token_num",
                "draft_tokens",
                "seq_lens_this_time",
                "seq_lens_encoder",
                "seq_lens_decoder",
                "max_dec_len"})
        .Attrs({"max_ngram_size: int", "max_draft_tokens: int"})
        .Outputs({"draft_tokens_out", "seq_lens_this_time_out"})
        .SetKernelFn(PD_KERNEL(NgramMatch))
        .SetInplaceMap({{"draft_tokens", "draft_tokens_out"}, {"seq_lens_this_time", "seq_lens_this_time_out"}});
