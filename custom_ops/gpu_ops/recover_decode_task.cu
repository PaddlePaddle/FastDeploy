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

__global__ void recover_decode_task(bool *stop_flags,
                                    int *seq_lens_this_time,
                                    int *seq_lens_encoder,
                                    int *seq_lens_decoder,
                                    int *step_seq_lens_decoder,
                                    int *block_tables,
                                    bool *is_block_step,
                                    const int bsz,
                                    const int block_num_per_seq,
                                    const int block_size) {
    int thread_idx = threadIdx.x;
    if (thread_idx < bsz) {
        if(is_block_step[thread_idx] == true) {
            int *block_table_now = block_tables + thread_idx * block_num_per_seq;
            if (block_table_now[step_seq_lens_decoder[thread_idx] / block_size] != -1) {
                // can be recovered for decoding
                is_block_step[thread_idx] = false;
                seq_lens_this_time[thread_idx]= 1;
                stop_flags[thread_idx] = false;
                seq_lens_encoder[thread_idx] = 0;
                seq_lens_decoder[thread_idx] = step_seq_lens_decoder[thread_idx];

            }
        }
    }
}

__global__ void recover_spec_decode_task(bool *stop_flags,
                                        int *seq_lens_this_time,
                                        int *seq_lens_encoder,
                                        int *seq_lens_decoder,
                                        int *step_seq_lens_decoder,
                                        int *block_tables,
                                        bool *is_block_step,
                                        int64_t *draft_tokens,
                                        const int64_t *step_draft_tokens,
                                        const int *step_seq_lens_this_time,
                                        const int bsz,
                                        const int block_num_per_seq,
                                        const int block_size,
                                        const int draft_tokens_len,
                                        const int num_extra_tokens) {
    int thread_idx = threadIdx.x;
    if (thread_idx < bsz) {
        if(is_block_step[thread_idx] == true) {
            int *block_table_now = block_tables + thread_idx * block_num_per_seq;
            int max_possible_block_idx = (step_seq_lens_decoder[thread_idx] + num_extra_tokens) / block_size;
            max_possible_block_idx = min(max_possible_block_idx, block_num_per_seq);
            if (block_table_now[max_possible_block_idx] != -1) {
                // can be recovered for decoding
                int64_t *draft_tokens_now = draft_tokens + thread_idx * draft_tokens_len;
                const int64_t *step_draft_tokens_now = step_draft_tokens + thread_idx * draft_tokens_len;
                is_block_step[thread_idx] = false;
                seq_lens_this_time[thread_idx] = step_seq_lens_this_time[thread_idx];
                stop_flags[thread_idx] = false;
                seq_lens_encoder[thread_idx] = 0;
                seq_lens_decoder[thread_idx] = step_seq_lens_decoder[thread_idx];
                for (int i = 0; i < seq_lens_this_time[thread_idx]; i++) {
                    draft_tokens_now[i] = step_draft_tokens_now[i];
                }

            }
        }
    }
}


void RecoverDecodeTask(const paddle::Tensor &stop_flags,
                   const paddle::Tensor &seq_lens_this_time,
                   const paddle::Tensor &seq_lens_encoder,
                   const paddle::Tensor &seq_lens_decoder,
                   const paddle::Tensor &step_seq_lens_decoder,
                   const paddle::Tensor &block_tables,
                   const paddle::Tensor &is_block_step,
                   const paddle::optional<paddle::Tensor> &draft_tokens,
                   const paddle::optional<paddle::Tensor> &step_draft_tokens,
                   const paddle::optional<paddle::Tensor> &step_seq_lens_this_time,
                   const int block_size,
                   const int max_draft_tokens) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    auto dev_ctx = static_cast<const phi::CustomContext*>(paddle::experimental::DeviceContextPool::Instance().Get(seq_lens_this_time.place()));
    auto cu_stream = dev_ctx->stream();
#else
    auto cu_stream = seq_lens_this_time.stream();
#endif
    const int bsz = seq_lens_this_time.shape()[0];
    const int block_num_per_seq = block_tables.shape()[1];
    if (draft_tokens) {
        const int draft_tokens_len = draft_tokens.get_ptr()->shape()[1];
        recover_spec_decode_task<<<1, 1024, 0, cu_stream>>>(
            const_cast<bool *>(stop_flags.data<bool>()),
            const_cast<int *>(seq_lens_this_time.data<int>()),
            const_cast<int *>(seq_lens_encoder.data<int>()),
            const_cast<int *>(seq_lens_decoder.data<int>()),
            const_cast<int *>(step_seq_lens_decoder.data<int>()),
            const_cast<int *>(block_tables.data<int>()),
            const_cast<bool *>(is_block_step.data<bool>()),
            const_cast<int64_t *>(draft_tokens.get_ptr()->data<int64_t>()),
            step_draft_tokens.get_ptr()->data<int64_t>(),
            step_seq_lens_this_time.get_ptr()->data<int>(),
            bsz,
            block_num_per_seq,
            block_size,
            draft_tokens_len,
            max_draft_tokens * 2 + 1);

    } else {
        recover_decode_task<<<1, 1024, 0, cu_stream>>>(
            const_cast<bool *>(stop_flags.data<bool>()),
            const_cast<int *>(seq_lens_this_time.data<int>()),
            const_cast<int *>(seq_lens_encoder.data<int>()),
            const_cast<int *>(seq_lens_decoder.data<int>()),
            const_cast<int *>(step_seq_lens_decoder.data<int>()),
            const_cast<int *>(block_tables.data<int>()),
            const_cast<bool *>(is_block_step.data<bool>()),
            bsz,
            block_num_per_seq,
            block_size);
    }
}

PD_BUILD_STATIC_OP(recover_decode_task)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_seq_lens_decoder",
             "block_tables",
             "is_block_step",
             paddle::Optional("draft_tokens"),
             paddle::Optional("step_draft_tokens"),
             paddle::Optional("step_seq_lens_this_time")})
    .Attrs({"block_size: int", "max_draft_tokens: int"})
    .Outputs({"seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "stop_flags_out",
              "is_block_step_out"})
    .SetInplaceMap({{"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"is_block_step", "is_block_step_out"}})
    .SetKernelFn(PD_KERNEL(RecoverDecodeTask));
