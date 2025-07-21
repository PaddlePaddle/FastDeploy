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

// 根据上一步计算出的可以复原的query_id进行状态恢复
__global__ void recover_block_system_cache(int *recover_block_list, // [bsz]
                                           int *recover_len,
                                           bool *stop_flags,
                                           int *seq_lens_this_time,
                                           int *ori_seq_lens_encoder,
                                           int *ori_seq_lens_decoder,
                                           int *seq_lens_encoder,
                                           int *seq_lens_decoder,
                                           int *block_tables,
                                           int *free_list,
                                           int *free_list_len,
                                           int64_t *input_ids,
                                           int64_t *pre_ids,
                                           int64_t *step_idx,
                                           int *encoder_block_lens,
                                           int *used_list_len,
                                           const int64_t *next_tokens,
                                           const int64_t *first_token_ids,
                                           const int bsz,
                                           const int block_num_per_seq,
                                           const int length,
                                           const int pre_id_length) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int ori_free_list_len;
    if (bid < recover_len[0]) {
        const int recover_id = recover_block_list[bid];
        const int ori_seq_len_encoder = ori_seq_lens_encoder[recover_id];
        const int step_idx_now = step_idx[recover_id];
        const int seq_len = ori_seq_len_encoder + step_idx_now;
        const int encoder_block_len = encoder_block_lens[recover_id];
        const int decoder_used_len = used_list_len[recover_id];
        int *block_table_now = block_tables + recover_id * block_num_per_seq;
        int64_t *input_ids_now = input_ids + recover_id * length;
        int64_t *pre_ids_now = pre_ids + recover_id * pre_id_length;
        if (tid == 0) {
            seq_lens_this_time[recover_id] = seq_len;
            seq_lens_encoder[recover_id] = seq_len;
            seq_lens_decoder[recover_id] = ori_seq_lens_decoder[recover_id];
            stop_flags[recover_id] = false;
            input_ids_now[ori_seq_len_encoder + step_idx_now - 1] = next_tokens[recover_id]; // next tokens
            input_ids_now[0] = first_token_ids[recover_id]; // set first prompt token
            const int ori_free_list_len_tid0 = atomicSub(free_list_len, decoder_used_len);
            ori_free_list_len = ori_free_list_len_tid0;
#ifdef DEBUG_STEP
            printf("seq_id: %d, ori_seq_len_encoder: %d, step_idx_now: %d, seq_len: %d, ori_free_list_len_tid0: %d, ori_free_list_len: %d\n",
                    recover_id, ori_seq_len_encoder, step_idx_now, seq_len, ori_free_list_len_tid0, ori_free_list_len);
#endif
        }
        __syncthreads();
        // 恢复block table
        for (int i = tid; i < decoder_used_len; i += blockDim.x) {
            block_table_now[encoder_block_len + i] = free_list[ori_free_list_len - i - 1];
        }
        // 恢复input_ids
        for (int i = tid; i < step_idx_now - 1; i += blockDim.x) {
            input_ids_now[ori_seq_len_encoder + i] = pre_ids_now[i + 1];
        }
    }

    if (bid == 0 && tid == 0) {
        recover_len[0] = 0;
    }
}

void StepSystemCache(const paddle::Tensor& stop_flags,
                     const paddle::Tensor& seq_lens_this_time,
                     const paddle::Tensor& ori_seq_lens_encoder,
                     const paddle::Tensor& ori_seq_lens_decoder,
                     const paddle::Tensor& seq_lens_encoder,
                     const paddle::Tensor& seq_lens_decoder,
                     const paddle::Tensor& block_tables, // [bsz, block_num_per_seq]
                     const paddle::Tensor& encoder_block_lens,
                     const paddle::Tensor& is_block_step,
                     const paddle::Tensor& step_block_list,
                     const paddle::Tensor& step_lens,
                     const paddle::Tensor& recover_block_list,
                     const paddle::Tensor& recover_lens,
                     const paddle::Tensor& need_block_list,
                     const paddle::Tensor& need_block_len,
                     const paddle::Tensor& used_list_len,
                     const paddle::Tensor& free_list,
                     const paddle::Tensor& free_list_len,
                     const paddle::Tensor& input_ids,
                     const paddle::Tensor& pre_ids,
                     const paddle::Tensor& step_idx,
                     const paddle::Tensor& next_tokens,
                     const paddle::Tensor& first_token_ids,
                     const int block_size,
                     const int encoder_decoder_block_num) {
    auto cu_stream = seq_lens_this_time.stream();
    const int bsz = seq_lens_this_time.shape()[0];
    const int block_num_per_seq = block_tables.shape()[1];
    const int length = input_ids.shape()[1];
    const int pre_id_length = pre_ids.shape()[1];
    constexpr int BlockSize = 256; // bsz <= 256
    const int max_decoder_block_num = length / block_size;
    // const int max_decoder_block_num = 2048 / block_size - encoder_decoder_block_num;
#ifdef DEBUG_STEP
    printf("bsz: %d, block_num_per_seq: %d, length: %d, max_decoder_block_num: %d\n", bsz, block_num_per_seq, length, max_decoder_block_num);
#endif
    free_and_dispatch_block<<<1, BlockSize, 0, cu_stream>>>(
        const_cast<bool*>(stop_flags.data<bool>()),
        const_cast<int*>(seq_lens_this_time.data<int>()),
        const_cast<int*>(seq_lens_decoder.data<int>()),
        const_cast<int*>(block_tables.data<int>()),
        const_cast<int*>(encoder_block_lens.data<int>()),
        const_cast<bool*>(is_block_step.data<bool>()),
        const_cast<int*>(step_block_list.data<int>()),
        const_cast<int*>(step_lens.data<int>()),
        const_cast<int*>(recover_block_list.data<int>()),
        const_cast<int*>(recover_lens.data<int>()),
        const_cast<int*>(need_block_list.data<int>()),
        const_cast<int*>(need_block_len.data<int>()),
        const_cast<int*>(used_list_len.data<int>()),
        const_cast<int*>(free_list.data<int>()),
        const_cast<int*>(free_list_len.data<int>()),
        const_cast<int64_t*>(first_token_ids.data<int64_t>()),
        bsz,
        block_size,
        block_num_per_seq,
        max_decoder_block_num
    );
#ifdef DEBUG_STEP
    cudaDeviceSynchronize();
#endif
    auto cpu_recover_lens = recover_lens.copy_to(paddle::CPUPlace(), false);
    const int grid_size = cpu_recover_lens.data<int>()[0];
#ifdef DEBUG_STEP
    printf("grid_size2 %d\n", grid_size);
#endif
    if (grid_size > 0) {
        recover_block_system_cache<<<grid_size, BlockSize, 0, cu_stream>>>(
            const_cast<int*>(recover_block_list.data<int>()),
            const_cast<int*>(recover_lens.data<int>()),
            const_cast<bool*>(stop_flags.data<bool>()),
            const_cast<int*>(seq_lens_this_time.data<int>()),
            const_cast<int*>(ori_seq_lens_encoder.data<int>()),
            const_cast<int*>(ori_seq_lens_decoder.data<int>()),
            const_cast<int*>(seq_lens_encoder.data<int>()),
            const_cast<int*>(seq_lens_decoder.data<int>()),
            const_cast<int*>(block_tables.data<int>()),
            const_cast<int*>(free_list.data<int>()),
            const_cast<int*>(free_list_len.data<int>()),
            const_cast<int64_t*>(input_ids.data<int64_t>()),
            const_cast<int64_t*>(pre_ids.data<int64_t>()),
            const_cast<int64_t*>(step_idx.data<int64_t>()),
            const_cast<int*>(encoder_block_lens.data<int>()),
            const_cast<int*>(used_list_len.data<int>()),
            next_tokens.data<int64_t>(),
            first_token_ids.data<int64_t>(),
            bsz,
            block_num_per_seq,
            length,
            pre_id_length
        );
#ifdef DEBUG_STEP
        cudaDeviceSynchronize();
#endif
    }
}

PD_BUILD_STATIC_OP(step_system_cache)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "ori_seq_lens_encoder",
             "ori_seq_lens_decoder",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "is_block_step",
             "step_block_list",
             "step_lens",
             "recover_block_list",
             "recover_lens",
             "need_block_list",
             "need_block_len",
             "used_list_len",
             "free_list",
             "free_list_len",
             "input_ids",
             "pre_ids",
             "step_idx",
             "next_tokens",
             "first_token_ids"})
    .Attrs({"block_size: int",
            "encoder_decoder_block_num: int"})
    .Outputs({"stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "block_tables_out",
              "encoder_block_lens_out",
              "is_block_step_out",
              "step_block_list_out",
              "step_lens_out",
              "recover_block_list_out",
              "recover_lens_out",
              "need_block_list_out",
              "need_block_len_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out",
              "input_ids_out",
              "first_token_ids_out"})
    .SetInplaceMap({{"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"block_tables", "block_tables_out"},
                    {"encoder_block_lens", "encoder_block_lens_out"},
                    {"is_block_step", "is_block_step_out"},
                    {"step_block_list", "step_block_list_out"},
                    {"step_lens", "step_lens_out"},
                    {"recover_block_list", "recover_block_list_out"},
                    {"recover_lens", "recover_lens_out"},
                    {"need_block_list", "need_block_list_out"},
                    {"need_block_len", "need_block_len_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"},
                    {"input_ids", "input_ids_out"},
                    {"first_token_ids", "first_token_ids_out"}})
    .SetKernelFn(PD_KERNEL(StepSystemCache));
