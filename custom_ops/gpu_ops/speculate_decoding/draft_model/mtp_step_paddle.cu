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

#include "helper.h"  // NOLINT
template<int NUM_THREADS, int MAX_BATCH_SIZE=256>
__global__ void mtp_free_and_dispatch_block(
    bool *base_model_stop_flags,
    bool *stop_flags,
    bool *batch_drop,
    int *seq_lens_this_time,
    int *seq_lens_decoder,
    int *block_tables,
    int *encoder_block_lens,
    int *used_list_len,
    int *free_list,
    int *free_list_len,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_draft_tokens) {

    typedef cub::BlockReduce<cub::KeyValuePair<int, int>, NUM_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    __shared__ int need_block_len;
    __shared__ int need_block_list[MAX_BATCH_SIZE];
    const int tid = threadIdx.x;

    if (tid < bsz) {
        if (tid == 0) {
            need_block_len = 0;
        }
        need_block_list[tid] = 0;
        int *block_table_now = block_tables + tid * block_num_per_seq;
        if (base_model_stop_flags[tid] || batch_drop[tid]) {
            // 回收block块
            const int encoder_block_len = encoder_block_lens[tid];
            const int decoder_used_len = used_list_len[tid];
            if (decoder_used_len > 0) {
                const int ori_free_list_len =
                    atomicAdd(free_list_len, decoder_used_len);
#ifdef DEBUG_STEP
                printf(
                    "free block seq_id: %d, free block num: %d, "
                    "encoder_block_len: %d, ori_free_list_len: %d\n",
                    tid,
                    decoder_used_len,
                    encoder_block_len,
                    ori_free_list_len);
#endif
                for (int i = 0; i < decoder_used_len; i++) {
                    free_list[ori_free_list_len + i] =
                        block_table_now[encoder_block_len + i];
                    block_table_now[encoder_block_len + i] = -1;
                }
                encoder_block_lens[tid] = 0;
                used_list_len[tid] = 0;
            }
        }
    }
    __syncthreads();
    if (tid < bsz) {
        int *block_table_now = block_tables + tid * block_num_per_seq;
        int max_possible_block_idx = (seq_lens_decoder[tid] + max_draft_tokens + 1) / block_size;
        if (!base_model_stop_flags[tid] && !batch_drop[tid] && max_possible_block_idx < block_num_per_seq &&
                   block_table_now[max_possible_block_idx] == -1) {
            int ori_need_block_len = atomicAdd(&need_block_len, 1);
            need_block_list[ori_need_block_len] = tid;
            // 统计需要分配block的位置和总数
            // const int ori_free_list_len = atomicSub(free_list_len, 1);
            // block_table_now[(seq_lens_decoder[tid] + max_draft_tokens + 1) / block_size] =
            //                 free_list[ori_free_list_len - 1];
            // used_list_len[tid] += 1;

#ifdef DEBUG_STEP
            printf("seq_id: %d need block\n", tid);
#endif
        }
    }
    __syncthreads();
#ifdef DEBUG_STEP
    if (tid == 0) {
        printf("need_block_len:%d, free_list_len: %d\n", need_block_len, free_list_len[0]);
    }
#endif
    // 这里直接从 bid 0 开始遍历
    while (need_block_len > free_list_len[0]) {
#ifdef DEBUG_STEP
        if (tid == 0) {
            printf("in while need_block_len:%d, free_list_len: %d\n", need_block_len, free_list_len[0]);
        }
#endif
        const int used_block_num =
            tid < bsz && !base_model_stop_flags[tid] ? used_list_len[tid] : 0;
        cub::KeyValuePair<int, int> kv_pair = {tid, used_block_num};
        kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, cub::ArgMax());

        if (tid == 0) {
            const int encoder_block_len = encoder_block_lens[kv_pair.key];
            int *block_table_now =
                block_tables + kv_pair.key * block_num_per_seq;
            for (int i = 0; i < kv_pair.value; i++) {
                free_list[free_list_len[0] + i] =
                    block_table_now[encoder_block_len + i];
                block_table_now[encoder_block_len + i] = -1;
            }
            const int ori_free_list_len = atomicAdd(free_list_len, kv_pair.value);

            printf(
                "MTP STEP need_block_len: %d. free_list_len: %d."
                "Drop bid: %d, free block num: %d, "
                "encoder_block_len: %d,"
                "After drop free_list_len %d \n",
                need_block_len,
                ori_free_list_len,
                kv_pair.key,
                kv_pair.value,
                encoder_block_len,
                free_list_len[0]);
            stop_flags[kv_pair.key] = true;
            batch_drop[kv_pair.key] = true;
            seq_lens_this_time[kv_pair.key] = 0;
            seq_lens_decoder[kv_pair.key] = 0;
            used_list_len[kv_pair.key] = 0;
        }
        __syncthreads();
    }

    if (tid < need_block_len) {
        const int need_block_id = need_block_list[tid];
        // 这里必须用 batch_drop, 不能用 stop_flags
        if (!batch_drop[need_block_id]) {
            used_list_len[need_block_id] += 1;
            const int ori_free_list_len = atomicSub(free_list_len, 1);
            int *block_table_now =
                    block_tables + need_block_id * block_num_per_seq;
#ifdef DEBUG_STEP
            printf("bid: %d allocate block_id %d. seq_lens_decoder:%d \n", need_block_id, free_list[ori_free_list_len - 1], seq_lens_decoder[need_block_id]);
#endif
            block_table_now[(seq_lens_decoder[need_block_id] +
                                 max_draft_tokens + 1) /
                                block_size] = free_list[ori_free_list_len - 1];
        }
    }
}

void MTPStepPaddle(
    const paddle::Tensor &base_model_stop_flags,
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &batch_drop,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &block_tables,  // [bsz, block_num_per_seq]
    const paddle::Tensor &encoder_block_lens,
    const paddle::Tensor &used_list_len,
    const paddle::Tensor &free_list,
    const paddle::Tensor &free_list_len,
    const int block_size,
    const int max_draft_tokens) {
    auto cu_stream = seq_lens_this_time.stream();
    const int bsz = seq_lens_this_time.shape()[0];
    const int block_num_per_seq = block_tables.shape()[1];
    constexpr int BlockSize = 512;  // bsz <= 256
#ifdef DEBUG_STEP
    printf(
        "bsz: %d, block_num_per_seq: %d, length: %d, max_decoder_block_num: "
        "%d\n",
        bsz,
        block_num_per_seq,
        length,
        max_decoder_block_num);
#endif
    mtp_free_and_dispatch_block<BlockSize, BlockSize><<<1, BlockSize, 0, cu_stream>>>(
        const_cast<bool *>(base_model_stop_flags.data<bool>()),
        const_cast<bool *>(stop_flags.data<bool>()),
        const_cast<bool *>(batch_drop.data<bool>()),
        const_cast<int *>(seq_lens_this_time.data<int>()),
        const_cast<int *>(seq_lens_decoder.data<int>()),
        const_cast<int *>(block_tables.data<int>()),
        const_cast<int *>(encoder_block_lens.data<int>()),
        const_cast<int *>(used_list_len.data<int>()),
        const_cast<int *>(free_list.data<int>()),
        const_cast<int *>(free_list_len.data<int>()),
        bsz,
        block_size,
        block_num_per_seq,
        max_draft_tokens);
}

PD_BUILD_STATIC_OP(mtp_step_paddle)
    .Inputs({"base_model_stop_flags",
             "stop_flags",
             "batch_drop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables",
             "encoder_block_lens",
             "used_list_len",
             "free_list",
             "free_list_len"})
    .Attrs({"block_size: int",
            "max_draft_tokens: int"})
    .Outputs({"block_tables_out",
              "stop_flags_out",
              "used_list_len_out",
              "free_list_out",
              "free_list_len_out"})
    .SetInplaceMap({{"block_tables", "block_tables_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"used_list_len", "used_list_len_out"},
                    {"free_list", "free_list_out"},
                    {"free_list_len", "free_list_len_out"}})
    .SetKernelFn(PD_KERNEL(MTPStepPaddle));
