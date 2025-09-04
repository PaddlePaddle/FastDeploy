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
#include "speculate_msg.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

__device__ __forceinline__ bool in_need_block_list_schedule(const int &qid,
                                            int *need_block_list,
                                            const int &need_block_len) {
    bool res = false;
    for (int i = 0; i < need_block_len; i++) {
        if (qid == need_block_list[i]) {
            res = true;
            need_block_list[i] = -1;
            break;
        }
    }
    return res;
}

__global__ void speculate_free_and_reschedule(bool *stop_flags,
                                   int *seq_lens_this_time,
                                   int *seq_lens_decoder,
                                   int *block_tables,
                                   int *encoder_block_lens,
                                   bool *is_block_step,
                                   int *step_block_list,  // [bsz]
                                   int *step_len,
                                   int *recover_block_list,
                                   int *recover_len,
                                   int *need_block_list,
                                   int *need_block_len,
                                   int *used_list_len,
                                   int *free_list,
                                   int *free_list_len,
                                   int64_t *first_token_ids,
                                   int* accept_num,
                                   const int bsz,
                                   const int block_size,
                                   const int block_num_per_seq,
                                   const int max_decoder_block_num,
                                   const int max_draft_tokens) {
    typedef cub::BlockReduce<cub::KeyValuePair<int, int>, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ bool step_max_block_flag;
    __shared__ int in_need_block_list_len;
    const int tid = threadIdx.x;
    if (tid < bsz) {
        if (tid == 0) {
            step_max_block_flag = false;
            in_need_block_list_len = 0;
        }
        int *block_table_now = block_tables + tid * block_num_per_seq;
        int max_possible_block_idx = (seq_lens_decoder[tid] + max_draft_tokens + 1 ) / block_size;
        if (stop_flags[tid]) {
            // 回收block块
            first_token_ids[tid] = -1;
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
        } else if (seq_lens_this_time[tid] != 0 && max_possible_block_idx < block_num_per_seq &&
                   block_table_now[(seq_lens_decoder[tid]  + max_draft_tokens  +
                                    1) /
                                   block_size] == -1) {
            // 统计需要分配block的位置和总数
#ifdef DEBUG_STEP
            printf("step seq_id:%d, ##### pin 1 #####\n", tid);
#endif
            const int ori_need_block_len = atomicAdd(need_block_len, 1);
            need_block_list[ori_need_block_len] = tid;
#ifdef DEBUG_STEP
            printf("seq_id: %d need block\n", tid);
#endif
        }
    }
#ifdef DEBUG_STEP
    printf("step seq_id:%d, ##### pin 2 #####\n", tid);
#endif
    __syncthreads();

    // 调度block，直到满足need_block_len
    while (need_block_len[0] > free_list_len[0]) {
        if (tid == 0) {
            printf("need_block_len: %d, free_list_len: %d\n",
                   need_block_len[0],
                   free_list_len[0]);
        }
        // 调度block，根据used_list_len从大到小回收block，直到满足need_block_len，已解码到最后一个block的query不参与调度（马上就结束）
        const int used_block_num =
            tid < bsz ? used_list_len[tid] : 0;
        cub::KeyValuePair<int, int> kv_pair = {tid, used_block_num};
        kv_pair = BlockReduce(temp_storage).Reduce(kv_pair, cub::ArgMax());
        if (tid == 0) {
            if (kv_pair.value == 0) {
                step_max_block_flag = true;
            } else {
                const int encoder_block_len = encoder_block_lens[kv_pair.key];
                printf("step max_id: %d, max_num: %d, encoder_block_len: %d\n",
                       kv_pair.key,
                       kv_pair.value,
                       encoder_block_len);
                int *block_table_now =
                    block_tables + kv_pair.key * block_num_per_seq;
                // 回收调度位的block
                for (int i = 0; i < kv_pair.value; i++) {
                    free_list[free_list_len[0] + i] =
                        block_table_now[encoder_block_len + i];
                    block_table_now[encoder_block_len + i] = -1;
                }
                step_block_list[step_len[0]] = kv_pair.key;
                // 如果调度位置本次也需要block，对应的处理
                if (in_need_block_list_schedule(
                        kv_pair.key,
                        need_block_list,
                        need_block_len[0] + in_need_block_list_len)) {
                    need_block_len[0] -= 1;
                    in_need_block_list_len += 1;
                }
                step_len[0] += 1;
                free_list_len[0] += kv_pair.value;
                stop_flags[kv_pair.key] = true;
                seq_lens_this_time[kv_pair.key] = 0;
                seq_lens_decoder[kv_pair.key] = 0;
                encoder_block_lens[kv_pair.key] = 0;
                used_list_len[kv_pair.key] = 0;
                printf(
                    "free block seq_id: %d, free block num: %d, "
                    "now_free_list_len: %d\n",
                    (int)kv_pair.key,
                    (int)kv_pair.value,
                    (int)free_list_len[0]);
            }
        }
        __syncthreads();
    }
#ifdef DEBUG_STEP
    printf("step seq_id:%d, ##### pin 3 #####\n", tid);
#endif
    // 为需要block的位置分配block，每个位置分配一个block
    if (tid < need_block_len[0] + in_need_block_list_len) {
        const int need_block_id = need_block_list[tid];
        if (need_block_id != -1) {
            if (!stop_flags[need_block_id]) {
                // 如果需要的位置正好是上一步中被释放的位置，不做处理
                used_list_len[need_block_id] += 1;
                const int ori_free_list_len = atomicSub(free_list_len, 1);
                int *block_table_now =
                    block_tables + need_block_id * block_num_per_seq;
                block_table_now[(seq_lens_decoder[need_block_id] +
                                 max_draft_tokens + 1) /
                                block_size] = free_list[ori_free_list_len - 1];
            }
            need_block_list[tid] = -1;
        }
    }
    __syncthreads();
    // reset need_block_len
    if (tid == 0) {
        need_block_len[0] = 0;
    }
}

// 为不修改接口调用方式，入参暂不改变
void SpeculateStepSchedule(const paddle::Tensor &stop_flags,
              const paddle::Tensor &seq_lens_this_time,
              const paddle::Tensor &ori_seq_lens_encoder,
              const paddle::Tensor &seq_lens_encoder,
              const paddle::Tensor &seq_lens_decoder,
              const paddle::Tensor &block_tables,  // [bsz, block_num_per_seq]
              const paddle::Tensor &encoder_block_lens,
              const paddle::Tensor &is_block_step,
              const paddle::Tensor &step_block_list,
              const paddle::Tensor &step_lens,
              const paddle::Tensor &recover_block_list,
              const paddle::Tensor &recover_lens,
              const paddle::Tensor &need_block_list,
              const paddle::Tensor &need_block_len,
              const paddle::Tensor &used_list_len,
              const paddle::Tensor &free_list,
              const paddle::Tensor &free_list_len,
              const paddle::Tensor &input_ids,
              const paddle::Tensor &pre_ids,
              const paddle::Tensor &step_idx,
              const paddle::Tensor &next_tokens,
              const paddle::Tensor &first_token_ids,
              const paddle::Tensor &accept_num,
              const int block_size,
              const int encoder_decoder_block_num,
              const int max_draft_tokens) {
    auto cu_stream = seq_lens_this_time.stream();
    const int bsz = seq_lens_this_time.shape()[0];
    const int block_num_per_seq = block_tables.shape()[1];
    const int length = input_ids.shape()[1];
    const int pre_id_length = pre_ids.shape()[1];
    constexpr int BlockSize = 256;  // bsz <= 256
    const int max_decoder_block_num = length / block_size - encoder_decoder_block_num; // 最大输出长度对应的block - 服务为解码分配的block数量
    auto step_lens_inkernel = paddle::full({1}, 0, paddle::DataType::INT32, stop_flags.place());
    auto step_bs_list = GetEmptyTensor({bsz}, paddle::DataType::INT32, stop_flags.place());
#ifdef DEBUG_STEP
    printf(
        "bsz: %d, block_num_per_seq: %d, length: %d, max_decoder_block_num: "
        "%d\n",
        bsz,
        block_num_per_seq,
        length,
        max_decoder_block_num);
#endif
    speculate_free_and_reschedule<<<1, BlockSize, 0, cu_stream>>>(
        const_cast<bool *>(stop_flags.data<bool>()),
        const_cast<int *>(seq_lens_this_time.data<int>()),
        const_cast<int *>(seq_lens_decoder.data<int>()),
        const_cast<int *>(block_tables.data<int>()),
        const_cast<int *>(encoder_block_lens.data<int>()),
        const_cast<bool *>(is_block_step.data<bool>()),
        const_cast<int *>(step_bs_list.data<int>()),
        const_cast<int *>(step_lens_inkernel.data<int>()),
        const_cast<int *>(recover_block_list.data<int>()),
        const_cast<int *>(recover_lens.data<int>()),
        const_cast<int *>(need_block_list.data<int>()),
        const_cast<int *>(need_block_len.data<int>()),
        const_cast<int *>(used_list_len.data<int>()),
        const_cast<int *>(free_list.data<int>()),
        const_cast<int *>(free_list_len.data<int>()),
        const_cast<int64_t *>(first_token_ids.data<int64_t>()),
        const_cast<int *>(accept_num.data<int>()),
        bsz,
        block_size,
        block_num_per_seq,
        max_decoder_block_num,
        max_draft_tokens);
#ifdef DEBUG_STEP
    cudaDeviceSynchronize();
#endif
    // save output
    auto step_lens_cpu = step_lens_inkernel.copy_to(paddle::CPUPlace(), false);
    if (step_lens_cpu.data<int>()[0] > 0) {
        auto step_bs_list_cpu = step_bs_list.copy_to(paddle::CPUPlace(), false);
        auto next_tokens = paddle::full({bsz}, -1, paddle::DataType::INT64, paddle::CPUPlace());
        for (int i = 0; i < step_lens_cpu.data<int>()[0]; i++) {
            const int step_bid = step_bs_list_cpu.data<int>()[i];
            next_tokens.data<int64_t>()[step_bid] = -3; // need reschedule
        }
        const int rank_id = static_cast<int>(stop_flags.place().GetDeviceId());
        printf("reschedule rank_id: %d, step_lens: %d", rank_id, step_lens_cpu.data<int>()[0]);
        const int64_t* x_data = next_tokens.data<int64_t>();
        static struct speculate_msgdata msg_sed;
        int msg_queue_id = rank_id;
        if (const char* inference_msg_queue_id_env_p =
                std::getenv("INFERENCE_MSG_QUEUE_ID")) {
            std::string inference_msg_queue_id_env_str(
                inference_msg_queue_id_env_p);
            int inference_msg_queue_id_from_env =
                std::stoi(inference_msg_queue_id_env_str);
            msg_queue_id = inference_msg_queue_id_from_env;
        } else {
            std::cout << "Failed to got INFERENCE_MSG_QUEUE_ID at env, use default."
                    << std::endl;
        }
        int inference_msg_id_from_env = 1;
        if (const char* inference_msg_id_env_p = std::getenv("INFERENCE_MSG_ID")) {
            std::string inference_msg_id_env_str(inference_msg_id_env_p);
            inference_msg_id_from_env = std::stoi(inference_msg_id_env_str);
            if (inference_msg_id_from_env == 2) {
                // 2 and -2 is preserve for no-output indication.
                throw std::runtime_error(
                    " INFERENCE_MSG_ID cannot be 2, please use other number.");
            }
            if (inference_msg_id_from_env < 0) {
                throw std::runtime_error(
                    " INFERENCE_MSG_ID cannot be negative, please use other "
                    "number.");
            }

        } else {
        }
        // static key_t key = ftok("/dev/shm", msg_queue_id);
        static key_t key = ftok("./", msg_queue_id);

        static int msgid = msgget(key, IPC_CREAT | 0666);
        msg_sed.mtype = 1;
        msg_sed.mtext[0] = inference_msg_id_from_env;
        msg_sed.mtext[1] = bsz;
        for (int i = 2; i < bsz + 2; i++) {
            msg_sed.mtext[i] = (int)x_data[i - 2];
        }
        if ((msgsnd(msgid, &msg_sed, (MAX_BSZ + 2) * 4, 0)) == -1) {
            printf("full msg buffer\n");
        }
    }
}

PD_BUILD_STATIC_OP(speculate_step_reschedule)
    .Inputs({"stop_flags",
             "seq_lens_this_time",
             "ori_seq_lens_encoder",
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
             "first_token_ids",
             "accept_num"})
    .Attrs({"block_size: int",
            "encoder_decoder_block_num: int",
            "max_draft_tokens: int"})
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
    .SetKernelFn(PD_KERNEL(SpeculateStepSchedule));
