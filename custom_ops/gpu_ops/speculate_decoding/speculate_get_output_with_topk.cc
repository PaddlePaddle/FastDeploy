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

#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#define MAX_BSZ 512
#define K 20
#define MAX_DRAFT_TOKEN_NUM 6

struct batch_msgdata {
    int tokens[MAX_DRAFT_TOKEN_NUM * (K + 1)];
    float scores[MAX_DRAFT_TOKEN_NUM * (K + 1)];
    int ranks[MAX_DRAFT_TOKEN_NUM];
};

struct msgdata {
    long mtype;
    int meta[3 + MAX_BSZ];  // stop_flag, message_flag, bsz, batch_token_nums
    batch_msgdata mtext[MAX_BSZ];
};

void SpeculateGetOutMmsgTopK(const paddle::Tensor& output_tokens,
                             const paddle::Tensor& output_scores,
                             const paddle::Tensor& output_ranks,
                             int real_k,
                             int64_t rank_id,
                             bool wait_flag) {
    struct msgdata msg_rcv;
    int msg_queue_id = 1;

    if (const char* inference_msg_queue_id_env_p =
            std::getenv("INFERENCE_MSG_QUEUE_ID")) {
        std::string inference_msg_queue_id_env_str(
            inference_msg_queue_id_env_p);
        int inference_msg_queue_id_from_env =
            std::stoi(inference_msg_queue_id_env_str);
#ifdef SPECULATE_GET_WITH_OUTPUT_DEBUG
        std::cout << "Your INFERENCE_MSG_QUEUE_ID is: "
                  << inference_msg_queue_id_from_env << std::endl;
#endif
        msg_queue_id = inference_msg_queue_id_from_env;
    }
    static key_t key = ftok("/dev/shm", msg_queue_id);

    static int msgid = msgget(key, IPC_CREAT | 0666);
#ifdef SPECULATE_GET_WITH_OUTPUT_DEBUG
    std::cout << "get_output_key: " << key << std::endl;
    std::cout << "get_output msgid: " << msgid << std::endl;
#endif

    int64_t* output_tokens_data =
        const_cast<int64_t*>(output_tokens.data<int64_t>());
    float* output_scores_data = const_cast<float*>(output_scores.data<float>());
    int64_t* output_ranks_data =
        const_cast<int64_t*>(output_ranks.data<int64_t>());
    int ret = -1;
    if (!wait_flag) {
        ret = msgrcv(
            msgid, &msg_rcv, sizeof(msg_rcv) - sizeof(long), 0, IPC_NOWAIT);
    } else {
        ret = msgrcv(msgid, &msg_rcv, sizeof(msg_rcv) - sizeof(long), 0, 0);
    }
    if (ret == -1) {
        // read none
        output_tokens_data[0] = -2;  // stop_flag
        output_tokens_data[1] = 0;   // message_flag, Target: 3, Draft: 4
        output_tokens_data[2] = 0;   // bsz
        return;
    }

    int bsz = msg_rcv.meta[2];
    output_tokens_data[0] = (int64_t)msg_rcv.meta[0];
    output_tokens_data[1] = (int64_t)msg_rcv.meta[1];
    output_tokens_data[2] = (int64_t)msg_rcv.meta[2];

    int output_tokens_offset = 3 + MAX_BSZ;
    for (int i = 0; i < bsz; i++) {
        int cur_token_num = msg_rcv.meta[3 + i];
        output_tokens_data[3 + i] = (int64_t)cur_token_num;  // batch_token_nums

        auto* cur_output_token = output_tokens_data + output_tokens_offset +
                                 i * (MAX_DRAFT_TOKEN_NUM * (K + 1));
        auto* cur_output_score =
            output_scores_data + i * (MAX_DRAFT_TOKEN_NUM * (K + 1));
        auto* cur_batch_msg_rcv = &msg_rcv.mtext[i];
        for (int j = 0; j < cur_token_num; j++) {
            for (int k = 0; k < real_k + 1; k++) {
                cur_output_token[j * (K + 1) + k] =
                    (int64_t)cur_batch_msg_rcv->tokens[j * (K + 1) + k];
                cur_output_score[j * (K + 1) + k] =
                    cur_batch_msg_rcv->scores[j * (K + 1) + k];
            }
            output_ranks_data[i * MAX_DRAFT_TOKEN_NUM + j] =
                (int64_t)cur_batch_msg_rcv->ranks[j];
        }
    }
#ifdef SPECULATE_GET_WITH_OUTPUT_DEBUG
    std::cout << "msg data: " << std::endl;
    std::cout << "stop_flag: " << output_tokens_data[0]
              << ", message_flag: " << output_tokens_data[1]
              << ", bsz: " << output_tokens_data[2] << std::endl;
    for (int i = 0; i < output_tokens_data[2]; i++) {
        int cur_token_num = output_tokens_data[3 + i];
        std::cout << "batch " << i << " token_num: " << cur_token_num
                  << std::endl;
        for (int j = 0; j < cur_token_num; j++) {
            std::cout << "tokens: ";
            for (int k = 0; k < K + 1; k++) {
                std::cout
                    << output_tokens_data[output_tokens_offset +
                                          i * MAX_DRAFT_TOKEN_NUM * (K + 1) +
                                          j * (K + 1) + k]
                    << " ";
            }
            std::cout << std::endl;
            std::cout << "scores: ";
            for (int k = 0; k < K + 1; k++) {
                std::cout
                    << output_scores_data[i * MAX_DRAFT_TOKEN_NUM * (K + 1) +
                                          j * (K + 1) + k]
                    << " ";
            }
            std::cout << std::endl;
            std::cout << "ranks: "
                      << output_ranks_data[i * MAX_DRAFT_TOKEN_NUM + j]
                      << std::endl;
        }
    }
    std::cout << std::endl;
#endif
    return;
}

PD_BUILD_STATIC_OP(speculate_get_output_topk)
    .Inputs({"output_tokens", "output_scores", "output_ranks"})
    .Attrs({"real_k: int", "rank_id: int64_t", "wait_flag: bool"})
    .Outputs({"output_tokens_out", "output_scores_out", "output_ranks_out"})
    .SetInplaceMap({{"output_tokens", "output_tokens_out"},
                    {"output_scores", "output_scores_out"},
                    {"output_ranks", "output_ranks_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateGetOutMmsgTopK));
