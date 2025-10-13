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
// #define SAVE_WITH_OUTPUT_DEBUG

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ * (K + 1) + 2];  // stop_flag, bsz, tokens
    float mtext_f[MAX_BSZ * (K + 1)];  // score
    int mtext_ranks[MAX_BSZ];  // ranks
};

void SaveOutMmsgTopK(const paddle::Tensor& x,
                     const paddle::Tensor& logprob_token_ids,     // [bsz, k+1]
                     const paddle::Tensor& logprob_scores,  // [bsz, k+1]
                     const paddle::Tensor& ranks,
                     const paddle::Tensor& not_need_stop,
                     int64_t rank_id) {
    if (rank_id > 0) {
        return;
    }
    auto x_cpu = x.copy_to(paddle::CPUPlace(), false);
    auto logprob_token_ids_cpu = logprob_token_ids.copy_to(paddle::CPUPlace(), false);
    auto logprob_scores_cpu = logprob_scores.copy_to(paddle::CPUPlace(), false);
    auto ranks_cpu = ranks.copy_to(paddle::CPUPlace(), false);
    int64_t* x_data = x_cpu.data<int64_t>();
    int64_t* logprob_token_ids_data = logprob_token_ids_cpu.data<int64_t>();
    float* logprob_scores_data = logprob_scores_cpu.data<float>();
    int64_t* ranks_data = ranks_cpu.data<int64_t>();
    static struct msgdata msg_sed;
    int msg_queue_id = 1;
    if (const char* inference_msg_queue_id_env_p =
            std::getenv("INFERENCE_MSG_QUEUE_ID")) {
        std::string inference_msg_queue_id_env_str(
            inference_msg_queue_id_env_p);
        int inference_msg_queue_id_from_env =
            std::stoi(inference_msg_queue_id_env_str);
        msg_queue_id = inference_msg_queue_id_from_env;
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Your INFERENCE_MSG_QUEUE_ID is: "
                  << inference_msg_queue_id_from_env << std::endl;
#endif
    } else {
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Failed to got INFERENCE_MSG_QUEUE_ID at env, use default."
                  << std::endl;
#endif
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
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout << "Your INFERENCE_MSG_ID is: " << inference_msg_id_from_env
                  << std::endl;
#endif
    } else {
#ifdef SAVE_WITH_OUTPUT_DEBUG
        std::cout
            << "Failed to got INFERENCE_MSG_ID at env, use (int)1 as default."
            << std::endl;
#endif
    }
    static key_t key = ftok("/dev/shm", msg_queue_id);
    static int msgid = msgget(key, IPC_CREAT | 0666);
#ifdef SAVE_WITH_OUTPUT_DEBUG
    std::cout << "save_output_key: " << key << std::endl;
    std::cout << "save msgid: " << msgid << std::endl;
#endif
    msg_sed.mtype = 1;
    bool not_need_stop_data = not_need_stop.data<bool>()[0];
    msg_sed.mtext[0] = not_need_stop_data ? inference_msg_id_from_env
                                          : -inference_msg_id_from_env;
    int bsz = x.shape()[0];
    int max_num_logprobs = logprob_token_ids.shape()[1];
    msg_sed.mtext[1] = bsz;
    for (int i = 0; i < bsz; i++) {
        for (int j = 0; j < K + 1; j++) {
            const int64_t offset = i * (K + 1) + j;
            if (j == 0) {
                msg_sed.mtext[offset + 2] = (int)x_data[i];
                msg_sed.mtext_f[offset] = logprob_scores_data[i * max_num_logprobs + j];
            } else if (j < max_num_logprobs) {
                msg_sed.mtext[offset + 2] = (int)logprob_token_ids_data[i * max_num_logprobs + j];
                msg_sed.mtext_f[offset] = logprob_scores_data[i * max_num_logprobs + j];
            } else {
                msg_sed.mtext[offset + 2] = -1;
                msg_sed.mtext_f[offset] = 0.0;
            }
        }
        msg_sed.mtext_ranks[i] = (int)ranks_data[i];
    }
#ifdef SAVE_WITH_OUTPUT_DEBUG
    std::cout << "msg data: ";
    for (int i = 0; i < bsz; i++) {
        std::cout << " " << (int)x_data[i];
    }
    std::cout << std::endl;
#endif
    if ((msgsnd(msgid,
                &msg_sed,
                (MAX_BSZ * (K + 1) + 2) * 4 + (MAX_BSZ * (K + 1)) * 4 + MAX_BSZ * 4,
                0)) == -1) {
        printf("full msg buffer\n");
    }
    return;
}

PD_BUILD_STATIC_OP(save_output_topk)
    .Inputs({"x", "topk_ids", "logprob_scores", "ranks", "not_need_stop"})
    .Attrs({"rank_id: int64_t"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"x", "x_out"}})
    .SetKernelFn(PD_KERNEL(SaveOutMmsgTopK));
