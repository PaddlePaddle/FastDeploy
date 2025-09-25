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

struct msgdata {
    long mtype;
    int mtext[MAX_BSZ * (K + 1) + 2];  // stop_flag, bsz, tokens
    float mtext_f[MAX_BSZ * (K + 1)];  // score
    int mtext_ranks[MAX_BSZ];  // ranks
};

void GetOutputTopK(const paddle::Tensor& x,
                   const paddle::Tensor& scores,
                   const paddle::Tensor& ranks,
                   int k,
                   int64_t rank_id,
                   bool wait_flag) {

    static struct msgdata msg_rcv;
    int msg_queue_id = 1;

    if (const char* inference_msg_queue_id_env_p =
            std::getenv("INFERENCE_MSG_QUEUE_ID")) {
        std::string inference_msg_queue_id_env_str(
            inference_msg_queue_id_env_p);
        int inference_msg_queue_id_from_env =
            std::stoi(inference_msg_queue_id_env_str);
#ifdef GET_OUTPUT_DEBUG
        std::cout << "Your INFERENCE_MSG_QUEUE_ID is: "
                  << inference_msg_queue_id_from_env << std::endl;
#endif
        msg_queue_id = inference_msg_queue_id_from_env;
    }
    static key_t key = ftok("/dev/shm", msg_queue_id);

    static int msgid = msgget(key, IPC_CREAT | 0666);
#ifdef GET_OUTPUT_DEBUG
    std::cout << "get_output_key: " << key << std::endl;
    std::cout << "get_output msgid: " << msgid << std::endl;
#endif

    int64_t* out_data = const_cast<int64_t*>(x.data<int64_t>());
    float* scores_data = const_cast<float*>(scores.data<float>());
    int64_t* ranks_data = const_cast<int64_t*>(ranks.data<int64_t>());
    int ret = -1;
    if (!wait_flag) {
        ret = msgrcv(msgid,
                     &msg_rcv,
                     (MAX_BSZ * (K + 1) + 2) * 4 + MAX_BSZ * (K + 1) * 4 + MAX_BSZ * 4,
                     0,
                     IPC_NOWAIT);
    } else {
        ret = msgrcv(msgid,
                     &msg_rcv,
                     (MAX_BSZ * (K + 1) + 2) * 4 + MAX_BSZ * (K + 1) * 4 +  MAX_BSZ * 4,
                     0,
                     0);
    }
    if (ret == -1) {
        // read none
        out_data[0] = -2;
        out_data[1] = 0;
        return;
    }

    int bsz = msg_rcv.mtext[1];
    out_data[0] = (int64_t)msg_rcv.mtext[0];
    out_data[1] = (int64_t)msg_rcv.mtext[1];

    for (int i = 0; i < bsz; i++) {
        for (int j = 0; j < k + 1; j++) {
            const int64_t offset = i * (K + 1) + j;
            out_data[offset + 2] = (int64_t)msg_rcv.mtext[offset + 2];
            scores_data[offset] = msg_rcv.mtext_f[offset];
        }
        ranks_data[i] = (int64_t)msg_rcv.mtext_ranks[i];
    }
    return;
}

PD_BUILD_STATIC_OP(get_output_topk)
    .Inputs({"x", "scores", "ranks"})
    .Attrs({"k: int", "rank_id: int64_t", "wait_flag: bool"})
    .Outputs({"x_out", "scores_out", "ranks_out"})
    .SetInplaceMap({{"x", "x_out"}, {"scores", "scores_out"}, {"ranks", "ranks_out"}})
    .SetKernelFn(PD_KERNEL(GetOutputTopK));
