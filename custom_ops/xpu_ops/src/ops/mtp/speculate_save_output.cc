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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include <stdio.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/types.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#define MAX_BSZ 256
#define MAX_DRAFT_TOKENS 6

struct msgdata {
  long mtype;  // NOLINT
  int mtext[MAX_BSZ * MAX_DRAFT_TOKENS + MAX_BSZ +
            2];  // stop_flag, bsz, tokens
};

void SpeculateSaveWithOutputMsg(const paddle::Tensor& accept_tokens,
                                const paddle::Tensor& accept_num,
                                const paddle::Tensor& not_need_stop,
                                int64_t rank_id,
                                int msg_queue_id,
                                bool save_each_rank) {
  // printf("enter save output");
  if (!save_each_rank && rank_id > 0) {
    return;
  }

  int max_draft_tokens = accept_tokens.shape()[1];

  auto accept_tokens_cpu = accept_tokens.copy_to(paddle::CPUPlace(), true);
  auto accept_num_cpu = accept_num.copy_to(paddle::CPUPlace(), true);
  int64_t* accept_tokens_data = accept_tokens_cpu.data<int64_t>();
  int* accept_num_data = accept_num_cpu.data<int>();

  if (const char* inference_msg_queue_id_env_p =
          std::getenv("INFERENCE_MSG_QUEUE_ID")) {
    std::string inference_msg_queue_id_env_str(inference_msg_queue_id_env_p);
    int inference_msg_queue_id_from_env =
        std::stoi(inference_msg_queue_id_env_str);
#ifdef GET_OUTPUT_DEBUG
    std::cout << "Your INFERENCE_MSG_QUEUE_ID is: "
              << inference_msg_queue_id_from_env << std::endl;
#endif
    msg_queue_id = inference_msg_queue_id_from_env;
  }
  static struct msgdata msg_sed;
  static key_t key = ftok("./", msg_queue_id);
  static int msgid = msgget(key, IPC_CREAT | 0666);

  msg_sed.mtype = 1;
  bool not_need_stop_data = not_need_stop.data<bool>()[0];

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
    std::cout << "Failed to got INFERENCE_MSG_ID at env, use (int)1 as default."
              << std::endl;
#endif
  }

  msg_sed.mtext[0] = not_need_stop_data ? inference_msg_id_from_env
                                        : -inference_msg_id_from_env;
  int bsz = accept_tokens.shape()[0];
  msg_sed.mtext[1] = bsz;

  for (int i = 2; i < MAX_BSZ + 2; i++) {
    if (i - 2 >= bsz) {
      msg_sed.mtext[i] = 0;
    } else {
      msg_sed.mtext[i] = static_cast<int>(accept_num_data[i - 2]);
    }
  }
  for (int i = MAX_BSZ + 2; i < MAX_BSZ * MAX_DRAFT_TOKENS + MAX_BSZ + 2; i++) {
    int token_id = i - MAX_BSZ - 2;
    int bid = token_id / MAX_DRAFT_TOKENS;
    int local_token_id = token_id % MAX_DRAFT_TOKENS;
    if (token_id / MAX_DRAFT_TOKENS >= bsz) {
      msg_sed.mtext[i] = 0;
    } else {
      msg_sed.mtext[i] =
          accept_tokens_data[bid * max_draft_tokens + local_token_id];
    }
  }
  if ((msgsnd(msgid,
              &msg_sed,
              (MAX_BSZ * MAX_DRAFT_TOKENS + MAX_BSZ + 2) * 4,
              0)) == -1) {
    printf("full msg buffer\n");
  }
  return;
}

void SpeculateSaveWithOutputMsgStatic(const paddle::Tensor& accept_tokens,
                                      const paddle::Tensor& accept_num,
                                      const paddle::Tensor& not_need_stop,
                                      int64_t rank_id,
                                      bool save_each_rank) {
  SpeculateSaveWithOutputMsg(
      accept_tokens, accept_num, not_need_stop, rank_id, 1, save_each_rank);
}

void SpeculateSaveWithOutputMsgDynamic(const paddle::Tensor& accept_tokens,
                                       const paddle::Tensor& accept_num,
                                       const paddle::Tensor& not_need_stop,
                                       int64_t rank_id,
                                       int msg_queue_id,
                                       bool save_each_rank) {
  SpeculateSaveWithOutputMsg(accept_tokens,
                             accept_num,
                             not_need_stop,
                             rank_id,
                             msg_queue_id,
                             save_each_rank);
}

PD_BUILD_OP(speculate_save_output)
    .Inputs({"accept_tokens", "accept_num", "not_need_stop"})
    .Attrs({"rank_id: int64_t", "save_each_rank: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"accept_tokens", "x_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateSaveWithOutputMsgStatic));

PD_BUILD_OP(speculate_save_output_dynamic)
    .Inputs({"accept_tokens", "accept_num", "not_need_stop"})
    .Attrs({"rank_id: int64_t", "msg_queue_id: int", "save_each_rank: bool"})
    .Outputs({"x_out"})
    .SetInplaceMap({{"accept_tokens", "x_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateSaveWithOutputMsgDynamic));
