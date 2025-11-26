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

void SpeculateSaveOutMmsgTopK(const paddle::Tensor& sampled_token_ids,
                              const paddle::Tensor& logprob_token_ids,
                              const paddle::Tensor& logprob_scores,
                              const paddle::Tensor& logprob_ranks,
                              const paddle::Tensor& token_num_per_batch,
                              const paddle::Tensor& cu_batch_token_offset,
                              const paddle::Tensor& not_need_stop,
                              const paddle::Tensor& seq_lens_decoder,
                              const paddle::Tensor& prompt_lens,
                              int message_flag,  // Target: 3, Draft: 4
                              int64_t rank_id,
                              bool save_each_rank) {
  if (!save_each_rank && rank_id > 0) {
    return;
  }
  auto sampled_token_ids_cpu =
      sampled_token_ids.copy_to(paddle::CPUPlace(), false);
  auto logprob_token_ids_cpu =
      logprob_token_ids.copy_to(paddle::CPUPlace(), false);
  auto logprob_scores_cpu = logprob_scores.copy_to(paddle::CPUPlace(), false);
  auto logprob_ranks_cpu = logprob_ranks.copy_to(paddle::CPUPlace(), false);
  auto token_num_per_batch_cpu =
      token_num_per_batch.copy_to(paddle::CPUPlace(), false);
  auto cu_batch_token_offset_cpu =
      cu_batch_token_offset.copy_to(paddle::CPUPlace(), false);
  auto seq_lens_decoder_cpu =
      seq_lens_decoder.copy_to(paddle::CPUPlace(), true);
  auto prompt_lens_cpu = prompt_lens.copy_to(paddle::CPUPlace(), true);
  int64_t* sampled_token_ids_data = sampled_token_ids_cpu.data<int64_t>();
  int64_t* logprob_token_ids_data = logprob_token_ids_cpu.data<int64_t>();
  float* logprob_scores_data = logprob_scores_cpu.data<float>();
  int64_t* logprob_ranks_data = logprob_ranks_cpu.data<int64_t>();
  int* token_num_per_batch_data = token_num_per_batch_cpu.data<int>();
  int* cu_batch_token_offset_data = cu_batch_token_offset_cpu.data<int>();
  int* seq_lens_decoder_data = seq_lens_decoder_cpu.data<int>();
  int64_t* prompt_lens_data = prompt_lens_cpu.data<int64_t>();

  static struct msgdata msg_sed;
  int msg_queue_id = 1;
  if (const char* inference_msg_queue_id_env_p =
          std::getenv("INFERENCE_MSG_QUEUE_ID")) {
    std::string inference_msg_queue_id_env_str(inference_msg_queue_id_env_p);
    int inference_msg_queue_id_from_env =
        std::stoi(inference_msg_queue_id_env_str);
    msg_queue_id = inference_msg_queue_id_from_env;
#ifdef SPECULATE_SAVE_WITH_OUTPUT_DEBUG
    std::cout << "Your INFERENCE_MSG_QUEUE_ID is: "
              << inference_msg_queue_id_from_env << std::endl;
#endif
  } else {
#ifdef SPECULATE_SAVE_WITH_OUTPUT_DEBUG
    std::cout << "Failed to got INFERENCE_MSG_QUEUE_ID at env, use default."
              << std::endl;
#endif
  }
  int inference_msg_id_from_env = 1;
  if (const char* inference_msg_id_env_p = std::getenv("INFERENCE_MSG_ID")) {
    std::string inference_msg_id_env_str(inference_msg_id_env_p);
    inference_msg_id_from_env = std::stoi(inference_msg_id_env_str);
    if (inference_msg_id_from_env == 2) {
      // 2 and -2 is perserve for no-output indication.
      throw std::runtime_error(
          " INFERENCE_MSG_ID cannot be 2, please use other number.");
    }
    if (inference_msg_id_from_env < 0) {
      throw std::runtime_error(
          " INFERENCE_MSG_ID cannot be negative, please use other "
          "number.");
    }
#ifdef SPECULATE_SAVE_WITH_OUTPUT_DEBUG
    std::cout << "Your INFERENCE_MSG_ID is: " << inference_msg_id_from_env
              << std::endl;
#endif
  } else {
#ifdef SPECULATE_SAVE_WITH_OUTPUT_DEBUG
    std::cout << "Failed to got INFERENCE_MSG_ID at env, use (int)1 as default."
              << std::endl;
#endif
  }
  static key_t key = ftok("/dev/shm", msg_queue_id);
  static int msgid = msgget(key, IPC_CREAT | 0666);
#ifdef SPECULATE_SAVE_WITH_OUTPUT_DEBUG
  std::cout << "save_output_key: " << key << std::endl;
  std::cout << "save msgid: " << msgid << std::endl;
#endif
  msg_sed.mtype = 1;
  msg_sed.meta[0] = not_need_stop.data<bool>()[0] ? inference_msg_id_from_env
                                                  : -inference_msg_id_from_env;
  msg_sed.meta[1] = message_flag;
  int bsz = token_num_per_batch.shape()[0];
  msg_sed.meta[2] = bsz;
  int max_num_logprobs = logprob_token_ids.shape()[1];
  for (int i = 0; i < bsz; i++) {
    int cur_token_num;
    if (seq_lens_decoder_data[i] < prompt_lens_data[i]) {
      cur_token_num = 0;
    } else {
      cur_token_num = token_num_per_batch_data[i];
    }
    msg_sed.meta[3 + i] = cur_token_num;
    auto* cur_batch_msg_sed = &msg_sed.mtext[i];
    int token_offset = cu_batch_token_offset_data[i];
    for (int j = 0; j < cur_token_num; j++) {
      auto* cur_tokens = &cur_batch_msg_sed->tokens[j * (K + 1)];
      auto* cur_scores = &cur_batch_msg_sed->scores[j * (K + 1)];
      for (int k = 0; k < K + 1; k++) {
        if (k == 0) {
          cur_tokens[k] = (int)sampled_token_ids_data[token_offset + j];
          cur_scores[k] = logprob_scores_data[(token_offset + j) * (K + 1) + k];
        } else if (k < max_num_logprobs) {
          cur_tokens[k] =
              (int)logprob_token_ids_data[(token_offset + j) * (K + 1) + k];
          cur_scores[k] = logprob_scores_data[(token_offset + j) * (K + 1) + k];
        } else {
          cur_tokens[k] = -1;
          cur_scores[k] = 0.0;
        }
      }
      cur_batch_msg_sed->ranks[j] = (int)logprob_ranks_data[token_offset + j];
    }
  }
#ifdef SPECULATE_SAVE_WITH_OUTPUT_DEBUG
  std::cout << "msg data: " << std::endl;
  std::cout << "stop_flag: " << msg_sed.meta[0]
            << ", message_flag: " << msg_sed.meta[1]
            << ", bsz: " << msg_sed.meta[2] << std::endl;
  for (int i = 0; i < bsz; i++) {
    int cur_token_num = msg_sed.meta[3 + i];
    auto* cur_batch_msg_sed = &msg_sed.mtext[i];
    std::cout << "batch " << i << " token_num: " << cur_token_num << std::endl;
    for (int j = 0; j < cur_token_num; j++) {
      auto* cur_tokens = &cur_batch_msg_sed->tokens[j * (K + 1)];
      auto* cur_scores = &cur_batch_msg_sed->scores[j * (K + 1)];
      std::cout << "tokens: ";
      for (int k = 0; k < K + 1; k++) {
        std::cout << cur_tokens[k] << " ";
      }
      std::cout << std::endl;
      std::cout << "scores: ";
      for (int k = 0; k < K + 1; k++) {
        std::cout << cur_scores[k] << " ";
      }
      std::cout << std::endl;
      std::cout << "ranks: " << cur_batch_msg_sed->ranks[j] << std::endl;
    }
  }
  std::cout << std::endl;
#endif
  if (msgsnd(msgid, &msg_sed, sizeof(msg_sed) - sizeof(long), 0) == -1) {
    printf("full msg buffer\n");
  }
}

PD_BUILD_STATIC_OP(speculate_save_output_topk)
    .Inputs({
        "sampled_token_ids",
        "logprob_token_ids",
        "logprob_scores",
        "logprob_ranks",
        "token_num_per_batch",
        "cu_batch_token_offset",
        "not_need_stop",
        "seq_lens_decoder",
        "prompt_lens",
    })
    .Attrs({"message_flag: int", "rank_id: int64_t", "save_each_rank: bool"})
    .SetKernelFn(PD_KERNEL(SpeculateSaveOutMmsgTopK));
