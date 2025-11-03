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

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/phi/core/enforce.h"
#include "speculate_msg.h"  // NOLINT
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

// 为不修改接口调用方式，入参暂不改变
void SpeculateStepSchedule(
    const paddle::Tensor &stop_flags,
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
  namespace api = baidu::xpu::api;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  api::Context *ctx = xpu_ctx->x_context();
  if (stop_flags.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }
  const int bsz = seq_lens_this_time.shape()[0];
  const int block_num_per_seq = block_tables.shape()[1];
  const int length = input_ids.shape()[1];
  const int pre_id_length = pre_ids.shape()[1];
  constexpr int BlockSize = 256;  // bsz <= 256
  const int max_decoder_block_num =
      length / block_size -
      encoder_decoder_block_num;  // 最大输出长度对应的block -
                                  // 服务为解码分配的block数量
  auto step_lens_inkernel =
      paddle::full({1}, 0, paddle::DataType::INT32, stop_flags.place());
  auto step_bs_list =
      paddle::full({bsz}, 0, paddle::DataType::INT32, stop_flags.place());
  int r = baidu::xpu::api::plugin::speculate_free_and_reschedule(
      ctx,
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
      bsz,
      block_size,
      block_num_per_seq,
      max_decoder_block_num,
      max_draft_tokens);
  PD_CHECK(r == 0, "speculate_free_and_reschedule  failed.");
  // save output
  auto step_lens_cpu = step_lens_inkernel.copy_to(paddle::CPUPlace(), false);
  if (step_lens_cpu.data<int>()[0] > 0) {
    auto step_bs_list_cpu = step_bs_list.copy_to(paddle::CPUPlace(), false);
    auto next_tokens =
        paddle::full({bsz}, -1, paddle::DataType::INT64, paddle::CPUPlace());
    for (int i = 0; i < step_lens_cpu.data<int>()[0]; i++) {
      const int step_bid = step_bs_list_cpu.data<int>()[i];
      next_tokens.data<int64_t>()[step_bid] = -3;  // need reschedule
    }
    const int rank_id = static_cast<int>(stop_flags.place().GetDeviceId());
    printf("reschedule rank_id: %d, step_lens: %d",
           rank_id,
           step_lens_cpu.data<int>()[0]);
    const int64_t *x_data = next_tokens.data<int64_t>();
    static struct speculate_msgdata msg_sed;
    int msg_queue_id = rank_id;
    if (const char *inference_msg_queue_id_env_p =
            std::getenv("INFERENCE_MSG_QUEUE_ID")) {
      std::string inference_msg_queue_id_env_str(inference_msg_queue_id_env_p);
      int inference_msg_queue_id_from_env =
          std::stoi(inference_msg_queue_id_env_str);
      msg_queue_id = inference_msg_queue_id_from_env;
    } else {
      std::cout << "Failed to got INFERENCE_MSG_QUEUE_ID at env, use default."
                << std::endl;
    }
    int inference_msg_id_from_env = 1;
    if (const char *inference_msg_id_env_p = std::getenv("INFERENCE_MSG_ID")) {
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
      msg_sed.mtext[i] = static_cast<int>(x_data[i - 2]);
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
