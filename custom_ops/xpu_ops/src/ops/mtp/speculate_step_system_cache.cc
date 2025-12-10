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

#include "speculate_step_helper.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void SpeculateStepSystemCachePaddle(
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &ori_seq_lens_encoder,
    const paddle::Tensor &ori_seq_lens_decoder,
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
  SpeculateStepPaddleBase(
      stop_flags,
      seq_lens_this_time,
      ori_seq_lens_encoder,
      paddle::make_optional<paddle::Tensor>(ori_seq_lens_decoder),
      seq_lens_encoder,
      seq_lens_decoder,
      block_tables,
      encoder_block_lens,
      is_block_step,
      step_block_list,
      step_lens,
      recover_block_list,
      recover_lens,
      need_block_list,
      need_block_len,
      used_list_len,
      free_list,
      free_list_len,
      input_ids,
      pre_ids,
      step_idx,
      next_tokens,
      first_token_ids,
      accept_num,
      block_size,
      encoder_decoder_block_num,
      max_draft_tokens);
}

PD_BUILD_STATIC_OP(speculate_step_system_cache)
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
    .SetKernelFn(PD_KERNEL(SpeculateStepSystemCachePaddle));
