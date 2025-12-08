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

#pragma once

#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

void SpeculateStepPaddleBase(
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &ori_seq_lens_encoder,
    const paddle::optional<paddle::Tensor> &ori_seq_lens_decoder,
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
    const int max_draft_tokens);
