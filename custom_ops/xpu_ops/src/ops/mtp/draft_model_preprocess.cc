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
#include "paddle/extension.h"
#include "paddle/phi/core/enforce.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;
void DraftModelPreprocess(const paddle::Tensor& draft_tokens,
                          const paddle::Tensor& input_ids,
                          const paddle::Tensor& stop_flags,
                          const paddle::Tensor& seq_lens_this_time,
                          const paddle::Tensor& seq_lens_encoder,
                          const paddle::Tensor& seq_lens_decoder,
                          const paddle::Tensor& step_idx,
                          const paddle::Tensor& not_need_stop,
                          const paddle::Tensor& pre_ids,
                          const paddle::Tensor& accept_tokens,
                          const paddle::Tensor& accept_num,
                          const paddle::Tensor& target_model_seq_lens_encoder,
                          const paddle::Tensor& target_model_seq_lens_decoder,
                          const paddle::Tensor& target_model_step_idx,
                          const paddle::Tensor& target_model_stop_flags,
                          const paddle::Tensor& max_dec_len,
                          const paddle::Tensor& target_model_draft_tokens,
                          const int num_model_step,
                          const bool is_splitwise_prefill) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  api::Context* ctx = static_cast<const phi::XPUContext*>(dev_ctx)->x_context();
  if (draft_tokens.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }
  int real_bsz = seq_lens_this_time.shape()[0];
  PADDLE_ENFORCE_LE(
      real_bsz,
      1024,
      phi::errors::InvalidArgument(
          "draft_model_preprocess: real_bsz (%d) exceeds 1024.", real_bsz));
  int accept_tokens_len = accept_tokens.shape()[1];
  int input_ids_len = input_ids.shape()[1];
  int draft_tokens_len = draft_tokens.shape()[1];
  int pre_ids_len = pre_ids.shape()[1];
  int target_model_draft_tokens_len = target_model_draft_tokens.shape()[1];
  auto not_need_stop_gpu =
      not_need_stop.copy_to(seq_lens_this_time.place(), false);

  int r = fastdeploy::plugin::draft_model_preprocess(
      ctx,
      const_cast<int64_t*>(draft_tokens.data<int64_t>()),
      const_cast<int64_t*>(input_ids.data<int64_t>()),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      const_cast<bool*>(not_need_stop_gpu.data<bool>()),
      const_cast<int64_t*>(pre_ids.data<int64_t>()),
      accept_tokens.data<int64_t>(),
      accept_num.data<int>(),
      target_model_seq_lens_encoder.data<int>(),
      target_model_seq_lens_decoder.data<int>(),
      target_model_step_idx.data<int64_t>(),
      target_model_stop_flags.data<bool>(),
      max_dec_len.data<int64_t>(),
      const_cast<int64_t*>(target_model_draft_tokens.data<int64_t>()),
      real_bsz,
      num_model_step,
      accept_tokens_len,
      draft_tokens_len,
      input_ids_len,
      target_model_draft_tokens_len,
      pre_ids_len,
      is_splitwise_prefill);

  PD_CHECK(r == 0, "fastdeploy::plugin::draft_model_preprocess failed.");
  auto not_need_stop_cpu =
      not_need_stop_gpu.copy_to(not_need_stop.place(), false);
  bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

PD_BUILD_STATIC_OP(draft_model_preprocess)
    .Inputs({"draft_tokens",
             "input_ids",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "not_need_stop",
             "pre_ids",
             "accept_tokens",
             "accept_num",
             "target_model_seq_lens_encoder",
             "target_model_seq_lens_decoder",
             "target_model_step_idx",
             "target_model_stop_flags",
             "max_dec_len",
             "target_model_draft_tokens"})
    .Outputs({"draft_tokens_out",
              "input_ids_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_idx_out",
              "not_need_stop_out",
              "pre_ids_out"})
    .Attrs({"num_model_step: int", "is_splitwise_prefill: bool"})
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"input_ids", "input_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"step_idx", "step_idx_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"pre_ids", "pre_ids_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelPreprocess));
