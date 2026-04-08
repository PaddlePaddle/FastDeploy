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

void DraftModelUpdate(const paddle::Tensor& inter_next_tokens,
                      const paddle::Tensor& draft_tokens,
                      const paddle::Tensor& pre_ids,
                      const paddle::Tensor& seq_lens_this_time,
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& seq_lens_decoder,
                      const paddle::Tensor& step_idx,
                      const paddle::Tensor& output_cum_offsets,
                      const paddle::Tensor& stop_flags,
                      const paddle::Tensor& not_need_stop,
                      const paddle::Tensor& max_dec_len,
                      const paddle::Tensor& end_ids,
                      const paddle::Tensor& base_model_draft_tokens,
                      const int max_seq_len,
                      const int substep) {
  // printf("enter clear \n");
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  baidu::xpu::api::Context* ctx =
      static_cast<const phi::XPUContext*>(dev_ctx)->x_context();

  if (draft_tokens.is_cpu()) {
    ctx = new baidu::xpu::api::Context(baidu::xpu::api::kCPU);
  }

  auto seq_lens_this_time_shape = seq_lens_this_time.shape();
  const int real_bsz = seq_lens_this_time_shape[0];
  auto not_need_stop_device =
      not_need_stop.copy_to(seq_lens_this_time.place(), false);
  const int end_ids_len = end_ids.shape()[0];
  const int max_draft_token = draft_tokens.shape()[1];
  const int pre_id_length = pre_ids.shape()[1];
  const int max_base_model_draft_token = base_model_draft_tokens.shape()[1];
  constexpr int BlockSize = 512;
  bool prefill_one_step_stop = false;
  if (const char* env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP")) {
    // std::cout << "Your PATH is: " << env_p << '\n';
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }

  int r = baidu::xpu::api::plugin::draft_model_update(
      ctx,
      inter_next_tokens.data<int64_t>(),
      const_cast<int64_t*>(draft_tokens.data<int64_t>()),
      const_cast<int64_t*>(pre_ids.data<int64_t>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      output_cum_offsets.data<int>(),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<bool*>(not_need_stop_device.data<bool>()),
      max_dec_len.data<int64_t>(),
      end_ids.data<int64_t>(),
      const_cast<int64_t*>(base_model_draft_tokens.data<int64_t>()),
      real_bsz,
      max_draft_token,
      pre_id_length,
      max_base_model_draft_token,
      end_ids_len,
      max_seq_len,
      substep,
      prefill_one_step_stop);

  PD_CHECK(r == 0, "draft_model_update  failed.");
}

PD_BUILD_STATIC_OP(draft_model_update)
    .Inputs({"inter_next_tokens",
             "draft_tokens",
             "pre_ids",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "output_cum_offsets",
             "stop_flags",
             "not_need_stop",
             "max_dec_len",
             "end_ids",
             "base_model_draft_tokens"})
    .Attrs({"max_seq_len: int", "substep: int"})
    .Outputs({"draft_tokens_out",
              "pre_ids_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_idx_out",
              "stop_flags_out",
              "not_need_stop_out",
              "base_model_draft_tokens_out"})
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"pre_ids", "pre_ids_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"step_idx", "step_idx_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"base_model_draft_tokens", "base_model_draft_tokens_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelUpdate));
