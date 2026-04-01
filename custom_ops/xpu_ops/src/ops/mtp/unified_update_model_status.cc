// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/common/flags.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "xpu/internal/infra_op.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;
void UnifiedUpdateModelStatus(const paddle::Tensor &seq_lens_encoder,
                              const paddle::Tensor &seq_lens_decoder,
                              const paddle::Tensor &has_running_seqs,
                              const paddle::Tensor &step_input_ids,
                              const paddle::Tensor &adaptive_step_input_len,
                              const paddle::Tensor &step_output_ids,
                              const paddle::Tensor &step_output_len,
                              const paddle::Tensor &stop_flags,
                              const paddle::Tensor &seq_lens_this_time,
                              const paddle::Tensor &is_paused,
                              const paddle::Tensor &mask_rollback,
                              const paddle::Tensor &token_ids_all,
                              const paddle::Tensor &prompt_lens,
                              const paddle::Tensor &step_idx,
                              const paddle::Tensor &end_tokens,
                              const paddle::Tensor &max_dec_len,
                              const bool is_naive_mode,
                              const bool prefill_one_step_stop) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  api::Context *ctx = xpu_ctx->x_context();

  // just for ut to run base line
  std::unique_ptr<baidu::xpu::api::Context> cpu_ctx;
  if (seq_lens_encoder.place().GetType() == phi::AllocationType::CPU) {
    cpu_ctx = std::make_unique<baidu::xpu::api::Context>(baidu::xpu::api::kCPU);
    ctx = cpu_ctx.get();
  }

  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  PADDLE_ENFORCE_LE(
      max_bsz,
      1024,
      phi::errors::InvalidArgument(
          "unified_update_model_status: max_bsz (%d) must be <= 1024 "
          "(single-block launch limit).",
          max_bsz));
  const int max_step_tokens = step_input_ids.shape()[1];
  const int max_model_len = token_ids_all.shape()[1];
  const int num_end_tokens = end_tokens.shape()[0];

  // has_running_seqs is CPU tensor, need to copy to GPU first
  auto has_running_seqs_xpu =
      has_running_seqs.copy_to(seq_lens_this_time.place(), false);
  int r = fastdeploy::plugin::unified_update_model_status(
      ctx,
      const_cast<int *>(seq_lens_encoder.data<int>()),
      const_cast<int *>(seq_lens_decoder.data<int>()),
      const_cast<bool *>(has_running_seqs_xpu.data<bool>()),
      const_cast<int *>(mask_rollback.data<int>()),
      const_cast<int64_t *>(step_input_ids.data<int64_t>()),
      const_cast<int *>(adaptive_step_input_len.data<int>()),
      const_cast<int64_t *>(step_output_ids.data<int64_t>()),
      const_cast<int *>(step_output_len.data<int>()),
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int *>(seq_lens_this_time.data<int>()),
      const_cast<bool *>(is_paused.data<bool>()),
      const_cast<int64_t *>(token_ids_all.data<int64_t>()),
      prompt_lens.data<int64_t>(),
      const_cast<int64_t *>(step_idx.data<int64_t>()),
      end_tokens.data<int64_t>(),
      max_dec_len.data<int64_t>(),
      real_bsz,
      max_bsz,
      max_step_tokens,
      max_model_len,
      num_end_tokens,
      is_naive_mode,
      prefill_one_step_stop);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "unified_update_model_status");
  // Copy result back to CPU
  auto has_running_seqs_cpu =
      has_running_seqs_xpu.copy_to(has_running_seqs.place(), false);
  bool *out_data = const_cast<bool *>(has_running_seqs.data<bool>());
  out_data[0] = has_running_seqs_cpu.data<bool>()[0];
}

PD_BUILD_STATIC_OP(unified_update_model_status)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "has_running_seqs",
             "step_input_ids",
             "adaptive_step_input_len",
             "step_output_ids",
             "step_output_len",
             "stop_flags",
             "seq_lens_this_time",
             "is_paused",
             "mask_rollback",
             "token_ids_all",
             "prompt_lens",
             "step_idx",
             "end_tokens",
             "max_dec_len"})
    .Attrs({"is_naive_mode: bool", "prefill_one_step_stop: bool"})
    .Outputs({"seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "has_running_seqs_out",
              "step_input_ids_out",
              "adaptive_step_input_len_out",
              "step_output_ids_out",
              "step_output_len_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "mask_rollback_out",
              "token_ids_all_out",
              "step_idx_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"has_running_seqs", "has_running_seqs_out"},
                    {"step_input_ids", "step_input_ids_out"},
                    {"adaptive_step_input_len", "adaptive_step_input_len_out"},
                    {"step_output_ids", "step_output_ids_out"},
                    {"step_output_len", "step_output_len_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"mask_rollback", "mask_rollback_out"},
                    {"token_ids_all", "token_ids_all_out"},
                    {"step_idx", "step_idx_out"}})
    .SetKernelFn(PD_KERNEL(UnifiedUpdateModelStatus));
