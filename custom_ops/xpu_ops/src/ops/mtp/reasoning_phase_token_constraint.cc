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
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

void ReasoningPhaseTokenConstraint(
    const paddle::Tensor& logits,  // inplace output
    const paddle::Tensor& token_ids_all,
    const paddle::Tensor& prompt_lens,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& allowed_tokens,
    const paddle::Tensor& reasoning_status,
    const paddle::Tensor& batch_id_per_token_output,
    const paddle::Tensor& cu_seqlens_q_output,
    const paddle::Tensor& enable_thinking,
    int64_t think_end_id,
    int64_t line_break_id) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  baidu::xpu::api::Context* ctx = xpu_ctx->x_context();

  std::unique_ptr<baidu::xpu::api::Context> cpu_ctx_guard;
  if (logits.is_cpu()) {
    cpu_ctx_guard.reset(new baidu::xpu::api::Context(baidu::xpu::api::kCPU));
    ctx = cpu_ctx_guard.get();
  }
  int bs = seq_lens_this_time.shape()[0];
  int token_num = logits.shape()[0];
  int vocab_size = logits.shape()[1];
  int max_seq_len = token_ids_all.shape()[1];
  int allowed_tokens_len = allowed_tokens.shape()[0];

  // Backup logits before enforce
  auto logits_tmp = logits.copy_to(logits.place(), false);

  switch (logits.type()) {
    case paddle::DataType::BFLOAT16: {
      using XPUType = typename XPUTypeTrait<paddle::bfloat16>::Type;
      typedef paddle::bfloat16 data_t;
      int r = fastdeploy::plugin::reasoning_phase_token_constraint(
          ctx,
          reinterpret_cast<const XPUType*>(logits_tmp.data<data_t>()),
          reinterpret_cast<XPUType*>(
              const_cast<data_t*>(logits.data<data_t>())),
          token_ids_all.data<int64_t>(),
          prompt_lens.data<int64_t>(),
          stop_flags.data<bool>(),
          seq_lens_encoder.data<int>(),
          step_idx.data<int64_t>(),
          allowed_tokens.data<int64_t>(),
          const_cast<int*>(reasoning_status.data<int32_t>()),
          batch_id_per_token_output.data<int>(),
          cu_seqlens_q_output.data<int>(),
          enable_thinking.data<bool>(),
          think_end_id,
          line_break_id,
          bs,
          token_num,
          vocab_size,
          max_seq_len,
          allowed_tokens_len);
      PD_CHECK(r == 0,
               "fastdeploy::plugin::reasoning_phase_token_constraint failed.");
    } break;
    case paddle::DataType::FLOAT16: {
      using XPUType = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 data_t;
      int r = fastdeploy::plugin::reasoning_phase_token_constraint(
          ctx,
          reinterpret_cast<const XPUType*>(logits_tmp.data<data_t>()),
          reinterpret_cast<XPUType*>(
              const_cast<data_t*>(logits.data<data_t>())),
          token_ids_all.data<int64_t>(),
          prompt_lens.data<int64_t>(),
          stop_flags.data<bool>(),
          seq_lens_encoder.data<int>(),
          step_idx.data<int64_t>(),
          allowed_tokens.data<int64_t>(),
          const_cast<int*>(reasoning_status.data<int32_t>()),
          batch_id_per_token_output.data<int>(),
          cu_seqlens_q_output.data<int>(),
          enable_thinking.data<bool>(),
          think_end_id,
          line_break_id,
          bs,
          token_num,
          vocab_size,
          max_seq_len,
          allowed_tokens_len);
      PD_CHECK(r == 0,
               "fastdeploy::plugin::reasoning_phase_token_constraint failed.");
    } break;
    case paddle::DataType::FLOAT32: {
      int r = fastdeploy::plugin::reasoning_phase_token_constraint(
          ctx,
          logits_tmp.data<float>(),
          const_cast<float*>(logits.data<float>()),
          token_ids_all.data<int64_t>(),
          prompt_lens.data<int64_t>(),
          stop_flags.data<bool>(),
          seq_lens_encoder.data<int>(),
          step_idx.data<int64_t>(),
          allowed_tokens.data<int64_t>(),
          const_cast<int*>(reasoning_status.data<int32_t>()),
          batch_id_per_token_output.data<int>(),
          cu_seqlens_q_output.data<int>(),
          enable_thinking.data<bool>(),
          think_end_id,
          line_break_id,
          bs,
          token_num,
          vocab_size,
          max_seq_len,
          allowed_tokens_len);
      PD_CHECK(r == 0,
               "fastdeploy::plugin::reasoning_phase_token_constraint failed.");
    } break;
    default:
      PD_THROW(
          "NOT supported data type. "
          "Only float16, bfloat16 and float32 are supported. ");
      break;
  }
}

PD_BUILD_STATIC_OP(reasoning_phase_token_constraint)
    .Inputs({"logits",
             "token_ids_all",
             "prompt_lens",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "step_idx",
             "allowed_tokens",
             "reasoning_status",
             "batch_id_per_token_output",
             "cu_seqlens_q_output",
             "enable_thinking"})
    .Outputs({"logits_out", "reasoning_status_out"})
    .Attrs({"think_end_id: int64_t", "line_break_id: int64_t"})
    .SetInplaceMap({{"logits", "logits_out"},
                    {"reasoning_status", "reasoning_status_out"}})
    .SetKernelFn(PD_KERNEL(ReasoningPhaseTokenConstraint));
