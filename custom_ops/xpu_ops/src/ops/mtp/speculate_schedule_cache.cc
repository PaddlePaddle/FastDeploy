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
void SpeculateScheduleCache(const paddle::Tensor &draft_tokens,
                            const paddle::Tensor &block_tables,
                            const paddle::Tensor &stop_flags,
                            const paddle::Tensor &prompt_lens,
                            const paddle::Tensor &seq_lens_this_time,
                            const paddle::Tensor &seq_lens_encoder,
                            const paddle::Tensor &seq_lens_decoder,
                            const paddle::Tensor &step_seq_lens_decoder,
                            const paddle::Tensor &step_draft_tokens,
                            const paddle::Tensor &step_seq_lens_this_time,
                            const paddle::Tensor &accept_num,
                            const paddle::Tensor &accept_tokens,
                            const paddle::Tensor &is_block_step,
                            const paddle::Tensor &not_need_stop,
                            const paddle::Tensor &stop_nums,
                            const int block_size,
                            const int max_draft_tokens) {
  namespace api = baidu::xpu::api;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  api::Context *ctx = xpu_ctx->x_context();
  if (stop_flags.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  const int accept_tokens_len = accept_tokens.shape()[1];
  const int draft_token_len = draft_tokens.shape()[1];
  const int block_num_per_seq = block_tables.shape()[1];
  const int max_next_step_tokens = 2 * max_draft_tokens + 2;
  constexpr int BlockSize = MAX_BSZ;  // bsz <= 512
  bool prefill_one_step_stop = false;
  if (const char *env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP_V1")) {
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }
  auto not_need_stop_gpu = not_need_stop.copy_to(stop_flags.place(), false);

  int r = baidu::xpu::api::plugin::speculate_schedule_cache(
      ctx,
      draft_tokens.data<int64_t>(),
      const_cast<int *>(block_tables.data<int>()),
      const_cast<bool *>(stop_flags.data<bool>()),
      prompt_lens.data<int64_t>(),
      const_cast<int *>(seq_lens_this_time.data<int>()),
      const_cast<int *>(seq_lens_encoder.data<int>()),
      const_cast<int *>(seq_lens_decoder.data<int>()),
      const_cast<int *>(step_seq_lens_decoder.data<int>()),
      const_cast<int64_t *>(step_draft_tokens.data<int64_t>()),
      const_cast<int *>(step_seq_lens_this_time.data<int>()),
      const_cast<int *>(accept_num.data<int>()),
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<bool *>(is_block_step.data<bool>()),
      const_cast<bool *>(not_need_stop_gpu.data<bool>()),
      stop_nums.data<int64_t>(),
      real_bsz,
      max_bsz,
      max_next_step_tokens,
      draft_token_len,
      accept_tokens_len,
      block_size,
      block_num_per_seq,
      prefill_one_step_stop);
  // kernel launch
  PD_CHECK(r == 0, "speculate_free_and_reschedule  failed.");

  auto not_need_stop_cpu =
      not_need_stop_gpu.copy_to(not_need_stop.place(), true);
  bool *not_need_stop_data = const_cast<bool *>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

// PD_BUILD_STATIC_OP(speculate_schedule_cache)
PD_BUILD_OP(speculate_schedule_cache)
    .Inputs({"draft_tokens",
             "block_tables",
             "stop_flags",
             "prompt_lens",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_seq_lens_decoder",
             "step_draft_tokens",
             "step_seq_lens_this_time",
             "accept_num",
             "accept_tokens",
             "is_block_step",
             "not_need_stop",
             "stop_nums"})
    .Attrs({"block_size: int", "max_draft_tokens: int"})
    .Outputs({"draft_tokens_out",
              "block_tables_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_seq_lens_decoder_out",
              "step_draft_tokens_out",
              "step_seq_lens_this_time_out",
              "accept_num_out",
              "accept_tokens_out",
              "is_block_step_out",
              "not_need_stop_out"})
    .SetInplaceMap({
        {"draft_tokens", "draft_tokens_out"},
        {"block_tables", "block_tables_out"},
        {"stop_flags", "stop_flags_out"},
        {"seq_lens_this_time", "seq_lens_this_time_out"},
        {"seq_lens_encoder", "seq_lens_encoder_out"},
        {"seq_lens_decoder", "seq_lens_decoder_out"},
        {"step_seq_lens_decoder", "step_seq_lens_decoder_out"},
        {"step_draft_tokens", "step_draft_tokens_out"},
        {"step_seq_lens_this_time", "step_seq_lens_this_time_out"},
        {"accept_num", "accept_num_out"},
        {"accept_tokens", "accept_tokens_out"},
        {"is_block_step", "is_block_step_out"},
        {"not_need_stop", "not_need_stop_out"},
    })
    .SetKernelFn(PD_KERNEL(SpeculateScheduleCache));
