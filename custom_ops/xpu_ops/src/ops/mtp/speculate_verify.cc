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
#include "paddle/common/flags.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "xpu/internal/infra_op.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;

void SpeculateVerify(const paddle::Tensor &sampled_token_ids,
                     const paddle::Tensor &accept_tokens,
                     const paddle::Tensor &accept_num,
                     const paddle::Tensor &step_idx,
                     const paddle::Tensor &stop_flags,
                     const paddle::Tensor &seq_lens_encoder,
                     const paddle::Tensor &seq_lens_decoder,
                     const paddle::Tensor &draft_tokens,
                     const paddle::Tensor &seq_lens_this_time,
                     const paddle::Tensor &verify_tokens,
                     const paddle::Tensor &verify_scores,
                     const paddle::Tensor &max_dec_len,
                     const paddle::Tensor &end_tokens,
                     const paddle::Tensor &is_block_step,
                     const paddle::Tensor &output_cum_offsets,
                     const paddle::Tensor &actual_candidate_len,
                     const paddle::Tensor &actual_draft_token_nums,
                     const paddle::Tensor &topp,
                     int max_seq_len,
                     int verify_window,
                     bool enable_topp,
                     bool benchmark_mode,
                     bool accept_all_drafts) {
  auto bsz = accept_tokens.shape()[0];
  int real_bsz = seq_lens_this_time.shape()[0];
  auto max_draft_tokens = draft_tokens.shape()[1];
  auto end_length = end_tokens.shape()[0];
  auto max_candidate_len = verify_tokens.shape()[1];

  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  api::Context *ctx =
      static_cast<const phi::XPUContext *>(dev_ctx)->x_context();
  bool xpu_ctx_flag = true;
  if (draft_tokens.is_cpu()) {
    ctx = new api::Context(api::kCPU);
    xpu_ctx_flag = false;
  }

  bool use_topk = false;
  char *env_var = getenv("SPECULATE_VERIFY_USE_TOPK");
  if (env_var) {
    use_topk = static_cast<bool>(std::stoi(env_var));
  }
  bool use_target_sampling = false;
  char *env_var_1 = getenv("SPECULATE_VERIFY_USE_TARGET_SAMPLING");
  if (env_var_1) {
    use_target_sampling = static_cast<bool>(std::stoi(env_var_1));
  }
  bool prefill_one_step_stop = false;
  if (const char *env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP")) {
    // std::cout << "Your PATH is: " << env_p << '\n';
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }
  // random
  int random_seed = 0;
  std::vector<int64_t> infer_seed(bsz, random_seed);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::vector<float> dev_curand_states_cpu;
  for (int i = 0; i < bsz; i++) {
    std::mt19937_64 engine(infer_seed[i]);
    dev_curand_states_cpu.push_back(dist(engine));
  }
  float *dev_curand_states_xpu;
  if (xpu_ctx_flag) {
    xpu::ctx_guard RAII_GUARD(ctx);
    dev_curand_states_xpu =
        RAII_GUARD.alloc<float>(dev_curand_states_cpu.size());
    xpu_memcpy(dev_curand_states_xpu,
               dev_curand_states_cpu.data(),
               dev_curand_states_cpu.size() * sizeof(float),
               XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  }

  auto dev_curand_states =
      !xpu_ctx_flag ? dev_curand_states_cpu.data() : dev_curand_states_xpu;
  int ret;
  if (use_topk) {
    if (enable_topp) {
      ret = baidu::xpu::api::plugin::speculate_verify<true, true>(
          ctx,
          sampled_token_ids.data<int64_t>(),
          const_cast<int64_t *>(accept_tokens.data<int64_t>()),
          const_cast<int *>(accept_num.data<int>()),
          const_cast<int64_t *>(step_idx.data<int64_t>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          draft_tokens.data<int64_t>(),
          actual_draft_token_nums.data<int>(),
          dev_curand_states,
          topp.data<float>(),
          seq_lens_this_time.data<int>(),
          verify_tokens.data<int64_t>(),
          verify_scores.data<float>(),
          max_dec_len.data<int64_t>(),
          end_tokens.data<int64_t>(),
          is_block_step.data<bool>(),
          output_cum_offsets.data<int>(),
          actual_candidate_len.data<int>(),
          real_bsz,
          max_draft_tokens,
          end_length,
          max_seq_len,
          max_candidate_len,
          verify_window,
          prefill_one_step_stop,
          benchmark_mode,
          accept_all_drafts,
          use_target_sampling);
      PD_CHECK(ret == 0, "speculate_verify failed.");
    } else {
      ret = baidu::xpu::api::plugin::speculate_verify<false, true>(
          ctx,
          sampled_token_ids.data<int64_t>(),
          const_cast<int64_t *>(accept_tokens.data<int64_t>()),
          const_cast<int *>(accept_num.data<int>()),
          const_cast<int64_t *>(step_idx.data<int64_t>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          draft_tokens.data<int64_t>(),
          actual_draft_token_nums.data<int>(),
          dev_curand_states,
          topp.data<float>(),
          seq_lens_this_time.data<int>(),
          verify_tokens.data<int64_t>(),
          verify_scores.data<float>(),
          max_dec_len.data<int64_t>(),
          end_tokens.data<int64_t>(),
          is_block_step.data<bool>(),
          output_cum_offsets.data<int>(),
          actual_candidate_len.data<int>(),
          real_bsz,
          max_draft_tokens,
          end_length,
          max_seq_len,
          max_candidate_len,
          verify_window,
          prefill_one_step_stop,
          benchmark_mode,
          accept_all_drafts,
          use_target_sampling);
    }
    PD_CHECK(ret == 0, "speculate_verify failed.");
  } else {
    if (enable_topp) {
      ret = baidu::xpu::api::plugin::speculate_verify<true, false>(
          ctx,
          sampled_token_ids.data<int64_t>(),
          const_cast<int64_t *>(accept_tokens.data<int64_t>()),
          const_cast<int *>(accept_num.data<int>()),
          const_cast<int64_t *>(step_idx.data<int64_t>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          draft_tokens.data<int64_t>(),
          actual_draft_token_nums.data<int>(),
          dev_curand_states,
          topp.data<float>(),
          seq_lens_this_time.data<int>(),
          verify_tokens.data<int64_t>(),
          verify_scores.data<float>(),
          max_dec_len.data<int64_t>(),
          end_tokens.data<int64_t>(),
          is_block_step.data<bool>(),
          output_cum_offsets.data<int>(),
          actual_candidate_len.data<int>(),
          real_bsz,
          max_draft_tokens,
          end_length,
          max_seq_len,
          max_candidate_len,
          verify_window,
          prefill_one_step_stop,
          benchmark_mode,
          accept_all_drafts,
          use_target_sampling);
      PD_CHECK(ret == 0, "speculate_verify failed.");
    } else {
      ret = baidu::xpu::api::plugin::speculate_verify<false, false>(
          ctx,
          sampled_token_ids.data<int64_t>(),
          const_cast<int64_t *>(accept_tokens.data<int64_t>()),
          const_cast<int *>(accept_num.data<int>()),
          const_cast<int64_t *>(step_idx.data<int64_t>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          draft_tokens.data<int64_t>(),
          actual_draft_token_nums.data<int>(),
          dev_curand_states,
          topp.data<float>(),
          seq_lens_this_time.data<int>(),
          verify_tokens.data<int64_t>(),
          verify_scores.data<float>(),
          max_dec_len.data<int64_t>(),
          end_tokens.data<int64_t>(),
          is_block_step.data<bool>(),
          output_cum_offsets.data<int>(),
          actual_candidate_len.data<int>(),
          real_bsz,
          max_draft_tokens,
          end_length,
          max_seq_len,
          max_candidate_len,
          verify_window,
          prefill_one_step_stop,
          benchmark_mode,
          accept_all_drafts,
          use_target_sampling);
    }
    PD_CHECK(ret == 0, "speculate_verify failed.");
  }
  if (draft_tokens.is_cpu()) {
    delete ctx;
  }
}

PD_BUILD_STATIC_OP(speculate_verify)
    .Inputs({"sampled_token_ids",
             "accept_tokens",
             "accept_num",
             "step_idx",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "stop_flags",
             "draft_tokens",
             "seq_lens_this_time",
             "verify_tokens",
             "verify_scores",
             "max_dec_len",
             "end_tokens",
             "is_block_step",
             "output_cum_offsets",
             "actual_candidate_len",
             "actual_draft_token_nums",
             "topp"})
    .Outputs({"accept_tokens_out",
              "accept_num_out",
              "step_idx_out",
              "stop_flags_out"})
    .Attrs({"max_seq_len: int",
            "verify_window: int",
            "enable_topp: bool",
            "benchmark_mode: bool",
            "accept_all_drafts: bool"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"accept_num", "accept_num_out"},
                    {"step_idx", "step_idx_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateVerify));
