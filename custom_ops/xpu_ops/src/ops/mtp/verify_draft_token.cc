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

// Verification kernel — outputs step_output_ids + step_output_len,
// and performs EOS / max_dec_len detection (read-only on step_idx).
// step_idx is NOT modified here; all state updates (including step_idx)
// are handled by unified_update_model_status.
//
// Verification strategies:
//   0 = TOPP         : draft token in top-p candidate set (+ verify_window
//   fallback) 1 = GREEDY       : draft token == top-1 token (strict argmax
//   match) 2 = TARGET_MATCH : draft token == target model's sampled token

#include <atomic>
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "paddle/extension.h"
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace api = baidu::xpu::api;

// Persistent seed/offset — mirrors GPU curand state lifecycle.
static std::atomic<uint64_t> g_seed{0};
static std::atomic<uint64_t> g_offset{0};

// ============================================================
// Host function
// ============================================================
void VerifyDraftTokens(
    // Core I/O
    const paddle::Tensor &step_output_ids,
    const paddle::Tensor &step_output_len,
    const paddle::Tensor &step_input_ids,
    // Target model outputs (optional, required for TARGET_MATCH)
    const paddle::optional<paddle::Tensor> &target_tokens,
    // Candidate set (optional, required for TOPP/GREEDY)
    const paddle::optional<paddle::Tensor> &candidate_ids,
    const paddle::optional<paddle::Tensor> &candidate_scores,
    const paddle::optional<paddle::Tensor> &candidate_lens,
    // Sampling params
    const paddle::Tensor &topp,
    // Metadata
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &end_tokens,
    const paddle::Tensor &is_block_step,
    const paddle::Tensor &cu_seqlens_q_output,
    const paddle::Tensor &reasoning_status,
    // max_dec_len / step_idx for EOS/max-len detection
    const paddle::Tensor &max_dec_len,
    const paddle::Tensor &step_idx,
    int max_seq_len,
    int verify_window,
    int verify_strategy,
    bool reject_all,
    bool accept_all) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  api::Context *ctx =
      static_cast<const phi::XPUContext *>(dev_ctx)->x_context();
  bool xpu_ctx_flag = true;
  if (step_output_ids.is_cpu()) {
    ctx = new api::Context(api::kCPU);
    xpu_ctx_flag = false;
  }

  auto bsz = step_output_ids.shape()[0];
  auto real_bsz = seq_lens_this_time.shape()[0];
  auto max_step_tokens = step_input_ids.shape()[1];
  auto end_length = end_tokens.shape()[0];
  // max_candidate_len: 1 if candidate_ids not provided, else from shape
  int max_candidate_len = candidate_ids ? candidate_ids->shape()[1] : 1;

  // curand state: only needed for TOPP(0) strategy (stochastic sampling)
  // Use persistent seed/offset (mirrors GPU curand lifecycle) so that
  // each call and each batch element produce distinct random numbers.
  uint64_t cur_seed = g_seed++;
  uint64_t cur_offset = g_offset++;
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::vector<float> dev_curand_states_cpu;
  for (int i = 0; i < bsz; i++) {
    std::mt19937_64 engine(cur_seed + i);
    engine.discard(cur_offset);
    dev_curand_states_cpu.push_back(dist(engine));
  }
  float *dev_curand_states = dev_curand_states_cpu.data();
  auto dev_curand_states_tensor =
      paddle::empty({static_cast<int64_t>(dev_curand_states_cpu.size())},
                    paddle::DataType::FLOAT32,
                    seq_lens_this_time.place());
  int ret;
  if (xpu_ctx_flag) {
    ret = api::do_host2device(ctx,
                              dev_curand_states_cpu.data(),
                              dev_curand_states_tensor.data<float>(),
                              dev_curand_states_cpu.size() * sizeof(float));
    PD_CHECK(ret == 0, "do_host2device failed.");
    dev_curand_states = dev_curand_states_tensor.data<float>();
  }

  // Get data pointers (nullptr if optional not provided)
  const int64_t *target_tokens_ptr =
      target_tokens ? target_tokens->data<int64_t>() : nullptr;
  const int64_t *candidate_ids_ptr =
      candidate_ids ? candidate_ids->data<int64_t>() : nullptr;
  const float *candidate_scores_ptr =
      candidate_scores ? candidate_scores->data<float>() : nullptr;
  const int *candidate_lens_ptr =
      candidate_lens ? candidate_lens->data<int>() : nullptr;

  // Validate parameters based on verify_strategy.
  // Note: empty_input_forward may lead to empty optional tensors — only
  // validate when bsz > 0 (i.e. there are active sequences).
  if (bsz > 0) {
    if (verify_strategy == 0 /* TOPP */) {
      if (!candidate_ids_ptr || !candidate_scores_ptr || !candidate_lens_ptr) {
        PD_THROW(
            "verify_strategy=TOPP (0) requires candidate_ids, "
            "candidate_scores, candidate_lens");
      }
    } else if (verify_strategy == 1 /* GREEDY */) {
      if (!target_tokens_ptr) {
        PD_THROW("verify_strategy=GREEDY (1) requires target_tokens (argmax)");
      }
    } else if (verify_strategy == 2 /* TARGET_MATCH */) {
      if (!target_tokens_ptr) {
        PD_THROW(
            "verify_strategy=TARGET_MATCH (2) requires target_tokens "
            "(sampled)");
      }
    }
  }
  ret = fastdeploy::plugin::verify_draft_tokens(
      ctx,
      // Core I/O
      const_cast<int64_t *>(step_output_ids.data<int64_t>()),
      const_cast<int *>(step_output_len.data<int>()),
      step_input_ids.data<int64_t>(),
      // Target model outputs
      target_tokens_ptr,
      // Candidate set
      candidate_ids_ptr,
      candidate_scores_ptr,
      candidate_lens_ptr,
      // Sampling params
      dev_curand_states,
      topp.data<float>(),
      // Metadata
      stop_flags.data<bool>(),
      seq_lens_encoder.data<int>(),
      seq_lens_this_time.data<int>(),
      end_tokens.data<int64_t>(),
      is_block_step.data<bool>(),
      cu_seqlens_q_output.data<int>(),
      reasoning_status.data<int>(),
      // max_dec_len / step_idx
      max_dec_len.data<int64_t>(),
      step_idx.data<int64_t>(),
      // Dimensions and config
      bsz,       // max_bsz
      real_bsz,  // real_bsz
      max_step_tokens,
      end_length,
      max_seq_len,
      max_candidate_len,
      verify_window,
      verify_strategy,
      reject_all,
      accept_all);
  if (step_output_ids.is_cpu()) {
    delete ctx;
  }
  PD_CHECK(ret == 0, "verify_draft_tokens failed.");
}

PD_BUILD_STATIC_OP(verify_draft_tokens)
    .Inputs({"step_output_ids",
             "step_output_len",
             "step_input_ids",
             paddle::Optional("target_tokens"),
             paddle::Optional("candidate_ids"),
             paddle::Optional("candidate_scores"),
             paddle::Optional("candidate_lens"),
             "topp",
             "stop_flags",
             "seq_lens_encoder",
             "seq_lens_this_time",
             "end_tokens",
             "is_block_step",
             "cu_seqlens_q_output",
             "reasoning_status",
             "max_dec_len",
             "step_idx"})
    .Outputs({"step_output_ids_out", "step_output_len_out"})
    .Attrs({"max_seq_len: int",
            "verify_window: int",
            "verify_strategy: int",
            "reject_all: bool",
            "accept_all: bool"})
    .SetInplaceMap({{"step_output_ids", "step_output_ids_out"},
                    {"step_output_len", "step_output_len_out"}})
    .SetKernelFn(PD_KERNEL(VerifyDraftTokens));
