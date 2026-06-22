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

/**
 * @brief Stop-sequence detection for speculative decoding.
 *
 * for bid in [0, bs):
 *   if step_idx[bid] + accept_num[bid] < min_tokens[bid]: skip
 *   if stop_flags[bid]: skip
 *   tokens = token_ids_all[bid][prompt_len:] ++ accept_tokens[bid]
 *   for each stop_seq in stop_seqs[bid]:
 *     for accept_idx in [-1, accept_num-2]:  // -1 = delayed match from prev
 * round if tokens[..accept_idx] ends with stop_seq: accept_nums[bid] =
 * accept_idx + 1 accept_tokens[bid][accept_idx] = end_id break
 *   // stop_flags is NOT set here; handled by downstream operators.
 */
void SpecGetStopFlagsMultiSeqs(const paddle::Tensor &accept_tokens,
                               const paddle::Tensor &accept_num,
                               const paddle::Tensor &token_ids_all,
                               const paddle::Tensor &prompt_lens,
                               const paddle::Tensor &step_idx,
                               const paddle::Tensor &stop_flags,
                               const paddle::Tensor &seq_lens,
                               const paddle::Tensor &stop_seqs,
                               const paddle::Tensor &stop_seqs_len,
                               const paddle::Tensor &end_ids,
                               const paddle::Tensor &min_tokens) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  api::Context *ctx =
      static_cast<const phi::XPUContext *>(dev_ctx)->x_context();
  if (accept_tokens.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }
  PD_CHECK(accept_tokens.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);

  std::vector<int64_t> shape = accept_tokens.shape();
  std::vector<int64_t> stop_seqs_shape = stop_seqs.shape();
  int bs_now = shape[0];
  // Align with GPU: stop_seqs shape is [bs, stop_seqs_bs, stop_seqs_max_len]
  int stop_seqs_bs = stop_seqs_shape[1];
  int stop_seqs_max_len = stop_seqs_shape[2];
  int max_model_len = token_ids_all.shape()[1];
  int accept_tokens_len = accept_tokens.shape()[1];

  int r = fastdeploy::plugin::speculate_set_stop_value_multi_seqs(
      ctx,
      const_cast<bool *>(stop_flags.data<bool>()),
      const_cast<int64_t *>(accept_tokens.data<int64_t>()),
      const_cast<int *>(accept_num.data<int>()),
      token_ids_all.data<int64_t>(),
      prompt_lens.data<int64_t>(),
      step_idx.data<int64_t>(),
      stop_seqs.data<int64_t>(),
      stop_seqs_len.data<int>(),
      seq_lens.data<int>(),
      end_ids.data<int64_t>(),
      min_tokens.data<int64_t>(),
      bs_now,
      accept_tokens_len,
      stop_seqs_bs,
      stop_seqs_max_len,
      max_model_len);
  PD_CHECK(r == 0,
           "fastdeploy::plugin::speculate_set_stop_value_multi_seqs failed.");
}

PD_BUILD_STATIC_OP(speculate_set_stop_value_multi_seqs)
    .Inputs({"accept_tokens",
             "accept_num",
             "token_ids_all",
             "prompt_lens",
             "step_idx",
             "stop_flags",
             "seq_lens",
             "stop_seqs",
             "stop_seqs_len",
             "end_ids",
             "min_tokens"})
    .Outputs({"accept_tokens_out", "stop_flags_out"})
    .SetInplaceMap({{"accept_tokens", "accept_tokens_out"},
                    {"stop_flags", "stop_flags_out"}})
    .SetKernelFn(PD_KERNEL(SpecGetStopFlagsMultiSeqs));
