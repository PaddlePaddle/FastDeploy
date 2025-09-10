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

void SpeculateTokenPenaltyMultiScores(
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& logits,
    const paddle::Tensor& penalty_scores,
    const paddle::Tensor& frequency_scores,
    const paddle::Tensor& presence_scores,
    const paddle::Tensor& temperatures,
    const paddle::Tensor& bad_tokens,
    const paddle::Tensor& cur_len,
    const paddle::Tensor& min_len,
    const paddle::Tensor& eos_token_id,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& output_padding_offset,
    const paddle::Tensor& output_cum_offsets,
    const int max_seq_len) {
  namespace api = baidu::xpu::api;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  if (pre_ids.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  int64_t bs = seq_lens_this_time.shape()[0];
  int64_t token_num = logits.shape()[0];
  PADDLE_ENFORCE_LE(bs,
                    640,
                    phi::errors::InvalidArgument(
                        "Only support bs <= 640, but received bsz is %d", bs));
  int64_t length = logits.shape()[1];
  int64_t length_id = pre_ids.shape()[1];
  int64_t length_bad_words = bad_tokens.shape()[0];
  int64_t end_length = eos_token_id.shape()[0];
  switch (logits.type()) {
    case paddle::DataType::BFLOAT16: {
      using XPUType = typename XPUTypeTrait<paddle::bfloat16>::Type;
      typedef paddle::bfloat16 data_t;
      int r = baidu::xpu::api::plugin::speculate_token_penalty_multi_scores(
          ctx,
          pre_ids.data<int64_t>(),
          reinterpret_cast<XPUType*>(
              const_cast<data_t*>(logits.data<data_t>())),
          reinterpret_cast<const XPUType*>(penalty_scores.data<data_t>()),
          reinterpret_cast<const XPUType*>(frequency_scores.data<data_t>()),
          reinterpret_cast<const XPUType*>(presence_scores.data<data_t>()),
          temperatures.data<float>(),
          cur_len.data<int64_t>(),
          min_len.data<int64_t>(),
          eos_token_id.data<int64_t>(),
          bad_tokens.data<int64_t>(),
          output_padding_offset.data<int>(),
          output_cum_offsets.data<int>(),
          bs,
          length,
          length_id,
          end_length,
          length_bad_words,
          token_num,
          max_seq_len);
      PD_CHECK(r == 0, "xpu::plugin::token_penalty_multi_scores failed.");
    } break;
    case paddle::DataType::FLOAT16: {
      using XPUType = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 data_t;
      int r = baidu::xpu::api::plugin::speculate_token_penalty_multi_scores(
          ctx,
          pre_ids.data<int64_t>(),
          reinterpret_cast<XPUType*>(
              const_cast<data_t*>(logits.data<data_t>())),
          reinterpret_cast<const XPUType*>(penalty_scores.data<data_t>()),
          reinterpret_cast<const XPUType*>(frequency_scores.data<data_t>()),
          reinterpret_cast<const XPUType*>(presence_scores.data<data_t>()),
          temperatures.data<float>(),
          cur_len.data<int64_t>(),
          min_len.data<int64_t>(),
          eos_token_id.data<int64_t>(),
          bad_tokens.data<int64_t>(),
          output_padding_offset.data<int>(),
          output_cum_offsets.data<int>(),
          bs,
          length,
          length_id,
          end_length,
          length_bad_words,
          token_num,
          max_seq_len);
      PD_CHECK(r == 0, "xpu::plugin::token_penalty_multi_scores failed.");
    } break;
    case paddle::DataType::FLOAT32: {
      int r = baidu::xpu::api::plugin::speculate_token_penalty_multi_scores(
          ctx,
          pre_ids.data<int64_t>(),
          const_cast<float*>(logits.data<float>()),
          penalty_scores.data<float>(),
          frequency_scores.data<float>(),
          presence_scores.data<float>(),
          temperatures.data<float>(),
          cur_len.data<int64_t>(),
          min_len.data<int64_t>(),
          eos_token_id.data<int64_t>(),
          bad_tokens.data<int64_t>(),
          output_padding_offset.data<int>(),
          output_cum_offsets.data<int>(),
          bs,
          length,
          length_id,
          end_length,
          length_bad_words,
          token_num,
          max_seq_len);
      PD_CHECK(r == 0, "xpu::plugin::token_penalty_multi_scores failed.");
    } break;
    default:
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and float32 are supported. ");
      break;
  }
}

PD_BUILD_STATIC_OP(speculate_get_token_penalty_multi_scores)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "temperatures",
             "bad_tokens",
             "cur_len",
             "min_len",
             "eos_token_id",
             "seq_lens_this_time",
             "output_padding_offset",
             "output_cum_offsets"})
    .Outputs({"logits_out"})
    .Attrs({"max_seq_len: int"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateTokenPenaltyMultiScores));
