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
#include "xpu/plugin.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

#define FIXED_TOPK_BASE(topk, ...) \
  case (topk): {                   \
    constexpr auto kTopK = topk;   \
    __VA_ARGS__;                   \
  } break

#define FIXED_TOPK(...)                      \
  FIXED_TOPK_BASE(2, ##__VA_ARGS__);         \
  FIXED_TOPK_BASE(3, ##__VA_ARGS__);         \
  FIXED_TOPK_BASE(4, ##__VA_ARGS__);         \
  FIXED_TOPK_BASE(5, ##__VA_ARGS__);         \
  FIXED_TOPK_BASE(8, ##__VA_ARGS__);         \
  FIXED_TOPK_BASE(10, ##__VA_ARGS__);        \
  default: {                                 \
    PD_THROW("Unsupported candidates_len."); \
  }

namespace api = baidu::xpu::api;
std::vector<paddle::Tensor> TopPCandidates(
    const paddle::Tensor& probs,
    const paddle::Tensor& top_p,
    const paddle::Tensor& output_padding_offset,
    int candidates_len,
    int max_seq_len) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  api::Context* ctx = xpu_ctx->x_context();
  if (probs.is_cpu()) {
    ctx = new api::Context(api::kCPU);
  }

  std::vector<int64_t> input_shape = probs.shape();
  const int token_num = input_shape[0];
  const int vocab_size = input_shape[1];

  auto verify_scores =
      paddle::empty({token_num, candidates_len}, probs.dtype(), probs.place());
  auto verify_tokens = paddle::empty(
      {token_num, candidates_len}, paddle::DataType::INT64, probs.place());
  auto actual_candidate_lens =
      paddle::empty({token_num}, paddle::DataType::INT32, probs.place());

  constexpr int TopKMaxLength = 2;
  int r;
  switch (probs.dtype()) {
    case paddle::DataType::BFLOAT16:
      using XPUTypeBF16 = typename XPUTypeTrait<bfloat16>::Type;
      typedef paddle::bfloat16 bf16_data_t;
      switch (candidates_len) {
        FIXED_TOPK(
            r = api::plugin::top_p_candidates<XPUTypeBF16,
                                              TopKMaxLength,
                                              kTopK>(
                ctx,
                reinterpret_cast<const XPUTypeBF16*>(probs.data<bf16_data_t>()),
                reinterpret_cast<const XPUTypeBF16*>(top_p.data<bf16_data_t>()),
                output_padding_offset.data<int>(),
                verify_tokens.data<int64_t>(),
                reinterpret_cast<XPUTypeBF16*>(
                    verify_scores.data<bf16_data_t>()),
                actual_candidate_lens.data<int>(),
                vocab_size,
                token_num,
                candidates_len,
                max_seq_len);
            PD_CHECK(r == 0, "xpu::plugin::top_p_candidates failed.");
            return {verify_scores, verify_tokens, actual_candidate_lens});
      }
    case paddle::DataType::FLOAT16:
      using XPUTypeFP16 = typename XPUTypeTrait<float16>::Type;
      typedef paddle::float16 fp16_data_t;
      switch (candidates_len) {
        FIXED_TOPK(
            r = api::plugin::top_p_candidates<XPUTypeFP16,
                                              TopKMaxLength,
                                              kTopK>(
                ctx,
                reinterpret_cast<const XPUTypeFP16*>(probs.data<fp16_data_t>()),
                reinterpret_cast<const XPUTypeFP16*>(top_p.data<fp16_data_t>()),
                output_padding_offset.data<int>(),
                verify_tokens.data<int64_t>(),
                reinterpret_cast<XPUTypeFP16*>(
                    verify_scores.data<fp16_data_t>()),
                actual_candidate_lens.data<int>(),
                vocab_size,
                token_num,
                candidates_len,
                max_seq_len);
            PD_CHECK(r == 0, "xpu::plugin::top_p_candidates failed.");
            return {verify_scores, verify_tokens, actual_candidate_lens});
      }
    case paddle::DataType::FLOAT32:
      switch (candidates_len) {
        FIXED_TOPK(
            r = api::plugin::top_p_candidates<float, TopKMaxLength, kTopK>(
                ctx,
                probs.data<float>(),
                top_p.data<float>(),
                output_padding_offset.data<int>(),
                verify_tokens.data<int64_t>(),
                verify_scores.data<float>(),
                actual_candidate_lens.data<int>(),
                vocab_size,
                token_num,
                candidates_len,
                max_seq_len);
            PD_CHECK(r == 0, "xpu::plugin::top_p_candidates failed.");
            return {verify_scores, verify_tokens, actual_candidate_lens});
      }
    default:
      PD_THROW("Unsupported data type.");
  }
}

std::vector<std::vector<int64_t>> TopPCandidatesInferShape(
    const std::vector<int64_t>& probs_shape,
    const std::vector<int64_t>& top_p_shape,
    const std::vector<int64_t>& output_padding_offset_shape,
    int max_candidates_len) {
  int token_num = probs_shape[0];
  return {{token_num, max_candidates_len},
          {token_num, max_candidates_len},
          {token_num}};
}

std::vector<paddle::DataType> TopPCandidatesInferDtype(
    const paddle::DataType& probs_dtype,
    const paddle::DataType& top_p_dtype,
    const paddle::DataType& output_padding_offset_dtype) {
  return {probs_dtype, paddle::DataType::INT64, paddle::DataType::INT32};
}

PD_BUILD_STATIC_OP(top_p_candidates)
    .Inputs({"probs", "top_p", "output_padding_offset"})
    .Outputs({"verify_scores", "verify_tokens", "actual_candidate_lens"})
    .Attrs({"candidates_len: int", "max_seq_len: int"})
    .SetKernelFn(PD_KERNEL(TopPCandidates))
    .SetInferShapeFn(PD_INFER_SHAPE(TopPCandidatesInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TopPCandidatesInferDtype));
