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

#include <blocks/moe_fc_block_eb.h>
#include <core/check.h>
#include <core/context.h>
#include <core/param.h>
#include <infer_ops.h>
#include <xft_api.h>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"
#include "utility/env.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

XPU_DECLARE_BOOL(MOE_FFN_USE_DENSE_INPUT, false);
XPU_DECLARE_BOOL(BKCL_DISPATCH_ALL_GATHER, false);

namespace xftblock = baidu::xpu::xftblock;
namespace api = baidu::xpu::api;

template <typename TX1, typename TX2, typename TW, typename TGEMM>
void MoeExpertFFNImpl(xftblock::Tensor* ffn_in,
                      xftblock::Tensor* token_num_info,
                      xftblock::Tensor* ffn1_weight,
                      xftblock::Tensor* ffn2_weight,
                      xftblock::Tensor* ffn1_bias,
                      xftblock::Tensor* ffn2_bias,
                      xftblock::Tensor* ffn2_out,
                      float* ffn2_act_scale,
                      TX2* ffn2_shift,
                      TX2* ffn2_smooth,
                      const int hadamard_blocksize) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
  auto rt_guard = xctx.get_rt_guard();
  auto xftblock_tx2 = xftblock::DataTypeToEnum<TX2>::value;

  int ret = -1;
  int expert_num = ffn1_weight->get_dim(0);
  int inter_dim = ffn1_weight->get_dim(1);
  int outer_dim = inter_dim / 2;

  bool is_padding_input = ffn_in->get_dims().size() == 3;
  auto ffn1_out_shape = ffn_in->get_dims();
  int hidden_dim = ffn1_out_shape[ffn1_out_shape.size() - 1];
  ffn1_out_shape[ffn1_out_shape.size() - 1] = inter_dim;
  xftblock::Tensor ffn1_out(rt_guard, xftblock_tx2, ffn1_out_shape);
  ret = xftblock::xft_moe_fc_block_eb<TX1, TW, TX2, float, int, TGEMM>(
      &xctx,
      ffn_in,
      ffn1_weight,
      &ffn1_out,
      ffn1_bias,
      is_padding_input ? nullptr : token_num_info,
      is_padding_input ? token_num_info : nullptr,
      expert_num,
      1,  // moe_topk
      0,  // group_size
      ffn1_out_shape.size() == 2 ? xftblock::MoeFCInputMode::DENSE
                                 : xftblock::MoeFCInputMode::SPARSE);
  PD_CHECK(ret == 0);

  int token_num = ffn_in->numel() / hidden_dim;
  auto swiglu_out_shape = ffn1_out_shape;
  swiglu_out_shape[swiglu_out_shape.size() - 1] /= 2;
  xftblock::Tensor swiglu_out(rt_guard, xftblock_tx2, swiglu_out_shape);
  ret = api::fast_swiglu<TX2>(xpu_ctx->x_context(),
                              ffn1_out.data<TX2>(),
                              swiglu_out.mutable_data<TX2>(),
                              {token_num, inter_dim},
                              1,
                              true);
  PD_CHECK(ret == 0);
  // TODO(mayang02): use fusion_smooth_transform
  if (ffn2_shift != nullptr) {
    ret = api::broadcast_add<TX2>(xpu_ctx->x_context(),
                                  ffn2_shift,
                                  swiglu_out.data<TX2>(),
                                  swiglu_out.mutable_data<TX2>(),
                                  {1, outer_dim},
                                  {token_num, outer_dim});
    PD_CHECK(ret == 0);
  }
  if (ffn2_smooth != nullptr) {
    ret = api::broadcast_mul<TX2>(xpu_ctx->x_context(),
                                  ffn2_smooth,
                                  swiglu_out.data<TX2>(),
                                  swiglu_out.mutable_data<TX2>(),
                                  {1, outer_dim},
                                  {token_num, outer_dim});
    PD_CHECK(ret == 0);
  }

  if (hadamard_blocksize > 0) {
    ret = infer_ops::fast_walsh_transform<TX2>(xpu_ctx->x_context(),
                                               swiglu_out.data<TX2>(),
                                               nullptr,
                                               nullptr,
                                               swiglu_out.mutable_data<TX2>(),
                                               hadamard_blocksize,
                                               token_num,
                                               outer_dim);
    PD_CHECK(ret == 0);
  }

  xftblock::Tensor ffn2_in(swiglu_out.mutable_data<TX2>(),
                           nullptr,
                           ffn2_act_scale,
                           xftblock_tx2,
                           swiglu_out_shape);
  ret = xftblock::xft_moe_fc_block_eb<TX2, TW, TX2, float, int, TGEMM>(
      &xctx,
      &ffn2_in,
      ffn2_weight,
      ffn2_out,
      nullptr,
      is_padding_input ? nullptr : token_num_info,
      is_padding_input ? token_num_info : nullptr,
      expert_num,
      1,  // moe_topk
      0,  // group_size
      ffn1_out_shape.size() == 2
          ? xftblock::MoeFCInputMode::DENSE
          : xftblock::MoeFCInputMode::SPARSE);  // bias_mode
  PD_CHECK(ret == 0);
}

static void convert_to_lod(xftblock::XFTContext* xctx,
                           xftblock::Tensor* token_num_info) {
  auto rt_guard = xctx->get_rt_guard();
  auto ctx = xctx->get_context();
  const int expert_num = token_num_info->numel();
  xftblock::Tensor tokens_num_lod(
      rt_guard, xftblock::DataType::DT_INT32, {expert_num + 1});
  int ret = api::constant(ctx, tokens_num_lod.data<int>(), expert_num + 1, 0);
  PD_CHECK(ret == 0);
  ret = api::cumsum<int>(ctx,
                         token_num_info->data<int>(),
                         tokens_num_lod.data<int>() + 1,
                         {expert_num},
                         false,
                         false,
                         0);
  PD_CHECK(ret == 0);
  *token_num_info = std::move(tokens_num_lod);
}

template <typename TX1, typename TX2, typename TW>
std::vector<paddle::Tensor> MoeExpertFFNKernel(
    const paddle::Tensor& ffn_in,
    const paddle::Tensor& token_num_info,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn2_bias,
    const paddle::optional<paddle::Tensor>& ffn1_act_scale,
    const paddle::optional<paddle::Tensor>& ffn2_act_scale,
    const paddle::optional<paddle::Tensor>& ffn1_weight_scale,
    const paddle::optional<paddle::Tensor>& ffn2_weight_scale,
    const paddle::optional<paddle::Tensor>& ffn2_shift,
    const paddle::optional<paddle::Tensor>& ffn2_smooth,
    const std::string& quant_method,
    const int hadamard_blocksize,
    const int valid_token_num) {
  using XPU_TX1 = typename XPUTypeTrait<TX1>::Type;
  using XPU_TX2 = typename XPUTypeTrait<TX2>::Type;
  using XPU_TW = typename XPUTypeTrait<TW>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
  auto rt_guard = xctx.get_rt_guard();

  int ret = -1;
  auto input_shape = ffn_in.shape();
  auto ffn1_w_shape = ffn1_weight.shape();
  int expert_num = ffn1_w_shape[0];
  int hidden_dim = input_shape[input_shape.size() - 1];
  int inter_dim = ffn1_w_shape[1];
  int outer_dim = inter_dim / 2;
  bool is_padding_input = input_shape.size() == 3;
  if (is_padding_input) {
    PD_CHECK(input_shape[0] == expert_num);
    PD_CHECK(token_num_info.numel() == expert_num,
             "token_num_info.numel() != expert_num, "
             "token_num_info.numel(): ",
             token_num_info.numel(),
             ", expert_num: ",
             expert_num);
  }

  bool is_w4 = quant_method == "w4a8" || quant_method == "weight_only_int4";
  auto xftblock_tx1 = xftblock::DataTypeToEnum<XPU_TX1>::value;
  auto xftblock_tx2 = xftblock::DataTypeToEnum<XPU_TX2>::value;
  auto xftblock_tw = xftblock::DataTypeToEnum<XPU_TW>::value;
  if (is_w4) {
    xftblock_tw = xftblock::DataTypeToEnum<int4_t>::value;
  }
  float* ffn1_act_scale_data =
      ffn1_act_scale.get_ptr() == nullptr
          ? nullptr
          : const_cast<float*>(ffn1_act_scale.get_ptr()->data<float>());
  float* ffn2_act_scale_data =
      ffn2_act_scale.get_ptr() == nullptr
          ? nullptr
          : const_cast<float*>(ffn2_act_scale.get_ptr()->data<float>());
  float* ffn1_w_scale_data =
      ffn1_weight_scale.get_ptr() == nullptr
          ? nullptr
          : const_cast<float*>(ffn1_weight_scale.get_ptr()->data<float>());
  xftblock::Tensor xffn1_w(const_cast<TW*>(ffn1_weight.data<TW>()),
                           nullptr,
                           ffn1_w_scale_data,
                           xftblock_tw,
                           {expert_num, inter_dim, hidden_dim});
  float* ffn2_w_scale_data =
      ffn2_weight_scale.get_ptr() == nullptr
          ? nullptr
          : const_cast<float*>(ffn2_weight_scale.get_ptr()->data<float>());
  xftblock::Tensor xffn2_w(const_cast<TW*>(ffn2_weight.data<TW>()),
                           nullptr,
                           ffn2_w_scale_data,
                           xftblock_tw,
                           {expert_num, hidden_dim, outer_dim});
  std::shared_ptr<xftblock::Tensor> xffn1_bias;
  if (ffn1_bias.get_ptr()) {
    xffn1_bias = std::make_shared<xftblock::Tensor>(
        const_cast<float*>(ffn1_bias.get_ptr()->data<float>()),
        xftblock::DataType::DT_FLOAT,
        ffn1_bias.get_ptr()->shape());
  }
  std::shared_ptr<xftblock::Tensor> xffn2_bias;
  if (ffn2_bias.get_ptr()) {
    xffn2_bias = std::make_shared<xftblock::Tensor>(
        const_cast<float*>(ffn2_bias.get_ptr()->data<float>()),
        xftblock::DataType::DT_FLOAT,
        ffn2_bias.get_ptr()->shape());
  }
  xftblock::Tensor xtoken_num_info(const_cast<int*>(token_num_info.data<int>()),
                                   xftblock::DataType::DT_INT32,
                                   token_num_info.shape());
  XPU_TX2* shift_data = nullptr;
  XPU_TX2* smooth_data = nullptr;
  if (ffn2_shift.get_ptr()) {
    shift_data = reinterpret_cast<XPU_TX2*>(
        const_cast<TX2*>(ffn2_shift.get_ptr()->data<TX2>()));
  }
  if (ffn2_smooth.get_ptr()) {
    smooth_data = reinterpret_cast<XPU_TX2*>(
        const_cast<TX2*>(ffn2_smooth.get_ptr()->data<TX2>()));
  }
  paddle::Tensor ffn2_out =
      paddle::empty_like(ffn_in, paddle::DataType::BFLOAT16);
  xftblock::Tensor xffn1_in;
  xftblock::Tensor xffn2_out;
  paddle::Tensor ffn1_in_dense;
  paddle::Tensor ffn1_in_scale_per_token;
  if (FLAGS_MOE_FFN_USE_DENSE_INPUT && is_padding_input) {
    convert_to_lod(&xctx, &xtoken_num_info);
    if (quant_method == "w4a8") {
      ffn1_in_scale_per_token = paddle::empty(
          {valid_token_num}, paddle::DataType::FLOAT32, ffn_in.place());
      ffn1_in_dense = paddle::empty({valid_token_num, hidden_dim},
                                    paddle::DataType::INT8,
                                    ffn_in.place());
      xffn1_in = xftblock::Tensor(ffn1_in_dense.data<int8_t>(),
                                  nullptr,
                                  ffn1_in_scale_per_token.data<float>(),
                                  xftblock::DataType::DT_INT8,
                                  {valid_token_num, hidden_dim});
      if (std::is_same<XPU_TX1, int8_t>::value) {
        PD_CHECK(ffn1_act_scale_data != nullptr,
                 "need ffn1_act_scale for x int8 per expert input");
        ret = infer_ops::sequence_unpad<float, int>(
            xpu_ctx->x_context(),
            ffn1_act_scale_data,
            ffn1_in_scale_per_token.data<float>(),
            xtoken_num_info.data<int>(),
            expert_num,
            input_shape[1],
            1,
            true);
        PD_CHECK(ret == 0);
        ret = infer_ops::sequence_unpad<int8_t, int>(
            xpu_ctx->x_context(),
            reinterpret_cast<const int8_t*>(ffn_in.data<int8_t>()),
            reinterpret_cast<int8_t*>(xffn1_in.data<int8_t>()),
            xtoken_num_info.data<int>(),
            expert_num,
            input_shape[1],
            input_shape[2],
            true);
        PD_CHECK(ret == 0);
      } else {
        ret = infer_ops::quant2d_per_expert<XPU_TX1>(
            xpu_ctx->x_context(),
            reinterpret_cast<const XPU_TX1*>(ffn_in.data<TX1>()),
            ffn1_act_scale_data,
            xtoken_num_info.data<int>(),
            reinterpret_cast<int8_t*>(xffn1_in.data<int8_t>()),
            ffn1_in_scale_per_token.data<float>(),
            expert_num,
            valid_token_num,
            hidden_dim,
            true,
            false,
            input_shape[1]);
        PD_CHECK(ret == 0);
      }
    } else {
      ffn1_in_dense = paddle::empty(
          {valid_token_num, hidden_dim}, ffn_in.dtype(), ffn_in.place());
      xffn1_in = xftblock::Tensor(ffn1_in_dense.data<TX1>(),
                                  nullptr,
                                  ffn1_act_scale_data,
                                  xftblock_tx1,
                                  {valid_token_num, hidden_dim});
      ret = infer_ops::sequence_unpad<XPU_TX1, int>(
          xpu_ctx->x_context(),
          reinterpret_cast<const XPU_TX1*>(ffn_in.data<TX1>()),
          reinterpret_cast<XPU_TX1*>(xffn1_in.data<XPU_TX1>()),
          xtoken_num_info.data<int>(),
          expert_num,
          input_shape[1],
          input_shape[2],
          true);
      PD_CHECK(ret == 0);
    }
    xffn2_out =
        xftblock::Tensor(rt_guard, xftblock_tx2, {valid_token_num, hidden_dim});
  } else if (FLAGS_BKCL_DISPATCH_ALL_GATHER && !is_padding_input &&
             quant_method == "w4a8") {
    convert_to_lod(&xctx, &xtoken_num_info);
    ffn1_in_scale_per_token = paddle::empty(
        {valid_token_num}, paddle::DataType::FLOAT32, ffn_in.place());
    ffn1_in_dense = paddle::empty(
        {valid_token_num, hidden_dim}, paddle::DataType::INT8, ffn_in.place());
    xffn1_in = xftblock::Tensor(ffn1_in_dense.data<int8_t>(),
                                nullptr,
                                ffn1_in_scale_per_token.data<float>(),
                                xftblock::DataType::DT_INT8,
                                {valid_token_num, hidden_dim});
    ret = infer_ops::quant2d_per_expert<XPU_TX1>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPU_TX1*>(ffn_in.data<TX1>()),
        ffn1_act_scale_data,
        xtoken_num_info.data<int>(),
        reinterpret_cast<int8_t*>(xffn1_in.data<int8_t>()),
        ffn1_in_scale_per_token.data<float>(),
        expert_num,
        valid_token_num,
        hidden_dim);
    PD_CHECK(ret == 0);
    xffn2_out =
        xftblock::Tensor(ffn2_out.data<TX2>(), xftblock_tx2, input_shape);
  } else {
    xffn1_in = xftblock::Tensor(const_cast<TX1*>(ffn_in.data<TX1>()),
                                nullptr,
                                ffn1_act_scale_data,
                                xftblock_tx1,
                                input_shape);
    xffn2_out = xftblock::Tensor(
        ffn2_out.mutable_data<TX2>(), xftblock_tx2, input_shape);
  }

#define FFN_IMPL(TX1, TX2, TW, TGEMM)                        \
  MoeExpertFFNImpl<TX1, TX2, TW, TGEMM>(&xffn1_in,           \
                                        &xtoken_num_info,    \
                                        &xffn1_w,            \
                                        &xffn2_w,            \
                                        xffn1_bias.get(),    \
                                        xffn2_bias.get(),    \
                                        &xffn2_out,          \
                                        ffn2_act_scale_data, \
                                        shift_data,          \
                                        smooth_data,         \
                                        hadamard_blocksize)
  if (quant_method == "weight_only_int8") {
    FFN_IMPL(XPU_TX1, XPU_TX2, int8_t, float);
  } else if (quant_method == "weight_only_int4") {
    FFN_IMPL(XPU_TX1, XPU_TX2, int4_t, int4_wo_int15);
  } else if (quant_method == "w4a8") {
    if (FLAGS_MOE_FFN_USE_DENSE_INPUT && is_padding_input) {
      FFN_IMPL(int8_t, XPU_TX2, int4_t, int4_wo_int8);
    } else if (FLAGS_BKCL_DISPATCH_ALL_GATHER && !is_padding_input) {
      FFN_IMPL(int8_t, XPU_TX2, int4_t, int4_wo_int8);
    } else {
      FFN_IMPL(XPU_TX1, XPU_TX2, int4_t, int4_wo_int8);
    }
  } else {
    FFN_IMPL(XPU_TX1, XPU_TX2, XPU_TW, float);
  }
#undef FFN_IMPL
  if (FLAGS_MOE_FFN_USE_DENSE_INPUT && is_padding_input) {
    ret = infer_ops::sequence_pad<XPU_TX2, int>(
        xpu_ctx->x_context(),
        const_cast<XPU_TX2*>(xffn2_out.data<XPU_TX2>()),
        reinterpret_cast<XPU_TX2*>(ffn2_out.data<TX2>()),
        xtoken_num_info.data<int>(),
        input_shape[0],
        input_shape[1],
        input_shape[2],
        false,
        0);
    PD_CHECK(ret == 0);
  }

  return {ffn2_out};
}

std::vector<paddle::Tensor> MoeExpertFFN(
    const paddle::Tensor& ffn_in,
    const paddle::Tensor& token_num_info,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn2_bias,
    const paddle::optional<paddle::Tensor>& ffn1_act_scale,
    const paddle::optional<paddle::Tensor>& ffn2_act_scale,
    const paddle::optional<paddle::Tensor>& ffn1_weight_scale,
    const paddle::optional<paddle::Tensor>& ffn2_weight_scale,
    const paddle::optional<paddle::Tensor>& ffn2_shift,
    const paddle::optional<paddle::Tensor>& ffn2_smooth,
    const std::string& quant_method,
    const int hadamard_blocksize,
    const int valid_token_num) {
  if (ffn_in.numel() == 0 || valid_token_num == 0) {
    paddle::Tensor ffn2_out =
        paddle::empty_like(ffn_in, paddle::DataType::BFLOAT16);
    return {ffn2_out};
  }

  const auto x_type = ffn_in.dtype();
  const auto w_type = ffn1_weight.dtype();

#define APPLY_FFN_KERNEL(TX1, TX2, TW)                        \
  return MoeExpertFFNKernel<TX1, TX2, TW>(ffn_in,             \
                                          token_num_info,     \
                                          ffn1_weight,        \
                                          ffn2_weight,        \
                                          ffn1_bias,          \
                                          ffn2_bias,          \
                                          ffn1_act_scale,     \
                                          ffn2_act_scale,     \
                                          ffn1_weight_scale,  \
                                          ffn2_weight_scale,  \
                                          ffn2_shift,         \
                                          ffn2_smooth,        \
                                          quant_method,       \
                                          hadamard_blocksize, \
                                          valid_token_num);
  if (x_type == paddle::DataType::BFLOAT16 &&
      w_type == paddle::DataType::BFLOAT16) {
    APPLY_FFN_KERNEL(paddle::bfloat16, paddle::bfloat16, paddle::bfloat16);
  } else if (x_type == paddle::DataType::BFLOAT16 &&
             w_type == paddle::DataType::INT8) {
    APPLY_FFN_KERNEL(paddle::bfloat16, paddle::bfloat16, int8_t);
  } else if (x_type == paddle::DataType::INT8 &&
             w_type == paddle::DataType::INT8) {
    APPLY_FFN_KERNEL(int8_t, paddle::bfloat16, int8_t);
  } else {
    PD_THROW("MoeExpertFFN not support x_type=",
             static_cast<int>(x_type),
             ", w_type=",
             static_cast<int>(w_type));
    return {};
  }
#undef APPLY_FFN_KERNEL
}

std::vector<std::vector<int64_t>> MoeExpertFFNInferShape(
    const std::vector<int64_t>& permute_input_shape,
    const std::vector<int64_t>& token_num_info_shape,
    const std::vector<int64_t>& ffn1_weight_shape,
    const std::vector<int64_t>& ffn2_weight_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_bias_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_act_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_act_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn1_weight_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_weight_scale_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_shift_shape,
    const paddle::optional<std::vector<int64_t>>& ffn2_smooth_shape) {
  return {permute_input_shape};
}

std::vector<paddle::DataType> MoeExpertFFNInferDtype(
    const paddle::DataType& permute_input_dtype,
    const paddle::DataType& token_num_info_dtype,
    const paddle::DataType& ffn1_weight_dtype,
    const paddle::DataType& ffn2_weight_dtype,
    const paddle::optional<paddle::DataType>& ffn1_bias_dtype,
    const paddle::optional<paddle::DataType>& ffn2_bias_dtype,
    const paddle::optional<paddle::DataType>& ffn1_act_scale_dtype,
    const paddle::optional<paddle::DataType>& ffn2_act_scale_dtype,
    const paddle::optional<paddle::DataType>& ffn1_weight_scale_dtype,
    const paddle::optional<paddle::DataType>& ffn2_weight_scale_dtype,
    const paddle::optional<paddle::DataType>& ffn2_shift_dtype,
    const paddle::optional<paddle::DataType>& ffn2_smooth_dtype) {
  if (permute_input_dtype == paddle::DataType::INT8) {
    return {paddle::DataType::BFLOAT16};
  } else {
    return {permute_input_dtype};
  }
}

PD_BUILD_STATIC_OP(moe_expert_ffn)
    .Inputs({"ffn_in",
             "token_num_info",
             "ffn1_weight",
             "ffn2_weight",
             paddle::Optional("ffn1_bias"),
             paddle::Optional("ffn2_bias"),
             paddle::Optional("ffn1_act_scale"),
             paddle::Optional("ffn2_act_scale"),
             paddle::Optional("ffn1_weight_scale"),
             paddle::Optional("ffn2_weight_scale"),
             paddle::Optional("ffn2_shift"),
             paddle::Optional("ffn2_smooth")})
    .Outputs({"ffn_out"})
    .Attrs({"quant_method:std::string",
            "hadamard_blocksize:int",
            "valid_token_num:int"})
    .SetKernelFn(PD_KERNEL(MoeExpertFFN))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeExpertFFNInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeExpertFFNInferDtype));
