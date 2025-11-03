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

#include <blocks/xft_blocks.h>
#include <infer_ops.h>
#include <functional>
#include "paddle/extension.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "utility/debug.h"
#include "utility/env.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

XPU_DECLARE_BOOL(ENABLE_XVLLM_SDNN_INFER, false);
namespace xftblock = baidu::xpu::xftblock;
namespace api = baidu::xpu::api;

template <typename TX, typename TW>
std::vector<paddle::Tensor> WeightOnlyLinearKernel(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& weight_scale,
    const paddle::optional<paddle::Tensor>& bias,
    const std::string& weight_dtype) {
  using XPU_TX = typename XPUTypeTrait<TX>::Type;
  using XPU_TW = typename XPUTypeTrait<TW>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
  auto rt_guard = xctx.get_rt_guard();
  auto xftblock_tx = xftblock::DataTypeToEnum<XPU_TX>::value;
  auto xftblock_tw = xftblock::DataTypeToEnum<XPU_TW>::value;

  int ret = -1;
  auto x_shape = x.shape();
  auto w_shape = weight.shape();
  int64_t n = w_shape[0];
  int64_t k = w_shape[1];
  int64_t m = x.numel() / k;
  if (weight_dtype == "int4_t") {
    n = n * 2;
  }
  paddle::Tensor out = paddle::empty({m, n}, x.dtype(), x.place());
  if (m == 0) {
    return {out};
  }

  paddle::Tensor bias_fp32;
  if (bias.get_ptr() && bias.get_ptr()->dtype() != paddle::DataType::FLOAT32) {
    bias_fp32 = paddle::empty({n}, paddle::DataType::FLOAT32, x.place());
    PD_CHECK(bias.get_ptr()->dtype() == x.dtype(), "bias.dtype != x.dtype");
    ret = api::cast<XPU_TX, float>(
        xpu_ctx->x_context(),
        reinterpret_cast<const XPU_TX*>(bias.get_ptr()->data<TX>()),
        bias_fp32.data<float>(),
        n);
    PD_CHECK(ret == 0, "cast");
  }

  xftblock::Tensor input_x(const_cast<TX*>(x.data<TX>()), xftblock_tx, {m, k});
  xftblock::Tensor input_w(const_cast<TW*>(weight.data<TW>()),
                           nullptr,
                           const_cast<float*>(weight_scale.data<float>()),
                           xftblock_tw,
                           {n, k});
  xftblock::Tensor output(const_cast<TX*>(out.data<TX>()), xftblock_tx, {m, n});
  std::shared_ptr<xftblock::Tensor> input_bias;
  if (bias.get_ptr()) {
    if (bias.get_ptr()->dtype() != paddle::DataType::FLOAT32) {
      input_bias = std::make_shared<xftblock::Tensor>(
          const_cast<float*>(bias_fp32.data<float>()),
          xftblock::DataType::DT_FLOAT,
          std::vector<int64_t>({n}));
    } else {
      input_bias = std::make_shared<xftblock::Tensor>(
          const_cast<float*>(bias.get_ptr()->data<float>()),
          xftblock::DataType::DT_FLOAT,
          std::vector<int64_t>({n}));
    }
  }
  bool use_sdnn = FLAGS_ENABLE_XVLLM_SDNN_INFER;
  if (x.dtype() == paddle::DataType::BFLOAT16) {
    ret = xftblock::
        xft_fc_block_cast_te_per_token<bfloat16, int8_t, bfloat16, float16>(
            &xctx,
            &input_x,
            &input_w,
            &output,
            input_bias.get(),
            api::Activation_t::LINEAR,
            false,
            true,
            1.0f,
            0.0f,
            0,
            1,
            false,
            false,
            use_sdnn);
    PD_CHECK(ret == 0, "xft_fc_block_cast_te_per_token");
  } else {
    ret = xftblock::xft_fc_block<XPU_TX, XPU_TW, XPU_TX, XPU_TX>(
        &xctx,
        &input_x,
        &input_w,
        &output,
        input_bias.get(),
        api::Activation_t::LINEAR,
        false,
        true,
        1.0f,
        0.0f,
        0,
        1,
        false,
        false);
    PD_CHECK(ret == 0, "xft_fc_block");
  }

  return {out};
}

std::vector<paddle::Tensor> WeightOnlyLinear(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& weight_scale,
    const paddle::optional<paddle::Tensor>& bias,
    const std::string& weight_dtype,
    const int arch,
    const int group_size) {
  const auto x_type = x.dtype();
  const auto w_type = weight.dtype();

#define APPLY_FFN_KERNEL(TX, TW)         \
  return WeightOnlyLinearKernel<TX, TW>( \
      x, weight, weight_scale, bias, weight_dtype);

  if (x_type == paddle::DataType::BFLOAT16 &&
      w_type == paddle::DataType::INT8) {
    APPLY_FFN_KERNEL(paddle::bfloat16, int8_t);
  } else if (x_type == paddle::DataType::FLOAT16 &&
             w_type == paddle::DataType::INT8) {
    APPLY_FFN_KERNEL(paddle::float16, int8_t);
  } else {
    PD_THROW("WeightOnlyLinear not support x_type=",
             static_cast<int>(x_type),
             ", w_type=",
             static_cast<int>(w_type));
    return {};
  }
#undef APPLY_FFN_KERNEL
}

std::vector<std::vector<int64_t>> WeightOnlyLinearInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& weight_shape,
    const std::vector<int64_t>& weight_scale_shape,
    const paddle::optional<std::vector<int64_t>>& bias_shape,
    const std::string& weight_dtype,
    const int arch,
    const int group_size) {
  PD_CHECK(weight_shape.size() == 2);
  int64_t n = weight_shape[0];
  int64_t k = weight_shape[1];
  int64_t x_numel = std::accumulate(x_shape.begin(),
                                    x_shape.end(),
                                    static_cast<int64_t>(1),
                                    std::multiplies<int64_t>());
  int64_t m = x_numel / k;
  if (weight_dtype == "int4") {
    n = n * 2;
  }
  return {{m, n}};
}

std::vector<paddle::DataType> WeightOnlyLinearInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& w_dtype,
    const paddle::DataType& weight_scale_dtype,
    const paddle::optional<paddle::DataType>& bias_dtype,
    const std::string& weight_dtype,
    const int arch,
    const int group_size) {
  return {x_dtype};
}

PD_BUILD_STATIC_OP(weight_only_linear_xpu)
    .Inputs({"x", "weight", "weight_scale", paddle::Optional("bias")})
    .Outputs({"out"})
    .Attrs({"weight_dtype:std::string", "arch:int", "group_size:int"})
    .SetKernelFn(PD_KERNEL(WeightOnlyLinear))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightOnlyLinearInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightOnlyLinearInferDtype));
