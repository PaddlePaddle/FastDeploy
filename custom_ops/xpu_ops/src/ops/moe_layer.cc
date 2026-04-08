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
#include <blocks/xft_blocks.h>
#include <core/check.h>
#include <core/context.h>
#include <core/param.h>
#include <infer_ops.h>
#include <paddle/extension.h>
#include <paddle/phi/backends/xpu/enforce_xpu.h>
#include <xft_api.h>

#include <paddle/phi/backends/xpu/xpu_context.h>
#include <string>
#include <vector>

namespace xftblock = baidu::xpu::xftblock;
namespace api = baidu::xpu::api;

template <typename TX, typename TW>
struct fused_moe_ffn_trait {
  using GEMM_TYPE = TW;
};
template <>
struct fused_moe_ffn_trait<bfloat16, bfloat16> {
  using GEMM_TYPE = float;
};
template <>
struct fused_moe_ffn_trait<bfloat16, int8_t> {
  using GEMM_TYPE = float;
};
template <>
struct fused_moe_ffn_trait<bfloat16, int4_t> {
  using GEMM_TYPE = int4_wo_int15;
};

template <typename TX, typename TW>
std::vector<paddle::Tensor> MoeLayerKernel(
    const paddle::Tensor &x,
    const paddle::Tensor &gate_weight,
    const paddle::optional<paddle::Tensor> &gate_correction_bias,
    const paddle::Tensor &up_gate_proj_weight,
    const paddle::Tensor &down_proj_weight,
    const paddle::optional<paddle::Tensor> &up_gate_proj_bias,
    const paddle::optional<paddle::Tensor> &down_proj_bias,
    const paddle::optional<paddle::Tensor> &up_gate_proj_weight_scale,
    const paddle::optional<paddle::Tensor> &down_proj_weight_scale,
    const paddle::optional<paddle::Tensor> &down_proj_in_scale,  // not support
    const std::string &quant_method,
    const int moe_top_k,
    const bool moe_group) {
  // std::cout << "[Op Debug] enter moe layer" << std::endl;
  using XPU_TX = typename XPUTypeTrait<TX>::Type;
  using XPU_TW = typename XPUTypeTrait<TW>::Type;
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext *>(dev_ctx);
  xftblock::XFTContext xctx(xpu_ctx->x_context(), nullptr);
  auto rt_guard = xctx.get_rt_guard();

  const auto xtype = x.dtype();
  auto x_dims = x.shape();
  auto up_gate_proj_dims = up_gate_proj_weight.shape();
  PD_CHECK(x_dims.size() == 2, "x_dims.size() should be 2.");
  PD_CHECK(up_gate_proj_dims.size() == 3,
           "up_gate_proj_dims.size() should be 3.");
  PD_CHECK(down_proj_in_scale.get_ptr() == nullptr,
           "down_proj_in_scale not support.");
  if (quant_method == "weight_only_int4") {
    PD_CHECK(x_dims[1] == up_gate_proj_dims[2] * 2,
             "x_dims[1] should equal to up_gate_proj_dims[2], (weight must be "
             "[e,n,k]).");
  } else {
    PD_CHECK(x_dims[1] == up_gate_proj_dims[2],
             "x_dims[1] should equal to up_gate_proj_dims[2], (weight must be "
             "[e,n,k]).");
  }

  int token_num = x_dims[0];
  int hidden_dim = x_dims[1];
  int expert_num = up_gate_proj_dims[0];
  int inter_dim = up_gate_proj_dims[1];
  int outer_dim = inter_dim / 2;

  paddle::Tensor fused_moe_out = paddle::empty_like(x);

  auto x_mpart_shape = x_dims;
  int MPART_SIZE = 2048;
  if (const char *env_val = std::getenv("XPU_MPART_SIZE")) {
    MPART_SIZE = std::atoi(env_val);
  }
  int bsz = x_dims[0];
  for (int m_part_start = 0; m_part_start < bsz; m_part_start += MPART_SIZE) {
    auto m_part_end = std::min(m_part_start + MPART_SIZE, bsz);
    auto x_offset = m_part_start * hidden_dim;
    x_mpart_shape[0] = m_part_end - m_part_start;
    int ret = -1;
    auto xftblock_tx = xftblock::DataTypeToEnum<XPU_TX>::value;
    auto xftblock_tw = xftblock::DataTypeToEnum<XPU_TW>::value;
    // input + output
    xftblock::Tensor xin(
        const_cast<TX *>(x.data<TX>() + x_offset), xftblock_tx, x_mpart_shape);

    xftblock::Tensor xout(fused_moe_out.mutable_data<TX>() + x_offset,
                          xftblock_tx,
                          x_mpart_shape);
    // gate
    xftblock::Tensor xgate_w(const_cast<float *>(gate_weight.data<float>()),
                             xftblock::DataType::DT_FLOAT,
                             gate_weight.shape());
    std::shared_ptr<xftblock::Tensor> xgate_correct_bias;
    if (gate_correction_bias.get_ptr()) {
      xgate_correct_bias = std::make_shared<xftblock::Tensor>(
          const_cast<float *>(gate_correction_bias.get_ptr()->data<float>()),
          xftblock::DataType::DT_FLOAT,
          gate_correction_bias.get_ptr()->shape());
    }

    // up_gate_proj + down_proj
    std::shared_ptr<xftblock::Tensor> xup_gate_proj_w, xdown_proj_w;

    if (std::is_same<TW, int4_t>::value) {
      xup_gate_proj_w = std::make_shared<xftblock::Tensor>(
          const_cast<int8_t *>(up_gate_proj_weight.data<int8_t>()),
          nullptr,
          const_cast<float *>(
              up_gate_proj_weight_scale.get_ptr()
                  ? up_gate_proj_weight_scale.get_ptr()->data<float>()
                  : nullptr),
          xftblock_tw,
          std::vector<int64_t>{expert_num, inter_dim, hidden_dim});

      xdown_proj_w = std::make_shared<xftblock::Tensor>(
          const_cast<int8_t *>(down_proj_weight.data<int8_t>()),
          nullptr,
          const_cast<float *>(
              down_proj_weight_scale.get_ptr()
                  ? down_proj_weight_scale.get_ptr()->data<float>()
                  : nullptr),
          xftblock_tw,
          std::vector<int64_t>{expert_num, hidden_dim, outer_dim});

    } else {
      xup_gate_proj_w = std::make_shared<xftblock::Tensor>(
          const_cast<TW *>(up_gate_proj_weight.data<TW>()),
          nullptr,
          const_cast<float *>(
              up_gate_proj_weight_scale.get_ptr()
                  ? up_gate_proj_weight_scale.get_ptr()->data<float>()
                  : nullptr),
          xftblock_tw,
          std::vector<int64_t>{expert_num, inter_dim, hidden_dim});

      xdown_proj_w = std::make_shared<xftblock::Tensor>(
          const_cast<TW *>(down_proj_weight.data<TW>()),
          nullptr,
          const_cast<float *>(
              down_proj_weight_scale.get_ptr()
                  ? down_proj_weight_scale.get_ptr()->data<float>()
                  : nullptr),
          xftblock_tw,
          std::vector<int64_t>{expert_num, hidden_dim, outer_dim});
    }
    std::shared_ptr<xftblock::Tensor> xup_gate_proj_bias;
    std::shared_ptr<xftblock::Tensor> xdown_proj_bias;
    if (up_gate_proj_bias.get_ptr()) {
      xup_gate_proj_bias = std::make_shared<xftblock::Tensor>(
          const_cast<float *>(up_gate_proj_bias.get_ptr()->data<float>()),
          xftblock::DataType::DT_FLOAT,
          up_gate_proj_bias.get_ptr()->shape());
    }
    if (down_proj_bias.get_ptr()) {
      xdown_proj_bias = std::make_shared<xftblock::Tensor>(
          const_cast<float *>(down_proj_bias.get_ptr()->data<float>()),
          xftblock::DataType::DT_FLOAT,
          down_proj_bias.get_ptr()->shape());
    }
    // std::cout << "[Op Debug] start init moe_ffn weight and bias" <<
    // std::endl; MoeFFNWeight
    xftblock::MoeFFNWeight moe_ffn_w_struct;
    moe_ffn_w_struct.gate_weight = &xgate_w;
    moe_ffn_w_struct.ffn_inter_weights = xup_gate_proj_w.get();
    moe_ffn_w_struct.ffn_inter_bias = xup_gate_proj_bias.get();
    moe_ffn_w_struct.ffn_outer_weights = xdown_proj_w.get();
    moe_ffn_w_struct.ffn_outer_bias = xdown_proj_bias.get();
    moe_ffn_w_struct.score_bias = xgate_correct_bias.get();
    // MoeFFNParam
    xftblock::MoeFFNParam moe_ffn_param;
    moe_ffn_param.expert_num = expert_num;
    moe_ffn_param.moe_top_k = moe_top_k;
    moe_ffn_param.fast_swiglu = true;

    // std::cout << "[Op Debug] pre in xvfblock moe_ffn" << std::endl;

    using XPU_TGEMM = typename fused_moe_ffn_trait<XPU_TX, XPU_TW>::GEMM_TYPE;
    ret =
        baidu::xpu::xftblock::moe_ffn_block_sorted_castte_per_token<XPU_TX,
                                                                    XPU_TW,
                                                                    XPU_TX,
                                                                    XPU_TGEMM>(
            &xctx, &xin, &xout, moe_ffn_w_struct, moe_ffn_param);
    PD_CHECK(ret == 0,
             "xftblock::moe_ffn_block_sorted_castte_per_token failed");
  }

  return {fused_moe_out};
}

std::vector<paddle::Tensor> MoeLayer(
    const paddle::Tensor &x,
    const paddle::Tensor &gate_weight,
    const paddle::optional<paddle::Tensor> &gate_correction_bias,
    const paddle::Tensor &up_gate_proj_weight,
    const paddle::Tensor &down_proj_weight,
    const paddle::optional<paddle::Tensor> &up_gate_proj_bias,
    const paddle::optional<paddle::Tensor> &down_proj_bias,
    const paddle::optional<paddle::Tensor> &up_gate_proj_weight_scale,
    const paddle::optional<paddle::Tensor> &down_proj_weight_scale,
    const paddle::optional<paddle::Tensor> &down_proj_in_scale,
    const std::string &quant_method,
    const int moe_top_k,
    const bool moe_group) {
  const auto x_type = x.dtype();
  const auto w_type = up_gate_proj_weight.dtype();

#define APPLY_MOE_LAYER_KERNEL(TX, TW)                     \
  return MoeLayerKernel<TX, TW>(x,                         \
                                gate_weight,               \
                                gate_correction_bias,      \
                                up_gate_proj_weight,       \
                                down_proj_weight,          \
                                up_gate_proj_bias,         \
                                down_proj_bias,            \
                                up_gate_proj_weight_scale, \
                                down_proj_weight_scale,    \
                                down_proj_in_scale,        \
                                quant_method,              \
                                moe_top_k,                 \
                                moe_group);

  // TODO(mayang02): how to use quant_method?
  if (x_type == paddle::DataType::BFLOAT16 &&
      w_type == paddle::DataType::BFLOAT16) {
    APPLY_MOE_LAYER_KERNEL(paddle::bfloat16, paddle::bfloat16);
  } else if (x_type == paddle::DataType::BFLOAT16 &&
             quant_method == "weight_only_int8") {
    APPLY_MOE_LAYER_KERNEL(paddle::bfloat16, int8_t);
  } else if (x_type == paddle::DataType::BFLOAT16 &&
             quant_method == "weight_only_int4") {
    APPLY_MOE_LAYER_KERNEL(paddle::bfloat16, int4_t);
  } else {
    PD_THROW("MoeLayer not support x_type=",
             static_cast<int>(x_type),
             ", w_type=",
             static_cast<int>(w_type),
             ", quant_method=",
             quant_method);
    return {};
  }
#undef APPLY_MOE_LAYER_KERNEL
}

std::vector<std::vector<int64_t>> MoeLayerInferShape(
    const std::vector<int64_t> &x_shape,
    const std::vector<int64_t> &gate_weight_shape,
    const paddle::optional<std::vector<int64_t>> &gate_correction_bias_shape,
    const std::vector<int64_t> &up_gate_proj_weight_shape,
    const std::vector<int64_t> &down_proj_weight_shape,
    const paddle::optional<std::vector<int64_t>> &up_gate_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>> &down_proj_bias_shape,
    const paddle::optional<std::vector<int64_t>>
        &up_gate_proj_weight_scale_shape,
    const paddle::optional<std::vector<int64_t>> &down_proj_weight_scale_shape,
    const paddle::optional<std::vector<int64_t>> &down_proj_in_scale_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> MoeLayerInferDtype(
    const paddle::DataType &x_dtype,
    const paddle::DataType &gate_weight_dtype,
    const paddle::optional<paddle::DataType> &gate_correction_bias_dtype,
    const paddle::DataType &up_gate_proj_weight_dtype,
    const paddle::DataType &down_proj_weight_dtype,
    const paddle::optional<paddle::DataType> &up_gate_proj_bias_dtype,
    const paddle::optional<paddle::DataType> &down_proj_bias_dtype,
    const paddle::optional<paddle::DataType> &up_gate_proj_weight_scale_dtype,
    const paddle::optional<paddle::DataType> &down_proj_weight_scale_dtype,
    const paddle::optional<paddle::DataType> &down_proj_in_scale_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(xpu_moe_layer)  // fused_moe
    .Inputs({"x",
             "gate_weight",
             paddle::Optional("gate_correction_bias"),
             "up_gate_proj_weight",
             "down_proj_weight",
             paddle::Optional("up_gate_proj_bias"),
             paddle::Optional("down_proj_bias"),
             paddle::Optional("up_gate_proj_weight_scale"),
             paddle::Optional("down_proj_weight_scale"),
             paddle::Optional("down_proj_in_scale")})
    .Outputs({"fused_moe_out"})
    .Attrs({"quant_method:std::string", "moe_top_k:int", "moe_group:bool"})
    .SetKernelFn(PD_KERNEL(MoeLayer))
    .SetInferShapeFn(PD_INFER_SHAPE(MoeLayerInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MoeLayerInferDtype));
