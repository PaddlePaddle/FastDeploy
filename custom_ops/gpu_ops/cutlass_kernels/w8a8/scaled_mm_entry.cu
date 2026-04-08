// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/scaled_mm_entry.cu

#pragma once
#include "helper.h"
#include <iostream>

void cutlass_scaled_mm_sm75(paddle::Tensor &c, paddle::Tensor const &a,
                            paddle::Tensor const &b,
                            paddle::Tensor const &a_scales,
                            paddle::Tensor const &b_scales,
                            paddle::optional<paddle::Tensor> const &bias);

void cutlass_scaled_mm_sm80(paddle::Tensor &c, paddle::Tensor const &a,
                            paddle::Tensor const &b,
                            paddle::Tensor const &a_scales,
                            paddle::Tensor const &b_scales,
                            paddle::optional<paddle::Tensor> const &bias);

void cutlass_scaled_mm_sm89(paddle::Tensor &c, paddle::Tensor const &a,
                            paddle::Tensor const &b,
                            paddle::Tensor const &a_scales,
                            paddle::Tensor const &b_scales,
                            paddle::optional<paddle::Tensor> const &bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_sm90(paddle::Tensor &c, paddle::Tensor const &a,
                            paddle::Tensor const &b,
                            paddle::Tensor const &a_scales,
                            paddle::Tensor const &b_scales,
                            paddle::optional<paddle::Tensor> const &bias);
#endif

void cutlass_scaled_mm_azp_sm75(paddle::Tensor& c, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm80(paddle::Tensor& c, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm89(paddle::Tensor& c, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_azp_sm90(paddle::Tensor& c, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}

void CutlassScaledMm(paddle::Tensor &c, paddle::Tensor const &a,
                     paddle::Tensor const &b, paddle::Tensor const &a_scales,
                     paddle::Tensor const &b_scales,
                     paddle::optional<paddle::Tensor> const &bias) {
  // Checks for conformality
  PD_CHECK(a.dims().size() == 2 && b.dims().size() == 2 &&
           c.dims().size() == 2);
  PD_CHECK(c.dims()[0] == a.dims()[0] && a.dims()[1] == b.dims()[1] &&
           b.dims()[0] == c.dims()[1]);

  // Check for strides and alignment
  PD_CHECK(a.strides()[1] == 1 && c.strides()[1] == 1); // Row-major
  PD_CHECK(b.strides()[1] == 1);                        // Column-major
  PD_CHECK(c.strides()[0] % 16 == 0 &&
           b.strides()[0] % 16 == 0); // 16 Byte Alignment

  if (bias) {
    PD_CHECK(bias->numel() == b.dims()[0] && bias->is_contiguous() &&
             bias->dims().size() == 1);
  }

  int32_t version_num = GetGPUComputeCapability(a.place().GetDeviceId());

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90 && version_num < 100) {
    // Hopper
    cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 75) {
    // Turing
    cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  PADDLE_THROW(phi::errors::Unimplemented(
      "No compiled cutlass_scaled_mm for a compute capability less than "
      "CUDA device capability: %d",
      version_num));
}

void CutlassScaledMmAzp(paddle::Tensor& c, paddle::Tensor const& a,
                           paddle::Tensor const& b,
                           paddle::Tensor const& a_scales,
                           paddle::Tensor const& b_scales,
                           paddle::Tensor const& azp_adj,
                           paddle::optional<paddle::Tensor> const& azp,
                           paddle::optional<paddle::Tensor> const& bias) {
  // Checks for conformality
  PD_CHECK(a.dims().size() == 2 && b.dims().size() == 2 &&
           c.dims().size() == 2);
  PD_CHECK(c.dims()[0] == a.dims()[0] && a.dims()[1] == b.dims()[1] &&
           b.dims()[0] == c.dims()[1]);
  PD_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.dims()[0]);
  PD_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.dims()[0]);

  // Check for strides and alignment
  PD_CHECK(a.strides()[1] == 1 && c.strides()[1] == 1); // Row-major
  PD_CHECK(b.strides()[1] == 1);                        // Column-major
  PD_CHECK(c.strides()[0] % 16 == 0 &&
           b.strides()[0] % 16 == 0); // 16 Byte Alignment
  PD_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    PD_CHECK(bias->numel() == b.dims()[0] && bias->is_contiguous());
  }
  if (azp) {
    PD_CHECK(azp->numel() == a.dims()[0] && azp->is_contiguous());
  }
  PD_CHECK(azp_adj.numel() == b.dims()[0] && azp_adj.is_contiguous());

  // azp & bias types
  PD_CHECK(azp_adj.dtype() == paddle::DataType::INT32);
  PD_CHECK(!azp || azp->dtype() == paddle::DataType::INT32);
  PD_CHECK(!bias || bias->dtype() == c.dtype(),
              "currently bias dtype must match output dtype ", c.dtype());

  int32_t version_num = GetGPUComputeCapability(a.place().GetDeviceId());

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_azp_sm89(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_azp_sm80(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  // Turing
  PD_CHECK(version_num >= 75);
  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  return;
#endif

  PADDLE_THROW(phi::errors::Unimplemented(
      "No compiled cutlass_scaled_mm_azp for a compute capability less than "
      "CUDA device capability: %d",
      version_num));
}


PD_BUILD_STATIC_OP(cutlass_scaled_mm)
    .Inputs({"c", "a", "b", "a_scales", "b_scales", paddle::Optional("bias")})
    .Outputs({"c_out"})
    .SetInplaceMap({{"c", "c_out"}})
    .SetKernelFn(PD_KERNEL(CutlassScaledMm));

PD_BUILD_STATIC_OP(cutlass_scaled_mm_azp)
    .Inputs({"c", "a", "b", "a_scales", "b_scales", "azp_adj", paddle::Optional("azp"), paddle::Optional("bias")})
    .Outputs({"c_out"})
    .SetInplaceMap({{"c", "c_out"}})
    .SetKernelFn(PD_KERNEL(CutlassScaledMmAzp));
