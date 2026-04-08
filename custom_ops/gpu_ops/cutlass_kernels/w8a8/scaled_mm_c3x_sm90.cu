// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm90.cu

#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm90a (Hopper).
*/

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90

void cutlass_scaled_mm_sm90(paddle::Tensor &c, paddle::Tensor const &a,
                            paddle::Tensor const &b,
                            paddle::Tensor const &a_scales,
                            paddle::Tensor const &b_scales,
                            paddle::optional<paddle::Tensor> const &bias) {
  dispatch_scaled_mm(c, a, b, a_scales, b_scales, bias,
                     fastdeploy::cutlass_scaled_mm_sm90_fp8,
                     fastdeploy::cutlass_scaled_mm_sm90_int8);
}

void cutlass_scaled_mm_azp_sm90(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);

  fastdeploy::cutlass_scaled_mm_azp_sm90_int8(out, a, b, a_scales, b_scales, azp_adj,
                                        azp, bias);
}

#endif
