// adapted from:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/scaled_mm_c3x_sm120.cu

#include "c3x/scaled_mm_helper.hpp"
#include "c3x/scaled_mm_kernels.hpp"

/*
   This file defines quantized GEMM operations using the CUTLASS 3.x API, for
   NVIDIA GPUs with sm120 (Blackwell).
*/

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120

void cutlass_scaled_mm_sm120(paddle::Tensor& c,
                             paddle::Tensor const& a,
                             paddle::Tensor const& b,
                             paddle::Tensor const& a_scales,
                             paddle::Tensor const& b_scales,
                             paddle::optional<paddle::Tensor> const& bias) {
  dispatch_scaled_mm(c,
                     a,
                     b,
                     a_scales,
                     b_scales,
                     bias,
                     fastdeploy::cutlass_scaled_mm_sm120_fp8,
                     nullptr,  // int8 not supported on SM120
                     fastdeploy::cutlass_scaled_mm_blockwise_sm120_fp8);
}

#endif
