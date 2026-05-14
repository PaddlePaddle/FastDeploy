// adapted from:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8.cu

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm100_fp8_dispatch.cuh"

namespace fastdeploy {

void cutlass_scaled_mm_sm100_fp8(paddle::Tensor &out,
                                 paddle::Tensor const &a,
                                 paddle::Tensor const &b,
                                 paddle::Tensor const &a_scales,
                                 paddle::Tensor const &b_scales,
                                 paddle::optional<paddle::Tensor> const &bias) {
  PD_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
  if (bias) {
    PD_CHECK(bias->dtype() == out.dtype(),
             "currently bias dtype must match output dtype ",
             out.dtype());
    return cutlass_scaled_mm_sm100_fp8_epilogue<true>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm100_fp8_epilogue<false>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace fastdeploy
