// adapted from:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8.cu

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_blockwise_sm100_fp8_dispatch.cuh"

namespace fastdeploy {

void cutlass_scaled_mm_blockwise_sm100_fp8(paddle::Tensor &out,
                                           paddle::Tensor const &a,
                                           paddle::Tensor const &b,
                                           paddle::Tensor const &a_scales,
                                           paddle::Tensor const &b_scales) {
  if (out.dtype() == paddle::DataType::BFLOAT16) {
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);
  } else {
    PD_CHECK(out.dtype() == paddle::DataType::FLOAT16);
    cutlass_gemm_blockwise_sm100_fp8_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace fastdeploy
