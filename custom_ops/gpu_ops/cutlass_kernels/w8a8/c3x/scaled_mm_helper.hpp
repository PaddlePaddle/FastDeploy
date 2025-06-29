// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_helper.hpp

#include "helper.h"

template <typename Fp8Func, typename Int8Func>
void dispatch_scaled_mm(paddle::Tensor &c, paddle::Tensor const &a,
                        paddle::Tensor const &b, paddle::Tensor const &a_scales,
                        paddle::Tensor const &b_scales,
                        paddle::optional<paddle::Tensor> const &bias,
                        Fp8Func fp8_func, Int8Func int8_func) {
    PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
    PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);

    int M = a.dims()[0], N = b.dims()[0], K = a.dims()[1];

    if ((a_scales.numel() == 1 || a_scales.numel() == a.dims()[0]) &&
        (b_scales.numel() == 1 || b_scales.numel() == b.dims()[0])) {
      // Standard per-tensor/per-token/per-channel scaling
      PD_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());
      if (a.dtype() == phi::DataType::FLOAT8_E4M3FN) {
        fp8_func(c, a, b, a_scales, b_scales, bias);
      } else {
        PD_CHECK(a.dtype() == paddle::DataType::INT8);
        if constexpr (!std::is_same_v<Int8Func, std::nullptr_t>) {
          int8_func(c, a, b, a_scales, b_scales, bias);
        } else {
          PD_CHECK(false, "Int8 not supported for this architecture");
        }
      }
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "No kernel for this combination of input dtypes is implemented."));
    }
}
