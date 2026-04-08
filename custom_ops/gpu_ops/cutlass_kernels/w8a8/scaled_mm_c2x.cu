// adapted from: https://github.com/vllm-project/vllm/blob/118ff921118cc81061a2af865a1e13840ceb6792/csrc/quantization/cutlass_w8a8/scaled_mm_c2x.cu

#include "helper.h"
#include <stddef.h>
#include "cutlass/cutlass.h"
#include "scaled_mm_c2x.cuh"
#include "scaled_mm_c2x_sm75_dispatch.cuh"
#include "scaled_mm_c2x_sm80_dispatch.cuh"
#include "scaled_mm_c2x_sm89_fp8_dispatch.cuh"
#include "scaled_mm_c2x_sm89_int8_dispatch.cuh"

#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c2x.hpp"

using namespace fastdeploy;

/*
   This file defines quantized GEMM operations using the CUTLASS 2.x API, for
   NVIDIA GPUs with SM versions prior to sm90 (Hopper).
*/

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm75_epilogue(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  PD_CHECK(a.dtype() == paddle::DataType::INT8);
  PD_CHECK(b.dtype() == paddle::DataType::INT8);

  if (out.dtype() == paddle::DataType::BFLOAT16) {
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    PD_CHECK(out.dtype() == paddle::DataType::FLOAT16);
    return cutlass_gemm_sm75_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm75(paddle::Tensor& out, paddle::Tensor const& a,
                            paddle::Tensor const& b,
                            paddle::Tensor const& a_scales,
                            paddle::Tensor const& b_scales,
                            paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);
  if (bias) {
    PD_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm75(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);

  if (azp) {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm75_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm80_epilogue(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  PD_CHECK(a.dtype() == paddle::DataType::INT8);
  PD_CHECK(b.dtype() == paddle::DataType::INT8);

  if (out.dtype() == paddle::DataType::BFLOAT16) {
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::bfloat16_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    PD_CHECK(out.dtype() == paddle::DataType::FLOAT16);
    return cutlass_gemm_sm80_dispatch<int8_t, cutlass::half_t, Epilogue>(
        out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

void cutlass_scaled_mm_sm80(paddle::Tensor& out, paddle::Tensor const& a,
                            paddle::Tensor const& b,
                            paddle::Tensor const& a_scales,
                            paddle::Tensor const& b_scales,
                            paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);
  if (bias) {
    PD_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm80(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);

  if (azp) {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm80_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}

template <template <typename, typename> typename Epilogue,
          typename... EpilogueArgs>
void cutlass_scaled_mm_sm89_epilogue(paddle::Tensor& out, paddle::Tensor const& a,
                                     paddle::Tensor const& b,
                                     EpilogueArgs&&... epilogue_args) {
  if (a.dtype() == paddle::DataType::INT8) {
    PD_CHECK(b.dtype() == paddle::DataType::INT8);

    if (out.dtype() == paddle::DataType::BFLOAT16) {
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::bfloat16_t,
                                             Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      assert(out.dtype() == paddle::DataType::FLOAT16);
      return cutlass_gemm_sm89_int8_dispatch<int8_t, cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  } else {
    PD_CHECK(a.dtype() == paddle::DataType::FLOAT8_E4M3FN);
    PD_CHECK(b.dtype() == paddle::DataType::FLOAT8_E4M3FN);

    if (out.dtype() == paddle::DataType::BFLOAT16) {
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::bfloat16_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    } else {
      PD_CHECK(out.dtype() == paddle::DataType::FLOAT16);
      return cutlass_gemm_sm89_fp8_dispatch<cutlass::float_e4m3_t,
                                            cutlass::half_t, Epilogue>(
          out, a, b, std::forward<EpilogueArgs>(epilogue_args)...);
    }
  }
}

void cutlass_scaled_mm_sm89(paddle::Tensor& out, paddle::Tensor const& a,
                            paddle::Tensor const& b,
                            paddle::Tensor const& a_scales,
                            paddle::Tensor const& b_scales,
                            paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);
  if (bias) {
    PD_CHECK(bias->dtype() == out.dtype(),
                "currently bias dtype must match output dtype ", out.dtype());
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBias>(
        out, a, b, a_scales, b_scales, *bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogue>(
        out, a, b, a_scales, b_scales);
  }
}

void cutlass_scaled_mm_azp_sm89(paddle::Tensor& out, paddle::Tensor const& a,
                                paddle::Tensor const& b,
                                paddle::Tensor const& a_scales,
                                paddle::Tensor const& b_scales,
                                paddle::Tensor const& azp_adj,
                                paddle::optional<paddle::Tensor> const& azp,
                                paddle::optional<paddle::Tensor> const& bias) {
  PD_CHECK(a_scales.dtype() == paddle::DataType::FLOAT32);
  PD_CHECK(b_scales.dtype() == paddle::DataType::FLOAT32);

  if (azp) {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzpToken>(
        out, a, b, a_scales, b_scales, azp_adj, *azp, bias);
  } else {
    return cutlass_scaled_mm_sm89_epilogue<c2x::ScaledEpilogueBiasAzp>(
        out, a, b, a_scales, b_scales, azp_adj, bias);
  }
}
