// adapted from:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_sm100_fp8_dispatch.cuh

#pragma once

#include "scaled_mm.cuh"
#include "cutlass_gemm_caller.cuh"
#include "cutlass_extensions/epilogue/scaled_mm_epilogues_c3x.hpp"

/**
 * This file defines Gemm kernel configurations for SM100 (fp8) based on the
 * Gemm shape.
 */

namespace fastdeploy {

using c3x::cutlass_gemm_caller;

template <typename ElementAB_,
          typename ElementD_,
          template <typename, typename, typename> typename Epilogue_,
          typename TileShape,
          typename ClusterShape,
          typename KernelSchedule,
          typename EpilogueSchedule,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_sm100_fp8 {
  using ElementAB = ElementAB_;
  using ElementC = ElementD_;
  using ElementD = ElementD_;
  using ElementAcc = typename std::
      conditional<std::is_same_v<ElementAB, int8_t>, int32_t, float>::type;

  using Epilogue = Epilogue_<ElementAcc, ElementD, TileShape>;
  using EVTCompute = typename Epilogue::EVTCompute;

  static constexpr int AlignmentAB =
      128 / cutlass::sizeof_bits<ElementAB>::value;
  static constexpr int AlignmentCD =
      128 / cutlass::sizeof_bits<ElementD>::value;

  static constexpr bool swap_ab = swap_ab_;

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_T = typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_T = typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          cutlass::arch::Sm100,
          cutlass::arch::OpClassTensorOp,
          TileShape,
          ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAcc,
          float,
          ElementC,
          conditional_t<swap_ab, LayoutC_Transpose, LayoutC>,
          AlignmentCD,
          ElementD,
          conditional_t<swap_ab, LayoutD_Transpose, LayoutD>,
          AlignmentCD,
          EpilogueSchedule,
          EVTCompute>::CollectiveOp;

  static constexpr size_t CEStorageSize =
      sizeof(typename CollectiveEpilogue::SharedStorage);
  using Stages = typename cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(CEStorageSize)>;

  using CollectiveMainloop =
      conditional_t<swap_ab,
                    typename cutlass::gemm::collective::CollectiveBuilder<
                        cutlass::arch::Sm100,
                        cutlass::arch::OpClassTensorOp,
                        ElementAB,
                        LayoutB_T,
                        AlignmentAB,
                        ElementAB,
                        LayoutA_T,
                        AlignmentAB,
                        ElementAcc,
                        TileShape,
                        ClusterShape,
                        Stages,
                        KernelSchedule>::CollectiveOp,
                    typename cutlass::gemm::collective::CollectiveBuilder<
                        cutlass::arch::Sm100,
                        cutlass::arch::OpClassTensorOp,
                        ElementAB,
                        LayoutA,
                        AlignmentAB,
                        ElementAB,
                        LayoutB,
                        AlignmentAB,
                        ElementAcc,
                        TileShape,
                        ClusterShape,
                        Stages,
                        KernelSchedule>::CollectiveOp>;

  using GemmKernel = enable_sm100f_only<
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                           CollectiveMainloop,
                                           CollectiveEpilogue,
                                           void>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_default {
  // M in (256, inf)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_256, _128, _128>;
  using ClusterShape = Shape<_2, _2, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogue,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M256 {
  // M in (64, 256]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogue,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M64_swap_ab {
  // M in (16, 64] and K >= 4096 — swap_ab enabled
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _64, _256>;
  using ClusterShape = Shape<_4, _1, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueColumnBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule,
                                              true>,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogue,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule,
                                              true>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M64 {
  // M = 64 and K < 4096 — no swap_ab
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_64, _64, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule>,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogue,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule>>;
};

template <typename InType, typename OutType, bool EnableBias>
struct sm100_fp8_config_M16_swap_ab {
  // M in [1, 16] — swap_ab enabled
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = Shape<_128, _32, _128>;
  using ClusterShape = Shape<_4, _1, _1>;
  using Cutlass3xGemm =
      conditional_t<EnableBias,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogueColumnBias,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule,
                                              true>,
                    cutlass_3x_gemm_sm100_fp8<InType,
                                              OutType,
                                              c3x::ScaledEpilogue,
                                              TileShape,
                                              ClusterShape,
                                              KernelSchedule,
                                              EpilogueSchedule,
                                              true>>;
};

template <typename Gemm, typename... EpilogueArgs>
void cutlass_gemm_caller_sm100_fp8(paddle::Tensor &out,
                                   paddle::Tensor const &a,
                                   paddle::Tensor const &b,
                                   EpilogueArgs &&...epilogue_params) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using GemmKernel = typename Gemm::GemmKernel;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;

  int32_t m = a.dims()[0], n = b.dims()[0], k = a.dims()[1];
  auto prob_shape =
      swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{},
      swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  auto a_ptr = static_cast<ElementAB *>(const_cast<void *>(a.data()));
  auto b_ptr = static_cast<ElementAB *>(const_cast<void *>(b.data()));
  auto c_ptr = static_cast<ElementD *>(const_cast<void *>(out.data()));

  typename GemmKernel::MainloopArguments mainloop_args =
      swap_ab ? typename GemmKernel::MainloopArguments{b_ptr,
                                                       b_stride,
                                                       a_ptr,
                                                       a_stride}
              : typename GemmKernel::MainloopArguments{
                    a_ptr, a_stride, b_ptr, b_stride};

  typename GemmKernel::EpilogueArguments epilogue_args{
      Gemm::Epilogue::prepare_args(
          std::forward<EpilogueArgs>(epilogue_params)...),
      c_ptr,
      c_stride,
      c_ptr,
      c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(
      a.place(), prob_shape, mainloop_args, epilogue_args);
}

template <typename InType,
          typename OutType,
          bool EnableBias,
          typename... EpilogueArgs>
inline void cutlass_gemm_sm100_fp8_dispatch(paddle::Tensor &out,
                                            paddle::Tensor const &a,
                                            paddle::Tensor const &b,
                                            paddle::Tensor const &a_scales,
                                            paddle::Tensor const &b_scales,
                                            EpilogueArgs &&...args) {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  PD_CHECK(a.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(b.dtype() == paddle::DataType::FLOAT8_E4M3FN);

  using Cutlass3xGemmDefault =
      typename sm100_fp8_config_default<InType, OutType, EnableBias>::
          Cutlass3xGemm;
  using Cutlass3xGemmM16SwapAB =
      typename sm100_fp8_config_M16_swap_ab<InType, OutType, EnableBias>::
          Cutlass3xGemm;
  using Cutlass3xGemmM64SwapAB =
      typename sm100_fp8_config_M64_swap_ab<InType, OutType, EnableBias>::
          Cutlass3xGemm;
  using Cutlass3xGemmM64 =
      typename sm100_fp8_config_M64<InType, OutType, EnableBias>::Cutlass3xGemm;
  using Cutlass3xGemmM256 =
      typename sm100_fp8_config_M256<InType, OutType, EnableBias>::
          Cutlass3xGemm;

  uint32_t const m = a.dims()[0];
  uint32_t const k = a.dims()[1];

  if (m <= 16) {
    // m in [1, 16] — swap_ab
    return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM16SwapAB>(
        out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 64) {
    // m in (16, 64]
    if (m == 64 && k < 4096) {
      // do not enable swap AB
      return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64>(
          out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
    }
    // swap_ab
    return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM64SwapAB>(
        out, a, b, b_scales, a_scales, std::forward<EpilogueArgs>(args)...);
  } else if (m <= 256) {
    // m in (64, 256]
    return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmM256>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  } else {
    // m in (256, inf)
    return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemmDefault>(
        out, a, b, a_scales, b_scales, std::forward<EpilogueArgs>(args)...);
  }
}

template <bool EnableBias, typename... EpilogueArgs>
void cutlass_scaled_mm_sm100_fp8_epilogue(paddle::Tensor &out,
                                          paddle::Tensor const &a,
                                          paddle::Tensor const &b,
                                          paddle::Tensor const &a_scales,
                                          paddle::Tensor const &b_scales,
                                          EpilogueArgs &&...epilogue_args) {
  PD_CHECK(a.dtype() == paddle::DataType::FLOAT8_E4M3FN);
  PD_CHECK(b.dtype() == paddle::DataType::FLOAT8_E4M3FN);

  if (out.dtype() == paddle::DataType::BFLOAT16) {
    return cutlass_gemm_sm100_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::bfloat16_t,
                                           EnableBias>(
        out,
        a,
        b,
        a_scales,
        b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  } else {
    PD_CHECK(out.dtype() == paddle::DataType::FLOAT16);
    return cutlass_gemm_sm100_fp8_dispatch<cutlass::float_e4m3_t,
                                           cutlass::half_t,
                                           EnableBias>(
        out,
        a,
        b,
        a_scales,
        b_scales,
        std::forward<EpilogueArgs>(epilogue_args)...);
  }
}

}  // namespace fastdeploy
