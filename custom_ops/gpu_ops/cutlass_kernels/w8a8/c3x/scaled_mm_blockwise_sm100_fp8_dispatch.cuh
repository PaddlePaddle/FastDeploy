// adapted from:
// https://github.com/vllm-project/vllm/blob/main/csrc/quantization/cutlass_w8a8/c3x/scaled_mm_blockwise_sm100_fp8_dispatch.cuh

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass_gemm_caller.cuh"

namespace fastdeploy {

using namespace cute;

// clang-format off
template <class OutType,
          int ScaleGranularityM,
          int ScaleGranularityN,
          int ScaleGranularityK,
          class MmaTileShape,
          class ClusterShape,
          class EpilogueScheduler,
          class MainloopScheduler,
          bool swap_ab_ = false>
struct cutlass_3x_gemm_fp8_blockwise_sm100 {
  static constexpr bool swap_ab = swap_ab_;
  using ElementAB = cutlass::float_e4m3_t;

  using ElementA = ElementAB;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutA_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutA>::type;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;

  using ElementB = ElementAB;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutB_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutB>::type;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;

  using ElementD = OutType;
  using LayoutD = cutlass::layout::RowMajor;
  using LayoutD_Transpose =
      typename cutlass::layout::LayoutTranspose<LayoutD>::type;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using LayoutC = LayoutD;
  using LayoutC_Transpose = LayoutD_Transpose;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ElementBlockScale = float;

  using ScaleConfig = conditional_t<
      swap_ab,
      cutlass::detail::Sm100BlockwiseScaleConfig<
          ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
          cute::UMMA::Major::K, cute::UMMA::Major::MN>,
      cutlass::detail::Sm100BlockwiseScaleConfig<
          ScaleGranularityM, ScaleGranularityN, ScaleGranularityK,
          cute::UMMA::Major::MN, cute::UMMA::Major::K>>;

  using LayoutSFA = decltype(ScaleConfig::deduce_layoutSFA());
  using LayoutSFB = decltype(ScaleConfig::deduce_layoutSFB());

  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;
  using ElementScalar = float;
  using DefaultOperation = cutlass::epilogue::fusion::LinearCombination<
      ElementD, ElementCompute, ElementC, ElementScalar, RoundStyle>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, MmaTileShape, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementC,
          conditional_t<swap_ab, LayoutC_Transpose, LayoutC>, AlignmentC,
          ElementD,
          conditional_t<swap_ab, LayoutD_Transpose, LayoutD>, AlignmentD,
          EpilogueScheduler, DefaultOperation>::CollectiveOp;

  using StageCountType = cutlass::gemm::collective::StageCountAuto;
  using CollectiveMainloop = conditional_t<
      swap_ab,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          ElementB, cute::tuple<LayoutB_Transpose, LayoutSFA>, AlignmentB,
          ElementA, cute::tuple<LayoutA_Transpose, LayoutSFB>, AlignmentA,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler>::CollectiveOp,
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass,
          ElementA, cute::tuple<LayoutA, LayoutSFA>, AlignmentA,
          ElementB, cute::tuple<LayoutB, LayoutSFB>, AlignmentB,
          ElementAccumulator, MmaTileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          MainloopScheduler>::CollectiveOp>;

  using KernelType =
      enable_sm100f_only<cutlass::gemm::kernel::GemmUniversal<
          Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>>;

  struct GemmKernel : public KernelType {};
};
// clang-format on

template <typename Gemm>
void cutlass_gemm_caller_blockwise_sm100(paddle::Tensor &out,
                                         paddle::Tensor const &a,
                                         paddle::Tensor const &b,
                                         paddle::Tensor const &a_scales,
                                         paddle::Tensor const &b_scales) {
  static constexpr bool swap_ab = Gemm::swap_ab;
  using GemmKernel = typename Gemm::GemmKernel;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using LayoutSFA = typename Gemm::LayoutSFA;
  using LayoutSFB = typename Gemm::LayoutSFB;
  using ScaleConfig = typename Gemm::ScaleConfig;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;
  using ElementBlockScale = typename Gemm::ElementBlockScale;

  int32_t m = a.dims()[0], n = b.dims()[0], k = a.dims()[1];

  StrideA a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  StrideB b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  StrideC c_stride = cutlass::make_cute_packed_stride(
      StrideC{},
      swap_ab ? cute::make_shape(n, m, 1) : cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA =
      swap_ab ? ScaleConfig::tile_atom_to_shape_SFA(make_shape(n, m, k, 1))
              : ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB =
      swap_ab ? ScaleConfig::tile_atom_to_shape_SFB(make_shape(n, m, k, 1))
              : ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementAB const *>(const_cast<void *>(a.data()));
  auto b_ptr = static_cast<ElementAB const *>(const_cast<void *>(b.data()));
  auto a_scales_ptr = static_cast<ElementBlockScale const *>(
      const_cast<void *>(a_scales.data()));
  auto b_scales_ptr = static_cast<ElementBlockScale const *>(
      const_cast<void *>(b_scales.data()));

  typename GemmKernel::MainloopArguments mainloop_args{};
  mainloop_args.layout_SFA = layout_SFA;
  mainloop_args.layout_SFB = layout_SFB;
  if (swap_ab) {
    mainloop_args.ptr_A = b_ptr;
    mainloop_args.dA = b_stride;
    mainloop_args.ptr_B = a_ptr;
    mainloop_args.dB = a_stride;
    mainloop_args.ptr_SFA = b_scales_ptr;
    mainloop_args.ptr_SFB = a_scales_ptr;
  } else {
    mainloop_args.ptr_A = a_ptr;
    mainloop_args.dA = a_stride;
    mainloop_args.ptr_B = b_ptr;
    mainloop_args.dB = b_stride;
    mainloop_args.ptr_SFA = a_scales_ptr;
    mainloop_args.ptr_SFB = b_scales_ptr;
  }
  auto prob_shape =
      swap_ab ? cute::make_shape(n, m, k, 1) : cute::make_shape(m, n, k, 1);

  auto c_ptr = static_cast<ElementD *>(const_cast<void *>(out.data()));
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(
      a.place(), prob_shape, mainloop_args, epilogue_args);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm100_fp8_dispatch(paddle::Tensor &out,
                                               paddle::Tensor const &a,
                                               paddle::Tensor const &b,
                                               paddle::Tensor const &a_scales,
                                               paddle::Tensor const &b_scales) {
  int32_t m = a.dims()[0], n = b.dims()[0], k = a.dims()[1], sms;
  cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, a.place().GetDeviceId());

  constexpr int TILE_K = 128;
  bool swap_ab = (m < 16) || (m % 4 != 0);
  bool use_tma_epilogue = (m * n) % 4 == 0;

  if (!swap_ab) {
    constexpr int TILE_N = 128;
    int tile_m = 256;
    auto ceil_div = [](int a, int b) { return (a + b - 1) / b; };
    if (ceil_div(n, TILE_N) * ceil_div(m, 64) <= sms) {
      tile_m = 64;
    } else if (ceil_div(n, TILE_N) * ceil_div(m, 128) <= sms) {
      tile_m = 128;
    }
    if (tile_m == 64) {
      if (use_tma_epilogue) {
        cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
            OutType,
            1,
            TILE_N,
            TILE_K,
            Shape<_64, Int<TILE_N>, Int<TILE_K>>,
            Shape<_1, _1, _1>,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>>(
            out, a, b, a_scales, b_scales);
      } else {
        cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
            OutType,
            1,
            TILE_N,
            TILE_K,
            Shape<_64, Int<TILE_N>, Int<TILE_K>>,
            Shape<_1, _1, _1>,
            cutlass::epilogue::BlockwiseNoSmemWarpSpecialized1Sm,
            cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>>(
            out, a, b, a_scales, b_scales);
      }
    } else if (tile_m == 128) {
      if (use_tma_epilogue) {
        cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
            OutType,
            1,
            TILE_N,
            TILE_K,
            Shape<_128, Int<TILE_N>, Int<TILE_K>>,
            Shape<_1, _1, _1>,
            cutlass::epilogue::TmaWarpSpecialized1Sm,
            cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>>(
            out, a, b, a_scales, b_scales);
      } else {
        cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
            OutType,
            1,
            TILE_N,
            TILE_K,
            Shape<_128, Int<TILE_N>, Int<TILE_K>>,
            Shape<_1, _1, _1>,
            cutlass::epilogue::BlockwiseNoSmemWarpSpecialized1Sm,
            cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100>>(
            out, a, b, a_scales, b_scales);
      }
    } else {  // tile_m == 256
      if (use_tma_epilogue) {
        cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
            OutType,
            1,
            TILE_N,
            TILE_K,
            Shape<_256, Int<TILE_N>, Int<TILE_K>>,
            Shape<_2, _1, _1>,
            cutlass::epilogue::TmaWarpSpecialized2Sm,
            cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100>>(
            out, a, b, a_scales, b_scales);
      } else {
        cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
            OutType,
            1,
            TILE_N,
            TILE_K,
            Shape<_256, Int<TILE_N>, Int<TILE_K>>,
            Shape<_2, _1, _1>,
            cutlass::epilogue::BlockwiseNoSmemWarpSpecialized2Sm,
            cutlass::gemm::KernelTmaWarpSpecializedBlockwise2SmSm100>>(
            out, a, b, a_scales, b_scales);
      }
    }
  } else {
    constexpr int TILE_M = 128;
    constexpr int TILE_N = 16;
    // TMA epilogue isn't compatible with swap_ab
    cutlass_gemm_caller_blockwise_sm100<cutlass_3x_gemm_fp8_blockwise_sm100<
        OutType,
        TILE_M,
        1,
        TILE_K,
        Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>,
        Shape<_1, _1, _1>,
        cutlass::epilogue::BlockwiseNoSmemWarpSpecialized1Sm,
        cutlass::gemm::KernelTmaWarpSpecializedBlockwise1SmSm100,
        true>>(out, a, b, a_scales, b_scales);
  }
}

}  // namespace fastdeploy
