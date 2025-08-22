
#include "../machete_mm_launcher.cuh"

namespace machete {


struct sch_256x128_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_256, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_128x256_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _256>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_256x32_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_256, _32>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_256x16_1x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_256, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_128x16_1x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _16>;
  using ClusterShape = Shape<_1, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_128x128_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _128>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_128x64_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _64>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_256x64_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_256, _64>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};

struct sch_128x32_2x1x1_TmaMI__TmaCoop_streamK {
  using TileShapeNM = Shape<_128, _32>;
  using ClusterShape = Shape<_2, _1, _1>;
  // TODO: Reimplement
  // using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
  using TileScheduler    = cutlass::gemm::StreamKScheduler;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
};



template<typename Sch>
using Kernel_f16u4f16f16voidvoidf16f32 = MacheteKernelTemplate<
  cutlass::half_t,  // ElementA
  cutlass::uint4b_t,  // ElementB
  cutlass::half_t,  // ElementD
  float, // Accumulator
  cutlass::half_t, // GroupScaleT
  cutlass::half_t, // GroupZeroT
  void, // ChannelScaleT
  void, // TokenScaleT
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  Sch>;


paddle::Tensor
impl_f16u4f16f16voidvoidf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_f16u4f16f16voidvoidf16f32<sch_128x32_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_f16u4f16f16voidvoidf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_f16u4f16f16voidvoidf16f32<sch_256x128_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_f16u4f16f16voidvoidf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_f16u4f16f16voidvoidf16f32<sch_128x16_1x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_f16u4f16f16voidvoidf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_f16u4f16f16voidvoidf16f32<sch_256x64_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_f16u4f16f16voidvoidf16f32_sch_256x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_f16u4f16f16voidvoidf16f32<sch_256x32_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_f16u4f16f16voidvoidf16f32_sch_256x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_f16u4f16f16voidvoidf16f32<sch_256x16_1x1x1_TmaMI__TmaCoop_streamK>>(args);
}
template<typename Sch>
using Kernel_bf16u4bf16bf16voidvoidbf16f32 = MacheteKernelTemplate<
  cutlass::bfloat16_t,  // ElementA
  cutlass::uint4b_t,  // ElementB
  cutlass::bfloat16_t,  // ElementD
  float, // Accumulator
  cutlass::bfloat16_t, // GroupScaleT
  cutlass::bfloat16_t, // GroupZeroT
  void, // ChannelScaleT
  void, // TokenScaleT
  cutlass::gemm::KernelTmaWarpSpecializedCooperative,
  Sch>;


paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_128x128_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x256_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_128x256_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_128x64_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x32_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_128x32_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x128_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_256x128_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_128x16_1x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_128x16_1x1x1_TmaMI__TmaCoop_streamK>>(args);
}
paddle::Tensor
impl_bf16u4bf16bf16voidvoidbf16f32_sch_256x64_2x1x1_TmaMI__TmaCoop_streamK(MMArgs args) {
  return run_impl<Kernel_bf16u4bf16bf16voidvoidbf16f32<sch_256x64_2x1x1_TmaMI__TmaCoop_streamK>>(args);
}

}; // namespace machete
