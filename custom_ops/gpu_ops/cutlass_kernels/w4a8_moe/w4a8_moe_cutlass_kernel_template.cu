/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic pop
#include <sstream>
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/kernel/default_intA_nf4B_traits.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/kernel/default_dequant_gemm_nf4.h"
#include "w4a8_gemm_grouped.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "w4a8_moe_gemm_kernel_template.h"
#include "w4a8_moe_cutlass_kernel.h"
#include "cuda_utils.h"
#include "glog/logging.h"
#include "w4a4_gemm_configs.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/epilogue_helpers.h"
#include "cutlass_heuristic_w4a4.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/compute_occupancy.h"
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "base64_encode.h"

#include "cutlass_kernels/w4a8_moe/cutlass_extensions/gemm/kernel/gemm_with_epilogue_visitor_interleaved_nf4.h"
#include "w4a8_moe_gemm_with_epilogue_visitor.h"


template <int val>
class IntegerType {
  public:
  static constexpr int value = val;
};

template <int v>
using Int = IntegerType<v>;

template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_w4a8_moe_gemm_kernelLauncher(
    const IntAType* A,
    const IntBType* B,
    cutlass::epilogue::QuantMode quant_mode,
    const OutputType* col_scale,
    const OutputType* row_scale,
    const int32_t* nf4_look_up_table,
    OutputType* C,
    int64_t* total_rows_before_expert,
    int total_rows_in_ll_else_minus1,
    int total_rows,
    int n,
    int k,
    int num_experts,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    int multi_processor_count,
    cudaStream_t stream,
    int* occupancy) {
  if (gemm_config.split_k_style == SplitKStyle::NO_SPLIT_K){
    static_assert(cutlass::platform::is_same<IntAType, int8_t>::value,
                  "input type must be int8_t");

    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if necessary.
    // using OutputElementType_      = OutputType;
    using OutputElementType_ = typename cutlass::platform::conditional<cutlass::platform::is_same<OutputType, __nv_bfloat16>::value,
                                                cutlass::bfloat16_t, OutputType>::type;

    using OutputElementType       = typename cutlass::platform::conditional<cutlass::platform::is_same<OutputElementType_, half>::value,
                                                cutlass::half_t, OutputElementType_>::type;

    using CutlassIntAType_ = IntAType;
    using CutlassIntAType = CutlassIntAType_;

    using CutlassIntBType_ = IntBType;
    using CutlassIntBType = CutlassIntBType_;

    // We need separate config for each architecture since we will target different tensorcore instructions. For float,
    // we do not target TCs.

    using MixedGemmArchTraits = cutlass::gemm::kernel::
        Int8Nf4GemmArchTraits<CutlassIntAType, CutlassIntBType, OutputElementType, arch>;

    using ElementAccumulator  = typename MixedGemmArchTraits::AccType;
    using ElementCompute  = float;


    // ==============
    using EpilogueOp =
        typename Epilogue<OutputElementType, MixedGemmArchTraits::ElementsPerAccessC, ElementCompute, EpilogueTag>::Op;

    using ThreadBlockSwizzle = typename cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle;
    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultInt8InterleavedGemm<
        CutlassIntAType,
        cutlass::layout::RowMajor,
        MixedGemmArchTraits::ElementsPerAccessA,
        CutlassIntBType,
        typename MixedGemmArchTraits::LayoutB,
        MixedGemmArchTraits::ElementsPerAccessB,
        OutputElementType,
        cutlass::layout::RowMajor,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        arch,
        ThreadblockShape,
        WarpShape,
        typename MixedGemmArchTraits::InstructionShape,
        EpilogueOp,
        ThreadBlockSwizzle,
        Stages,
        true,
        typename MixedGemmArchTraits::Operator>::GemmKernel;
    using GemmKernel = cutlass::gemm::kernel::MoeW4A8Gemm<typename GemmKernel_::Mma,
                                                          typename GemmKernel_::Epilogue,
                                                          typename GemmKernel_::ThreadblockSwizzle,
                                                          arch,  // Ensure top level arch is used for dispatch
                                                          GemmKernel_::kSplitKSerial,
                                                          cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>;
    using AlphaColTileIterator =
        cutlass::epilogue::threadblock::PredicatedTileIterator<
            cutlass::epilogue::threadblock::OutputTileOptimalThreadMap<
                typename GemmKernel::Epilogue::OutputTileIterator::ThreadMap::
                    Shape,
                typename GemmKernel::Epilogue::OutputTileIterator::ThreadMap::
                    Count,
                GemmKernel::Epilogue::OutputTileIterator::ThreadMap::kThreads,
                GemmKernel::Epilogue::OutputTileIterator::kElementsPerAccess,
                cutlass::sizeof_bits<OutputElementType>::value>,
            OutputElementType>;

    using EpilogueVisitor = typename cutlass::epilogue::threadblock::EpilogueVisitorPerRowPerColNf4<
        ThreadblockShape,
        GemmKernel::kThreadCount,
        AlphaColTileIterator,
        typename GemmKernel::Epilogue::OutputTileIterator,
        ElementAccumulator,
        ElementCompute,
        EpilogueOp>;

    /// Epilogue
    using Epilogue = typename cutlass::epilogue::threadblock::
        EpilogueWithVisitorFromExistingEpilogue<EpilogueVisitor, typename GemmKernel::Epilogue>::Epilogue;

    // GEMM
    using GemmWithEpilogueVisitorKernel =
        cutlass::gemm::kernel::MoeW4A8GemmWithEpilogueVisitorInterleavedNf4<typename GemmKernel::Mma,
                                                                            Epilogue,
                                                                            ThreadBlockSwizzle,
                                                                            cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>;


    if (occupancy != nullptr) {
        *occupancy = compute_occupancy_for_kernel<GemmWithEpilogueVisitorKernel>();
        return;
    }

    using Gemm = cutlass::gemm::device::W4A8MoeGemmUniversalBase<GemmWithEpilogueVisitorKernel>;

    const int ldb =
        cutlass::platform::is_same<cutlass::layout::RowMajor, typename MixedGemmArchTraits::LayoutB>::value ?
            n :
            k * GemmKernel::kInterleave;

    typename EpilogueOp::Params linear_scaling_params;

    // printf("init end\n");
    int occupancy_ = std::min(2, Gemm::maximum_active_blocks());
    if (occupancy_ == 0) {
      throw std::runtime_error(
          "[w4a8MoE Runner] GPU lacks the shared memory resources to run "
          "GroupedGEMM kernel");
    }

    const int threadblock_count = multi_processor_count * occupancy_;

    typename Gemm::Arguments args{cutlass::gemm::GemmUniversalMode::kBatched,
                                  num_experts,
                                  threadblock_count,
                                  {total_rows, n, k},
                                  1,
                                  {reinterpret_cast<CutlassIntAType*>(const_cast<IntAType*>(A)), k},
                                  {reinterpret_cast<CutlassIntBType*>(const_cast<IntBType*>(B)), ldb},
                                  quant_mode,
                                  {reinterpret_cast<OutputElementType*>(const_cast<OutputType*>(col_scale)), 0},
                                  {reinterpret_cast<OutputElementType*>(const_cast<OutputType*>(row_scale)), 0},
                                  {const_cast<int32_t*>(nf4_look_up_table), 0},
                                  {reinterpret_cast<OutputElementType*>(C), n},
                                  {reinterpret_cast<OutputElementType*>(C), n},
                                  total_rows_before_expert,
                                  total_rows_in_ll_else_minus1,
                                  n,
                                  k,
                                  (int64_t)0,
                                  (int64_t)0,
                                  typename EpilogueVisitor::Arguments(linear_scaling_params, 0, 0, 0)};

    // This assertion is enabled because because for the column interleaved layout, K MUST be a multiple of
    // threadblockK. The reason for this is that the default pitchlinear iterators are used to handle walking over the
    // interleaved matrix. The way masking in handled in these do not map to the interleaved layout. We need to write
    // our own predicated iterator in order to relax this limitation.
    if (GemmKernel::kInterleave > 1
        && ((k % MixedGemmArchTraits::ThreadblockK)
            || ((k / gemm_config.split_k_factor) % MixedGemmArchTraits::ThreadblockK))) {
        throw std::runtime_error("Temp assertion: k must be multiple of threadblockK");
    }

    Gemm gemm;
    if (gemm.get_workspace_size(args) > workspace_bytes) {
        std::cout<<
            "Requested split-k but workspace size insufficient. Falling back to non-split-k implementation."<<std::endl;
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }
    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg = "intA_intB cutlass kernel will fail for params. Error: "
                              + std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[FT Error][intA_intB Runner] " + err_msg);
    }
    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to initialize cutlass intA_intB gemm. Error: " + std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[FT Error][intA_intB Runner] " + err_msg);
    }
    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg =
            "Failed to run cutlass intA_intB gemm. Error: " + std::string(cutlassGetStatusString(run_status));
        CUTLASS_TRACE_HOST("  [FT Error][intA_intB Runner] " << cutlassGetStatusString(run_status));
        // throw std::runtime_error("[FT Error][intA_intB Runner] " + err_msg);
    }
    CUTLASS_TRACE_HOST("  finish run kernel " << cutlassGetStatusString(run_status));
  }
}

template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
void dispatch_gemm_config(const IntAType* A,
                          const IntBType* B,
                          cutlass::epilogue::QuantMode quant_mode,
                          const OutputType* col_scale,
                          const OutputType* row_scale,
                          const int32_t* nf4_look_up_table,
                          OutputType* C,
                          int64_t* total_rows_before_expert,
                          int64_t total_rows_in_ll_else_minus1,
                          int64_t total_rows,
                          int64_t gemm_n,
                          int64_t gemm_k,
                          int num_experts,
                          CutlassGemmConfig gemm_config,
                          char* workspace,
                          size_t workspace_bytes,
                          int multi_processor_count,
                          cudaStream_t stream,
                          int* occupancy = nullptr) {

  auto dispatch_by_stage = [&](auto temp_args) {
    using DispatcherStages = dispatch_stages<OutputType,
                                             IntAType,
                                             IntBType,
                                             arch,
                                             EpilogueTag,
                                             ThreadblockShape,
                                             WarpShape,
                                             decltype(temp_args)::value>;
    DispatcherStages::dispatch(A,
                               B,
                               quant_mode,
                               col_scale,
                               row_scale,
                               nf4_look_up_table,
                               C,
                               total_rows_before_expert,
                               total_rows_in_ll_else_minus1,
                               total_rows,
                               gemm_n,
                               gemm_k,
                               num_experts,
                               gemm_config,
                               workspace,
                               workspace_bytes,
                               multi_processor_count,
                               stream,
                               occupancy);
  };
  switch (gemm_config.stages) {
    case 2:
      dispatch_by_stage(Int<2>());
      break;
    case 3:
      dispatch_by_stage(Int<3>());
      break;
    case 4:
      dispatch_by_stage(Int<4>());
      break;
    case 5:
      dispatch_by_stage(Int<5>());
      break;
    case 6:
      dispatch_by_stage(Int<6>());
      break;
    case 7:
      dispatch_by_stage(Int<7>());
      break;
    default:
      std::string err_msg = "dispatch_gemm_config does not support stages " +
                            std::to_string(gemm_config.stages);
      throw std::runtime_error("[W4A8MoE][dispatch_gemm_config] " +
                               err_msg);
      break;
  }
}

template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename arch,
          typename EpilogueTag>
void dispatch_moe_gemm_to_cutlass(
    const IntAType* A,
    const IntBType* B,
    cutlass::epilogue::QuantMode quant_mode,
    const OutputType* col_scale,
    const OutputType* row_scale,
    const int32_t* nf4_look_up_table,
    OutputType* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows_in_ll_else_minus1,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    CutlassGemmConfig gemm_config,
    char* workspace_ptr,
    const size_t workspace_bytes,
    // int sm_version,
    int multi_processor_count,
    cudaStream_t stream,
    int* occupancy = nullptr) {
  // VLOG(1)<<__PRETTY_FUNCTION__;

  auto dispatch_by_tile = [&](auto ThreadblockShapeM,
                              auto ThreadblockShapeN,
                              auto ThreadblockShapeK,
                              auto WarpShapeM,
                              auto WarpShapeN,
                              auto WarpShapeK) {
      dispatch_gemm_config<
          OutputType,
          IntAType,
          IntBType,
          arch,
          EpilogueTag,
          cutlass::gemm::GemmShape<decltype(ThreadblockShapeM)::value,
                                   decltype(ThreadblockShapeN)::value,
                                   decltype(ThreadblockShapeK)::value>,
          cutlass::gemm::GemmShape<decltype(WarpShapeM)::value,
                                   decltype(WarpShapeN)::value,
                                   decltype(WarpShapeK)::value>>
                                   (A,
                                    B,
                                    quant_mode,
                                    col_scale,
                                    row_scale,
                                    nf4_look_up_table,
                                    C,
                                    total_rows_before_expert,
                                    total_rows_in_ll_else_minus1,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    gemm_config,
                                    workspace_ptr,
                                    workspace_bytes,
                                    multi_processor_count,
                                    stream,
                                    occupancy);
  };

  switch (gemm_config.tile_config) {
    case CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
      dispatch_by_tile(Int<16>(), Int<64>(), Int<64>(),
                       Int<16>(), Int<32>(), Int<64>());
      break;
    case CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
      dispatch_by_tile(Int<32>(), Int<128>(), Int<64>(),
                       Int<32>(), Int<32>(), Int<64>());
      break;
    case CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
      dispatch_by_tile(Int<64>(), Int<128>(), Int<64>(),
                       Int<64>(), Int<32>(), Int<64>());
      break;
    case CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
      dispatch_by_tile(Int<128>(), Int<128>(), Int<64>(),
                       Int<128>(), Int<32>(), Int<64>());
      break;
    case CutlassTileConfig::CtaShape32x512x64_WarpShape32x128x64:
      dispatch_by_tile(Int<32>(), Int<512>(), Int<64>(),
                       Int<32>(), Int<128>(), Int<64>());
      break;
    case CutlassTileConfig::CtaShape32x256x64_WarpShape32x64x64:
      dispatch_by_tile(Int<32>(), Int<256>(), Int<64>(),
                       Int<32>(), Int<64>(), Int<64>());
      break;
    case CutlassTileConfig::CtaShape64x256x64_WarpShape64x64x64:
      dispatch_by_tile(Int<64>(), Int<256>(), Int<64>(),
                       Int<64>(), Int<64>(), Int<64>());
      break;
    // case CutlassTileConfig::CtaShape128x256x64_WarpShape128x64x64:
    //   dispatch_by_tile(Int<128>(), Int<256>(), Int<64>(),
    //                    Int<128>(), Int<64>(), Int<64>());
    //   break;
    // config for M_16000_N_12288_K_6144 in encoder
    // case CutlassTileConfig::CtaShape256x128x64_WarpShape64x64x64:
    //   dispatch_by_tile(Int<256>(), Int<128>(), Int<64>(),
    //                    Int<128>(), Int<32>(), Int<64>());
    //   break;
    // case CutlassTileConfig::CtaShape128x256x64_WarpShape64x64x64:
    //   dispatch_by_tile(Int<128>(), Int<256>(), Int<64>(),
    //                    Int<64>(), Int<32>(), Int<64>());
    //   break;
    case CutlassTileConfig::Undefined:
      throw std::runtime_error(
          "[dispatch_moe_gemm_to_cutlass] gemm config undefined.");
      break;
    case CutlassTileConfig::ChooseWithHeuristic:
      throw std::runtime_error(
          "[dispatch_moe_gemm_to_cutlass] gemm config should have "
          "already been set by heuristic.");
      break;
    default:
      throw std::runtime_error(
          "[dispatch_moe_gemm_to_cutlass] Config is invalid for same "
          "type MoE tensorop GEMM.");
      break;
  }
}


template <typename OutputType, typename IntAType, typename IntBType>
W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::W4A8MoeGemmRunner() {
  int device{-1};
  check_cuda_error(cudaGetDevice(&device));
  sm_ = getSMVersion();
  // sm_ = 80;
  check_cuda_error(cudaDeviceGetAttribute(
      &multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
  std::string FLAGS_cutlass_w4a8_moe_best_config="";
  if (getenv("FLAGS_cutlass_w4a8_moe_best_config")) {
        FLAGS_cutlass_w4a8_moe_best_config = getenv("FLAGS_cutlass_w4a8_moe_best_config");
  }
  if(tuned_configs_from_file.empty() && FLAGS_cutlass_w4a8_moe_best_config!="") {
    std::string config_file_path = FLAGS_cutlass_w4a8_moe_best_config;
    if (config_file_path.find(".config")!=std::string::npos) {
      std::ifstream config_file(FLAGS_cutlass_w4a8_moe_best_config);
        if (config_file.is_open()) {
          VLOG(1)<<"Get tuned w4a8 moe gemm config from: "<<config_file_path;
          std::string config_string;
          while(std::getline(config_file, config_string)) {
            // decode one line of base64 string
            config_string = base64_decode(config_string);
            VLOG(1)<<"decode config_string: " << config_string;
            std::stringstream ss(config_string);
            std::string item;
            std::vector<int> vec_configs;
            while(std::getline(ss, item, ',')) {
              try {
                  int value = std::stoi(item);
                  vec_configs.push_back(value);
              } catch (const std::invalid_argument& e) {
                  std::cerr << "Invalid argument: " << item << " is not an integer." << std::endl;
                  return;
              } catch (const std::out_of_range& e) {
                  std::cerr << "Out of range: " << item << " is out of the range of representable values." << std::endl;
                  return;
              }
            }
            W4A8MoeGEMMConfig search_config;
            search_config.total_rows = vec_configs[0];
            search_config.n = vec_configs[1];
            search_config.k = vec_configs[2];
            search_config.num_experts = vec_configs[3];
            search_config.tile_config = static_cast<CutlassTileConfig>(vec_configs[4]);
            search_config.split_k_style = static_cast<SplitKStyle>(vec_configs[5]);
            search_config.split_k_factor = vec_configs[6];
            search_config.stages = vec_configs[7];
            tuned_configs_from_file.push_back(search_config);
            VLOG(1)<<"tuned_configs_from_file: "<<search_config.total_rows<<","<<search_config.n<<","<<search_config.k<<","<<search_config.num_experts<<","<<static_cast<int>(search_config.tile_config)<<"," << static_cast<int>(search_config.split_k_style)<<","<<search_config.split_k_factor<<","<<search_config.stages;

          }
        } else {
          VLOG(1)<<"No tuned w4a8 gemm config.";
        }

    } else {
      FILE * fp;
      fp = fopen(config_file_path.c_str(), "r");
      if(fp) {
        VLOG(1)<<"Get tuned w4a8 moe gemm config from: "<<config_file_path;
        int tile_config, split_k_style, split_k_factor, stages;
        int total_rows_tmp, k_tmp, n_tmp, num_experts_tmp;
        while(1) {
            fscanf(fp, "%d, %d, %d, %d, %d, %d, %d, %d", &total_rows_tmp, &n_tmp, &k_tmp, &num_experts_tmp, &tile_config, &split_k_style, &split_k_factor, &stages);
            W4A8MoeGEMMConfig search_config;
            search_config.total_rows = total_rows_tmp;
            search_config.n = n_tmp;
            search_config.k = k_tmp;
            search_config.num_experts = num_experts_tmp;
            search_config.tile_config = static_cast<CutlassTileConfig>(tile_config);
            search_config.split_k_style = static_cast<SplitKStyle>(split_k_style);
            search_config.split_k_factor = split_k_factor;
            search_config.stages = stages;
            tuned_configs_from_file.push_back(search_config);
            VLOG(1)<<"tuned_configs_from_file: "<<total_rows_tmp<<","<<n_tmp<<","<<k_tmp<<","<<num_experts_tmp<<","<<tile_config<<"," << split_k_style<<","<<split_k_factor<<","<<stages;
            if (feof(fp))
            break;
        }
      } else if(FLAGS_cutlass_w4a8_moe_best_config=="") {
          VLOG(1)<<"No tuned w4a8 gemm config.";
      }
    }
  }
}

template <typename OutputType, typename IntAType, typename IntBType>
W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::~W4A8MoeGemmRunner() {
}



template <typename OutputType, typename IntAType, typename IntBType>
template <typename EpilogueTag>
void W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::dispatch_to_arch<EpilogueTag>(
    const IntAType* A,
    const IntBType* B,
    cutlass::epilogue::QuantMode quant_mode,
    const OutputType* col_scale,
    const OutputType* row_scale,
    const int32_t* nf4_look_up_table,
    OutputType* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows_in_ll_else_minus1,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    int num_experts,
    CutlassGemmConfig gemm_config,
    char* workspace_ptr,
    const size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy) {

  // only sm80 here
  dispatch_moe_gemm_to_cutlass<OutputType,
                              IntAType,
                              IntBType,
                              cutlass::arch::Sm80,
                              EpilogueTag>(A,
                                          B,
                                          quant_mode,
                                          col_scale,
                                          row_scale,
                                          nf4_look_up_table,
                                          C,
                                          total_rows_before_expert,
                                          total_rows_in_ll_else_minus1,
                                          total_rows,
                                          gemm_n,
                                          gemm_k,
                                          num_experts,
                                          gemm_config,
                                          workspace_ptr,
                                          workspace_bytes,
                                          multi_processor_count_,
                                          stream,
                                          occupancy);


}

template <typename OutputType, typename IntAType, typename IntBType>
template <typename EpilogueTag>
void W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::run_gemm<EpilogueTag>(
    const IntAType* A,
    const IntBType* B,
    cutlass::epilogue::QuantMode quant_mode,
    const OutputType* col_scale,
    const OutputType* row_scale,
    const int32_t* nf4_look_up_table,
    OutputType* C,
    int64_t* total_rows_before_expert,
    int64_t total_rows_in_ll_else_minus1,
    int64_t total_rows,
    int64_t gemm_n,
    int64_t gemm_k,
    char* workspace_ptr,
    const size_t workspace_bytes,
    int num_experts,
    cudaStream_t stream,
    CutlassGemmConfig gemm_config) {
  VLOG(1)<<__PRETTY_FUNCTION__;
  static constexpr bool is_weight_only = true; //todo(yuanxiaolan)
  bool is_weight_only_encoder = total_rows >= 512 ? true : false;

  VLOG(1) << "gemm_config tile_config"
          << static_cast<int>(gemm_config.tile_config);
  VLOG(1) << "gemm_config split_k_style"
          << static_cast<int>(gemm_config.split_k_style);
  VLOG(1) << "gemm_config split_k_factor " << gemm_config.split_k_factor;
  VLOG(1) << "gemm_config stages " << gemm_config.stages;

  if(gemm_config.tile_config != CutlassTileConfig::Undefined) {
    dispatch_to_arch<EpilogueTag>(A,
                                    B,
                                    quant_mode,
                                    col_scale,
                                    row_scale,
                                    nf4_look_up_table,
                                    C,
                                    total_rows_before_expert,
                                    total_rows_in_ll_else_minus1,
                                    total_rows,
                                    gemm_n,
                                    gemm_k,
                                    num_experts,
                                    gemm_config,
                                    workspace_ptr,
                                    workspace_bytes,
                                    stream);
    return;
  }

  std::vector<CutlassGemmConfig> candidate_configs =
      get_candidate_configs_nf4(80, is_weight_only, is_weight_only_encoder, false);
  std::vector<int> occupancies(candidate_configs.size());

  for (size_t ii = 0; ii < candidate_configs.size(); ++ii) {
    dispatch_to_arch<EpilogueTag>(A,
                                  B,
                                  quant_mode,
                                  col_scale,
                                  row_scale,
                                  nf4_look_up_table,
                                  C,
                                  total_rows_before_expert,
                                  total_rows_in_ll_else_minus1,
                                  total_rows,
                                  gemm_n,
                                  gemm_k,
                                  num_experts,
                                  candidate_configs[ii],
                                  workspace_ptr,
                                  workspace_bytes,
                                  stream,
                                  &occupancies[ii]);
  }

  int local_device{-1};
  int local_multi_processor_count{0};
  check_cuda_error(cudaGetDevice(&local_device));
  // sm_ = getSMVersion();
  check_cuda_error(cudaDeviceGetAttribute(
      &local_multi_processor_count, cudaDevAttrMultiProcessorCount, local_device));

  CutlassGemmConfig chosen_config =
      estimate_best_config_from_occupancies_w4a4(candidate_configs,
                                            occupancies,
                                            total_rows,
                                            gemm_n,
                                            gemm_k,
                                            num_experts,
                                            split_k_limit,
                                            workspace_bytes,
                                            local_multi_processor_count,
                                            is_weight_only);

  VLOG(1) << "chosen_config tile_config "
          << static_cast<int>(chosen_config.tile_config);
  VLOG(1) << "chosen_config split_k_style "
          << static_cast<int>(chosen_config.split_k_style);
  VLOG(1) << "chosen_config split_k_factor " << chosen_config.split_k_factor;
  VLOG(1) << "chosen_config stages " << chosen_config.stages;

  VLOG(1) << "total_rows  " << total_rows << "gemm_n  " << gemm_n << "gemm_k  "
          << gemm_k;


  dispatch_to_arch<EpilogueTag>(A,
                                B,
                                quant_mode,
                                col_scale,
                                row_scale,
                                nf4_look_up_table,
                                C,
                                total_rows_before_expert,
                                total_rows_in_ll_else_minus1,
                                total_rows,
                                gemm_n,
                                gemm_k,
                                num_experts,
                                chosen_config,
                                workspace_ptr,
                                workspace_bytes,
                                stream);
}

// template <typename OutputType, typename IntAType, typename IntBType>
// void W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::moe_gemm_bias_act(    const IntAType* A,
//     const IntBType* B,
//     QuantMode quant_mode,
//     const OutputType* col_scale,
//     const OutputType* row_scale,
//     const OutputType* biases,
//     const int32_t* nf4_look_up_table,
//     OutputType* C,
//     int64_t* total_rows_before_expert,
//     int64_t total_rows,
//     int64_t gemm_n,
//     int64_t gemm_k,
//     char* workspace_ptr,
//     const size_t workspace_bytes,
//     int num_experts,
//     cudaStream_t stream,
//     CutlassGemmConfig gemm_config) {
//   run_gemm<EpilogueOpNoBias>(A,
//                              B,
//                              quant_mode,
//                              col_scale,
//                              row_scale,
//                              nf4_look_up_table,
//                              C,
//                             total_rows_before_expert,
//                             total_rows,
//                             gemm_n,
//                             gemm_k,
//                             workspace_ptr,
//                              workspace_bytes,
//                              num_experts,
//                             stream,
//                             gemm_config);
// }

template <typename OutputType, typename IntAType, typename IntBType>
void W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::moe_gemm(
  const IntAType* A,
  const IntBType* B,
  cutlass::epilogue::QuantMode quant_mode,
  const OutputType* col_scale,
  const OutputType* row_scale,
  const int32_t* nf4_look_up_table,
  OutputType* C,
  int64_t* total_rows_before_expert,
  int64_t total_rows_in_ll_else_minus1,
  int64_t total_rows,
  int64_t gemm_n,
  int64_t gemm_k,
  char* workspace_ptr,
  const size_t workspace_bytes,
  int num_experts,
  cudaStream_t stream,
  CutlassGemmConfig gemm_config) {
  CutlassGemmConfig gemm_config_from_file_and_param = gemm_config;
  if(!tuned_configs_from_file.empty()){
    bool match=false;
    int best_total_rows, best_n, best_k, best_num_experts;
    int max_config_total_rows_in_file=0;
    W4A8MoeGEMMConfig max_total_rows_config;
    for(const auto& tuned_config:tuned_configs_from_file) {
        // choose the smallest config_m with config_m >=m
        if(tuned_config.total_rows <= total_rows && tuned_config.n==gemm_n && tuned_config.k==gemm_k && tuned_config.num_experts==num_experts) {
          best_total_rows=tuned_config.total_rows;
          best_n=tuned_config.n;
          best_k=tuned_config.k;
          best_num_experts=tuned_config.num_experts;
          gemm_config_from_file_and_param.tile_config = tuned_config.tile_config;
          gemm_config_from_file_and_param.split_k_style = tuned_config.split_k_style;
          gemm_config_from_file_and_param.split_k_factor = tuned_config.split_k_factor;
          gemm_config_from_file_and_param.stages = tuned_config.stages;
          match=true;
        }
        if(tuned_config.total_rows > max_config_total_rows_in_file && tuned_config.n==gemm_n && tuned_config.k==gemm_k && tuned_config.num_experts==num_experts){
            max_config_total_rows_in_file = tuned_config.total_rows;
            max_total_rows_config = tuned_config;
        }
    }
    if(!match){
      if (max_total_rows_config.n==gemm_n && max_total_rows_config.k==gemm_k && max_total_rows_config.num_experts==num_experts) {
        best_total_rows = max_config_total_rows_in_file;
        gemm_config_from_file_and_param.tile_config = max_total_rows_config.tile_config;
        gemm_config_from_file_and_param.split_k_style = max_total_rows_config.split_k_style;
        gemm_config_from_file_and_param.split_k_factor = max_total_rows_config.split_k_factor;
        gemm_config_from_file_and_param.stages = max_total_rows_config.stages;
      }
    }
    VLOG(1) <<"W4A8 moe gemm "
            <<"total_rows: "<<total_rows<<" n: "<<gemm_n<<" k: "<<gemm_k
            <<"Using gemm config from config file: config_total_rows: "<< best_total_rows<<" config_n: "<< best_n << " config_k: "<< best_k
            <<"tile_config: "<<static_cast<int>(gemm_config_from_file_and_param.tile_config)
            <<"split_k_style: "<<static_cast<int>(gemm_config_from_file_and_param.split_k_style)
            <<"split_k_factor: "<<static_cast<int>(gemm_config_from_file_and_param.split_k_factor)
            <<"stages: "<<static_cast<int>(gemm_config_from_file_and_param.stages);
  } else {
    VLOG(1) << "tuned_configs_from_file is empty, use W4A8 gemm config in param";
  }
  run_gemm<EpilogueOpNoBias>(A,
                             B,
                             quant_mode,
                             col_scale,
                             row_scale,
                             nf4_look_up_table,
                             C,
                             total_rows_before_expert,
                             total_rows_in_ll_else_minus1,
                             total_rows,
                             gemm_n,
                             gemm_k,
                             workspace_ptr,
                             workspace_bytes,
                             num_experts,
                             stream,
                             gemm_config_from_file_and_param);
}

template <typename OutputType, typename IntAType, typename IntBType>
std::vector<typename W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::W4A8MoeGEMMConfig> W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::tuned_configs_from_file = {};

template <typename OutputType, typename IntAType, typename IntBType>
int W4A8MoeGemmRunner<OutputType, IntAType, IntBType>::getWorkspaceSize(
    const int m, const int n, const int k) {
  // These are the min tile sizes for each config, which would launch the
  // maximum number of blocks
  const int max_grid_m = (m + 31) / 32;
  const int max_grid_n = (n + 127) / 128;
  // We need 4 bytes per block in the worst case. We launch split_k_limit in z
  // dim.
  return max_grid_m * max_grid_n * split_k_limit * 4;
}


template class W4A8MoeGemmRunner<half, int8_t, cutlass::uint4b_t>;
template class W4A8MoeGemmRunner<__nv_bfloat16, int8_t, cutlass::uint4b_t>;
