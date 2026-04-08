/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include "common/cudaUtils.h"

#include "cutlass/gemm/kernel/default_gemm.h"

#include "cutlass_extensions/compute_occupancy.h"
#include "cutlass_extensions/epilogue_helpers.h"
#include "cutlass_extensions/gemm_configs.h"

#include "cutlass_extensions/gemm/device/gemm_universal_base_compat.h"
#include "cutlass_extensions/gemm/kernel/default_fpA_intB_traits.h"
#include "cutlass_extensions/gemm/kernel/fpA_intB_gemm.h"
#include "cutlass_extensions/gemm/threadblock/default_mma.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

#include "cutlass_kernels/cutlass_heuristic.h"
#include "cutlass_kernels/cutlass_type_conversion.h"
#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"

namespace kernels {
namespace cutlass_kernels {

template <
        typename ActivationType,
        typename WeightType,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType,
        typename arch,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename EpilogueTag,
        typename ThreadblockShape,
        typename WarpShape,
        int Stages>
void generic_mixed_gemm_kernelLauncher(
        ActivationType const* A,
        WeightType const* B,
        ScaleZeroType const* weight_scales,
        ScaleZeroType const* weight_zero_points,
        BiasType const* biases,
        float const alpha,
        OutputType* C,
        int m,
        int n,
        int k,
        int const group_size,
        cutlass_extensions::CutlassGemmConfig gemm_config,
        void* workspace,
        size_t workspace_bytes,
        cudaStream_t stream,
        int* occupancy = nullptr) {
    // The cutlass type for the input elements. This is needed to convert to cutlass::half_t if
    // necessary.
    using CutlassActivationType = typename CudaToCutlassTypeAdapter<ActivationType>::type;
    using CutlassWeightType = typename CudaToCutlassTypeAdapter<WeightType>::type;
    using CutlassScaleZeroType = typename CudaToCutlassTypeAdapter<ScaleZeroType>::type;
    using CutlassBiasType = typename CudaToCutlassTypeAdapter<BiasType>::type;
    using CutlassOutputType = typename CudaToCutlassTypeAdapter<OutputType>::type;

    // We need separate config for each architecture since we will target different tensorcore
    // instructions. For float, we do not target TCs.
    using MixedGemmArchTraits = cutlass::gemm::kernel::
            MixedGemmArchTraits<CutlassActivationType, CutlassWeightType, arch>;
    using ElementAccumulator = typename MixedGemmArchTraits::AccType;

    constexpr int ElementsPerAccessC = 128 / cutlass::sizeof_bits<CutlassOutputType>::value;
    using EpilogueOp = typename cutlass_extensions::
            Epilogue<CutlassOutputType, ElementsPerAccessC, ElementAccumulator, EpilogueTag>::Op;

    using Operator = typename MixedGemmArchTraits::Operator;
    using TaggedOperator = typename cutlass::arch::TagOperator<Operator, QuantOp>::TaggedOperator;

    using GemmKernel_ = typename cutlass::gemm::kernel::DefaultGemm<
            CutlassActivationType,
            cutlass::layout::RowMajor,
            MixedGemmArchTraits::ElementsPerAccessA,
            CutlassWeightType,
            typename MixedGemmArchTraits::LayoutB,
            MixedGemmArchTraits::ElementsPerAccessB,
            CutlassOutputType,
            cutlass::layout::RowMajor,
            ElementAccumulator,
            cutlass::arch::OpClassTensorOp,
            arch,
            ThreadblockShape,
            WarpShape,
            typename MixedGemmArchTraits::InstructionShape,
            EpilogueOp,
            typename cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
            Stages,
            true,
            TaggedOperator>::GemmKernel;

    using GemmKernel = cutlass::gemm::kernel::GemmFpAIntB<
            typename GemmKernel_::Mma,
            typename GemmKernel_::Epilogue,
            typename GemmKernel_::ThreadblockSwizzle,
            arch,  // Ensure top level arch is used for dispatch
            GemmKernel_::kSplitKSerial>;

    if (occupancy != nullptr) {
        *occupancy = cutlass_extensions::compute_occupancy_for_kernel<GemmKernel>();
        return;
    }

    using Gemm = cutlass::gemm::device::GemmUniversalBaseCompat<GemmKernel>;

    int const ldb =
            cutlass::platform::
                    is_same<cutlass::layout::RowMajor, typename MixedGemmArchTraits::LayoutB>::value
            ? n
            : k * GemmKernel::kInterleave;

    if (weight_scales == nullptr) {
        throw std::runtime_error("Weight scales must always be set to a non-null value.");
    }

    if constexpr (cutlass::isFinegrained(QuantOp)) {
        if constexpr (cutlass::platform::is_same<CutlassActivationType, cutlass::float_e4m3_t>::
                              value) {
            if (group_size != 128) {
                throw std::runtime_error(
                        "Only group size 128 supported for fine grained W4A(fp)8 kernels.");
            }
        }
        if (group_size != 64 && group_size != 128) {
            throw std::runtime_error(
                    "Only group size 64 and 128 supported for fine grained kernels.");
        }

        if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY) {
            if (weight_zero_points != nullptr) {
                throw std::runtime_error(
                        "Weight zero pointer must be a nullptr for scale only fine grained");
            }
        } else if constexpr (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
            if (weight_zero_points == nullptr) {
                throw std::runtime_error(
                        "Weight zero pointer must be valid for scale and bias fine grained");
            }
        }
    } else {
        if (group_size != k) {
            throw std::runtime_error("Invalid group size for per column scaling kernels.");
        }

        if (weight_zero_points != nullptr) {
            throw std::runtime_error(
                    "Weight zero-points must be null when running per column scaling");
        }
    }

    int const ld_scale_zero = cutlass::isFinegrained(QuantOp) ? n : 0;
    ElementAccumulator output_op_beta =
            (biases == nullptr) ? ElementAccumulator(0.f) : ElementAccumulator(1.f);
    typename Gemm::Arguments args(
            {m, n, k},
            group_size,
            {reinterpret_cast<CutlassActivationType*>(const_cast<ActivationType*>(A)), k},
            {reinterpret_cast<CutlassWeightType*>(const_cast<WeightType*>(B)), ldb},
            {reinterpret_cast<CutlassScaleZeroType*>(const_cast<ScaleZeroType*>(weight_scales)),
             ld_scale_zero},
            {reinterpret_cast<CutlassScaleZeroType*>(
                     const_cast<ScaleZeroType*>(weight_zero_points)),
             ld_scale_zero},
            {reinterpret_cast<CutlassBiasType*>(const_cast<BiasType*>(biases)), 0},
            {reinterpret_cast<CutlassOutputType*>(C), n},
            gemm_config.split_k_factor,
            {ElementAccumulator(alpha), output_op_beta});

    // This assertion is enabled because because for the column interleaved layout, K MUST be a
    // multiple of threadblockK. The reason for this is that the default pitchlinear iterators are
    // used to handle walking over the interleaved matrix. The way masking in handled in these do
    // not map to the interleaved layout. We need to write our own predicated iterator in order to
    // relax this limitation.
    if (GemmKernel::kInterleave > 1 &&
        ((k % MixedGemmArchTraits::ThreadblockK) ||
         ((k / gemm_config.split_k_factor) % MixedGemmArchTraits::ThreadblockK))) {
        throw std::runtime_error(
                "Assertion: k[" + std::to_string(k) + "] must be multiple of threadblockK[" +
                std::to_string(MixedGemmArchTraits::ThreadblockK) + "]");
    }

    Gemm gemm;

    if (gemm.get_workspace_size(args) > workspace_bytes) {
        std::cerr << "Requested split-k but workspace size insufficient. Falling back to "
                     "non-split-k implementation."
                  << std::endl;
        // If requested split-k factor will require more workspace bytes, revert to standard gemm.
        args.batch_count = 1;
    }

    auto can_implement = gemm.can_implement(args);
    if (can_implement != cutlass::Status::kSuccess) {
        std::string err_msg = "fp8_int4 cutlass kernel will fail for params. Error: " +
                std::string(cutlassGetStatusString(can_implement));
        throw std::runtime_error("[fp8_int4 Runner] " + err_msg);
    }

    auto init_status = gemm.initialize(args, workspace, stream);
    if (init_status != cutlass::Status::kSuccess) {
        std::string err_msg = "Failed to initialize cutlass fp8_int4 gemm. Error: " +
                std::string(cutlassGetStatusString(init_status));
        throw std::runtime_error("[fp8_int4 Runner] " + err_msg);
    }

    auto run_status = gemm.run(stream);
    if (run_status != cutlass::Status::kSuccess) {
        std::string err_msg = "Failed to run cutlass fp8_int4 gemm. Error: " +
                std::string(cutlassGetStatusString(run_status));
        throw std::runtime_error("[fp8_int4 Runner] " + err_msg);
    }
}

template <
        typename ActivationType,
        typename WeightType,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType,
        typename arch,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename EpilogueTag,
        typename ThreadblockShape,
        typename WarpShape>
void dispatch_gemm_config(
        ActivationType const* A,
        WeightType const* B,
        ScaleZeroType const* weight_scales,
        ScaleZeroType const* weight_zero_points,
        BiasType const* biases,
        float const alpha,
        OutputType* C,
        int m,
        int n,
        int k,
        int const group_size,
        cutlass_extensions::CutlassGemmConfig gemm_config,
        void* workspace,
        size_t workspace_bytes,
        cudaStream_t stream,
        int* occupancy = nullptr) {
    switch (gemm_config.stages) {
    case 2:
        throw std::runtime_error(
                "[filter_and_run_mixed_gemm] Cutlass fp8_int4 gemm not supported for arch " +
                std::to_string(arch::kMinComputeCapability) + " with stages set to 2");
        break;
    case 3:
        generic_mixed_gemm_kernelLauncher<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                ThreadblockShape,
                WarpShape,
                3>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    case 4:
        generic_mixed_gemm_kernelLauncher<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                ThreadblockShape,
                WarpShape,
                4>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    default:
        std::string err_msg = "dispatch_gemm_config does not support stages " +
                std::to_string(gemm_config.stages);
        throw std::runtime_error("[dispatch_gemm_config] " + err_msg);
        break;
    }
}

template <
        typename ActivationType,
        typename WeightType,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType,
        typename arch,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename EpilogueTag>
void dispatch_gemm_to_cutlass(
        ActivationType const* A,
        WeightType const* B,
        ScaleZeroType const* weight_scales,
        ScaleZeroType const* weight_zero_points,
        BiasType const* biases,
        float const alpha,
        OutputType* C,
        int m,
        int n,
        int k,
        int const group_size,
        void* workspace,
        size_t workspace_bytes,
        cutlass_extensions::CutlassGemmConfig gemm_config,
        cudaStream_t stream,
        int* occupancy = nullptr) {
    // Note that SIMT configs are omitted here since they are not supported for fp8_int4.
    // We also only instantiate configs here where threadblockShapeM == warpShapeM since those
    // usually perform the best for mixed type gemms.
    constexpr int tile_shape_k = 128 * 8 / cutlass::sizeof_bits<ActivationType>::value;
    switch (gemm_config.tile_config) {
    case cutlass_extensions::CutlassTileConfig::CtaShape16x128x64_WarpShape16x32x64:
        dispatch_gemm_config<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                cutlass::gemm::GemmShape<16, 128, tile_shape_k>,
                cutlass::gemm::GemmShape<16, 32, tile_shape_k>>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape16x256x64_WarpShape16x64x64:
        dispatch_gemm_config<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                cutlass::gemm::GemmShape<16, 256, tile_shape_k>,
                cutlass::gemm::GemmShape<16, 64, tile_shape_k>>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape32x128x64_WarpShape32x32x64:
        dispatch_gemm_config<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                cutlass::gemm::GemmShape<32, 128, tile_shape_k>,
                cutlass::gemm::GemmShape<32, 32, tile_shape_k>>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64:
        dispatch_gemm_config<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                cutlass::gemm::GemmShape<64, 128, tile_shape_k>,
                cutlass::gemm::GemmShape<64, 32, tile_shape_k>>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::CtaShape128x128x64_WarpShape128x32x64:
        dispatch_gemm_config<
                ActivationType,
                WeightType,
                ScaleZeroType,
                BiasType,
                OutputType,
                arch,
                QuantOp,
                EpilogueTag,
                cutlass::gemm::GemmShape<128, 128, tile_shape_k>,
                cutlass::gemm::GemmShape<128, 32, tile_shape_k>>(
                A,
                B,
                weight_scales,
                weight_zero_points,
                biases,
                alpha,
                C,
                m,
                n,
                k,
                group_size,
                gemm_config,
                workspace,
                workspace_bytes,
                stream,
                occupancy);
        break;
    case cutlass_extensions::CutlassTileConfig::Undefined:
        throw std::runtime_error("[fp8_int4][dispatch_gemm_to_cutlass] gemm config undefined.");
        break;
    case cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic:
        throw std::runtime_error(
                "[fp8_int4][dispatch_gemm_to_cutlass] gemm config should have already been set by "
                "heuristic.");
        break;
    default:
        printf("gemm_config.tile_config: %d", int(gemm_config.tile_config));
        throw std::runtime_error(
                "[fp8_int4][dispatch_gemm_to_cutlass] Config is invalid for mixed type GEMM.");
        break;
    }
}

template <
        typename ActivationType,
        typename WeightType,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType>
CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::
        CutlassFpAIntBGemmRunner() {
    // printf(__PRETTY_FUNCTION__);
    int device{-1};
    PADDLE_ENFORCE_GPU_SUCCESS(cudaGetDevice(&device));
    sm_ = common::getSMVersion();
    PADDLE_ENFORCE_GPU_SUCCESS(cudaDeviceGetAttribute(
            &multi_processor_count_, cudaDevAttrMultiProcessorCount, device));
}

template <
        typename ActivationType,
        typename WeightType,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType>
CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::
        ~CutlassFpAIntBGemmRunner() {
    // printf(__PRETTY_FUNCTION__);
}

template <
        typename ActivationType,
        typename WeightType,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType>
template <typename EpilogueTag>
void CutlassFpAIntBGemmRunner<
        ActivationType,
        WeightType,
        QuantOp,
        ScaleZeroType,
        BiasType,
        OutputType>::
        dispatch_to_arch<EpilogueTag>(
                ActivationType const* A,
                WeightType const* B,
                ScaleZeroType const* weight_scales,
                ScaleZeroType const* weight_zero_points,
                BiasType const* biases,
                float const alpha,
                OutputType* C,
                int m,
                int n,
                int k,
                int const group_size,
                cutlass_extensions::CutlassGemmConfig gemm_config,
                void* workspace_ptr,
                const size_t workspace_bytes,
                cudaStream_t stream,
                int* occupancy) {
    dispatch_gemm_to_cutlass<
            ActivationType,
            WeightType,
            ScaleZeroType,
            BiasType,
            OutputType,
            cutlass::arch::Sm89,
            QuantOp,
            EpilogueTag>(
            A,
            B,
            weight_scales,
            weight_zero_points,
            biases,
            alpha,
            C,
            m,
            n,
            k,
            group_size,
            workspace_ptr,
            workspace_bytes,
            gemm_config,
            stream,
            occupancy);
}

template <
        typename ActivationType,
        typename WeightType,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType>
void CutlassFpAIntBGemmRunner<
        ActivationType,
        WeightType,
        QuantOp,
        ScaleZeroType,
        BiasType,
        OutputType>::
        gemm(void const* A,
             void const* B,
             void const* weight_scales,
             void const* weight_zero_points,
             void const* biases,
             float const alpha,
             void* C,
             int m,
             int n,
             int k,
             int const group_size,
             cutlass_extensions::CutlassGemmConfig gemmConfig,
             void* workspace_ptr,
             const size_t workspace_bytes,
             cudaStream_t stream) {
    // printf(__PRETTY_FUNCTION__);
    if (gemmConfig.tile_config == cutlass_extensions::CutlassTileConfig::ChooseWithHeuristic) {
        std::vector<cutlass_extensions::CutlassGemmConfig> configs = getConfigs(k);
        std::vector<int> occupancies(configs.size());
        for (size_t i = 0; i < configs.size(); ++i) {
            dispatch_to_arch<cutlass_extensions::EpilogueOpBias>(
                    (ActivationType const*)A,
                    (WeightType const*)B,
                    (ScaleZeroType const*)weight_scales,
                    (ScaleZeroType const*)weight_zero_points,
                    (BiasType const*)biases,
                    alpha,
                    (OutputType*)C,
                    m,
                    n,
                    k,
                    group_size,
                    configs[i],
                    workspace_ptr,
                    workspace_bytes,
                    stream,
                    &occupancies[i]);
        }
        auto best_config = estimate_best_config_from_occupancies(
                configs,
                occupancies,
                m,
                n,
                k,
                1,
                SPLIT_K_LIMIT,
                workspace_bytes,
                multi_processor_count_,
                true);
        dispatch_to_arch<cutlass_extensions::EpilogueOpBias>(
                (ActivationType const*)A,
                (WeightType const*)B,
                (ScaleZeroType const*)weight_scales,
                (ScaleZeroType const*)weight_zero_points,
                (BiasType const*)biases,
                alpha,
                (OutputType*)C,
                m,
                n,
                k,
                group_size,
                best_config,
                workspace_ptr,
                workspace_bytes,
                stream,
                nullptr);
    } else {
        dispatch_to_arch<cutlass_extensions::EpilogueOpBias>(
                (ActivationType const*)A,
                (WeightType const*)B,
                (ScaleZeroType const*)weight_scales,
                (ScaleZeroType const*)weight_zero_points,
                (BiasType const*)biases,
                alpha,
                (OutputType*)C,
                m,
                n,
                k,
                group_size,
                gemmConfig,
                workspace_ptr,
                workspace_bytes,
                stream,
                nullptr);
    }
}

template <
        typename ActivationType,
        typename WeightType,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType>
std::vector<cutlass_extensions::CutlassGemmConfig>
CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::
        getConfigs(int k) const {
    // printf(__PRETTY_FUNCTION__);
    cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam config_type_param =
            cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER;
    config_type_param =
            static_cast<cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam>(
                    config_type_param |
                    cutlass_extensions::CutlassGemmConfig::CandidateConfigTypeParam::WEIGHT_ONLY);
    std::vector<cutlass_extensions::CutlassGemmConfig> candidateConfigs =
            get_candidate_configs(sm_, SPLIT_K_LIMIT, config_type_param);

    // filter configs that are not supported on sm89
    std::vector<cutlass_extensions::CutlassGemmConfig> rets;
    for (auto config : candidateConfigs) {
        // sm89 doesn't support stages 2
        if (config.stages == 2) {
            continue;
        }

        if (config.stages >= 5) {
            continue;
        }
        if (config.split_k_style != cutlass_extensions::SplitKStyle::NO_SPLIT_K) {
            int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
            if (k_size % 128) {
                continue;
            }
        }
        rets.push_back(config);
    }
    return rets;
}

template <
        typename ActivationType,
        typename WeightType,
        cutlass::WeightOnlyQuantOp QuantOp,
        typename ScaleZeroType,
        typename BiasType,
        typename OutputType>
size_t
CutlassFpAIntBGemmRunner<ActivationType, WeightType, QuantOp, ScaleZeroType, BiasType, OutputType>::
        getWorkspaceSize(int const m, int const n, int const k) {
    // printf(__PRETTY_FUNCTION__);
    // These are the min tile sizes for each config, which would launch the maximum number of blocks
    int const max_grid_m = cutlass::ceil_div(m, MIN_M_TILE);
    int const max_grid_n = cutlass::ceil_div(n, MIN_N_TILE);
    // We need 4 bytes per block in the worst case. We launch split_k_limit in z dim.
    return static_cast<size_t>(max_grid_m * max_grid_n * SPLIT_K_LIMIT * 4);
}

}  // namespace cutlass_kernels
}  // namespace kernels
