// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

template <int kStages, class GemmType, class OutputType, class SmemLayoutA,
          class SmemLayoutE,
          class SmemLayoutB, class SmemLayoutC>
struct SharedStorage {
    union {
        struct {
            cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutA>> smem_a;
            cute::array_aligned<uint32_t, cute::cosize_v<SmemLayoutE>> smem_e;
            cute::array_aligned<GemmType, cute::cosize_v<SmemLayoutB>> smem_b;
        };
        cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutC>> smem_c;
    };

  struct {
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline;
  };
};

template<int kBlockM_, int kBlockN_, int kBlockK_,
        int kNWarps_, int kStages_,
        int kTiles_, int M_,
        int TokenPackSize_,
        int TAIL_N_ = 0,
        int kClusterM_ = 1,
        typename elem_type=cutlass::float_e4m3_t,
        typename OutputType = cutlass::bfloat16_t>
struct Kernel_traits {
    using Element = elem_type;
    using ElementAccum = float;
    using ElementOutput = OutputType;
    static_assert(cutlass::sizeof_bits_v<Element> == 8);

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
    static constexpr int NumMmaThreads = kNThreads - NumProducerThreads;

    static_assert(kNWarps_ == 12);

    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kTiles = kTiles_;
    static constexpr int TokenPackSize = TokenPackSize_;
    static constexpr int TAIL_N = TAIL_N_;
    static constexpr int M = M_;

    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kBlockK>>;
    using TileShape_MNK_TAIL = Shape<Int<kBlockM>, Int<TAIL_N>, Int<kBlockK>>;
    static constexpr int kClusterM = kClusterM_;
    using ClusterShape_MNK = Shape<Int<kClusterM>, _1, _1>;

    static constexpr int kStages = kStages_;
    static_assert(kStages > 1);

    using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;

    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
        AtomLayoutMNK{}));

    using Mma = decltype(cute::GMMA::ss_op_selector_sparse<Element, Element, ElementAccum, TileShape_MNK>());

    using Mma_TAIL = decltype(cute::GMMA::ss_op_selector_sparse<Element, Element, ElementAccum, TileShape_MNK_TAIL>());

    using SmemLayoutAtomA = decltype(
        cutlass::gemm::collective::detail::rs_smem_selector<
            GMMA::Major::K, Element, Int<kBlockM / 2>, Int<kBlockK>>());

    using SmemLayoutA = decltype(
        tile_to_shape(SmemLayoutAtomA{},
            make_shape(Int<kBlockM / 2>{}, Int<kBlockK>{}, Int<kStages>{})));

    using SmemLayoutAtomB = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
            GMMA::Major::K, Element, decltype(cute::get<1>(TileShape_MNK{})),
            decltype(cute::get<2>(TileShape_MNK{}))>());

    using SmemLayoutB = decltype(
        tile_to_shape(SmemLayoutAtomB{},
            make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));

    using SmemLayoutAtomB_TAIL = decltype(
        cutlass::gemm::collective::detail::rs_smem_selector<
            GMMA::Major::K, Element, decltype(cute::get<1>(TileShape_MNK_TAIL{})),
            decltype(cute::get<2>(TileShape_MNK_TAIL{}))>());

    using SmemLayoutB_TAIL = decltype(
        tile_to_shape(SmemLayoutAtomB_TAIL{},
            make_shape(
                shape<1>(TileShape_MNK_TAIL{}),
                shape<2>(TileShape_MNK_TAIL{}),
                Int<kStages>{})
            ));
    using SmemLayoutAtomC = decltype(
        cutlass::gemm::collective::detail::ss_smem_selector<
        GMMA::Major::K, ElementOutput,
        decltype(cute::get<0>(TileShape_MNK{})),
        decltype(cute::get<1>(TileShape_MNK{}))>());

    using SmemLayoutC = decltype(tile_to_shape(SmemLayoutAtomC{}, select<0, 1>(TileShape_MNK{})));

    using SmemLayoutE = Layout<Shape<Int<NumMmaThreads>, Int<kBlockK / 64>, Int<kStages>>>;

    using SharedStorage = SharedStorage<
        kStages, Element, ElementOutput, SmemLayoutA, SmemLayoutE, SmemLayoutB, SmemLayoutC>;

    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;

    static constexpr int kNumVecElem = ceil_div(128, sizeof_bits_v<OutputType>);
    static constexpr int kNumThreadsPerRow = kBlockN / kNumVecElem;
    static constexpr int kNumRows = NumMmaThreads / kNumThreadsPerRow;
    using TiledCopyCAtom = cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, OutputType>;
    using TiledCopyCThrLayout = decltype(cute::make_layout(
        cute::make_shape(Int<kNumRows>{}, Int<kNumThreadsPerRow>{}),
        LayoutRight{}));
    using TiledCopyCValLayout = decltype(cute::make_layout(
        cute::make_shape(_1{}, Int<kNumVecElem>{}),
        LayoutRight{}));
    using TiledCopyC = decltype(make_tiled_copy(
        TiledCopyCAtom{},
        TiledCopyCThrLayout{}, // Thr layout
        TiledCopyCValLayout{} // Val layout
    ));
};
