// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

using namespace cute;

struct Flash_mask_params {
  void *__restrict__ q_ptr;
  void *__restrict__ k_ptr;
  void *__restrict__ v_ptr;
  void *__restrict__ o_ptr;
  int *__restrict__ cu_seq_q;
  int *__restrict__ cu_seq_k;
  int *__restrict__ mask;
  int *seq_len_encoder;
  int head_num;
  int kv_head_num;
  int max_seq_len_q;
  int max_seq_len_k;
  int batch_size;
  int gqa_group_size;
  float scale_softmax_log2;
};

template <int kStages,
          class Gemm1Type,
          class Gemm2Type,
          class OutputType,
          class SmemLayoutQ,
          class SmemLayoutK,
          class SmemLayoutV,
          class SmemLayoutO>
struct SharedStorageQKVO {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  union {
    cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
    cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
  };
  struct {
    cutlass::arch::ClusterTransactionBarrier barrier_Q;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
    typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
  };
};

template <int kHeadDim_,
          int kBlockM_,
          int kBlockN_,
          int kNWarps_,
          int kStages_,
          bool NeedMask_,
          typename elem_type = cutlass::half_t,
          typename out_type = cutlass::half_t>
struct Flash_mask_kernel_traits {
  using Element = elem_type;
  using output_type = out_type;
  using ElementAccum = float;
  using index_t = int32_t;

  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static_assert(kHeadDim % 32 == 0);
  using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
  using ClusterShape_MNK = Shape<Int<1>, Int<1>, Int<1>>;
  static constexpr int kStages = kStages_;
  static constexpr int NeedMask = NeedMask_;

  using AtomLayoutMNK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::
          ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
      AtomLayoutMNK{}));
  using TiledMma1 = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<Element,
                                 Element,
                                 ElementAccum,
                                 decltype(select<0, 2, 1>(TileShape_MNK{})),
                                 GMMA::Major::K,
                                 GMMA::Major::MN>(),
      AtomLayoutMNK{}));

  using SmemLayoutAtomQ =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutQ =
      decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));

  using SmemLayoutAtomK =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutK =
      decltype(tile_to_shape(SmemLayoutAtomK{},
                             make_shape(shape<1>(TileShape_MNK{}),
                                        shape<2>(TileShape_MNK{}),
                                        Int<kStages>{})));

  using SmemLayoutAtomV =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               Element,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutV =
      decltype(tile_to_shape(SmemLayoutAtomV{},
                             make_shape(shape<1>(TileShape_MNK{}),
                                        shape<2>(TileShape_MNK{}),
                                        Int<kStages>{})));

  using SmemLayoutAtomO =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               output_type,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutO =
      decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{})));

  using SmemCopyAtomQ = Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
  using SmemCopyAtomO = Copy_Atom<cute::SM90_U32x4_STSM_N, output_type>;

  using SharedStorage = SharedStorageQKVO<kStages,
                                          Element,
                                          Element,
                                          output_type,
                                          SmemLayoutQ,
                                          SmemLayoutK,
                                          SmemLayoutV,
                                          SmemLayoutO>;

  static constexpr int NumProducerThreads = cutlass::NumThreadsPerWarpGroup;
  static constexpr int NumMmaThreads = kNThreads - NumProducerThreads;
  static constexpr int kNumVecElem = ceil_div(128, sizeof_bits_v<output_type>);
  static constexpr int kNumThreadsPerRow = kHeadDim / kNumVecElem;
  static_assert(NumMmaThreads % kNumThreadsPerRow == 0);
  static constexpr int kNumRows = NumMmaThreads / kNumThreadsPerRow;
  using TiledCopyOAtom =
      cute::Copy_Atom<cute::UniversalCopy<cutlass::uint128_t>, output_type>;
  using TiledCopyOThrLayout = decltype(cute::make_layout(
      cute::make_shape(Int<kNumRows>{}, Int<kNumThreadsPerRow>{}),
      LayoutRight{}));
  using TiledCopyOValLayout = decltype(cute::make_layout(
      cute::make_shape(_1{}, Int<kNumVecElem>{}), LayoutRight{}));
  using GmemTiledCopyO = decltype(make_tiled_copy(
      TiledCopyOAtom{}, TiledCopyOThrLayout{}, TiledCopyOValLayout{}));

  using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
  using PipelineState = typename cutlass::PipelineState<kStages>;
};
