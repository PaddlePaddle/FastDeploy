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

#pragma once

#include "cutlass/gemm_coord.h"
#include "cutlass/trace.h"

#include "cutlass_extensions/gemm/threadblock/wint2x_unzip.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

template <typename ElementT, typename ScaleElementT, int Rows, int Columns,
          int Stages, int NumThreads, WintQuantMethod Method>
struct TileDequanter {
  using WeightQuantTraits = WintQuantTraits<ElementT, Method>;
  using MmaElementT = typename WeightQuantTraits::MmaWeightType;
  using QuantArguments = typename WeightQuantTraits::Arguments;

  using UnzipAndDequantFunctor =
      UnzipAndDequantFunctor<MmaElementT, Method, Rows, Columns, NumThreads>;

  static constexpr bool kUseSharedMemory = true;

  static constexpr int kRows = Rows;
  static constexpr int kColumns = Columns;
  static constexpr int kStages = Stages;

  MmaElementT *out_smem_ptr{nullptr};

  char *pointer{nullptr};
  int64_t ldm{0};
  cutlass::MatrixCoord tb_offset;
  cutlass::MatrixCoord extent;

  ScaleElementT *super_scale_ptr{nullptr};
  cutlass::MatrixCoord tb_offset_scale;

  QuantArguments quant_args;

  int64_t block_start_rows[kStages];
  bool need_preload{true};
  UnzipAndDequantFunctor unzip_functor;

  CUTLASS_DEVICE
  TileDequanter(MmaElementT *out_smem_ptr, char *pointer, int64_t ldm,
                const cutlass::MatrixCoord &extent,
                const cutlass::MatrixCoord &tb_offset,
                ScaleElementT *super_scale_ptr,
                const cutlass::MatrixCoord &tb_offset_scale,
                const QuantArguments &quant_args)
      : out_smem_ptr(out_smem_ptr), pointer(pointer), ldm(ldm), extent(extent),
        tb_offset(tb_offset), super_scale_ptr(super_scale_ptr),
        tb_offset_scale(tb_offset_scale), quant_args(quant_args) {}

  CUTLASS_DEVICE
  MmaElementT *GetOutPtr() { return out_smem_ptr; }

  CUTLASS_DEVICE
  void AddTileOffset(const cutlass::MatrixCoord &tile_offset) {
    tb_offset.row() += tile_offset.row() * kRows;
    tb_offset.column() += tile_offset.column() * kColumns;
    tb_offset_scale.column() += tile_offset.column() * kColumns;
  }

  CUTLASS_DEVICE
  void Load(uint8_t *zipped_smem_ptr, uint8_t *column_wise_smem_ptr, int stage) {
    int zipped_row = WeightQuantTraits::CaclPackedDim(tb_offset.row());
    if (tb_offset.row() >= extent.row() ||
        tb_offset.column() >= extent.column()) {
      return;
    }

    block_start_rows[stage % kStages] = tb_offset.row();

    using ZippedT = typename WeightQuantTraits::WeightType;
    ZippedT *in_ptr = reinterpret_cast<ZippedT *>(pointer) + zipped_row * ldm +
                      tb_offset.column();
    ScaleElementT *scale_ptr = super_scale_ptr + tb_offset_scale.column();

    if constexpr (Method == WintQuantMethod::kWeightOnlyInt2) {
      const uint8_t *local_scale_ptr = quant_args.local_scale_ptr +
                                       (tb_offset.row() / 128) * ldm +
                                       tb_offset_scale.column();
      const float *code_scale_ptr =
          quant_args.code_scale_ptr + tb_offset_scale.column();
      const float *code_zp_ptr =
          quant_args.code_zp_ptr + tb_offset_scale.column();

      typename UnzipAndDequantFunctor::Arguments args(zipped_smem_ptr, column_wise_smem_ptr);
      unzip_functor.LoadAsync(in_ptr, local_scale_ptr, code_scale_ptr, code_zp_ptr,
                              scale_ptr, &args, ldm, need_preload);
      need_preload = false;
    } else {
      // CUTLASS_TRACE_DEVICE("Not Supported!");
    }
  }

  CUTLASS_DEVICE
  void UnpackAndDequant(uint8_t *zipped_smem_ptr, uint8_t *column_wise_smem_ptr, int stage) {
    int64_t block_start_row = block_start_rows[stage % kStages];
    if (block_start_row >= extent.row()) {
      return;
    }

    if constexpr (Method == WintQuantMethod::kWeightOnlyInt2) {
      typename UnzipAndDequantFunctor::Arguments args(zipped_smem_ptr, column_wise_smem_ptr);
      unzip_functor.ComputeVectorized(args, out_smem_ptr, block_start_row);
    } else {
      // CUTLASS_TRACE_DEVICE("Not Supported!");
    }
  }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
