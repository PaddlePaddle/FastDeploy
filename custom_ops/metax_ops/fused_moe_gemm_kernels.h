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

#include "maca_version.h"
#include "mctlass/numeric_conversion.h"
#include "mctlassEx/mctlassEx.h"

namespace phi {

template <typename T>
struct mctlassExDataTraits;

template <>
struct mctlassExDataTraits<maca_bfloat16> {
  static constexpr mctlassExDataType type =
      mctlassExDataType::MCTLASS_EX_DATATYPE_BF16;
};

template <>
struct mctlassExDataTraits<int8_t> {
  static constexpr mctlassExDataType type =
      mctlassExDataType::MCTLASS_EX_DATATYPE_INT8;
};

template <typename T, typename WeightType>
class McMoeGemmRunner {
 public:
  McMoeGemmRunner() {}

  void mc_grouped_gemm_basic_kernel(const T* ptrA,
                                    mctlassExOrder_t majorA,
                                    const WeightType* ptrB,
                                    mctlassExOrder_t majorB,
                                    const T* ptrScale,
                                    const T* ptrBias,
                                    T* ptrC,
                                    mctlassExOrder_t majorC,
                                    const int* ptrSegInd,
                                    int* ptrMNumTilesInd,
                                    int numExperts,
                                    int m,  // expanded_active_expert_rows
                                    int n,  // inter_dim
                                    int k,  // hidden_size
                                    mcStream_t stream) {
    mctlassExHandle_t handle;
    mctlassExHandleCreate(&handle);

    mctlassExDataType DataType_ = mctlassExDataTraits<T>::type;
    mctlassExDataType WeightType_ = mctlassExDataTraits<WeightType>::type;

    mctlassExMatrixLayout_t matLayoutA;
    mctlassExMatrixLayout_t matLayoutB;
    mctlassExMatrixLayout_t matLayoutC;

    // mat A: (m, k)
    mctlassExMatrixLayoutCreate(&matLayoutA, DataType_, m, k, k);
    mctlassExMatrixLayoutSetAttribute(
        matLayoutA,
        mctlassExMatrixLayoutAttribute_t::MCTLASS_EX_MATRIX_LAYOUT_ORDER,
        &majorA,
        sizeof(mctlassExOrder_t));
    mctlassExMatrixLayoutSetAttribute(
        matLayoutA,
        mctlassExMatrixLayoutAttribute_t::MCTLASS_EX_MATRIX_LAYOUT_BATCH_COUNT,
        &numExperts,
        sizeof(int));
    // mat B: (num_experts, n, k)
    mctlassExMatrixLayoutCreate(&matLayoutB, WeightType_, k, n, k);
    mctlassExMatrixLayoutSetAttribute(
        matLayoutB,
        mctlassExMatrixLayoutAttribute_t::MCTLASS_EX_MATRIX_LAYOUT_ORDER,
        &majorB,
        sizeof(mctlassExOrder_t));
    mctlassExMatrixLayoutSetAttribute(
        matLayoutB,
        mctlassExMatrixLayoutAttribute_t::MCTLASS_EX_MATRIX_LAYOUT_BATCH_COUNT,
        &numExperts,
        sizeof(int));
    // mat C: (m, n)
    mctlassExMatrixLayoutCreate(&matLayoutC, DataType_, m, n, n);
    mctlassExMatrixLayoutSetAttribute(
        matLayoutC,
        mctlassExMatrixLayoutAttribute_t::MCTLASS_EX_MATRIX_LAYOUT_ORDER,
        &majorC,
        sizeof(mctlassExOrder_t));
    mctlassExMatrixLayoutSetAttribute(
        matLayoutC,
        mctlassExMatrixLayoutAttribute_t::MCTLASS_EX_MATRIX_LAYOUT_BATCH_COUNT,
        &numExperts,
        sizeof(int));
    // bias: (num_experts, n)
    // scale: (num, n)

    mctlassExDesc_t mctlass_desc;
    mctlassExCreateDesc(&mctlass_desc);
    mctlassExDataType input_type = DataType_;
    mctlassExDataType scale_type = WeightType_;
    mctlassExDataType compute_type =
        mctlassExDataType::MCTLASS_EX_DATATYPE_FP32;
    mctlassExEpilogueType epilogue_type =
        mctlassExEpilogueType::MCTLASS_EX_EPILOGUE_TYPE_DEFAULT;
    if (ptrBias) {
      epilogue_type = mctlassExEpilogueType::MCTLASS_EX_EPILOGUE_TYPE_BIAS;
    }
    // set scale
    mctlassExDescSetAttribute(
        mctlass_desc,
        mctlassExDescAttributes_t::MCTLASS_EX_DESC_B_SCALE_POINTER,
        &ptrScale,
        sizeof(ptrScale));
    mctlassExDescSetAttribute(
        mctlass_desc,
        mctlassExDescAttributes_t::MCTLASS_EX_DESC_B_SCALE_TYPE,
        &input_type,
        sizeof(mctlassExDataType));
    // set bias
    if (ptrBias) {
      mctlassExDescSetAttribute(
          mctlass_desc,
          mctlassExDescAttributes_t::MCTLASS_EX_DESC_BIAS_POINTER,
          &ptrBias,
          sizeof(ptrBias));
    }
    // set coumpute type
    mctlassExDescSetAttribute(
        mctlass_desc,
        mctlassExDescAttributes_t::MCTLASS_EX_DESC_COMPUTE_TYPE,
        &compute_type,
        sizeof(mctlassExDataType));
    // set epilogue type
    mctlassExDescSetAttribute(
        mctlass_desc,
        mctlassExDescAttributes_t::MCTLASS_EX_DESC_EPILOGUE_TYPE,
        &epilogue_type,
        sizeof(mctlassExEpilogueType));

    const mctlassExContiguousGroupedGemmAlgo_t algo =
        mctlassExContiguousGroupedGemmAlgo_t::
            MCTLASS_EX_CONTIGUOUS_GROUPED_ALGO_DEFAULT;
    mctlassExContiguousGroupedDesc_t contiguous_group_desc;
    mctlassExContiguousGroupedDescCreate(&contiguous_group_desc,
#if MACA_VERSION_GT(3, 3, 2, 0)
                                         const_cast<int*>(ptrSegInd),
#else
                                         ptrSegInd,
#endif
                                         nullptr,
                                         ptrMNumTilesInd,
                                         1);

    int blocksizeM;
    mctlassExContiguousGroupedGemmGetBlocksizeM(handle,
                                                mctlass_desc,
                                                matLayoutA,
                                                matLayoutB,
                                                matLayoutC,
                                                &algo,
                                                &blocksizeM);
    mctlassExContiguousGroupedGemmComputeMNumTilesIndptr(handle,
                                                         mctlass_desc,
                                                         matLayoutA,
                                                         matLayoutB,
                                                         matLayoutC,
                                                         &algo,
                                                         contiguous_group_desc,
                                                         numExperts,
                                                         blocksizeM,
                                                         stream);

    mctlassExContiguousGroupedGemmBasic(handle,
                                        mctlass_desc,
                                        ptrA,
                                        matLayoutA,
                                        ptrB,
                                        matLayoutB,
                                        ptrC,
                                        matLayoutC,
                                        contiguous_group_desc,
                                        &algo,
                                        nullptr,
                                        0,
                                        stream);

    mctlassExHandleDestroy(handle);
    mctlassExMatrixLayoutDestroy(matLayoutA);
    mctlassExMatrixLayoutDestroy(matLayoutB);
    mctlassExMatrixLayoutDestroy(matLayoutC);
    mctlassExContiguousGroupedDescDestroy(contiguous_group_desc);
    mctlassExDestroyDesc(mctlass_desc);
  }
};

template class McMoeGemmRunner<maca_bfloat16, int8_t>;

}  // namespace phi
