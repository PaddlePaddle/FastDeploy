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

/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

/*! \file
  \brief Epilogue visitor for threadblock scoped INT8 GEMMs that uses one
  scaling factor per row, and one per column.

  original file:
  3rdparty/cutlass/include/cutlass/epilogue/threadblock/epilogue_visitor_with_softmax.h

*/

#pragma once

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "../epilogue_quant_helper.h"
#include "cutlass/arch/memory.h"
#include "cutlass/arch/memory_sm75.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/numeric_conversion.h"

namespace cutlass {
namespace epilogue {
namespace threadblock {
template <typename T>
[[gnu::warning("your type here")]]
bool print_type_1111() { return false; }

template <typename ThreadblockShape_,
          int ThreadCount,
          typename ScaleTileIterator_,
          typename OutputTileIterator_,
          typename ElementAccumulator_,
          typename ElementCompute_,
          typename ElementwiseFunctor_,
          bool UseMasking_ = false>
class EpilogueVisitorPerRowPerColNf4 {
 public:
  using ThreadblockShape = ThreadblockShape_;
  static int const kThreadCount = ThreadCount;

  using ScaleTileIterator = ScaleTileIterator_;
  using OutputTileIterator = OutputTileIterator_;
  using ElementwiseFunctor = ElementwiseFunctor_;

  static int const kIterations = OutputTileIterator::kIterations;
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  using ElementOutput = typename OutputTileIterator::Element;
  using LayoutOutput = cutlass::layout::RowMajor;
  using ElementAccumulator = ElementAccumulator_;

  using AlphaScaleElementType = typename ScaleTileIterator::Element;

  using ElementCompute = ElementCompute_;
  using AccumulatorFragment = Array<ElementAccumulator, kElementsPerAccess>;
  using ComputeFragment = Array<ElementCompute_, kElementsPerAccess>;
  using ScaleFragment = Array<AlphaScaleElementType, kElementsPerAccess>;
  // static_assert(print_type_111<OutputTileIterator>())
  using OutputVector = Array<ElementOutput, kElementsPerAccess>;

  static int const kThreadsPerRow =
      OutputTileIterator::ThreadMap::Detail::kAccessWidth;
  static bool const kHasMultiStepsInRow =
      (OutputTileIterator::ThreadMap::Iterations::kColumn > 1);

  /// Argument structure
  struct Arguments {
    typename ElementwiseFunctor::Params elementwise;
    int64_t batch_stride_alpha;
    int64_t batch_stride_C;
    int64_t batch_stride_D;

    //
    // Methods
    //
    Arguments() : batch_stride_alpha(0), batch_stride_C(0), batch_stride_D(0) {}

    Arguments(typename ElementwiseFunctor::Params elementwise_)
        : elementwise(elementwise_),
          batch_stride_alpha(0),
          batch_stride_C(0),
          batch_stride_D(0) {}

    Arguments(typename ElementwiseFunctor::Params elementwise_,
              int64_t batch_stride_alpha_,
              int64_t batch_stride_C_,
              int64_t batch_stride_D_)
        : elementwise(elementwise_),
          batch_stride_alpha(batch_stride_alpha_),
          batch_stride_C(batch_stride_C_),
          batch_stride_D(batch_stride_D_) {}
  };

  struct Params {
    typename ElementwiseFunctor::Params elementwise;
    int64_t batch_stride_alpha;
    int64_t batch_stride_C;
    int64_t batch_stride_D;
    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params() {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args)
        : elementwise(args.elementwise),
          batch_stride_alpha(args.batch_stride_alpha),
          batch_stride_C(args.batch_stride_C),
          batch_stride_D(args.batch_stride_D) {}
  };

  /// Shared storage
  struct SharedStorage {};

 private:
  Params const& params_;
  SharedStorage& shared_storage_;
  MatrixCoord extent_;
  MatrixCoord extent_real_;
  ElementwiseFunctor elementwise_;

  const bool per_token_quant_;
  const bool per_channel_quant_;

  AlphaScaleElementType* ptr_alpha_row_;
  AlphaScaleElementType* ptr_alpha_col_;
  ScaleTileIterator iterator_alpha_col_;
  OutputTileIterator iterator_C_;
  OutputTileIterator iterator_D_;

  AlphaScaleElementType element_alpha_row_ = (AlphaScaleElementType)1.0f;
  AlphaScaleElementType element_alpha_col_ = (AlphaScaleElementType)1.0f;
  typename ScaleTileIterator::Fragment fragment_alpha_col_;
  typename OutputTileIterator::Fragment fragment_C_;
  typename OutputTileIterator::Fragment fragment_D_;

  ElementAccumulator beta_;

  int column_offset_;

  MatrixCoord thread_offset_;

 public:
  CUTLASS_DEVICE
  EpilogueVisitorPerRowPerColNf4(
      Params const& params,
      SharedStorage& shared_storage,
      cutlass::MatrixCoord const& problem_size,
      int thread_idx,
      int warp_idx,
      int lane_idx,
      typename ScaleTileIterator::Params params_alpha_col,
      typename OutputTileIterator::Params params_C,
      typename OutputTileIterator::Params params_D,
      cutlass::epilogue::QuantMode quant_mode,
      AlphaScaleElementType* ptr_alpha_row,
      AlphaScaleElementType* ptr_alpha_col,
      typename OutputTileIterator::Element* ptr_C,
      typename OutputTileIterator::Element* ptr_D,
      cutlass::MatrixCoord const& threadblock_offset = cutlass::MatrixCoord(0,
                                                                            0),
      int column_offset = 0,
      cutlass::MatrixCoord const& problem_size_real = cutlass::MatrixCoord(0,
                                                                           0))
      : params_(params),
        shared_storage_(shared_storage),
        extent_(problem_size),
        elementwise_(params.elementwise),
        per_token_quant_(quant_mode == cutlass::epilogue::QuantMode::PerTokenQuant ||
                         quant_mode == cutlass::epilogue::QuantMode::PerTokenChannelQuant),
        per_channel_quant_(quant_mode == cutlass::epilogue::QuantMode::PerChannelQuant ||
                           quant_mode == cutlass::epilogue::QuantMode::PerTokenChannelQuant),
        ptr_alpha_row_(ptr_alpha_row),
        ptr_alpha_col_(ptr_alpha_col),
        iterator_alpha_col_(params_alpha_col,
                            ptr_alpha_col,
                            problem_size,
                            thread_idx,
                            threadblock_offset),
        iterator_C_(
            params_C, ptr_C, problem_size, thread_idx, threadblock_offset),
        iterator_D_(
            params_D, ptr_D, problem_size, thread_idx, threadblock_offset),
        extent_real_(problem_size_real) {
    if (ptr_C) {
      iterator_C_.enable_mask();
    } else {
      iterator_C_.clear_mask();
    }
    // NOTE(wangbojun) Currently, this kernel don't hanve implantention for
    // adding elementwise beta, we keep this here for future usage beta_ =
    // (params.elementwise.beta_ptr ? *params.elementwise.beta_ptr :
    // params.elementwise.beta); if (beta_ == ElementAccumulator()) {
    //     iterator_C_.clear_mask();
    // }
  }

  /// Helper to indicate split-K behavior
  CUTLASS_DEVICE
  void set_k_partition(
      int split_k_index,     ///< Index of this threadblock within split-K
                             ///< partitioned scheme
      int split_k_slices) {  ///< Total number of split-K slices
  }

  /// Called to set the batch index
  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {
    iterator_alpha_col_.add_pointer_offset(batch_idx *
                                           params_.batch_stride_alpha);
    iterator_C_.add_pointer_offset(batch_idx * params_.batch_stride_C);
    iterator_D_.add_pointer_offset(batch_idx * params_.batch_stride_D);
  }

  /// Called at the start of the epilogue just before iterating over accumulator
  /// slices
  CUTLASS_DEVICE
  void begin_epilogue() {
    if (per_channel_quant_) {
      iterator_alpha_col_.load(fragment_alpha_col_);
    } else if (ptr_alpha_col_ != nullptr) {
      arch::global_load<AlphaScaleElementType, sizeof(AlphaScaleElementType)>(
          element_alpha_col_, ptr_alpha_col_, true);
    }

    if (!per_token_quant_ && ptr_alpha_row_ != nullptr) {
      arch::global_load<AlphaScaleElementType, sizeof(AlphaScaleElementType)>(
          element_alpha_row_, ptr_alpha_row_, true);
    }
  }

  /// Called at the start of one step before starting accumulator exchange
  CUTLASS_DEVICE
  void begin_step(int step_idx) {
    fragment_D_.clear();
    // NOTE(wangbojun) fargement C and iterator C is used for C added version
    // fragment_C_.clear();
    // iterator_C_.load(fragment_C_);

    // load alpha_row in begin_step only when per token(row) scaling is used
    if (per_token_quant_) {
      int thread_offset_row =
          iterator_D_.thread_start_row() +
          OutputTileIterator::ThreadMap::iteration_offset(0).row();

      arch::global_load<AlphaScaleElementType, sizeof(AlphaScaleElementType)>(
          element_alpha_row_,
          ptr_alpha_row_ + thread_offset_row,
          thread_offset_row < extent_.row());
    }
  }
  CUTLASS_DEVICE
  void begin_step_for_reduce(OutputTileIterator destination_iterator) {
    // load alpha_row in begin_step only when per token(row) scaling is used
    if (per_token_quant_) {
      int thread_offset_row =
          destination_iterator.thread_start_row() +
          OutputTileIterator::ThreadMap::iteration_offset(0).row();
      // element_alpha_row_ = ptr_alpha_row_[thread_offset_row];
      arch::global_load<AlphaScaleElementType, sizeof(AlphaScaleElementType)>(
          element_alpha_row_,
          ptr_alpha_row_ + thread_offset_row,
          thread_offset_row < extent_.row());
    }
  }
  /// Called at the start of a row
  CUTLASS_DEVICE
  void begin_row(int row_idx) {
    // Clear accumulators for max and sum when starting a whole row
  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(int iter_idx,
             int row_idx,
             int column_idx,
             int frag_idx,
             AccumulatorFragment const& accum) {
    NumericArrayConverter<ElementCompute,
                          ElementAccumulator,
                          kElementsPerAccess>
        source_converter;
    NumericArrayConverter<ElementCompute,
                          AlphaScaleElementType,
                          kElementsPerAccess>
        scale_converter;

    ComputeFragment result = source_converter(accum);

    if (per_channel_quant_) {
      ComputeFragment alpha_col = scale_converter(
          reinterpret_cast<ScaleFragment*>(&fragment_alpha_col_)[frag_idx]);
      result = per_token_channel_scale_accumulator_(
          result, alpha_col, element_alpha_row_);
    } else {
      result = per_token_scale_accumulator_(
          result, element_alpha_col_, element_alpha_row_);
    }

    /* // Convert to the output, without C added */
    NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
        output_converter;
    OutputVector& output =
        reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx];
    output               = output_converter(result);
    /* // Convert to the output, with non zero C added */
    // NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
    //     output_converter;
    // auto result_tmp = output_converter(result);
    // OutputVector& output =
    //     reinterpret_cast<OutputVector*>(&fragment_D_)[frag_idx];


    // OutputVector& vector_c =
    //     reinterpret_cast<OutputVector*>(&fragment_C_)[frag_idx];

    // CUTLASS_PRAGMA_UNROLL
    // for (int ii = 0; ii < kElementsPerAccess; ++ii) {
    //   output[ii] = result_tmp[ii] + vector_c[ii];
    // }
  }

  /// Called after accumulators have been exchanged for each accumulator vector
  CUTLASS_DEVICE
  void visit(AccumulatorFragment const &accum,
             int reduce_fragment_idx,
            OutputTileIterator &destination_iterator) {
    NumericArrayConverter<ElementCompute,
                          ElementAccumulator,
                          kElementsPerAccess>
        source_converter;
    NumericArrayConverter<ElementCompute,
                          AlphaScaleElementType,
                          kElementsPerAccess>
        scale_converter;

    ComputeFragment result = source_converter(accum);

    // if(threadIdx.x<32){
    //   printf("#### %d-%d-%d--%d-%d-%d, reduced accu:%d-%d-%d-%d-%d-%d-%d-%d, dequant: accu:%f-%f-%f-%f-%f-%f-%f-%f  \n",
    //           blockIdx.x,blockIdx.y,blockIdx.z,
    //           threadIdx.x,threadIdx.y,threadIdx.z,
    //           accum[0],
    //           accum[1],
    //           accum[2],
    //           accum[3],
    //           accum[4],
    //           accum[5],
    //           accum[6],
    //           accum[7],
    //           static_cast<float>(result[0]),
    //           static_cast<float>(result[1]),
    //           static_cast<float>(result[2]),
    //           static_cast<float>(result[3]),
    //           static_cast<float>(result[4]),
    //           static_cast<float>(result[5]),
    //           static_cast<float>(result[6]),
    //           static_cast<float>(result[7])
    //           );
    // }

    if (per_channel_quant_) {
      ComputeFragment alpha_col = scale_converter(
          reinterpret_cast<ScaleFragment*>(&fragment_alpha_col_)[0]);
      result = per_token_channel_scale_accumulator_(
          result, alpha_col, element_alpha_row_);
    } else {
      result = per_token_scale_accumulator_(
          result, element_alpha_col_, element_alpha_row_);
    }
    // just for bug, pass
    // if(threadIdx.x<32){
    //   printf("#### %d-%d-%d--%d-%d-%d, reduced accu:%d-%d-%d-%d-%d-%d-%d-%d, dequant: accu:%f-%f-%f-%f-%f-%f-%f-%f  \n",
    //           blockIdx.x,blockIdx.y,blockIdx.z,
    //           threadIdx.x,threadIdx.y,threadIdx.z,
    //           accum[0],
    //           accum[1],
    //           accum[2],
    //           accum[3],
    //           accum[4],
    //           accum[5],
    //           accum[6],
    //           accum[7],
    //           static_cast<float>(result[0]),
    //           static_cast<float>(result[1]),
    //           static_cast<float>(result[2]),
    //           static_cast<float>(result[3]),
    //           static_cast<float>(result[4]),
    //           static_cast<float>(result[5]),
    //           static_cast<float>(result[6]),
    //           static_cast<float>(result[7])
    //           );
    // }
    /* // Convert to the output */
    // NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess>
    //     output_converter;
    // auto result_tmp = output_converter(result);

    // if(threadIdx.x<32){
    //   printf("#### %d-%d-%d--%d-%d-%d, reduced accu:%d-%d-%d-%d-%d-%d-%d-%d, dequant: accu:%f-%f-%f-%f-%f-%f-%f-%f  \n",
    //           blockIdx.x,blockIdx.y,blockIdx.z,
    //           threadIdx.x,threadIdx.y,threadIdx.z,
    //           accum[0],
    //           accum[1],
    //           accum[2],
    //           accum[3],
    //           accum[4],
    //           accum[5],
    //           accum[6],
    //           accum[7],
    //           static_cast<float>(result[0]),
    //           static_cast<float>(result[1]),
    //           static_cast<float>(result[2]),
    //           static_cast<float>(result[3]),
    //           static_cast<float>(result[4]),
    //           static_cast<float>(result[5]),
    //           static_cast<float>(result[6]),
    //           static_cast<float>(result[7])
    //           );
    // }

    typename OutputTileIterator::Fragment output_fragment;
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii<output_fragment.size(); ++ii){
      output_fragment[ii]= static_cast<typename OutputTileIterator::Fragment::Element>(result[ii]);
    }
    // OutputVector& output =
    //     reinterpret_cast<OutputVector*>(&output_fragment)[0];

    // CUTLASS_PRAGMA_UNROLL
    // for (int ii = 0; ii < kElementsPerAccess; ++ii) {
    //   output[ii] = result_tmp[ii];
    // }

    // if(threadIdx.x<32){
    //   printf("#### %d-%d-%d--%d-%d-%d, reduced accu:%d-%d-%d-%d-%d-%d-%d-%d, dequant: accu:%f-%f-%f-%f-%f-%f-%f-%f  \n",
    //           blockIdx.x,blockIdx.y,blockIdx.z,
    //           threadIdx.x,threadIdx.y,threadIdx.z,
    //           accum[0],
    //           accum[1],
    //           accum[2],
    //           accum[3],
    //           accum[4],
    //           accum[5],
    //           accum[6],
    //           accum[7],
    //           static_cast<float>(output_fragment[0]),
    //           static_cast<float>(output_fragment[1]),
    //           static_cast<float>(output_fragment[2]),
    //           static_cast<float>(output_fragment[3]),
    //           static_cast<float>(output_fragment[4]),
    //           static_cast<float>(output_fragment[5]),
    //           static_cast<float>(output_fragment[6]),
    //           static_cast<float>(output_fragment[7])
    //           );
    // }
    destination_iterator.store(output_fragment);
  }
  /// Called at the end of a row
  CUTLASS_DEVICE
  void end_row(int row_idx) {
    /* using ConvertSumOutput = cutlass::NumericConverter<ElementSum,
     * ElementSoftmaxCompute>; */
    /* using ConvertNormOutput = cutlass::NumericConverter<ElementNorm,
     * ElementSoftmaxCompute>; */

    /* ConvertSumOutput   convert_sum_output; */
    /* ConvertNormOutput  convert_norm_output; */

    /* // Compute accumulate sum only in the last step */
    /* accum_sum_ = warp_reduce_sum_(accum_sum_); */

    /* bool is_first_thread_in_tile = ((threadIdx.x % kThreadsPerRow) == 0); */
    /* bool row_guard = thread_offset_.row() < extent_.row(); */
    /* bool is_write_thread = row_guard && is_first_thread_in_tile; */

    /* int block_batch = blockIdx.z; */

    /* ElementNorm *curr_ptr_max = ptr_Max_ + thread_offset_.row() +
     * column_offset_ + block_batch * params_.batch_stride_Max; */
    /* ElementSum *curr_ptr_sum = ptr_Sum_ + thread_offset_.row() +
     * column_offset_ + block_batch * params_.batch_stride_Sum; */

    /* arch::global_store<ElementNorm, sizeof(ElementNorm)>( */
    /*           convert_norm_output(accum_max_), */
    /*           (void *)curr_ptr_max, */
    /*           is_write_thread); */

    /* arch::global_store<ElementSum, sizeof(ElementSum)>( */
    /*           convert_sum_output(accum_sum_), */
    /*           (void *)curr_ptr_sum, */
    /*           is_write_thread); */

    /* // Clear accumulators for max and sum when finishing a whole row */
    /* clear_accum_(); */
  }

  /// Called after all accumulator elements have been visited
  CUTLASS_DEVICE
  void end_step(int step_idx) {
    iterator_D_.store(fragment_D_);
    ++iterator_D_;
    ++iterator_C_;
  }

  /// Called after all steps have been completed
  CUTLASS_DEVICE
  void end_epilogue() {}

 private:
  CUTLASS_DEVICE
  ComputeFragment per_token_channel_scale_accumulator_(
      ComputeFragment const& accum,
      ComputeFragment const& scale_col,
      AlphaScaleElementType const& scale_row) {
    // if(threadIdx.x<32){
    //   printf("#### per_token_channel_scale_accumulator,  %d-%d-%d--%d-%d-%d, quanted accu:%f-%f-%f-%f-%f-%f-%f-%f, scale_col:%f-%f-%f-%f-%f-%f-%f-%f  \n",
    //          blockIdx.x,blockIdx.y,blockIdx.z,
    //          threadIdx.x,threadIdx.y,threadIdx.z,
    //          static_cast<float>(accum[0]),
    //          static_cast<float>(accum[1]),
    //          static_cast<float>(accum[2]),
    //          static_cast<float>(accum[3]),
    //          static_cast<float>(accum[4]),
    //          static_cast<float>(accum[5]),
    //          static_cast<float>(accum[6]),
    //          static_cast<float>(accum[7]),
    //          static_cast<float>(scale_col[0]),
    //          static_cast<float>(scale_col[1]),
    //          static_cast<float>(scale_col[2]),
    //          static_cast<float>(scale_col[3]),
    //          static_cast<float>(scale_col[4]),
    //          static_cast<float>(scale_col[5]),
    //          static_cast<float>(scale_col[6]),
    //          static_cast<float>(scale_col[7]));
    //   }
    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ComputeFragment::kElements; ++i) {
      // printf("#### per_token_channel_scale_accumulator_, %f %f %f \n",
      //       static_cast<float>(accum[i]),
      //       static_cast<float>(scale_col[i]),
      //       static_cast<float>(scale_row));
      result[i] =
          accum[i] * (scale_col[i] * static_cast<ElementCompute>(scale_row));
    }

    return result;
  }

  CUTLASS_DEVICE
  ComputeFragment per_token_scale_accumulator_(
      ComputeFragment const& accum,
      AlphaScaleElementType const& scale_col,
      AlphaScaleElementType const& scale_row) {
    ComputeFragment result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < ComputeFragment::kElements; ++i) {
      result[i] = accum[i] * (scale_col * scale_row);
    }

    return result;
  }
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass
