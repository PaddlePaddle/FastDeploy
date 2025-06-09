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

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/trace.h"
#include <glog/logging.h>
#include "cutlass/gemm/kernel/params_universal_base.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/epilogue/epilogue_quant_helper.h"
#include "cutlass_kernels/w4a8_moe/cutlass_extensions/epilogue/threadblock/epilogue_per_row_per_col_scale_nf4.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,           ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          GroupScheduleMode GroupScheduleMode_
          >
struct MoeW4A8GemmWithEpilogueVisitorInterleavedNf4 {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueVisitor = typename Epilogue::Visitor;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ElementA = typename Mma::IteratorA::Element;
  using LayoutA = typename Mma::IteratorA::Layout;
  using TensorRefA = TensorRef<ElementA, LayoutA>;

  using ElementB = typename Mma::IteratorB::Element;
  using LayoutB = typename Mma::IteratorB::Layout;
  using TensorRefB = TensorRef<ElementB, LayoutB>;

  using ElementCompute = typename EpilogueVisitor::ElementCompute;
  using LayoutAlphaCol = cutlass::layout::RowMajor;
  using LayoutAlphaRow = cutlass::layout::ColumnMajor;
  using TensorRefAlphaCol =
      TensorRef<typename EpilogueVisitor::ScaleTileIterator::Element,
                LayoutAlphaCol>;
  using TensorRefAlphaRow =
      TensorRef<typename EpilogueVisitor::ScaleTileIterator::Element,
                LayoutAlphaRow>;
  using TensorRefNf4LookUpTable = TensorRef<int32_t, cutlass::layout::RowMajor>;

  using ElementC = typename EpilogueVisitor::ElementOutput;
  using LayoutC = typename Epilogue::Layout;
  using TensorRefC = TensorRef<ElementC, LayoutC>;

  static ComplexTransform const kTransformA = Mma::kTransformA;
  static ComplexTransform const kTransformB = Mma::kTransformB;
  using Operator = typename Mma::Operator;

  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;
  using EpilogueOutputOp = typename Epilogue::Visitor::
      ElementwiseFunctor;  // Define type so GemmUniversalBase doesn't complain
  static int const kStages = Mma::kStages;
  static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
  static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
  static int const kAlignmentC = EpilogueVisitor::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  static constexpr int kInterleave =
      Mma::IteratorB::Shape::kRow / Mma::Shape::kK;

  /// Split-K preserves splits that are 128b aligned
  static int const kSplitKAlignment = const_max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  using ProblemVisitor = GemmMoeProblemVisitor<ThreadblockShape,
                                              kGroupScheduleMode,
                                              kThreadCount,
                                              kThreadCount,
                                              kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments : UniversalArgumentsBase {
    //
    // Data members
    //

    int problem_count;
    int threadblock_count;
    GemmCoord problem_size;

    TensorRefA ref_A;
    TensorRefB ref_B;
    epilogue::QuantMode quant_mode;
    TensorRefAlphaCol ref_alpha_col;
    TensorRefAlphaRow ref_alpha_row;
    TensorRefNf4LookUpTable ref_nf4_look_up_table;
    TensorRefC ref_C;
    TensorRefC ref_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    int64_t* total_rows_before_expert;
    int64_t total_rows;
    int64_t gemm_n;
    int64_t gemm_k;

    // Only used by device-level operator
    GemmCoord* host_problem_sizes;

    typename EpilogueVisitor::Arguments epilogue_visitor;

    //
    // Methods
    //

    Arguments() {}

    /// constructs an arguments structure
    Arguments(cutlass::gemm::GemmUniversalMode mode_,
              int problem_count,
              int threadblock_count,
              GemmCoord problem_size_,
              int batch_count_,
              TensorRefA ref_A_,
              TensorRefB ref_B_,
              epilogue::QuantMode quant_mode_,
              TensorRefAlphaCol ref_alpha_col_,
              TensorRefAlphaRow ref_alpha_row_,
              TensorRefNf4LookUpTable ref_nf4_look_up_table_,
              TensorRefC ref_C_,
              TensorRefC ref_D_,
              int64_t* total_rows_before_expert,
              int64_t total_rows,
              int64_t gemm_n,
              int64_t gemm_k,
              int64_t batch_stride_A_,
              int64_t batch_stride_B_,
              typename EpilogueVisitor::Arguments epilogue_visitor_,
              GemmCoord* host_problem_sizes = nullptr)
        : UniversalArgumentsBase(mode_,
                                 problem_size_,
                                 /*serial_split_k_factor=*/batch_count_,
                                 /*batch_stride_D=*/0),
          problem_count(problem_count),
          threadblock_count(threadblock_count),
          problem_size(problem_size_),
          ref_A(ref_A_),
          ref_B(ref_B_),
          quant_mode(quant_mode_),
          ref_alpha_col(ref_alpha_col_),
          ref_alpha_row(ref_alpha_row_),
          ref_nf4_look_up_table(ref_nf4_look_up_table_),
          ref_C(ref_C_),
          ref_D(ref_D_),
          total_rows_before_expert(total_rows_before_expert),
          total_rows(total_rows),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          batch_stride_A(batch_stride_A_),
          batch_stride_B(batch_stride_B_),
          epilogue_visitor(epilogue_visitor_),
          host_problem_sizes(nullptr) {}
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure

  /// Parameters structure
  struct Params : UniversalParamsBase<ThreadblockSwizzle,
                                      ThreadblockShape,
                                      ElementA,
                                      ElementB,
                                      ElementC,
                                      LayoutA,
                                      LayoutB> {

    using ParamsBase = UniversalParamsBase<ThreadblockSwizzle,
                                           ThreadblockShape,
                                           ElementA,
                                           ElementB,
                                           ElementC,
                                           LayoutA,
                                           LayoutB>;

    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorNF4LookUpTable::Params params_nf4_look_up_table;
    typename EpilogueVisitor::ScaleTileIterator::Params params_alpha_col;
    typename EpilogueVisitor::ScaleTileIterator::Params params_alpha_row;
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;


    void* ptr_A;
    void* ptr_B;
    epilogue::QuantMode quant_mode;
    typename EpilogueVisitor::ScaleTileIterator::Element* ptr_alpha_col;
    typename EpilogueVisitor::ScaleTileIterator::Element* ptr_alpha_row;
    typename Mma::IteratorNF4LookUpTable::TensorRef ref_nf4_look_up_table;
    ElementC* ptr_C;
    ElementC* ptr_D;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    typename EpilogueVisitor::Params epilogue_visitor;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() = default;

    Params(Arguments const& args, int device_sms, int sm_occupancy, void* workspace = nullptr, int tile_count=0)
        : ParamsBase(args, device_sms, sm_occupancy),
          problem_visitor(args.total_rows_before_expert,
                          args.total_rows,
                          args.gemm_n,
                          args.gemm_k,
                          args.problem_count,
                          workspace,
                          tile_count),
          threadblock_count(args.threadblock_count),
          params_A(args.ref_A.layout()),
          params_B(args.ref_B.layout()),
          params_alpha_col(args.ref_alpha_col.layout()),
          params_alpha_row(args.ref_alpha_col.layout()),
          params_C(args.ref_C.layout()),
          params_D(args.ref_D.layout()),
          ptr_A(args.ref_A.data()),
          ptr_B(args.ref_B.data()),
          quant_mode(args.quant_mode),
          ptr_alpha_col(args.ref_alpha_col.data()),
          ptr_alpha_row(args.ref_alpha_row.data()),
          ref_nf4_look_up_table(args.ref_nf4_look_up_table),
          ptr_C(args.ref_C.data()),
          ptr_D(args.ref_D.data()),
          batch_stride_A(args.batch_stride_A),
          batch_stride_B(args.batch_stride_B),
          epilogue_visitor(args.epilogue_visitor) { }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
    typename Mma::SharedStorage main_loop;

    struct {
      typename Epilogue::SharedStorage epilogue;
      typename EpilogueVisitor::SharedStorage visitor;
    } epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  MoeW4A8GemmWithEpilogueVisitorInterleavedNf4() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
    CUTLASS_TRACE_HOST("MoeW4A8GemmWithEpilogueVisitorInterleavedNf4::can_implement()");

    static int const kAlignmentA = Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB = Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC =
        EpilogueVisitor::OutputTileIterator::kElementsPerAccess;

    bool isAMisaligned = false;
    bool isBMisaligned = false;
    bool isCMisaligned = false;

    if (platform::is_same<LayoutA, layout::RowMajor>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    } else if (platform::is_same<LayoutA, layout::ColumnMajor>::value) {
      isAMisaligned = problem_size.m() % kAlignmentA;
    } else if (platform::is_same<LayoutA,
                                 layout::ColumnMajorInterleaved<32>>::value ||
               platform::is_same<LayoutA,
                                 layout::ColumnMajorInterleaved<64>>::value) {
      isAMisaligned = problem_size.k() % kAlignmentA;
    }

    if (platform::is_same<LayoutB, layout::RowMajor>::value) {
      isBMisaligned = problem_size.n() % kAlignmentB;
    } else if (platform::is_same<LayoutB, layout::ColumnMajor>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    } else if (platform::is_same<LayoutB,
                                 layout::RowMajorInterleaved<32>>::value ||
               platform::is_same<LayoutB,
                                 layout::RowMajorInterleaved<64>>::value) {
      isBMisaligned = problem_size.k() % kAlignmentB;
    }

    if (platform::is_same<LayoutC, layout::RowMajor>::value) {
         isCMisaligned  = problem_size.n() % kAlignmentC;
    } else if (platform::is_same<LayoutC, layout::ColumnMajor>::value) {
      isCMisaligned = problem_size.m() % kAlignmentC;
    } else if (platform::is_same<LayoutC,
                                 layout::ColumnMajorInterleaved<32>>::value ||
               platform::is_same<LayoutC,
                                 layout::ColumnMajorInterleaved<64>>::value) {
      isCMisaligned = problem_size.n() % kAlignmentC;
    }

    if (isAMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for A operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isBMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for B operand");
      return Status::kErrorMisalignedOperand;
    }

    if (isCMisaligned) {
      CUTLASS_TRACE_HOST("  returning kErrorMisalignedOperand for C operand");
      return Status::kErrorMisalignedOperand;
    }

    CUTLASS_TRACE_HOST("  returning kSuccess");

    return Status::kSuccess;
  }

  static Status can_implement(Arguments const& args) {
    return can_implement(args.problem_size);
  }

  static size_t get_extra_workspace_size(
      Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

  CUTLASS_DEVICE
  static void invoke(Params const& params, SharedStorage& shared_storage) {
    MoeW4A8GemmWithEpilogueVisitorInterleavedNf4 op;
    op(params, shared_storage);
  }
#define SPLIT_K_ENABLED 1

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {

      using ElementA = typename Mma::IteratorA::Element;
      using LayoutA = typename Mma::IteratorA::Layout;
      using ElementB = typename Mma::IteratorB::Element;
      using LayoutB = typename Mma::IteratorB::Layout;

      static constexpr int kInterleave =
          Mma::IteratorB::Shape::kRow / Mma::Shape::kK;
      static_assert(
          platform::is_same<LayoutB, layout::RowMajor>::value &&
                  kInterleave == 1 ||
              platform::is_same<LayoutB, layout::ColumnMajor>::value &&
                  kInterleave >= 1,
          "B must be row major/col major OR col major interleaved.");

      //
      // Problem visitor.
      //
      ProblemVisitor problem_visitor(
          params.problem_visitor, shared_storage.problem_visitor, blockIdx.x);
      const int64_t gemm_k = params.problem_visitor.gemm_k;
      const int64_t gemm_n = params.problem_visitor.gemm_n;
      int64_t bytes_per_expert_matrix =
          (gemm_k * gemm_n / 8) * cutlass::sizeof_bits<ElementB>::value;

      // Outer 'persistent' loop to iterate over tiles
      while (problem_visitor.next_tile()) {
      // // Compute threadblock location
      ThreadblockSwizzle threadblock_swizzle;

      cutlass::gemm::GemmCoord threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        GemmCoord problem_size = problem_visitor.problem_size();
        int32_t problem_idx = problem_visitor.problem_index();
        int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());

        GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

        cutlass::MatrixCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma::Shape::kM,  // NOLINT
            int(cta_idx % grid_shape.n()) * Mma::Shape::kN  // NOLINT
            );

        // if (threadIdx.x == 0) {
        //   printf("%d-%d-%d problem_size: %d, %d problem_idx: %d, cta_idx: %d\n", blockIdx.x,blockIdx.y,blockIdx.z, problem_size.m(), problem_size.n(), problem_idx, cta_idx);
        // }

        // Load element pointers. Exchange pointers and strides if working on
        // the transpose
        int64_t rows_to_jump = 0;
        if (params.problem_visitor.total_rows < 0) {
          rows_to_jump = problem_idx == 0 ? 0 : params.problem_visitor.last_row_for_problem[problem_idx - 1];
        } else {
          rows_to_jump = problem_idx * (params.problem_visitor.total_rows / params.problem_visitor.problem_count);
        }

        ElementA* ptr_A =
            reinterpret_cast<ElementA*>(params.ptr_A) + rows_to_jump * gemm_k;
        typename LayoutA::LongIndex ldm_A = gemm_k;

        char* byte_ptr_B = ((char*)params.ptr_B) +                 // NOLINT
                           problem_idx * bytes_per_expert_matrix;  // NOLINT
        ElementB* ptr_B = reinterpret_cast<ElementB*>(byte_ptr_B);
        typename LayoutB::LongIndex ldm_B =
            platform::is_same<layout::RowMajor, LayoutB>::value
                ? gemm_n
                : gemm_k * kInterleave;


      int offset_k = 0;
      int problem_size_k = params.problem_size.k();


  // Maybe need to modify? Author zhengzekang.
  #if SPLIT_K_ENABLED
      //
      // Fetch pointers based on mode.
      //
      if (params.mode == GemmUniversalMode::kGemm ||
          params.mode == GemmUniversalMode::kGemmSplitKParallel) {
        if (threadblock_tile_offset.k() + 1 < params.grid_tiled_shape.k()) {
          problem_size_k = (threadblock_tile_offset.k() + 1) * params.gemm_k_size;
        }
        offset_k = threadblock_tile_offset.k() * params.gemm_k_size;
      } else if (params.mode == GemmUniversalMode::kBatched) {
        ptr_A += threadblock_tile_offset.k() * params.batch_stride_A;
        ptr_B += threadblock_tile_offset.k() * params.batch_stride_B;
      } else if (params.mode == GemmUniversalMode::kArray) {
        ptr_A = static_cast<ElementA* const*>(
            params.ptr_A)[threadblock_tile_offset.k()];
        ptr_B = static_cast<ElementB* const*>(
            params.ptr_B)[threadblock_tile_offset.k()];
      }
  #endif
    // if(threadIdx.x==0){
    //     printf("##### block: %d-%d-%d, offset_k:%d, threadblock_tile_offset.m-n-k():%d-%d-%d, params.gemm_k_size:%d \n",
    //             blockIdx.x, blockIdx.y, blockIdx.z,
    //             offset_k,
    //             threadblock_tile_offset.m(),
    //             threadblock_tile_offset.n(),
    //             threadblock_tile_offset.k(),
    //             params.gemm_k_size
    //             );
    //   }

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{
          threadblock_offset.row(),
          0,
      };


      cutlass::MatrixCoord tb_offset_B{
          0,
          threadblock_offset.column() / kInterleave};

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
          params.params_A,
          ptr_A,
          {problem_size.m(), problem_size_k},
          thread_idx,
          tb_offset_A);

      typename Mma::IteratorB iterator_B(
          params.params_B,
          ptr_B,
          {problem_size_k * kInterleave, params.problem_size.n() / kInterleave},
          thread_idx,
          tb_offset_B);
      typename Mma::IteratorNF4LookUpTable iterator_nf4_look_up_table =
        Mma::IteratorNF4LookUpTable(
          params.params_nf4_look_up_table,
        params.ref_nf4_look_up_table.data(),
          {0,16},
          threadIdx.x,
          {0,0}
        );

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      //
      // Main loop
      //
      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();

      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations =
          (problem_size_k  + Mma::Shape::kK - 1) / Mma::Shape::kK;
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, iterator_nf4_look_up_table, accumulators);
      // if(threadIdx.x==0){
      //   printf("##### block: %d-%d-%d, offset-m-n-k:%d-%d-%d \n",
      //           blockIdx.x, blockIdx.y, blockIdx.z,
      //           threadblock_tile_offset.m(),
      //           threadblock_tile_offset.n(),
      //           threadblock_tile_offset.k()
      //           );
      // }
      //
      // Masked tile iterators constructed from members
      //

      threadblock_tile_offset =
          threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        ElementC* ptr_C =
            reinterpret_cast<ElementC*>(params.ptr_C) + rows_to_jump * gemm_n;
        ElementC* ptr_D =
            reinterpret_cast<ElementC*>(params.ptr_D) + rows_to_jump * gemm_n;

      using Element_scale = typename EpilogueVisitor::ScaleTileIterator::Element;
      Element_scale* ptr_alpha_row = params.ptr_alpha_row == nullptr ? params.ptr_alpha_row : reinterpret_cast<Element_scale*>(params.ptr_alpha_row) + rows_to_jump;
      Element_scale* ptr_alpha_col = reinterpret_cast<Element_scale*>(params.ptr_alpha_col) + problem_idx * params.problem_size.n();
      // if (threadIdx.x == 0)
      // printf("##### block: %d-%d-%d, ptr_alpha_row:%p,(%f) ptr_alpha_col:%p,(%f)\n", blockIdx.x, blockIdx.y, blockIdx.z, ptr_alpha_row, static_cast<float>(*ptr_alpha_row), ptr_alpha_col, static_cast<float>(*ptr_alpha_col));
      //
      // Construct the epilogue visitor
      //

      EpilogueVisitor epilogue_visitor(params.epilogue_visitor,
                                     shared_storage.epilogue.visitor,
                                     problem_size.mn(),
                                     thread_idx,
                                     warp_idx,
                                     lane_idx,
                                     params.params_alpha_col,
                                     params.params_C,
                                     params.params_D,
                                     params.quant_mode,
                                     ptr_alpha_row,
                                     ptr_alpha_col,
                                     ptr_C,
                                     ptr_D,
                                     threadblock_offset,
                                     blockIdx.y * params.problem_size.m());

      if (params.mode == GemmUniversalMode::kGemm) {
        // Indicate which position in a serial reduction the output operator is
        // currently updating
        epilogue_visitor.set_k_partition(threadblock_tile_offset.k(),
                                        params.grid_tiled_shape.k());
      } else if (params.mode == GemmUniversalMode::kBatched ||
                params.mode == GemmUniversalMode::kArray) {
        epilogue_visitor.set_batch_index(threadblock_tile_offset.k());
      }

      // Construct the epilogue
      Epilogue epilogue(
          shared_storage.epilogue.epilogue, thread_idx, warp_idx, lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      epilogue(epilogue_visitor, accumulators);

        // Next tile
        problem_visitor.advance(gridDim.x);
      }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
