/***************************************************************************************************
 * Copyright (c) 2017-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *notice, this list of conditions and the following disclaimer in the
 *documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its
 *contributors may be used to endorse or promote products derived from this
 *software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief
*/

#pragma once

#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"

#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/trace.h"

#include "cutlass_extensions/gemm/kernel/gemm_moe_problem_visitor.h"
#include "cutlass_extensions/tile_interleaved_layout.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
// This section exists to that we can use the same kernel code for regular gemm
// and dequantizing gemms. It will dispatch to the dequantizing gemm if the Mma
// type has an Iterator for scales in global.
template <typename...>
using void_t = void;

template <typename Mma, typename = void>
struct use_dq_gemm : platform::false_type {};

template <typename Mma>
struct use_dq_gemm<Mma, void_t<typename Mma::IteratorScale>>
    : platform::true_type {};

// SFINAE overload for dequantizing gemm
template <
    typename Mma,
    typename ElementScale,
    typename platform::enable_if<use_dq_gemm<Mma>::value, bool>::type = true>
CUTLASS_DEVICE static void run_mma(Mma mma,
                                   int gemm_k_iterations,
                                   typename Mma::FragmentC& accum,  // NOLINT
                                   typename Mma::IteratorA iterator_A,
                                   typename Mma::IteratorB iterator_B,
                                   typename Mma::FragmentC const& src_accum,
                                   ElementScale* weight_scale_ptr,
                                   MatrixCoord scale_extent,
                                   const int thread_idx,
                                   MatrixCoord tb_offset_scale) {
  typename Mma::IteratorScale iterator_scale(
      Mma::IteratorScale::Layout(scale_extent.column()),
      weight_scale_ptr,
      scale_extent,
      thread_idx,
      tb_offset_scale);

  mma(gemm_k_iterations,
      accum,
      iterator_A,
      iterator_B,
      iterator_scale,
      src_accum);
}

// SFINAE overload for normal gemm. This completely ignores the scale parameters
template <
    typename Mma,
    typename ElementScale,
    typename platform::enable_if<!use_dq_gemm<Mma>::value, bool>::type = true>
CUTLASS_DEVICE static void run_mma(Mma mma,
                                   int gemm_k_iterations,
                                   typename Mma::FragmentC& accum,  // NOLINT
                                   typename Mma::IteratorA iterator_A,
                                   typename Mma::IteratorB iterator_B,
                                   typename Mma::FragmentC const& src_accum,
                                   ElementScale* weight_scale_ptr,
                                   MatrixCoord scale_extent,
                                   const int thread_idx,
                                   MatrixCoord tb_offset_scale) {
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, src_accum);
}

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          typename KernelArch,  ///! The Architecture this kernel is compiled
                                /// for. Used since SIMT kernels lose top-level
                                /// arch.
          GroupScheduleMode GroupScheduleMode_  ///! Type of scheduling to //
                                                /// NOLINT perform
          >
struct MoeFCGemm {
 public:
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  // Optional transpose
  using MapArguments =
      kernel::detail::MapArguments<typename Mma::IteratorA::Element,
                                   typename Mma::IteratorA::Layout,
                                   Mma::kTransformA,
                                   Mma::IteratorA::AccessType::kElements,
                                   typename Mma::IteratorB::Element,
                                   typename Mma::IteratorB::Layout,
                                   Mma::kTransformB,
                                   Mma::IteratorB::AccessType::kElements,
                                   typename Mma::LayoutC,
                                   kTransposed>;

  // Public-facing type definitions related to operand element type, layout, and
  // complex conjugate operation. Must interact with the 'kTransposed' notion.
  static_assert(!kTransposed, "Transpose problem not supported");
  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;
  using ElementScale = ElementC;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC =
      Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmMoeProblemVisitor<ThreadblockShape,
                                               kGroupScheduleMode,
                                               kThreadCount,
                                               kThreadCount,
                                               kTransposed>;

  //
  // Structures
  //

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //

    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementScale* weight_scales;
    ElementC* ptr_C;
    ElementC* ptr_D;

    int64_t* total_rows_before_expert;
    int64_t total_rows;
    int64_t gemm_n;
    int64_t gemm_k;

    WintQuantMethod quant_method;

    // Extra arguments for wint2.0
    uint8_t* local_scale;
    float* code_scale;
    float* code_zp;

    // Only used by device-level operator
    GemmCoord* host_problem_sizes;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments()
        : problem_count(0),
          threadblock_count(0),
          ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          total_rows_before_expert(nullptr),
          total_rows(-1),
          gemm_n(0),
          gemm_k(0),
          quant_method(WintQuantMethod::kNone),
          local_scale(nullptr),
          code_scale(nullptr),
          code_zp(nullptr),
          host_problem_sizes(nullptr) {}

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(int problem_count,
              int threadblock_count,
              typename EpilogueOutputOp::Params output_op,
              const ElementA* ptr_A,
              const ElementB* ptr_B,
              const ElementScale* weight_scales,
              const ElementC* ptr_C,
              ElementC* ptr_D,
              int64_t* total_rows_before_expert,
              int64_t total_rows,
              int64_t gemm_n,
              int64_t gemm_k,
              WintQuantMethod quant_method,
              const uint8_t* local_scale,
              const float* code_scale,
              const float* code_zp,
              GemmCoord* host_problem_sizes = nullptr)
        : problem_count(problem_count),
          threadblock_count(threadblock_count),
          output_op(output_op),
          ptr_A(const_cast<ElementA*>(ptr_A)),
          ptr_B(const_cast<ElementB*>(ptr_B)),
          weight_scales(const_cast<ElementScale*>(weight_scales)),
          ptr_C(const_cast<ElementC*>(ptr_C)),
          ptr_D(ptr_D),
          total_rows_before_expert(total_rows_before_expert),
          total_rows(total_rows),
          gemm_n(gemm_n),
          gemm_k(gemm_k),
          quant_method(quant_method),
          local_scale(const_cast<uint8_t*>(local_scale)),
          code_scale(const_cast<float*>(code_scale)),
          code_zp(const_cast<float*>(code_zp)),
          host_problem_sizes(nullptr) {
      if (quant_method != WintQuantMethod::kNone || platform::is_same<uint8_t, ElementB>::value ||
          platform::is_same<uint4b_t, ElementB>::value) {
        assert(weight_scales);
      }
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    typename ProblemVisitor::Params problem_visitor;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementScale* weight_scales;
    ElementC* ptr_C;
    ElementC* ptr_D;

    WintQuantMethod quant_method;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
        : ptr_A(nullptr),
          ptr_B(nullptr),
          weight_scales(nullptr),
          ptr_C(nullptr),
          ptr_D(nullptr),
          quant_method(WintQuantMethod::kNone) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args,
           void* workspace = nullptr,
           int tile_count = 0)  // NOLINT
        : problem_visitor(args.total_rows_before_expert,
                          args.total_rows,
                          args.gemm_n,
                          args.gemm_k,
                          args.problem_count,
                          workspace,
                          tile_count),
          threadblock_count(args.threadblock_count),
          output_op(args.output_op),
          ptr_A(args.ptr_A),
          ptr_B(args.ptr_B),
          weight_scales(args.weight_scales),
          ptr_C(args.ptr_C),
          ptr_D(args.ptr_D),
          quant_method(args.quant_method) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const& args,
                void* workspace = nullptr,
                int tile_count = 0) {
      problem_visitor =
          typename ProblemVisitor::Params(args.total_rows_before_expert,
                                          args.total_rows,
                                          args.gemm_n,
                                          args.gemm_k,
                                          args.problem_count,
                                          workspace,
                                          tile_count);
      threadblock_count = args.threadblock_count;
      output_op = args.output_op;
      ptr_A = args.ptr_A;
      ptr_B = args.ptr_B;
      weight_scales = args.weight_scales;
      ptr_C = args.ptr_C;
      ptr_D = args.ptr_D;
      quant_method = args.quant_method;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename ProblemVisitor::SharedStorage problem_visitor;
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  MoeFCGemm() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size) {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const& args) {
    if (args.quant_method != WintQuantMethod::kNone || platform::is_same<uint8_t, ElementB>::value ||
        platform::is_same<uint4b_t, ElementB>::value) {
      if (args.weight_scales == nullptr) {
        CUTLASS_TRACE_HOST(
            "MoeFCGemm::can_implement() - weight scales are required for "
            "uint8_t and uint4b_t");
        return Status::kInvalid;
      }
    } else if (args.weight_scales != nullptr) {
      CUTLASS_TRACE_HOST(
          "MoeFCGemm::can_implement() - weight scales are ignored for all "
          "types except uint8_t and uint4b_t");
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

  static size_t get_extra_workspace_size(
      Arguments const& args, cutlass::gemm::GemmCoord const& grid_tiled_shape) {
    return 0;
  }

  // The dummy template parameter is not used and exists so that we can compile
  // this code using a standard earlier than C++17. Prior to C++17, fully
  // specialized templates HAD to exists in a namespace
  template <bool B, typename dummy = void>
  struct KernelRunner {
    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      CUTLASS_NOT_IMPLEMENTED();
    }
  };

  template <typename dummy>
  struct KernelRunner<true, dummy> {

    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      //
      // These types shadow the type-level definitions and support the ability
      // to implement a 'transposed' GEMM that computes the transposed problems.
      //

      using ElementA = typename Mma::IteratorA::Element;
      using LayoutA = typename Mma::IteratorA::Layout;
      using ElementB = typename Mma::IteratorB::Element;
      using LayoutB = typename Mma::IteratorB::Layout;
      using ElementC = typename Epilogue::OutputTileIterator::Element;
      using LayoutC = typename Epilogue::OutputTileIterator::Layout;

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
        GemmCoord problem_size = problem_visitor.problem_size();
        int32_t problem_idx = problem_visitor.problem_index();
        int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());

        GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

        // threadblock_offset of C
        cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma::Shape::kM,  // NOLINT
            int(cta_idx % grid_shape.n()) * Mma::Shape::kN,  // NOLINT
            0);

        // Load element pointers. Exchange pointers and strides if working on
        // the transpose
        int64_t rows_to_jump = 0;

        if (params.problem_visitor.total_rows < 0) {
          rows_to_jump = problem_idx == 0 ? 0 : params.problem_visitor.last_row_for_problem[problem_idx - 1];
        } else {
          rows_to_jump = problem_idx * (params.problem_visitor.total_rows / params.problem_visitor.problem_count);
        }

        // begin address offset for A for current tile
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

        // Compute initial location in logical coordinates
        // the begin threadblock_offset of A, which holds the same row id with C
        cutlass::MatrixCoord tb_offset_A{
            threadblock_offset.m(),
            0,
        };

        // the begin threadblock_offset of B, which holds the same column id with C
        cutlass::MatrixCoord tb_offset_B{0,
                                         threadblock_offset.n() / kInterleave};

        // the begin threadblock_offset of scale, which holds the same column id with C, but with no row id
        cutlass::MatrixCoord tb_offset_scale{0, threadblock_offset.n()};

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(LayoutA(ldm_A),
                                           ptr_A,
                                           {problem_size.m(), problem_size.k()},
                                           thread_idx,
                                           tb_offset_A);

        typename Mma::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            {problem_size.k() * kInterleave, problem_size.n() / kInterleave},
            thread_idx,
            tb_offset_B);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

        int lane_idx = threadIdx.x % 32;

        //
        // Matrix multiply phase
        //

        // Construct thread-scoped matrix multiply
        auto CreateMMA = [&]() {
          if constexpr (use_dq_gemm<Mma>::value)
            return Mma(
                shared_storage.main_loop, -1, thread_idx, warp_idx, lane_idx);
          else
            return Mma(
                shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
        };
        Mma mma = CreateMMA();

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations =
            (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Wait for all threads to finish their epilogue phases from the
        // previous tile.
        __syncthreads();

        // Compute threadblock-scoped matrix multiply-add
        ElementScale* weight_scale_ptr =
            params.weight_scales + problem_idx * problem_size.n();
        run_mma<Mma>(mma,
                     gemm_k_iterations,
                     accumulators,
                     iterator_A,
                     iterator_B,
                     accumulators,
                     weight_scale_ptr,
                     {1, problem_size.n()},
                     thread_idx,
                     tb_offset_scale);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        ElementC* ptr_C =
            reinterpret_cast<ElementC*>(params.ptr_C) + problem_idx * gemm_n;
        ElementC* ptr_D =
            reinterpret_cast<ElementC*>(params.ptr_D) + rows_to_jump * gemm_n;

        LayoutC layout_C(0);
        LayoutC layout_D(gemm_n);

        typename Epilogue::OutputTileIterator::Params params_C(layout_C);
        typename Epilogue::OutputTileIterator::Params params_D(layout_D);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn());

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn());

        Epilogue epilogue(
            shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_D, accumulators, iterator_C);

        // Next tile
        problem_visitor.advance(gridDim.x);
      }
    }
  };

  /*
    To improve compilation speed, we do not compile the device operator if the
    CUDA_ARCH does not correspond to the ArchTag of the cutlass kernel operator.
  */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params,
                  SharedStorage& shared_storage) {  // NOLINT
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700) && (__CUDA_ARCH__ < 750)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm70>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750) && (__CUDA_ARCH__ < 800)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm75>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 910)
    static constexpr bool compile_needed =
        platform::is_same<KernelArch, arch::Sm80>::value;
    KernelRunner<compile_needed>::run_kernel(params, shared_storage);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,  ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,            ///! Epilogue
          typename ThreadblockSwizzle_,  ///! Threadblock swizzling function
          typename KernelArch,  ///! The Architecture this kernel is compiled
                                /// for. Used since SIMT kernels lose top-level
                                /// arch.
          GroupScheduleMode GroupScheduleMode_  ///! Type of scheduling to //
                                                /// NOLINT perform
          >
struct Wint2xMoeFCGemm : public MoeFCGemm<Mma_, Epilogue_, ThreadblockSwizzle_, KernelArch, GroupScheduleMode_> {
 public:
  using Base = MoeFCGemm<Mma_, Epilogue_, ThreadblockSwizzle_, KernelArch, GroupScheduleMode_>;
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed = false;

  // Optional transpose
  using MapArguments = typename Base::MapArguments;

  // Public-facing type definitions related to operand element type, layout, and
  // complex conjugate operation. Must interact with the 'kTransposed' notion.
  static_assert(!kTransposed, "Transpose problem not supported");

  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC = typename MapArguments::LayoutC;
  using ElementScale = ElementC;

  // Type definitions about the mainloop.
  using Operator = typename Mma::Operator;
  using OperatorClass = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag = typename Mma::ArchTag;

  static int const kStages = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC =
      Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = typename Base::ProblemVisitor;
  using Arguments = typename Base::Arguments;

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params : Base::Params {
    // Extra arguments for wint2.0
    uint8_t* local_scale;
    float* code_scale;
    float* code_zp;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params() : Base::Params(), local_scale(nullptr), code_scale(nullptr), code_zp(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args,
           void* workspace = nullptr,
           int tile_count = 0)  // NOLINT
        : Base::Params(args, workspace, tile_count),
          local_scale(args.local_scale),
          code_scale(args.code_scale),
          code_zp(args.code_zp) {}

    CUTLASS_HOST_DEVICE
    void update(Arguments const& args,
                void* workspace = nullptr,
                int tile_count = 0) {
      Base::update(args, workspace, tile_count);

      local_scale = args.local_scale;
      code_scale = args.code_scale;
      code_zp = args.code_zp;
    }
  };

  /// Shared memory storage structure
  using SharedStorage = typename Base::SharedStorage;

 public:

  //
  // Methods
  //

  CUTLASS_DEVICE
  Wint2xMoeFCGemm() {}

  static Status can_implement(Arguments const& args) {
    if (args.quant_method != WintQuantMethod::kWeightOnlyInt2) {
      CUTLASS_TRACE_HOST(
          "Wint2xMoeFCGemm::can_implement() - only support weight_only_int2!");
      return Status::kInvalid;
    } else if (args.weight_scales == nullptr || args.local_scale == nullptr) {
      CUTLASS_TRACE_HOST(
          "Wint2xMoeFCGemm::can_implement() - weight_scales and local_scale is expected to be not nullptr!");
      return Status::kInvalid;
    }
    return Status::kSuccess;
  }

  // The dummy template parameter is not used and exists so that we can compile
  // this code using a standard earlier than C++17. Prior to C++17, fully
  // specialized templates HAD to exists in a namespace
  template <WintQuantMethod QuantMethod, bool B, typename dummy = void>
  struct KernelRunner {
    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      CUTLASS_NOT_IMPLEMENTED();
    }
  };

  template <WintQuantMethod QuantMethod, typename dummy>
  struct KernelRunner<QuantMethod, true, dummy> {
    using WeightQuantTraits = WintQuantTraits<ElementA, QuantMethod>;
    using MmaQuantArguments = typename Mma::QuantParamsAccessor::Arguments;

    CUTLASS_DEVICE
    static MmaQuantArguments prepare_quant_args(
        Params const& params, cutlass::gemm::GemmCoord const& threadblock_offset,
        int64_t problem_idx, const int32_t gemm_k, const int32_t gemm_n, const int thread_idx) {
      // the begin threadblock_offset of scale, which holds the same column id with C, but with no row id
      cutlass::MatrixCoord tb_offset_scale{0, threadblock_offset.n()};
      cutlass::MatrixCoord tb_offset_local_scale{0, threadblock_offset.n() * 2};

      ElementScale* weight_scale_ptr = params.weight_scales + problem_idx * gemm_n;
      typename Mma::QuantParamsAccessor::IteratorSuperScale iterator_super_scale(
          Mma::QuantParamsAccessor::LayoutSuperScale(gemm_n),
          weight_scale_ptr,
          {1, gemm_n},
          thread_idx,
          tb_offset_scale);

      int local_scale_pointer_offset = ((ThreadblockShape::kK + 127) / 128) * (gemm_n * 2);
      int64_t offset_in_bytes = problem_idx * gemm_k * gemm_n / 128;
      uint4b_t *local_scale_ptr = reinterpret_cast<uint4b_t *>(params.local_scale + offset_in_bytes);

      typename Mma::QuantParamsAccessor::IteratorLocalScale iterator_local_scale(
          Mma::QuantParamsAccessor::LayoutLocalScale(gemm_n * 2),
          local_scale_ptr,
          {(gemm_k + 127) / 128, gemm_n * 2},
          thread_idx,
          tb_offset_local_scale);

      float* code_scale_ptr = params.code_scale + problem_idx * gemm_n;
      typename Mma::QuantParamsAccessor::IteratorCodeScaleZp iterator_code_scale(
          Mma::QuantParamsAccessor::LayoutCodeScaleZp(gemm_n),
          code_scale_ptr,
          {1, gemm_n},
          thread_idx,
          tb_offset_scale);

      float* code_zp_ptr = params.code_zp + problem_idx * gemm_n;
      typename Mma::QuantParamsAccessor::IteratorCodeScaleZp iterator_code_zp(
          Mma::QuantParamsAccessor::LayoutCodeScaleZp(gemm_n),
          code_zp_ptr,
          {1, gemm_n},
          thread_idx,
          tb_offset_scale);

      MmaQuantArguments mma_quant_args(
          iterator_super_scale, iterator_local_scale, iterator_code_scale, iterator_code_zp, local_scale_pointer_offset);
      return mma_quant_args;
    }

    CUTLASS_DEVICE
    static void run_kernel(Params const& params,
                           SharedStorage& shared_storage) {  // NOLINT
      //
      // These types shadow the type-level definitions and support the ability
      // to implement a 'transposed' GEMM that computes the transposed problems.
      //

      using ElementA = typename Mma::IteratorA::Element;
      using LayoutA = typename Mma::IteratorA::Layout;
      using ElementB = typename Mma::IteratorB::Element;
      using LayoutB = typename Mma::IteratorB::Layout;
      using ElementC = typename Epilogue::OutputTileIterator::Element;
      using LayoutC = typename Epilogue::OutputTileIterator::Layout;
      using QuantElementB = typename WeightQuantTraits::WeightType;
      using MmaElementB = typename WeightQuantTraits::MmaWeightType;

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
      // wint2.5 and wint2.0 is quantized and packed along k dimension with group_size 64.
      const int64_t quant_gemm_k = WintQuantTraits<ElementA, QuantMethod>::CaclPackedDim(gemm_k);
      int64_t bytes_per_expert_matrix = (quant_gemm_k * gemm_n / 8) * cutlass::sizeof_bits<QuantElementB>::value;

      // Outer 'persistent' loop to iterate over tiles
      while (problem_visitor.next_tile()) {
        GemmCoord problem_size = problem_visitor.problem_size();
        int32_t problem_idx = problem_visitor.problem_index();
        int32_t cta_idx = int32_t(problem_visitor.threadblock_idx());

        GemmCoord grid_shape = problem_visitor.grid_shape(problem_size);

        // threadblock_offset of C
        cutlass::gemm::GemmCoord threadblock_offset(
            int(cta_idx / grid_shape.n()) * Mma::Shape::kM,  // NOLINT
            int(cta_idx % grid_shape.n()) * Mma::Shape::kN,  // NOLINT
            0);

        // Load element pointers. Exchange pointers and strides if working on
        // the transpose
        int64_t rows_to_jump = 0;

        if (params.problem_visitor.total_rows < 0) {
          rows_to_jump = problem_idx == 0 ? 0 : params.problem_visitor.last_row_for_problem[problem_idx - 1];
        } else {
          rows_to_jump = problem_idx * (params.problem_visitor.total_rows / params.problem_visitor.problem_count);
        }

        // begin address offset for A for current tile
        ElementA* ptr_A =
            reinterpret_cast<ElementA*>(params.ptr_A) + rows_to_jump * gemm_k;
        typename LayoutA::LongIndex ldm_A = gemm_k;

        // Compute initial location in logical coordinates
        // the begin threadblock_offset of A, which holds the same row id with C
        cutlass::MatrixCoord tb_offset_A{threadblock_offset.m(), 0};

        // begin address offset for B for current problem_idx, totally num_experts problems
        char* byte_ptr_B = ((char*)params.ptr_B) +                 // NOLINT
                           problem_idx * bytes_per_expert_matrix;  // NOLINT
        ElementB* ptr_B = reinterpret_cast<ElementB*>(byte_ptr_B);
        typename LayoutB::LongIndex ldm_B =
            platform::is_same<layout::RowMajor, LayoutB>::value
                ? gemm_n
                : gemm_k * kInterleave;

        // the begin threadblock_offset of B, which holds the same column id with C
        cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.n() / kInterleave};
        cutlass::MatrixCoord extent_B{problem_size.k() * kInterleave, problem_size.n() / kInterleave};

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(LayoutA(ldm_A),
                                           ptr_A,
                                           {problem_size.m(), problem_size.k()},
                                           thread_idx,
                                           tb_offset_A);

        typename Mma::IteratorB iterator_B(
            LayoutB(ldm_B),
            ptr_B,
            extent_B,
            thread_idx,
            tb_offset_B);

        MmaQuantArguments mma_quant_args = prepare_quant_args(
            params, threadblock_offset, problem_idx, gemm_k, gemm_n, thread_idx);

        typename Mma::FragmentC accumulators;
        accumulators.clear();

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        //
        // Matrix multiply phase
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations =
            (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Wait for all threads to finish their epilogue phases from the
        // previous tile.
        __syncthreads();

        // Compute threadblock-scoped matrix multiply-add
        mma(gemm_k_iterations,
            accumulators,
            iterator_A,
            iterator_B,
            mma_quant_args,
            accumulators);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        ElementC* ptr_C =
            params.ptr_C ? reinterpret_cast<ElementC*>(params.ptr_C) + problem_idx * gemm_n : nullptr;
        ElementC* ptr_D =
            reinterpret_cast<ElementC*>(params.ptr_D) + rows_to_jump * gemm_n;

        LayoutC layout_C(0);
        LayoutC layout_D(gemm_n);

        typename Epilogue::OutputTileIterator::Params params_C(layout_C);
        typename Epilogue::OutputTileIterator::Params params_D(layout_D);

        // Tile iterator loading from source tensor.
        typename Epilogue::OutputTileIterator iterator_C(
            params_C,
            ptr_C,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn());

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            params_D,
            ptr_D,
            problem_size.mn(),
            thread_idx,
            threadblock_offset.mn());

        Epilogue epilogue(
            shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, iterator_D, accumulators, iterator_C);

        // Next tile
        problem_visitor.advance(gridDim.x);
      }
    }
  };

  /*
    To improve compilation speed, we do not compile the device operator if the
    CUDA_ARCH does not correspond to the ArchTag of the cutlass kernel operator.
  */
  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params,
                  SharedStorage& shared_storage) {  // NOLINT
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (__CUDA_ARCH__ < 910)
    KernelRunner<WintQuantMethod::kWeightOnlyInt2, true>::run_kernel(params, shared_storage);
#else
    CUTLASS_NOT_IMPLEMENTED();
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
