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

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma once
#include "w4a8_moe_gemm_kernel.h"
#include "cutlass/arch/arch.h"

template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
void generic_w4a8_moe_gemm_kernelLauncher(const IntAType* A,
    const IntBType* B,
    cutlass::epilogue::QuantMode quant_mode,
    const OutputType* col_scale,
    const OutputType* row_scale,
    const int32_t* nf4_look_up_table,
    OutputType* C,
    int64_t* total_rows_before_expert,
    int total_rows,
    int total_rows_in_ll_else_minus1,
    int n,
    int k,
    int num_experts,
    CutlassGemmConfig gemm_config,
    char* workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    int* occupancy);


template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages,
          typename Enable = void>
struct dispatch_stages {
  static void dispatch(const IntAType* A,
                       const IntBType* B,
                       cutlass::epilogue::QuantMode quant_mode,
                       const OutputType* col_scale,
                       const OutputType* row_scale,
                       const int32_t* nf4_look_up_table,
                       OutputType* C,
                       int64_t* total_rows_before_expert,
                       int n,
                       int k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    std::string err_msg = "Cutlass fpA_intB gemm. Not instantiates for arch " +
                          std::to_string(arch::kMinComputeCapability) +
                          " with stages set to " + std::to_string(Stages);
    throw std::runtime_error("[dispatch_stages::dispatch] " +
                             err_msg);
  }
};

template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename arch,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape>
struct dispatch_stages<OutputType,
                        IntAType,
                        IntBType,
                        arch,
                        EpilogueTag,
                        ThreadblockShape,
                        WarpShape,
                        2> {
  static void dispatch(const IntAType* A,
                       const IntBType* B,
                       cutlass::epilogue::QuantMode quant_mode,
                       const OutputType* col_scale,
                       const OutputType* row_scale,
                       const int32_t* nf4_look_up_table,
                       OutputType* C,
                       int64_t* total_rows_before_expert,
                       int total_rows,
                       int total_rows_in_ll_else_minus1,
                       int n,
                       int k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_w4a8_moe_gemm_kernelLauncher<OutputType,
                                            IntAType,
                                            IntBType,
                                            arch,
                                            EpilogueTag,
                                            ThreadblockShape,
                                            WarpShape,
                                            2>(A,
                                       B,
                                       quant_mode,
                                       col_scale,
                                       row_scale,
                                       nf4_look_up_table,
                                       C,
                                       total_rows_before_expert,
                                       total_rows,
                                       total_rows_in_ll_else_minus1,
                                       n,
                                       k,
                                       num_experts,
                                       gemm_config,
                                       workspace,
                                       workspace_bytes,
                                       multi_processor_count,
                                       stream,
                                       occupancy);
  }
};

template <typename OutputType,
          typename IntAType,
          typename IntBType,
          typename EpilogueTag,
          typename ThreadblockShape,
          typename WarpShape,
          int Stages>
struct dispatch_stages<OutputType,
                        IntAType,
                        IntBType,
                        cutlass::arch::Sm80,
                        EpilogueTag,
                        ThreadblockShape,
                        WarpShape,
                       Stages,
                       typename std::enable_if<(Stages > 2)>::type> {
  static void dispatch(const IntAType* A,
                       const IntBType* B,
                       cutlass::epilogue::QuantMode quant_mode,
                       const OutputType* col_scale,
                       const OutputType* row_scale,
                       const int32_t* nf4_look_up_table,
                       OutputType* C,
                       int64_t* total_rows_before_expert,
                       int total_rows,
                       int total_rows_in_ll_else_minus1,
                       int n,
                       int k,
                       int num_experts,
                       CutlassGemmConfig gemm_config,
                       char* workspace,
                       size_t workspace_bytes,
                       int multi_processor_count,
                       cudaStream_t stream,
                       int* occupancy = nullptr) {
    generic_w4a8_moe_gemm_kernelLauncher<OutputType,
                                            IntAType,
                                            IntBType,
                                            cutlass::arch::Sm80,
                                            EpilogueTag,
                                            ThreadblockShape,
                                            WarpShape,
                                            Stages>(A,
                                            B,
                                            quant_mode,
                                             col_scale,
                                             row_scale,
                                             nf4_look_up_table,
                                            C,
                                            total_rows_before_expert,
                                            total_rows,
                                            total_rows_in_ll_else_minus1,
                                            n,
                                            k,
                                            num_experts,
                                            gemm_config,
                                             workspace,
                                             workspace_bytes,
                                            multi_processor_count,
                                            stream,
                                            occupancy);
  }
};


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
                          int64_t gemm_n,
                          int64_t gemm_k,
                          int num_experts,
                          CutlassGemmConfig gemm_config,
                           char* workspace,
                           size_t workspace_bytes,
                          int multi_processor_count,
                          cudaStream_t stream,
                          int* occupancy = nullptr);
