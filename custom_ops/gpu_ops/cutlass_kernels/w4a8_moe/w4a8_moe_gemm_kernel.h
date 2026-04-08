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
#pragma once

#include "cutlass_kernels/w4a8_moe/cutlass_extensions/epilogue/epilogue_quant_helper.h"
#include "w4a4_gemm_configs.h"
#include <string>
#include <vector>

using CutlassGemmConfig = CutlassGemmConfig;
template <typename OutputType,
          typename IntAType, /*The type used for activations/scales/compute*/
          typename IntBType /* The type for the MoE weights */>
class W4A8MoeGemmRunner {
 public:
  W4A8MoeGemmRunner();
  ~W4A8MoeGemmRunner();

  void moe_gemm_bias_act(const IntAType* A,
                     const IntBType* B,
                     cutlass::epilogue::QuantMode quant_mode,
                     const OutputType* col_scale,
                     const OutputType* row_scale,
                     const OutputType* biases,
                     const int32_t* nf4_look_up_table,
                     OutputType* C,
                     int64_t* total_rows_before_expert,
                     int m,
                     int n,
                     int k,
                     int num_experts,
                     std::string activation_type,
                     char* workspace_ptr,
                     const size_t workspace_bytes,
                     cudaStream_t stream);

  void moe_gemm(const IntAType* A,
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
            CutlassGemmConfig gemm_config = CutlassGemmConfig{CutlassTileConfig::CtaShape64x128x64_WarpShape64x32x64,
                                                              SplitKStyle::NO_SPLIT_K,
                                                              1,
                                                              5});
 private:

  template <typename EpilogueTag>
  void dispatch_to_arch(const IntAType* A,
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
                        int* occupancy = nullptr);

  template <typename EpilogueTag>
  void run_gemm(const IntAType* A,
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
                CutlassGemmConfig gemm_config);

  int getWorkspaceSize(
    const int m, const int n, const int k);

 private:
  static constexpr int split_k_limit = 4;
  struct W4A8MoeGEMMConfig {
    int total_rows;
    int n;
    int k;
    int num_experts;
    CutlassTileConfig tile_config;
    SplitKStyle split_k_style;
    int split_k_factor;
    int stages;
  };
  static std::vector<W4A8MoeGEMMConfig> tuned_configs_from_file;

  int sm_ = 80;
  int multi_processor_count_;
};
