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

#include <string>
#include <vector>
#include "helper.h"
#include "moe_fast_hardamard_impl_common.h"

template <typename T, typename OutT>
void MoeFastHardamardWrapper(const T *x_data,
                             const int64_t *expert_idx_per_token,
                             const int64_t *recv_expert_count,
                             const T *shift,
                             const T *smooth,
                             const float *quant_scales,
                             const int quant_round_type,
                             const float quant_max_bound,
                             const float quant_min_bound,
                             const int64_t token_num,
                             const int64_t dim,
                             const int num_max_tokens_per_expert,
                             bool used_in_ep_low_latency,
                             const int hadamard_block_size,
                             OutT *out,
                             cudaStream_t &stream) {
  bool FLAGS_hardamard_use_diagonal_block_matrix = true;

  constexpr int kThreads = 128;
  if (FLAGS_hardamard_use_diagonal_block_matrix) {
    const int VecSize = hadamard_block_size / kThreads;
    const int logN = int(ceil(std::log2(kThreads * VecSize)));
    constexpr int kNChunks = 1;
    DISPATCH_SP_VS(VecSize, VEC_SIZE, {DISPATCH_SP_logN(logN, kLogN, {
                     MoeFastHardamardImplWrapper<T,
                                                 OutT,
                                                 kLogN,
                                                 VEC_SIZE,
                                                 kNChunks,
                                                 kThreads,
                                                 true>(
                         x_data,
                         expert_idx_per_token,
                         recv_expert_count,
                         shift,
                         smooth,
                         quant_scales,
                         quant_round_type,
                         quant_max_bound,
                         quant_min_bound,
                         token_num,
                         dim,
                         num_max_tokens_per_expert,
                         used_in_ep_low_latency,
                         out,
                         stream);
                   })});
  } else {
    if (!((dim / 28) & (dim / 28 - 1))) {
      VLOG(1) << "28 * 2^n";
      const int logN = int(ceil(std::log2(dim / 28)));
      constexpr int kNChunks = 28;
      DISPATCH_SP_logN(logN, kLogN, {
        constexpr int VecSize = (1 << kLogN) / kThreads;
        MoeFastHardamardImplWrapper<T,
                                    OutT,
                                    kLogN,
                                    VecSize,
                                    kNChunks,
                                    kThreads,
                                    false>(x_data,
                                           expert_idx_per_token,
                                           recv_expert_count,
                                           shift,
                                           smooth,
                                           quant_scales,
                                           quant_round_type,
                                           quant_max_bound,
                                           quant_min_bound,
                                           token_num,
                                           dim,
                                           num_max_tokens_per_expert,
                                           used_in_ep_low_latency,
                                           out,
                                           stream);
      });
    } else if (!((dim / 36) & (dim / 36 - 1))) {
      VLOG(1) << "36 * 2^n";
      const int logN = int(ceil(std::log2(dim / 36)));
      constexpr int kNChunks = 36;
      DISPATCH_SP_logN(logN, kLogN, {
        constexpr int VecSize = (1 << kLogN) / kThreads;
        MoeFastHardamardImplWrapper<T,
                                    OutT,
                                    kLogN,
                                    VecSize,
                                    kNChunks,
                                    kThreads,
                                    false>(x_data,
                                           expert_idx_per_token,
                                           recv_expert_count,
                                           shift,
                                           smooth,
                                           quant_scales,
                                           quant_round_type,
                                           quant_max_bound,
                                           quant_min_bound,
                                           token_num,
                                           dim,
                                           num_max_tokens_per_expert,
                                           used_in_ep_low_latency,
                                           out,
                                           stream);
      });
    } else {
      VLOG(1) << "2^n";
      const int logN = int(ceil(std::log2(dim)));
      constexpr int VecSize = 16 / sizeof(T);
      DISPATCH_logN(logN, kLogN, {
        constexpr int kNChunks = (1 << kLogN) / (kThreads * VecSize);
        MoeFastHardamardImplWrapper<T,
                                    OutT,
                                    kLogN,
                                    VecSize,
                                    kNChunks,
                                    kThreads,
                                    false>(x_data,
                                           expert_idx_per_token,
                                           recv_expert_count,
                                           shift,
                                           smooth,
                                           quant_scales,
                                           quant_round_type,
                                           quant_max_bound,
                                           quant_min_bound,
                                           token_num,
                                           dim,
                                           num_max_tokens_per_expert,
                                           used_in_ep_low_latency,
                                           out,
                                           stream);
      });
    }
  }
}

template void MoeFastHardamardWrapper<phi::dtype::float16, phi::dtype::float16>(
    const phi::dtype::float16 *x_data,
    const int64_t *expert_idx_per_token,
    const int64_t *recv_expert_count,
    const phi::dtype::float16 *shift,
    const phi::dtype::float16 *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    const int num_max_tokens_per_expert,
    bool used_in_ep_low_latency,
    const int hadamard_block_size,
    phi::dtype::float16 *out,
    cudaStream_t &stream);

template void MoeFastHardamardWrapper<phi::dtype::float16, int8_t>(
    const phi::dtype::float16 *x_data,
    const int64_t *expert_idx_per_token,
    const int64_t *recv_expert_count,
    const phi::dtype::float16 *shift,
    const phi::dtype::float16 *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    const int num_max_tokens_per_expert,
    bool used_in_ep_low_latency,
    const int hadamard_block_size,
    int8_t *out,
    cudaStream_t &stream);

template void
MoeFastHardamardWrapper<phi::dtype::bfloat16, phi::dtype::bfloat16>(
    const phi::dtype::bfloat16 *x_data,
    const int64_t *expert_idx_per_token,
    const int64_t *recv_expert_count,
    const phi::dtype::bfloat16 *shift,
    const phi::dtype::bfloat16 *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    const int num_max_tokens_per_expert,
    bool used_in_ep_low_latency,
    const int hadamard_block_size,
    phi::dtype::bfloat16 *out,
    cudaStream_t &stream);

template void MoeFastHardamardWrapper<phi::dtype::bfloat16, int8_t>(
    const phi::dtype::bfloat16 *x_data,
    const int64_t *expert_idx_per_token,
    const int64_t *recv_expert_count,
    const phi::dtype::bfloat16 *shift,
    const phi::dtype::bfloat16 *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    const int num_max_tokens_per_expert,
    bool used_in_ep_low_latency,
    const int hadamard_block_size,
    int8_t *out,
    cudaStream_t &stream);

template void
MoeFastHardamardWrapper<phi::dtype::bfloat16, phi::dtype::float8_e4m3fn>(
    const phi::dtype::bfloat16 *x_data,
    const int64_t *expert_idx_per_token,
    const int64_t *recv_expert_count,
    const phi::dtype::bfloat16 *shift,
    const phi::dtype::bfloat16 *smooth,
    const float *quant_scales,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound,
    const int64_t token_num,
    const int64_t dim,
    const int num_max_tokens_per_expert,
    bool used_in_ep_low_latency,
    const int hadamard_block_size,
    phi::dtype::float8_e4m3fn *out,
    cudaStream_t &stream);
