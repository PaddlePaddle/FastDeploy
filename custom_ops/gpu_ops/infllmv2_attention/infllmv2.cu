// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/extension.h"
#include "infllmv2_attention/infllmv2_impl.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

namespace {

template <paddle::DataType D>
struct InfLLMTypeTraits;

template <>
struct InfLLMTypeTraits<paddle::DataType::FLOAT32> {
  using NativeType = float;
  using PaddleType = float;
};

template <>
struct InfLLMTypeTraits<paddle::DataType::FLOAT16> {
  using NativeType = half;
  using PaddleType = phi::dtype::float16;
};

template <>
struct InfLLMTypeTraits<paddle::DataType::BFLOAT16> {
  using NativeType = __nv_bfloat16;
  using PaddleType = phi::dtype::bfloat16;
};

void CheckCUDA(const paddle::Tensor& tensor, const char* name) {
  PD_CHECK(tensor.is_gpu(), "InfLLM-V2 ", name, " must be a CUDA tensor.");
}

void CheckInt32(const paddle::Tensor& tensor, const char* name) {
  PD_CHECK(tensor.dtype() == paddle::DataType::INT32,
           "InfLLM-V2 ",
           name,
           " must use int32 metadata.");
}

void CheckFloatingType(const paddle::Tensor& tensor, const char* name) {
  const auto dtype = tensor.dtype();
  PD_CHECK(dtype == paddle::DataType::FLOAT32 ||
               dtype == paddle::DataType::FLOAT16 ||
               dtype == paddle::DataType::BFLOAT16,
           "InfLLM-V2 ",
           name,
           " only supports fp32, fp16, or bf16.");
}

void CheckCommonMetadata(const paddle::Tensor& block_tables,
                         const paddle::Tensor& seq_lens_decoder,
                         const paddle::Tensor& seq_lens_this_time,
                         const paddle::Tensor& batch_id_per_token,
                         const paddle::Tensor& cu_seqlens_q,
                         int tokens) {
  for (const auto* tensor : {&block_tables,
                             &seq_lens_decoder,
                             &seq_lens_this_time,
                             &batch_id_per_token,
                             &cu_seqlens_q}) {
    CheckCUDA(*tensor, "runtime metadata");
    CheckInt32(*tensor, "runtime metadata");
  }
  PD_CHECK(block_tables.shape().size() == 2,
           "InfLLM-V2 block_tables must be rank 2.");
  PD_CHECK(seq_lens_decoder.shape().size() == 1 &&
               seq_lens_this_time.shape().size() == 1 &&
               batch_id_per_token.shape().size() == 1 &&
               cu_seqlens_q.shape().size() == 1,
           "InfLLM-V2 sequence metadata must be rank 1.");
  const int64_t batch_size = block_tables.shape()[0];
  PD_CHECK(batch_size > 0 && block_tables.shape()[1] > 0,
           "InfLLM-V2 block_tables dimensions must be positive.");
  PD_CHECK(
      seq_lens_decoder.shape()[0] == batch_size &&
          seq_lens_this_time.shape()[0] == batch_size,
      "InfLLM-V2 sequence-length tensors must match block_tables batch size.");
  PD_CHECK(cu_seqlens_q.shape()[0] == batch_size + 1,
           "InfLLM-V2 cu_seqlens_q must have batch_size + 1 elements.");
  PD_CHECK(batch_id_per_token.shape()[0] == tokens,
           "InfLLM-V2 batch_id_per_token length must match query tokens.");
}

template <paddle::DataType D>
void LaunchUpdateCompressedK(const paddle::Tensor& current_tokens,
                             const paddle::Tensor& key_cache,
                             paddle::Tensor& compressed_k,
                             paddle::Tensor& compressed_k2,
                             const paddle::Tensor& block_tables,
                             const paddle::Tensor& seq_lens_decoder,
                             const paddle::Tensor& batch_id_per_token,
                             const paddle::Tensor& cu_seqlens_q,
                             int kernel_size,
                             int kernel_stride) {
  using Traits = InfLLMTypeTraits<D>;
  using NativeT = typename Traits::NativeType;
  using PaddleT = typename Traits::PaddleType;
  const int tokens = static_cast<int>(current_tokens.shape()[0]);
  const int kv_heads = static_cast<int>(key_cache.shape()[1]);
  const int head_dim = static_cast<int>(key_cache.shape()[3]);
  if (tokens == 0) {
    return;
  }
  dim3 grid(tokens, kv_heads, 2);
  const int threads = std::min(256, std::max(32, head_dim));
  fastdeploy::InfLLMV2UpdateCompressedKKernel<NativeT>
      <<<grid, threads, 0, key_cache.stream()>>>(
          reinterpret_cast<const NativeT*>(key_cache.data<PaddleT>()),
          reinterpret_cast<NativeT*>(compressed_k.data<PaddleT>()),
          reinterpret_cast<NativeT*>(compressed_k2.data<PaddleT>()),
          block_tables.data<int>(),
          seq_lens_decoder.data<int>(),
          batch_id_per_token.data<int>(),
          cu_seqlens_q.data<int>(),
          tokens,
          static_cast<int>(block_tables.shape()[0]),
          static_cast<int>(key_cache.shape()[0]),
          static_cast<int>(block_tables.shape()[1]),
          kv_heads,
          static_cast<int>(key_cache.shape()[2]),
          head_dim,
          kernel_size,
          kernel_stride);
}

template <paddle::DataType D>
void LaunchSelectBlocks(const paddle::Tensor& query,
                        const paddle::Tensor& compressed_k,
                        const paddle::Tensor& compressed_k2,
                        const paddle::Tensor& block_tables,
                        const paddle::Tensor& seq_lens_decoder,
                        const paddle::Tensor& batch_id_per_token,
                        const paddle::Tensor& cu_seqlens_q,
                        paddle::Tensor& topk_indices,
                        paddle::Tensor& block_scores,
                        paddle::Tensor& selected_counts,
                        paddle::Tensor& coarse_lse,
                        paddle::Tensor& coarse_partial_max,
                        paddle::Tensor& coarse_partial_sum,
                        int block_size,
                        int kernel_size,
                        int kernel_stride,
                        int topk,
                        int dense_len,
                        int init_blocks,
                        int local_blocks) {
  using Traits = InfLLMTypeTraits<D>;
  using NativeT = typename Traits::NativeType;
  using PaddleT = typename Traits::PaddleType;
  const int tokens = static_cast<int>(query.shape()[0]);
  const int query_heads = static_cast<int>(query.shape()[1]);
  const int head_dim = static_cast<int>(query.shape()[2]);
  const int kv_heads = static_cast<int>(compressed_k.shape()[1]);
  const int max_blocks = static_cast<int>(block_tables.shape()[1]);
  const int coarse_splits = static_cast<int>(coarse_partial_max.shape()[2]);
  if (tokens == 0) {
    return;
  }
  const int group_size = query_heads / kv_heads;
  constexpr int coarse_threads = 256;
  if constexpr (!std::is_same_v<NativeT, float>) {
    if (group_size == 16 && head_dim % 16 == 0) {
      constexpr int tensor_tile = 16;
      const size_t tensor_shared_k_bytes =
          static_cast<size_t>(tensor_tile) * head_dim * sizeof(NativeT);
      const size_t tensor_scores_offset =
          (tensor_shared_k_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
      const size_t tensor_shared_bytes =
          tensor_scores_offset + tensor_tile * tensor_tile * sizeof(float);
      const dim3 tensor_coarse_grid(tokens, kv_heads, coarse_splits);
      fastdeploy::InfLLMV2CoarseLSETensorCoreSplitKernel<NativeT>
          <<<tensor_coarse_grid, 32, tensor_shared_bytes, query.stream()>>>(
              reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(compressed_k2.data<PaddleT>()),
              block_tables.data<int>(),
              seq_lens_decoder.data<int>(),
              batch_id_per_token.data<int>(),
              cu_seqlens_q.data<int>(),
              coarse_partial_max.data<float>(),
              coarse_partial_sum.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              max_blocks,
              query_heads,
              kv_heads,
              block_size,
              head_dim,
              kernel_size,
              kernel_stride,
              coarse_splits);
      constexpr int combine_threads = 128;
      const dim3 combine_grid(tokens, query_heads);
      fastdeploy::
          InfLLMV2CoarseLSECombineKernel<<<combine_grid,
                                           combine_threads,
                                           2 * combine_threads * sizeof(float),
                                           query.stream()>>>(
              coarse_partial_max.data<float>(),
              coarse_partial_sum.data<float>(),
              batch_id_per_token.data<int>(),
              coarse_lse.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              query_heads,
              coarse_splits);
    } else {
      const dim3 coarse_grid(tokens, query_heads);
      fastdeploy::InfLLMV2CoarseLSEKernel<NativeT>
          <<<coarse_grid,
             coarse_threads,
             2 * coarse_threads * sizeof(float),
             query.stream()>>>(
              reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(compressed_k2.data<PaddleT>()),
              block_tables.data<int>(),
              seq_lens_decoder.data<int>(),
              batch_id_per_token.data<int>(),
              cu_seqlens_q.data<int>(),
              coarse_lse.data<float>(),
              coarse_partial_max.data<float>(),
              coarse_partial_sum.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              max_blocks,
              query_heads,
              kv_heads,
              block_size,
              head_dim,
              kernel_size,
              kernel_stride,
              coarse_splits);
    }
  } else {
    const dim3 coarse_grid(tokens, query_heads);
    fastdeploy::InfLLMV2CoarseLSEKernel<NativeT>
        <<<coarse_grid,
           coarse_threads,
           2 * coarse_threads * sizeof(float),
           query.stream()>>>(
            reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
            reinterpret_cast<const NativeT*>(compressed_k2.data<PaddleT>()),
            block_tables.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            cu_seqlens_q.data<int>(),
            coarse_lse.data<float>(),
            coarse_partial_max.data<float>(),
            coarse_partial_sum.data<float>(),
            tokens,
            static_cast<int>(block_tables.shape()[0]),
            max_blocks,
            query_heads,
            kv_heads,
            block_size,
            head_dim,
            kernel_size,
            kernel_stride,
            coarse_splits);
  }

  constexpr int score_blocks_per_cta = 4;
  const dim3 score_grid(
      tokens,
      kv_heads,
      (max_blocks + score_blocks_per_cta - 1) / score_blocks_per_cta);
  const int score_windows = block_size / kernel_stride + 1;
  const int score_threads = group_size * 32;
  const size_t shared_k_bytes =
      static_cast<size_t>(score_windows) * head_dim * sizeof(NativeT);
  const size_t score_shared_bytes =
      ((shared_k_bytes + alignof(float) - 1) & ~(alignof(float) - 1)) +
      static_cast<size_t>(score_windows) * group_size * sizeof(float);
  if constexpr (!std::is_same_v<NativeT, float>) {
    if (group_size == 16 && head_dim % 16 == 0 && score_windows <= 16) {
      constexpr int tensor_score_warps = 4;
      constexpr int tensor_tile = 16;
      const size_t tensor_shared_k_bytes =
          static_cast<size_t>(tensor_tile) * head_dim * sizeof(NativeT);
      const size_t tensor_scores_offset =
          (tensor_shared_k_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
      const size_t tensor_warp_bytes =
          tensor_scores_offset + tensor_tile * tensor_tile * sizeof(float);
      fastdeploy::InfLLMV2BlockScoreTensorCoreKernel<NativeT>
          <<<score_grid,
             tensor_score_warps * 32,
             tensor_score_warps * tensor_warp_bytes,
             query.stream()>>>(
              reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(compressed_k.data<PaddleT>()),
              block_tables.data<int>(),
              seq_lens_decoder.data<int>(),
              batch_id_per_token.data<int>(),
              cu_seqlens_q.data<int>(),
              coarse_lse.data<float>(),
              block_scores.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              max_blocks,
              query_heads,
              kv_heads,
              block_size,
              head_dim,
              kernel_size,
              kernel_stride,
              init_blocks,
              local_blocks);
    } else {
      fastdeploy::InfLLMV2BlockScoreKernel<NativeT>
          <<<score_grid, score_threads, score_shared_bytes, query.stream()>>>(
              reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(compressed_k.data<PaddleT>()),
              block_tables.data<int>(),
              seq_lens_decoder.data<int>(),
              batch_id_per_token.data<int>(),
              cu_seqlens_q.data<int>(),
              coarse_lse.data<float>(),
              block_scores.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              max_blocks,
              query_heads,
              kv_heads,
              block_size,
              head_dim,
              kernel_size,
              kernel_stride,
              init_blocks,
              local_blocks);
    }
  } else {
    fastdeploy::InfLLMV2BlockScoreKernel<NativeT>
        <<<score_grid, score_threads, score_shared_bytes, query.stream()>>>(
            reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
            reinterpret_cast<const NativeT*>(compressed_k.data<PaddleT>()),
            block_tables.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            cu_seqlens_q.data<int>(),
            coarse_lse.data<float>(),
            block_scores.data<float>(),
            tokens,
            static_cast<int>(block_tables.shape()[0]),
            max_blocks,
            query_heads,
            kv_heads,
            block_size,
            head_dim,
            kernel_size,
            kernel_stride,
            init_blocks,
            local_blocks);
  }

  const dim3 topk_grid(tokens, kv_heads);
  fastdeploy::InfLLMV2TopKKernel<<<topk_grid, 256, 0, query.stream()>>>(
      block_scores.data<float>(),
      seq_lens_decoder.data<int>(),
      batch_id_per_token.data<int>(),
      cu_seqlens_q.data<int>(),
      topk_indices.data<int>(),
      selected_counts.data<int>(),
      tokens,
      static_cast<int>(block_tables.shape()[0]),
      kv_heads,
      max_blocks,
      static_cast<int>(topk_indices.shape()[2]),
      block_size,
      topk,
      dense_len,
      local_blocks);
}

template <paddle::DataType D>
void LaunchSparseAttention(const paddle::Tensor& query,
                           const paddle::Tensor& key_cache,
                           const paddle::Tensor& value_cache,
                           const paddle::Tensor& block_tables,
                           const paddle::Tensor& seq_lens_decoder,
                           const paddle::Tensor& batch_id_per_token,
                           const paddle::Tensor& cu_seqlens_q,
                           const paddle::Tensor& topk_indices,
                           paddle::Tensor& out,
                           paddle::Tensor& partial_acc,
                           paddle::Tensor& partial_max,
                           paddle::Tensor& partial_sum) {
  using Traits = InfLLMTypeTraits<D>;
  using NativeT = typename Traits::NativeType;
  using PaddleT = typename Traits::PaddleType;
  const int tokens = static_cast<int>(query.shape()[0]);
  const int query_heads = static_cast<int>(query.shape()[1]);
  const int head_dim = static_cast<int>(query.shape()[2]);
  const int capacity = static_cast<int>(topk_indices.shape()[2]);
  const int block_size = static_cast<int>(key_cache.shape()[2]);
  if (tokens == 0) {
    return;
  }
  const int kv_heads = static_cast<int>(key_cache.shape()[1]);
  const int group_size = query_heads / kv_heads;
  const int splits = static_cast<int>(partial_acc.shape()[2]);
  const int blocks_per_split = (capacity + splits - 1) / splits;
  const int split_threads = group_size * 32;
  const size_t split_shared_bytes =
      static_cast<size_t>(2 * 8 * head_dim) * sizeof(NativeT);
  const dim3 split_grid(tokens, kv_heads, splits);
  if constexpr (!std::is_same_v<NativeT, float>) {
    if (group_size == 16 && head_dim % 16 == 0) {
      constexpr int tensor_tile = 16;
      constexpr int tensor_warps = 4;
      const size_t tensor_values_bytes =
          static_cast<size_t>(2 * tensor_tile * head_dim) * sizeof(NativeT);
      const size_t tensor_scores_offset =
          (tensor_values_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
      const size_t tensor_shared_bytes =
          tensor_scores_offset + group_size * tensor_tile * sizeof(float) +
          group_size * tensor_tile * sizeof(NativeT) +
          group_size * head_dim * sizeof(float) +
          tensor_warps * group_size * tensor_tile * sizeof(float) +
          3 * group_size * sizeof(float);
      fastdeploy::InfLLMV2SparseAttentionTensorCoreSplitKVKernel<NativeT>
          <<<split_grid, 4 * 32, tensor_shared_bytes, query.stream()>>>(
              reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(key_cache.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(value_cache.data<PaddleT>()),
              block_tables.data<int>(),
              seq_lens_decoder.data<int>(),
              batch_id_per_token.data<int>(),
              cu_seqlens_q.data<int>(),
              topk_indices.data<int>(),
              partial_acc.data<float>(),
              partial_max.data<float>(),
              partial_sum.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              static_cast<int>(key_cache.shape()[0]),
              static_cast<int>(block_tables.shape()[1]),
              query_heads,
              kv_heads,
              block_size,
              head_dim,
              capacity,
              splits,
              blocks_per_split);
    } else {
      fastdeploy::InfLLMV2SparseAttentionSplitKVKernel<NativeT>
          <<<split_grid, split_threads, split_shared_bytes, query.stream()>>>(
              reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(key_cache.data<PaddleT>()),
              reinterpret_cast<const NativeT*>(value_cache.data<PaddleT>()),
              block_tables.data<int>(),
              seq_lens_decoder.data<int>(),
              batch_id_per_token.data<int>(),
              cu_seqlens_q.data<int>(),
              topk_indices.data<int>(),
              partial_acc.data<float>(),
              partial_max.data<float>(),
              partial_sum.data<float>(),
              tokens,
              static_cast<int>(block_tables.shape()[0]),
              static_cast<int>(key_cache.shape()[0]),
              static_cast<int>(block_tables.shape()[1]),
              query_heads,
              kv_heads,
              block_size,
              head_dim,
              capacity,
              splits,
              blocks_per_split);
    }
  } else {
    fastdeploy::InfLLMV2SparseAttentionSplitKVKernel<NativeT>
        <<<split_grid, split_threads, split_shared_bytes, query.stream()>>>(
            reinterpret_cast<const NativeT*>(query.data<PaddleT>()),
            reinterpret_cast<const NativeT*>(key_cache.data<PaddleT>()),
            reinterpret_cast<const NativeT*>(value_cache.data<PaddleT>()),
            block_tables.data<int>(),
            seq_lens_decoder.data<int>(),
            batch_id_per_token.data<int>(),
            cu_seqlens_q.data<int>(),
            topk_indices.data<int>(),
            partial_acc.data<float>(),
            partial_max.data<float>(),
            partial_sum.data<float>(),
            tokens,
            static_cast<int>(block_tables.shape()[0]),
            static_cast<int>(key_cache.shape()[0]),
            static_cast<int>(block_tables.shape()[1]),
            query_heads,
            kv_heads,
            block_size,
            head_dim,
            capacity,
            splits,
            blocks_per_split);
  }
  constexpr int combine_threads = 256;
  const dim3 combine_grid(tokens, query_heads);
  fastdeploy::InfLLMV2SparseAttentionCombineKernel<NativeT>
      <<<combine_grid,
         combine_threads,
         combine_threads * sizeof(float),
         query.stream()>>>(partial_acc.data<float>(),
                           partial_max.data<float>(),
                           partial_sum.data<float>(),
                           reinterpret_cast<NativeT*>(out.data<PaddleT>()),
                           tokens,
                           query_heads,
                           head_dim,
                           splits);
}

}  // namespace

std::vector<paddle::Tensor> InfLLMV2UpdateCompressedK(
    const paddle::Tensor& current_tokens,
    const paddle::Tensor& key_cache,
    paddle::Tensor& compressed_k,
    paddle::Tensor& compressed_k2,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    int kernel_size,
    int kernel_stride) {
  CheckCUDA(current_tokens, "current_tokens");
  CheckCUDA(key_cache, "key_cache");
  CheckCUDA(compressed_k, "compressed_k");
  CheckCUDA(compressed_k2, "compressed_k2");
  CheckFloatingType(key_cache, "key_cache");
  PD_CHECK(current_tokens.dtype() == key_cache.dtype() &&
               compressed_k.dtype() == key_cache.dtype() &&
               compressed_k2.dtype() == key_cache.dtype(),
           "InfLLM-V2 update tensors must have matching dtypes.");
  PD_CHECK(
      current_tokens.shape().size() == 2 || current_tokens.shape().size() == 3,
      "InfLLM-V2 current_tokens must be rank 2 or rank 3.");
  PD_CHECK(key_cache.shape().size() == 4,
           "InfLLM-V2 key_cache must have shape "
           "[physical_blocks, kv_heads, block_size, head_dim].");
  PD_CHECK(
      compressed_k.shape().size() == 4 && compressed_k2.shape().size() == 4,
      "InfLLM-V2 compressed K caches must be rank 4.");
  PD_CHECK(kernel_size > 0 && kernel_stride > 0,
           "InfLLM-V2 kernel_size and kernel_stride must be positive.");
  const int64_t block_size = key_cache.shape()[2];
  PD_CHECK(
      block_size % kernel_stride == 0 && block_size % (4 * kernel_stride) == 0,
      "InfLLM-V2 block_size must be divisible by kernel_stride and "
      "4 * kernel_stride.");
  const std::vector<int64_t> expected_fine = {key_cache.shape()[0],
                                              key_cache.shape()[1],
                                              block_size / kernel_stride,
                                              key_cache.shape()[3]};
  const std::vector<int64_t> expected_coarse = {
      key_cache.shape()[0],
      key_cache.shape()[1],
      block_size / (4 * kernel_stride),
      key_cache.shape()[3]};
  PD_CHECK(
      compressed_k.shape() == expected_fine,
      "InfLLM-V2 compressed_k shape does not match the fine summary layout.");
  PD_CHECK(compressed_k2.shape() == expected_coarse,
           "InfLLM-V2 compressed_k2 shape does not match the coarse summary "
           "layout.");
  CheckCommonMetadata(block_tables,
                      seq_lens_decoder,
                      seq_lens_this_time,
                      batch_id_per_token,
                      cu_seqlens_q,
                      static_cast<int>(current_tokens.shape()[0]));

  switch (key_cache.dtype()) {
    case paddle::DataType::FLOAT32:
      LaunchUpdateCompressedK<paddle::DataType::FLOAT32>(current_tokens,
                                                         key_cache,
                                                         compressed_k,
                                                         compressed_k2,
                                                         block_tables,
                                                         seq_lens_decoder,
                                                         batch_id_per_token,
                                                         cu_seqlens_q,
                                                         kernel_size,
                                                         kernel_stride);
      break;
    case paddle::DataType::FLOAT16:
      LaunchUpdateCompressedK<paddle::DataType::FLOAT16>(current_tokens,
                                                         key_cache,
                                                         compressed_k,
                                                         compressed_k2,
                                                         block_tables,
                                                         seq_lens_decoder,
                                                         batch_id_per_token,
                                                         cu_seqlens_q,
                                                         kernel_size,
                                                         kernel_stride);
      break;
    case paddle::DataType::BFLOAT16:
      LaunchUpdateCompressedK<paddle::DataType::BFLOAT16>(current_tokens,
                                                          key_cache,
                                                          compressed_k,
                                                          compressed_k2,
                                                          block_tables,
                                                          seq_lens_decoder,
                                                          batch_id_per_token,
                                                          cu_seqlens_q,
                                                          kernel_size,
                                                          kernel_stride);
      break;
    default:
      PD_THROW("InfLLM-V2 update only supports fp32, fp16, or bf16.");
  }
  return {compressed_k, compressed_k2};
}

std::vector<paddle::Tensor> InfLLMV2SelectBlocks(
    const paddle::Tensor& query,
    const paddle::Tensor& compressed_k,
    const paddle::Tensor& compressed_k2,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    paddle::Tensor& topk_indices,
    paddle::Tensor& block_scores,
    paddle::Tensor& selected_counts,
    paddle::Tensor& coarse_lse,
    paddle::Tensor& coarse_partial_max,
    paddle::Tensor& coarse_partial_sum,
    int block_size,
    int kernel_size,
    int kernel_stride,
    int topk,
    int dense_len,
    int init_blocks,
    int local_blocks) {
  for (const paddle::Tensor* tensor :
       std::vector<const paddle::Tensor*>{&query,
                                          &compressed_k,
                                          &compressed_k2,
                                          &topk_indices,
                                          &block_scores,
                                          &selected_counts,
                                          &coarse_lse,
                                          &coarse_partial_max,
                                          &coarse_partial_sum}) {
    CheckCUDA(*tensor, "Stage 1 tensor");
  }
  CheckFloatingType(query, "query");
  PD_CHECK(query.shape().size() == 3,
           "InfLLM-V2 query must have shape [tokens, heads, head_dim].");
  PD_CHECK(
      compressed_k.shape().size() == 4 && compressed_k2.shape().size() == 4,
      "InfLLM-V2 compressed K caches must be rank 4.");
  PD_CHECK(query.dtype() == compressed_k.dtype() &&
               query.dtype() == compressed_k2.dtype(),
           "InfLLM-V2 query and compressed K dtypes must match.");
  PD_CHECK(block_size > 0 && kernel_size > 0 && kernel_stride > 0 && topk > 0 &&
               dense_len >= 0 && init_blocks >= 0 && local_blocks >= 0,
           "InfLLM-V2 Stage 1 attributes are outside their valid ranges.");
  PD_CHECK(
      block_size % kernel_stride == 0 && block_size % (4 * kernel_stride) == 0,
      "InfLLM-V2 block_size must be divisible by kernel_stride and "
      "4 * kernel_stride.");
  const int64_t tokens = query.shape()[0];
  const int64_t query_heads = query.shape()[1];
  const int64_t head_dim = query.shape()[2];
  const int64_t kv_heads = compressed_k.shape()[1];
  const int64_t max_blocks = block_tables.shape()[1];
  PD_CHECK(kv_heads > 0 && query_heads % kv_heads == 0,
           "InfLLM-V2 query heads must be divisible by KV heads.");
  PD_CHECK(query_heads / kv_heads <= 32,
           "InfLLM-V2 Stage 1 supports at most 32 query heads per KV head.");
  PD_CHECK(head_dim > 0 && head_dim <= 256 &&
               head_dim == compressed_k.shape()[3] &&
               head_dim == compressed_k2.shape()[3],
           "InfLLM-V2 query and compressed K head dimensions must match.");
  PD_CHECK(compressed_k.shape()[2] == block_size / kernel_stride &&
               compressed_k2.shape()[2] == block_size / (4 * kernel_stride),
           "InfLLM-V2 compressed K semantic strides are incompatible with "
           "block_size.");
  PD_CHECK(topk_indices.dtype() == paddle::DataType::INT32 &&
               selected_counts.dtype() == paddle::DataType::INT32,
           "InfLLM-V2 topk_indices and selected_counts must be int32.");
  PD_CHECK(block_scores.dtype() == paddle::DataType::FLOAT32 &&
               coarse_lse.dtype() == paddle::DataType::FLOAT32 &&
               coarse_partial_max.dtype() == paddle::DataType::FLOAT32 &&
               coarse_partial_sum.dtype() == paddle::DataType::FLOAT32,
           "InfLLM-V2 Stage 1 score workspaces must be float32.");
  PD_CHECK(topk_indices.shape().size() == 3 &&
               topk_indices.shape()[0] == tokens &&
               topk_indices.shape()[1] == kv_heads,
           "InfLLM-V2 topk_indices must have shape "
           "[tokens, kv_heads, capacity].");
  PD_CHECK(block_scores.shape() ==
               std::vector<int64_t>({tokens, kv_heads, max_blocks}),
           "InfLLM-V2 block_scores shape is invalid.");
  PD_CHECK(selected_counts.shape() == std::vector<int64_t>({tokens, kv_heads}),
           "InfLLM-V2 selected_counts shape is invalid.");
  PD_CHECK(coarse_lse.shape() == std::vector<int64_t>({tokens, query_heads}),
           "InfLLM-V2 coarse_lse shape is invalid.");
  PD_CHECK(coarse_partial_max.shape().size() == 3 &&
               coarse_partial_max.shape() == coarse_partial_sum.shape() &&
               coarse_partial_max.shape()[0] == tokens &&
               coarse_partial_max.shape()[1] == query_heads &&
               coarse_partial_max.shape()[2] > 0,
           "InfLLM-V2 coarse partial workspace shape is invalid.");
  const int64_t required_capacity = std::min<int64_t>(
      max_blocks,
      std::max<int64_t>(topk + local_blocks,
                        (dense_len + block_size - 1) / block_size));
  PD_CHECK(topk_indices.shape()[2] >= required_capacity,
           "InfLLM-V2 topk_indices capacity is too small for sparse and "
           "select-all modes.");
  PD_CHECK(max_blocks <= 2048,
           "InfLLM-V2 Stage 1 supports at most 2048 candidate blocks.");
  CheckCommonMetadata(block_tables,
                      seq_lens_decoder,
                      seq_lens_this_time,
                      batch_id_per_token,
                      cu_seqlens_q,
                      static_cast<int>(tokens));

  switch (query.dtype()) {
    case paddle::DataType::FLOAT32:
      LaunchSelectBlocks<paddle::DataType::FLOAT32>(query,
                                                    compressed_k,
                                                    compressed_k2,
                                                    block_tables,
                                                    seq_lens_decoder,
                                                    batch_id_per_token,
                                                    cu_seqlens_q,
                                                    topk_indices,
                                                    block_scores,
                                                    selected_counts,
                                                    coarse_lse,
                                                    coarse_partial_max,
                                                    coarse_partial_sum,
                                                    block_size,
                                                    kernel_size,
                                                    kernel_stride,
                                                    topk,
                                                    dense_len,
                                                    init_blocks,
                                                    local_blocks);
      break;
    case paddle::DataType::FLOAT16:
      LaunchSelectBlocks<paddle::DataType::FLOAT16>(query,
                                                    compressed_k,
                                                    compressed_k2,
                                                    block_tables,
                                                    seq_lens_decoder,
                                                    batch_id_per_token,
                                                    cu_seqlens_q,
                                                    topk_indices,
                                                    block_scores,
                                                    selected_counts,
                                                    coarse_lse,
                                                    coarse_partial_max,
                                                    coarse_partial_sum,
                                                    block_size,
                                                    kernel_size,
                                                    kernel_stride,
                                                    topk,
                                                    dense_len,
                                                    init_blocks,
                                                    local_blocks);
      break;
    case paddle::DataType::BFLOAT16:
      LaunchSelectBlocks<paddle::DataType::BFLOAT16>(query,
                                                     compressed_k,
                                                     compressed_k2,
                                                     block_tables,
                                                     seq_lens_decoder,
                                                     batch_id_per_token,
                                                     cu_seqlens_q,
                                                     topk_indices,
                                                     block_scores,
                                                     selected_counts,
                                                     coarse_lse,
                                                     coarse_partial_max,
                                                     coarse_partial_sum,
                                                     block_size,
                                                     kernel_size,
                                                     kernel_stride,
                                                     topk,
                                                     dense_len,
                                                     init_blocks,
                                                     local_blocks);
      break;
    default:
      PD_THROW("InfLLM-V2 Stage 1 only supports fp32, fp16, or bf16.");
  }
  return {topk_indices,
          block_scores,
          selected_counts,
          coarse_lse,
          coarse_partial_max,
          coarse_partial_sum};
}

std::vector<paddle::Tensor> InfLLMV2AttentionForward(
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& topk_indices,
    paddle::Tensor& out,
    paddle::Tensor& partial_acc,
    paddle::Tensor& partial_max,
    paddle::Tensor& partial_sum) {
  for (const paddle::Tensor* tensor :
       std::vector<const paddle::Tensor*>{&query,
                                          &key_cache,
                                          &value_cache,
                                          &topk_indices,
                                          &out,
                                          &partial_acc,
                                          &partial_max,
                                          &partial_sum}) {
    CheckCUDA(*tensor, "Stage 2 tensor");
  }
  CheckFloatingType(query, "query");
  PD_CHECK(query.shape().size() == 3,
           "InfLLM-V2 query must have shape [tokens, heads, head_dim].");
  PD_CHECK(key_cache.shape().size() == 4 && value_cache.shape().size() == 4,
           "InfLLM-V2 K/V cache must be paged rank-4 tensors.");
  PD_CHECK(query.dtype() == key_cache.dtype() &&
               query.dtype() == value_cache.dtype() &&
               query.dtype() == out.dtype(),
           "InfLLM-V2 query, K/V cache, and output dtypes must match.");
  PD_CHECK(key_cache.shape() == value_cache.shape(),
           "InfLLM-V2 key_cache and value_cache shapes must match.");
  const int64_t tokens = query.shape()[0];
  const int64_t query_heads = query.shape()[1];
  const int64_t head_dim = query.shape()[2];
  const int64_t kv_heads = key_cache.shape()[1];
  PD_CHECK(kv_heads > 0 && query_heads % kv_heads == 0,
           "InfLLM-V2 query heads must be divisible by KV heads.");
  PD_CHECK(head_dim == key_cache.shape()[3] && head_dim <= 256,
           "InfLLM-V2 Stage 2 requires matching head_dim in [1, 256].");
  PD_CHECK(topk_indices.dtype() == paddle::DataType::INT32 &&
               topk_indices.shape().size() == 3 &&
               topk_indices.shape()[0] == tokens &&
               topk_indices.shape()[1] == kv_heads &&
               topk_indices.shape()[2] > 0,
           "InfLLM-V2 topk_indices must have shape "
           "[tokens, kv_heads, capacity] and dtype int32.");
  PD_CHECK(out.shape() == query.shape(),
           "InfLLM-V2 output workspace shape must match query.");
  PD_CHECK(partial_acc.dtype() == paddle::DataType::FLOAT32 &&
               partial_max.dtype() == paddle::DataType::FLOAT32 &&
               partial_sum.dtype() == paddle::DataType::FLOAT32,
           "InfLLM-V2 Stage 2 partial workspaces must be float32.");
  PD_CHECK(
      partial_acc.shape().size() == 4 && partial_acc.shape()[0] == tokens &&
          partial_acc.shape()[1] == query_heads && partial_acc.shape()[2] > 0 &&
          partial_acc.shape()[2] <= topk_indices.shape()[2] &&
          partial_acc.shape()[3] == head_dim,
      "InfLLM-V2 partial_acc shape is invalid.");
  PD_CHECK(partial_max.shape().size() == 3 &&
               partial_max.shape() == partial_sum.shape() &&
               partial_max.shape()[0] == tokens &&
               partial_max.shape()[1] == query_heads &&
               partial_max.shape()[2] == partial_acc.shape()[2],
           "InfLLM-V2 partial max/sum shapes are invalid.");
  CheckCommonMetadata(block_tables,
                      seq_lens_decoder,
                      seq_lens_this_time,
                      batch_id_per_token,
                      cu_seqlens_q,
                      static_cast<int>(tokens));

  switch (query.dtype()) {
    case paddle::DataType::FLOAT32:
      LaunchSparseAttention<paddle::DataType::FLOAT32>(query,
                                                       key_cache,
                                                       value_cache,
                                                       block_tables,
                                                       seq_lens_decoder,
                                                       batch_id_per_token,
                                                       cu_seqlens_q,
                                                       topk_indices,
                                                       out,
                                                       partial_acc,
                                                       partial_max,
                                                       partial_sum);
      break;
    case paddle::DataType::FLOAT16:
      LaunchSparseAttention<paddle::DataType::FLOAT16>(query,
                                                       key_cache,
                                                       value_cache,
                                                       block_tables,
                                                       seq_lens_decoder,
                                                       batch_id_per_token,
                                                       cu_seqlens_q,
                                                       topk_indices,
                                                       out,
                                                       partial_acc,
                                                       partial_max,
                                                       partial_sum);
      break;
    case paddle::DataType::BFLOAT16:
      LaunchSparseAttention<paddle::DataType::BFLOAT16>(query,
                                                        key_cache,
                                                        value_cache,
                                                        block_tables,
                                                        seq_lens_decoder,
                                                        batch_id_per_token,
                                                        cu_seqlens_q,
                                                        topk_indices,
                                                        out,
                                                        partial_acc,
                                                        partial_max,
                                                        partial_sum);
      break;
    default:
      PD_THROW("InfLLM-V2 Stage 2 only supports fp32, fp16, or bf16.");
  }
  return {out, partial_acc, partial_max, partial_sum};
}

std::vector<std::vector<int64_t>> InfLLMV2UpdateCompressedKInferShape(
    const std::vector<int64_t>& current_tokens_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& compressed_k_shape,
    const std::vector<int64_t>& compressed_k2_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    int kernel_size,
    int kernel_stride) {
  return {compressed_k_shape, compressed_k2_shape};
}

std::vector<paddle::DataType> InfLLMV2UpdateCompressedKInferDtype(
    const paddle::DataType& current_tokens_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& compressed_k_dtype,
    const paddle::DataType& compressed_k2_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype) {
  return {compressed_k_dtype, compressed_k2_dtype};
}

std::vector<std::vector<int64_t>> InfLLMV2SelectBlocksInferShape(
    const std::vector<int64_t>& query_shape,
    const std::vector<int64_t>& compressed_k_shape,
    const std::vector<int64_t>& compressed_k2_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& topk_indices_shape,
    const std::vector<int64_t>& block_scores_shape,
    const std::vector<int64_t>& selected_counts_shape,
    const std::vector<int64_t>& coarse_lse_shape,
    const std::vector<int64_t>& coarse_partial_max_shape,
    const std::vector<int64_t>& coarse_partial_sum_shape,
    int block_size,
    int kernel_size,
    int kernel_stride,
    int topk,
    int dense_len,
    int init_blocks,
    int local_blocks) {
  return {topk_indices_shape,
          block_scores_shape,
          selected_counts_shape,
          coarse_lse_shape,
          coarse_partial_max_shape,
          coarse_partial_sum_shape};
}

std::vector<paddle::DataType> InfLLMV2SelectBlocksInferDtype(
    const paddle::DataType& query_dtype,
    const paddle::DataType& compressed_k_dtype,
    const paddle::DataType& compressed_k2_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& topk_indices_dtype,
    const paddle::DataType& block_scores_dtype,
    const paddle::DataType& selected_counts_dtype,
    const paddle::DataType& coarse_lse_dtype,
    const paddle::DataType& coarse_partial_max_dtype,
    const paddle::DataType& coarse_partial_sum_dtype) {
  return {topk_indices_dtype,
          block_scores_dtype,
          selected_counts_dtype,
          coarse_lse_dtype,
          coarse_partial_max_dtype,
          coarse_partial_sum_dtype};
}

std::vector<std::vector<int64_t>> InfLLMV2AttentionForwardInferShape(
    const std::vector<int64_t>& query_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& topk_indices_shape,
    const std::vector<int64_t>& out_shape,
    const std::vector<int64_t>& partial_acc_shape,
    const std::vector<int64_t>& partial_max_shape,
    const std::vector<int64_t>& partial_sum_shape) {
  return {out_shape, partial_acc_shape, partial_max_shape, partial_sum_shape};
}

std::vector<paddle::DataType> InfLLMV2AttentionForwardInferDtype(
    const paddle::DataType& query_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& topk_indices_dtype,
    const paddle::DataType& out_dtype,
    const paddle::DataType& partial_acc_dtype,
    const paddle::DataType& partial_max_dtype,
    const paddle::DataType& partial_sum_dtype) {
  return {out_dtype, partial_acc_dtype, partial_max_dtype, partial_sum_dtype};
}

PD_BUILD_STATIC_OP(infllmv2_update_compressed_k)
    .Inputs({"current_tokens",
             "key_cache",
             "compressed_k",
             "compressed_k2",
             "block_tables",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q"})
    .Outputs({"compressed_k_out", "compressed_k2_out"})
    .SetInplaceMap({{"compressed_k", "compressed_k_out"},
                    {"compressed_k2", "compressed_k2_out"}})
    .Attrs({"kernel_size: int", "kernel_stride: int"})
    .SetKernelFn(PD_KERNEL(InfLLMV2UpdateCompressedK))
    .SetInferShapeFn(PD_INFER_SHAPE(InfLLMV2UpdateCompressedKInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InfLLMV2UpdateCompressedKInferDtype));

PD_BUILD_STATIC_OP(infllmv2_select_blocks)
    .Inputs({"query",
             "compressed_k",
             "compressed_k2",
             "block_tables",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q",
             "topk_indices",
             "block_scores",
             "selected_counts",
             "coarse_lse",
             "coarse_partial_max",
             "coarse_partial_sum"})
    .Outputs({"topk_indices_out",
              "block_scores_out",
              "selected_counts_out",
              "coarse_lse_out",
              "coarse_partial_max_out",
              "coarse_partial_sum_out"})
    .SetInplaceMap({{"topk_indices", "topk_indices_out"},
                    {"block_scores", "block_scores_out"},
                    {"selected_counts", "selected_counts_out"},
                    {"coarse_lse", "coarse_lse_out"},
                    {"coarse_partial_max", "coarse_partial_max_out"},
                    {"coarse_partial_sum", "coarse_partial_sum_out"}})
    .Attrs({"block_size: int",
            "kernel_size: int",
            "kernel_stride: int",
            "topk: int",
            "dense_len: int",
            "init_blocks: int",
            "local_blocks: int"})
    .SetKernelFn(PD_KERNEL(InfLLMV2SelectBlocks))
    .SetInferShapeFn(PD_INFER_SHAPE(InfLLMV2SelectBlocksInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InfLLMV2SelectBlocksInferDtype));

PD_BUILD_STATIC_OP(infllmv2_attention_forward)
    .Inputs({"query",
             "key_cache",
             "value_cache",
             "block_tables",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q",
             "topk_indices",
             "out",
             "partial_acc",
             "partial_max",
             "partial_sum"})
    .Outputs(
        {"out_alias", "partial_acc_out", "partial_max_out", "partial_sum_out"})
    .SetInplaceMap({{"out", "out_alias"},
                    {"partial_acc", "partial_acc_out"},
                    {"partial_max", "partial_max_out"},
                    {"partial_sum", "partial_sum_out"}})
    .SetKernelFn(PD_KERNEL(InfLLMV2AttentionForward))
    .SetInferShapeFn(PD_INFER_SHAPE(InfLLMV2AttentionForwardInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(InfLLMV2AttentionForwardInferDtype));
