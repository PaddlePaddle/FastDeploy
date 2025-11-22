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

#include "cache.hpp"
#include "paddle/extension.h"

namespace dynamic_quant_int2 {

template <typename T,
          typename ScaleType,
          int kBlockSize,
          int kHeadDim,
          int kThreads>
void __global__ write_encoder_c2_cache_kernel(T *k_input,
                                              T *v_input,
                                              uint8_t *cache_k_c2,
                                              uint8_t *cache_v_c2,
                                              const int *cu_seq_k,
                                              const int *encoder_seqs_len,
                                              const int *decoder_seqs_len,
                                              const int *block_tables,
                                              const int64_t *prompt_lens,
                                              const int max_num_blocks_per_seq,
                                              const int data_num_per_block,
                                              const int c16_remain_seq_len,
                                              const int head_num,
                                              const int kv_head_num) {
  const int bidb = blockIdx.x;
  const int bidh = blockIdx.y;
  const int block_idx = blockIdx.z;

  write_c2_cache_kernel<T, ScaleType, kBlockSize, kHeadDim, kThreads, true>(
      k_input,
      v_input,
      cache_k_c2,
      cache_v_c2,
      cu_seq_k,
      encoder_seqs_len,
      decoder_seqs_len,
      block_tables,
      prompt_lens,
      max_num_blocks_per_seq,
      data_num_per_block,
      c16_remain_seq_len,
      head_num,
      kv_head_num,
      bidb,
      bidh,
      block_idx);
}

template <typename T,
          typename ScaleType,
          int kBlockSize,
          int kHeadDim,
          int kThreads>
void __global__ write_encoder_c16_cache_kernel(const T *k_input,
                                               const T *v_input,
                                               T *cache_k_c16,
                                               T *cache_v_c16,
                                               const int *cu_seq_k,
                                               const int *encoder_seqs_len,
                                               const int *decoder_seqs_len,
                                               const int *block_tables,
                                               const int64_t *prompt_lens,
                                               const int max_num_blocks_per_seq,
                                               const int data_num_per_block,
                                               const int c16_remain_seq_len,
                                               const int head_num,
                                               const int kv_head_num) {
  const int bidb = blockIdx.x;
  const int bidh = blockIdx.y;
  const int block_idx = blockIdx.z;
  const int tidx = threadIdx.x;
  const int seq_len_encoder = encoder_seqs_len[bidb];
  const int seq_len_decoder = decoder_seqs_len[bidb];
  const int prompt_len = prompt_lens[bidb];

  if (seq_len_encoder == 0) {
    return;
  }

  const int c16_cache_max_len = c16_remain_seq_len + kBlockSize;

  const int c16_cache_len = prompt_len < c16_cache_max_len
                                ? prompt_len
                                : c16_remain_seq_len + prompt_len % kBlockSize;
  const int c2_cache_len = prompt_len - c16_cache_len;

  int token_idx = block_idx * kBlockSize + c2_cache_len;

  if (seq_len_decoder > c2_cache_len) {
    token_idx += seq_len_decoder - c2_cache_len;
  }

  if (token_idx >= prompt_len ||
      seq_len_encoder + seq_len_decoder <= c2_cache_len) {
    return;
  }

  const int kPackSize = 16 / sizeof(T);
  constexpr int data_per_row = kHeadDim / kPackSize;
  const int row_idx = tidx / data_per_row;
  const int col_idx = tidx % data_per_row * kPackSize;

  int store_idx = (bidb * c16_cache_max_len + token_idx - c2_cache_len) *
                      kv_head_num * kHeadDim +
                  bidh * kHeadDim;

  int load_idx =
      (cu_seq_k[bidb] + token_idx - seq_len_decoder) * kv_head_num * kHeadDim +
      bidh * kHeadDim;

  if (bidh < kv_head_num) {
    for (int i = row_idx; i < kBlockSize; i += (kThreads / data_per_row)) {
      const int load_row = token_idx + i;
      if (load_row < prompt_len) {
        *reinterpret_cast<int4 *>(cache_k_c16 + store_idx +
                                  i * kv_head_num * kHeadDim + col_idx) =
            *reinterpret_cast<const int4 *>(
                k_input + load_idx + i * kv_head_num * kHeadDim + col_idx);
      }
    }
  } else {
    store_idx -= kv_head_num * kHeadDim;
    load_idx -= kv_head_num * kHeadDim;
    for (int i = row_idx; i < kBlockSize; i += (kThreads / data_per_row)) {
      const int load_row = token_idx + i;
      if (load_row < prompt_len) {
        *reinterpret_cast<int4 *>(cache_v_c16 + store_idx +
                                  i * kv_head_num * kHeadDim + col_idx) =
            *reinterpret_cast<const int4 *>(
                v_input + load_idx + i * kv_head_num * kHeadDim + col_idx);
      }
    }
  }
}

template <typename T, typename ScaleType>
void write_encoder_cache(T *k_input,
                         T *v_input,
                         uint8_t *cache_k_c2,
                         uint8_t *cache_v_c2,
                         T *cache_k_c16,
                         T *cache_v_c16,
                         const int *cu_seq_k,
                         const int *encoder_seqs_len,
                         const int *decoder_seq_len,
                         const int *block_tables,
                         const int64_t *prompt_lens,
                         const int max_num_blocks_per_seq,
                         const int data_num_per_block,
                         const int c16_remain_seq_len,
                         const int bsz,
                         const int head_num,
                         const int kv_head_num,
                         const int head_dim,
                         const int encoder_seq_len,
                         cudaStream_t stream) {
  constexpr int kBlockSize = 64;
  constexpr int kHeadDim = 128;
  constexpr int kThreads = 128;
  constexpr int smem_size = 2 * kBlockSize * kHeadDim * sizeof(T);

  int block_num = (encoder_seq_len + kBlockSize - 1) / kBlockSize;
  dim3 gird_dim;
  gird_dim.x = bsz;
  gird_dim.y = kv_head_num;
  gird_dim.z = block_num;
  write_encoder_c2_cache_kernel<T, ScaleType, kBlockSize, kHeadDim, kThreads>
      <<<gird_dim, kThreads, smem_size, stream>>>(k_input,
                                                  v_input,
                                                  cache_k_c2,
                                                  cache_v_c2,
                                                  cu_seq_k,
                                                  encoder_seqs_len,
                                                  decoder_seq_len,
                                                  block_tables,
                                                  prompt_lens,
                                                  max_num_blocks_per_seq,
                                                  data_num_per_block,
                                                  c16_remain_seq_len,
                                                  head_num,
                                                  kv_head_num);

  gird_dim.x = bsz;
  gird_dim.y = kv_head_num * 2;
  gird_dim.z = (c16_remain_seq_len + kBlockSize) / kBlockSize;

  write_encoder_c16_cache_kernel<T, ScaleType, kBlockSize, kHeadDim, kThreads>
      <<<gird_dim, kThreads, smem_size, stream>>>(k_input,
                                                  v_input,
                                                  cache_k_c16,
                                                  cache_v_c16,
                                                  cu_seq_k,
                                                  encoder_seqs_len,
                                                  decoder_seq_len,
                                                  block_tables,
                                                  prompt_lens,
                                                  max_num_blocks_per_seq,
                                                  data_num_per_block,
                                                  c16_remain_seq_len,
                                                  head_num,
                                                  kv_head_num);
}

void WriteEncoderCache(const paddle::Tensor &k_input,
                       const paddle::Tensor &v_input,
                       const paddle::Tensor &cache_k_c2,
                       const paddle::Tensor &cache_v_c2,
                       const paddle::Tensor &cache_k_c16,
                       const paddle::Tensor &cache_v_c16,
                       const paddle::Tensor &cu_seq_k,
                       const paddle::Tensor &encoder_seqs_len,
                       const paddle::Tensor &decoder_seqs_len,
                       const paddle::Tensor &block_table,
                       const paddle::Tensor &prompt_lens,
                       const int c16_remain_seq_len,
                       const int head_num,
                       const int kv_head_num,
                       const int head_dim,
                       const int max_seq_q,
                       const std::string &cache_quant_type_str) {
  using scale_type = cutlass::float_e4m3_t;
  constexpr int kBlockSize = 64;
  const int max_num_blocks_per_seq = block_table.dims()[1];
  const int data_num_per_block =
      kBlockSize * head_dim / 4 + kBlockSize / 32 * head_dim * 4;
  const int bsz = encoder_seqs_len.dims()[0];

  if (k_input.dtype() == paddle::DataType::FLOAT16) {
    using input_type = phi::dtype::float16;
    write_encoder_cache<input_type, scale_type>(
        const_cast<input_type *>(k_input.data<input_type>()),
        const_cast<input_type *>(v_input.data<input_type>()),
        const_cast<uint8_t *>(cache_k_c2.data<uint8_t>()),
        const_cast<uint8_t *>(cache_v_c2.data<uint8_t>()),
        const_cast<input_type *>(cache_k_c16.data<input_type>()),
        const_cast<input_type *>(cache_v_c16.data<input_type>()),
        cu_seq_k.data<int>(),
        encoder_seqs_len.data<int>(),
        decoder_seqs_len.data<int>(),
        block_table.data<int>(),
        prompt_lens.data<int64_t>(),
        max_num_blocks_per_seq,
        data_num_per_block,
        c16_remain_seq_len,
        bsz,
        head_num,
        kv_head_num,
        head_dim,
        max_seq_q,
        k_input.stream());
  } else {
    PD_THROW("BF16 is not supported\n");
  }
}
}  // namespace dynamic_quant_int2

PD_BUILD_OP(dynamic_quant_int2_write_encoder)
    .Inputs({"k_input",
             "v_input",
             "cache_k_c2",
             "cache_v_c2",
             "cache_k_c16",
             "cache_v_c16",
             "cu_seq_k",
             "encoder_seqs_len",
             "decoder_seqs_len",
             "block_table",
             "prompt_lens"})
    .Attrs({"c16_remain_seq_len: int",
            "head_num: int",
            "kv_head_num: int",
            "head_dim: int",
            "max_seq_q: int",
            "cache_quant_type_str: std::string"})
    .Outputs({"cache_k_c2_out",
              "cache_v_c2_out",
              "cache_k_c16_out",
              "cache_v_c16_out"})
    .SetInplaceMap({{"cache_k_c2", "cache_k_c2_out"},
                    {"cache_v_c2", "cache_v_c2_out"},
                    {"cache_k_c16", "cache_k_c16_out"},
                    {"cache_v_c16", "cache_v_c16_out"}})
    .SetKernelFn(PD_KERNEL(dynamic_quant_int2::WriteEncoderCache));
