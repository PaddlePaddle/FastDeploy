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

#include "cache.hpp"
#include "paddle/extension.h"

namespace dynamic_quant_cache_attn {

template <typename input_type,
          typename output_type,
          typename ScaleType,
          int kBlockSize,
          int kHeadDim>
__global__ void write_decoder_c16_cache_kernel(const input_type* qkv_out,
                                               const input_type* qkv_bias,
                                               const float* rotary_embs,
                                               output_type* q_input,
                                               const int* cu_seq_q,
                                               const int* seq_lens_encoder,
                                               const int* seq_lens_decoder,
                                               uint8_t* cache_k_c2,
                                               uint8_t* cache_v_c2,
                                               output_type* cache_k_c16,
                                               output_type* cache_v_c16,
                                               const int* block_tables,
                                               const int64_t* step_idx,
                                               const int head_num,
                                               const int kv_head_num,
                                               const int c16_remain_seq_len,
                                               const int max_num_blocks_per_seq,
                                               const int data_num_per_block,
                                               const int max_input_length) {
  constexpr int kPackSize = 4;
  using src_type = Vec<input_type, kPackSize>;
  using dst_type = Vec<output_type, kPackSize>;

  using rope_type = Vec<float, kPackSize / 2>;
  using pack_half =
      std::conditional_t<std::is_same<input_type, phi::dtype::float16>::value,
                         __half2,
                         nv_bfloat162>;

  src_type src, bias;
  rope_type sin, cos;
  dst_type dst;

  int bidh = blockIdx.y;
  const int bidb = blockIdx.x;
  const int tidx = threadIdx.x;
  const int seq_len_decoder = seq_lens_decoder[bidb];

  if (seq_len_decoder == 0) {
    return;
  }

  const int c16_max_cache_seq_len = c16_remain_seq_len + kBlockSize;

  const int step = step_idx[bidb];

  int store_token_idx;
  if (seq_len_decoder + 1 < c16_max_cache_seq_len) {
    store_token_idx = seq_len_decoder;
  } else {
    store_token_idx =
        (seq_len_decoder + 1) % kBlockSize + c16_remain_seq_len - 1;
  }

  if (bidh >= head_num && store_token_idx + 1 == c16_remain_seq_len &&
      step > 0 && seq_len_decoder > c16_remain_seq_len - 1) {
    write_c2_cache_kernel<output_type,
                          ScaleType,
                          kBlockSize,
                          kHeadDim,
                          128,
                          false>(cache_k_c16,
                                 cache_v_c16,
                                 cache_k_c2,
                                 cache_v_c2,
                                 cu_seq_q,
                                 seq_lens_encoder,
                                 seq_lens_decoder,
                                 block_tables,
                                 nullptr,
                                 max_num_blocks_per_seq,
                                 data_num_per_block,
                                 c16_remain_seq_len,
                                 head_num,
                                 kv_head_num,
                                 bidb,
                                 bidh - head_num,
                                 0);
  }

  __syncthreads();

  if (tidx >= 32) {
    return;
  }

  const int bias_idx = bidh * kHeadDim + tidx * kPackSize;

  src.load_from(qkv_out +
                cu_seq_q[bidb] * (head_num + 2 * kv_head_num) * kHeadDim +
                bias_idx);

  if (qkv_bias != nullptr) {
    bias.load_from(qkv_bias + bias_idx);
    src.add(bias);
  }

  if (bidh < head_num) {
    const float* cos_rope =
        rotary_embs + seq_len_decoder * (kHeadDim / 2) + tidx * (kPackSize / 2);
    const float* sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

    sin.load_from(sin_rope);
    cos.load_from(cos_rope);
    apply_rotary_embedding<input_type, output_type, kPackSize>(
        src, dst, cos, sin);

    dst.store_to(q_input + bidb * head_num * kHeadDim + bias_idx);
  } else {
    const float* cos_rope =
        rotary_embs + seq_len_decoder * (kHeadDim / 2) + tidx * (kPackSize / 2);
    const float* sin_rope = cos_rope + max_input_length * (kHeadDim / 2);

    sin.load_from(sin_rope);
    cos.load_from(cos_rope);
    apply_rotary_embedding<input_type, output_type, kPackSize>(
        src, dst, cos, sin);

    src.load_from(qkv_out +
                  cu_seq_q[bidb] * (head_num + 2 * kv_head_num) * kHeadDim +
                  bias_idx + kv_head_num * kHeadDim);
    if (qkv_bias != nullptr) {
      bias.load_from(qkv_bias + bias_idx + kv_head_num * kHeadDim);
      src.add(bias);
    }

    bidh -= head_num;
    const int store_idx = (bidb * c16_max_cache_seq_len + store_token_idx) *
                              kv_head_num * kHeadDim +
                          bidh * kHeadDim + tidx * kPackSize;
    dst.store_to(cache_k_c16 + store_idx);

    for (int i = 0; i < kPackSize; i++) {
      dst.data.elt[i] = static_cast<output_type>(src.data.elt[i]);
    }
    dst.store_to(cache_v_c16 + store_idx);
  }
}

template <typename input_type, typename output_type, typename ScaleType>
void write_decoder_c16_cache(const input_type* qkv_out,
                             const input_type* qkv_bias,
                             const float* rotary_embs,
                             output_type* q_input,
                             const int* cu_seq_q,
                             const int* seq_lens_encoder,
                             const int* seq_lens_decoder,
                             uint8_t* cache_k_c2,
                             uint8_t* cache_v_c2,
                             output_type* cache_k_c16,
                             output_type* cache_v_c16,
                             const int* block_tables,
                             const int64_t* step_idx,
                             const int head_num,
                             const int kv_head_num,
                             const int c16_remain_seq_len,
                             const int max_num_blocks_per_seq,
                             const int data_num_per_block,
                             const int max_input_length,
                             const int bsz,
                             cudaStream_t stream) {
  constexpr int kHeadDim = 128;
  constexpr int kThreads = 128;
  constexpr int kBlockSize = 64;

  dim3 gird_dim;
  gird_dim.x = bsz;
  gird_dim.y = head_num + kv_head_num;

  constexpr int smem_size = 2 * kBlockSize * kHeadDim * sizeof(input_type);

  write_decoder_c16_cache_kernel<input_type,
                                 output_type,
                                 ScaleType,
                                 kBlockSize,
                                 kHeadDim>
      <<<gird_dim, kThreads, smem_size, stream>>>(qkv_out,
                                                  qkv_bias,
                                                  rotary_embs,
                                                  q_input,
                                                  cu_seq_q,
                                                  seq_lens_encoder,
                                                  seq_lens_decoder,
                                                  cache_k_c2,
                                                  cache_v_c2,
                                                  cache_k_c16,
                                                  cache_v_c16,
                                                  block_tables,
                                                  step_idx,
                                                  head_num,
                                                  kv_head_num,
                                                  c16_remain_seq_len,
                                                  max_num_blocks_per_seq,
                                                  data_num_per_block,
                                                  max_input_length);
}

std::vector<paddle::Tensor> WriteDecoderCache(
    const paddle::Tensor& qkv_out,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& cache_k_c2,
    const paddle::Tensor& cache_v_c2,
    const paddle::Tensor& cache_k_c16,
    const paddle::Tensor& cache_v_c16,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& encoder_seqs_len,
    const paddle::Tensor& decoder_seqs_len,
    const paddle::Tensor& block_table,
    const paddle::Tensor& step_idx,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const int c16_remain_seq_len,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int max_input_length,
    const std::string& cache_quant_type_str) {
  paddle::Tensor q_input =
      paddle::empty({encoder_seqs_len.dims()[0], head_num, head_dim},
                    paddle::DataType::FLOAT16,
                    qkv_out.place());
  using scale_type = cutlass::float_e4m3_t;
  constexpr int kBlockSize = 64;
  const int data_num_per_block =
      kBlockSize * head_dim / 4 + kBlockSize / 32 * head_dim * 4;

  if (qkv_out.dtype() == paddle::DataType::BFLOAT16) {
    using input_type = phi::dtype::bfloat16;
    write_decoder_c16_cache<input_type, phi::dtype::float16, scale_type>(
        const_cast<input_type*>(qkv_out.data<input_type>()),
        qkv_bias ? const_cast<input_type*>(qkv_bias.get().data<input_type>())
                 : nullptr,
        rotary_embs.data<float>(),
        q_input.data<phi::dtype::float16>(),
        cu_seq_q.data<int>(),
        encoder_seqs_len.data<int>(),
        decoder_seqs_len.data<int>(),
        const_cast<uint8_t*>(cache_k_c2.data<uint8_t>()),
        const_cast<uint8_t*>(cache_v_c2.data<uint8_t>()),
        const_cast<phi::dtype::float16*>(
            cache_k_c16.data<phi::dtype::float16>()),
        const_cast<phi::dtype::float16*>(
            cache_v_c16.data<phi::dtype::float16>()),
        block_table.data<int>(),
        step_idx.data<int64_t>(),
        head_num,
        kv_head_num,
        c16_remain_seq_len,
        block_table.dims()[1],
        data_num_per_block,
        max_input_length,
        encoder_seqs_len.dims()[0],
        qkv_out.stream());
  } else {
    PD_THROW("BF16 is not supported\n");
  }

  return {q_input};
}
}  // namespace dynamic_quant_cache_attn

PD_BUILD_OP(dynamic_quant_int2_write_decoder)
    .Inputs({"qkv_out",
             "rotary_embs",
             "cache_k_c2",
             "cache_v_c2",
             "cache_k_c16",
             "cache_v_c16",
             "cu_seq_q",
             "encoder_seqs_len",
             "decoder_seqs_len",
             "block_table",
             "step_idx",
             paddle::Optional("qkv_bias")})
    .Attrs({"c16_remain_seq_len: int",
            "head_num: int",
            "kv_head_num: int",
            "head_dim: int",
            "max_input_length: int",
            "cache_quant_type_str: std::string"})
    .Outputs({"q_input",
              "cache_k_c2_out",
              "cache_v_c2_out",
              "cache_k_c16_out",
              "cache_v_c16_out"})
    .SetInplaceMap({{"cache_k_c2", "cache_k_c2_out"},
                    {"cache_v_c2", "cache_v_c2_out"},
                    {"cache_k_c16", "cache_k_c16_out"},
                    {"cache_v_c16", "cache_v_c16_out"}})
    .SetKernelFn(PD_KERNEL(dynamic_quant_cache_attn::WriteDecoderCache));
