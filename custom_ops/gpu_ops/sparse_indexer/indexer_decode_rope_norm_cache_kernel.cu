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

#include "indexer_decode_rope_norm_cache_kernel.h"
#include "append_attn/utils.cuh"

template <typename T, typename QKV_TYPE, int HEAD_DIM>
void indexer_decode_rope_norm_cache(const QKV_TYPE* qkv,
                                      T* key_cache,
                                      T* qkv_out,
                                      const int* block_tables,
                                      const int* cu_seqlens_q,
                                      const int* seq_lens,
                                      const int* seq_lens_encoder,
                                      const float* cos_emb,
                                      const float* sin_emb,
                                      const float* qkv_out_scales,
                                      const T* qkv_biases,
                                      const int max_seq_len,
                                      const int max_blocks_per_seq,
                                      const int num_heads,
                                      const int kv_num_heads,
                                      const int dim_head,
                                      const int block_size,
                                      const int bsz,
                                      const cudaStream_t& stream,
                                      const bool use_neox_style,
                                      const bool rope_3d,
                                      const float* q_norm_weight,
                                      const float* k_norm_weight,
                                      const float rms_norm_eps) {
  const uint32_t elem_nums =
      use_neox_style ? bsz * (num_heads + kv_num_heads) * dim_head / 2
                     : bsz * (num_heads + kv_num_heads) * dim_head;
  // constexpr int HEAD_DIM = 128;

  constexpr int PackSize = HEAD_DIM / kWarpSize;
  const int pack_num = elem_nums / PackSize;
  const int blocksize = 128;
  int grid_size = 1;
  GetNumBlocks<128>(pack_num, &grid_size);
  dim3 block_dim(kWarpSize, blocksize / kWarpSize, 1);
  // grid_size: 2, block_dim: 32,4,1, PackSize: 4, elem_nums: 768
  
  // grid_size: 3, block_dim: 32,4,1, PackSize: 2, elem_nums: 768
  // printf("grid_size: %d, block_dim: %d,%d,%d, PackSize: %d, elem_nums: %d\n", grid_size, block_dim.x,block_dim.y,block_dim.z, PackSize, elem_nums);
  launchWithPdlWhenEnabled(
      indexer_decode_cache_T_rope_qk_norm_kernel<T, PackSize>,
      grid_size,
      block_dim,
      0,
      stream,
      reinterpret_cast<const T*>(qkv),
      key_cache,
      qkv_out,
      block_tables,
      cu_seqlens_q,
      seq_lens,
      seq_lens_encoder,
      cos_emb,
      sin_emb,
      max_seq_len,
      max_blocks_per_seq,
      num_heads,
      dim_head,
      block_size,
      elem_nums,
      kv_num_heads,
      rope_3d,
      q_norm_weight,
      k_norm_weight,
      rms_norm_eps);
}

template <typename T, typename QKV_TYPE>
void IndexerDecoderRoPENormWriteCacheKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const float rms_norm_eps) {
  typedef cascade_attn_type_traits<T> traits_;
  typedef cascade_attn_type_traits<QKV_TYPE> qkt_nv_type_;
  typedef typename traits_::type DataType_;
  typedef typename qkt_nv_type_::type QKV_Data_TYPE;
  const QKV_TYPE* qkv_ptr = qkv.data<QKV_TYPE>();

  auto max_blocks_per_seq = meta_data.max_blocks_per_seq;
  auto bsz = meta_data.batch_size;
  auto block_size = meta_data.block_size;
  auto dim_head = meta_data.head_dims;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;

  const float* cos_emb =
      rotary_embs ? rotary_embs.get().data<float>() : nullptr;
  const float* sin_emb;
  int rotary_dim = dim_head;
  if (rotary_embs) {
    sin_emb =
        use_neox_rotary_style
            ? rotary_embs.get().data<float>() + max_seq_len * dim_head
            : rotary_embs.get().data<float>() + max_seq_len * dim_head / 2;
    rotary_dim =
        rotary_embs.get().dims()[rotary_embs.get().dims().size() - 1] * 2;
    if (rotary_dim < dim_head) {
      if (!use_neox_rotary_style || qkv_out_scales || q_norm_weight ||
          k_norm_weight || cache_quant_type_str != "none") {
        PADDLE_THROW(phi::errors::Fatal(
            "partial_rotary_factor < 1.0 only supports neox_rotary_style=True, "
            "qkv_out_scales is None, q_norm_weight/k_norm_weight) is None, and "
            "cache_quant_type_str is 'none'."));
      }
      sin_emb = rotary_embs.get().data<float>() + max_seq_len * rotary_dim / 2;
    }
  }

  if (q_norm_weight && k_norm_weight) {
    if (cache_quant_type_str == "none") {
      if (dim_head == 128) {

       indexer_decode_rope_norm_cache<DataType_,QKV_TYPE,128>(
          reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
          reinterpret_cast<DataType_*>(key_cache_out->data<T>()),
          reinterpret_cast<DataType_*>(qkv_out->data<T>()),
          block_tables.data<int>(),
          cu_seqlens_q.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          cos_emb,
          sin_emb,
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(qkv_biases.get().data<T>()))
                     : nullptr,
          max_seq_len,
          max_blocks_per_seq,
          num_heads,
          kv_num_heads,
          dim_head,
          block_size,
          bsz,
          stream,
          use_neox_rotary_style,
          rope_3d,
          q_norm_weight ? q_norm_weight.get().data<float>() : nullptr,
          k_norm_weight ? k_norm_weight.get().data<float>() : nullptr,
          rms_norm_eps);
      }else if(dim_head ==64) {
       indexer_decode_rope_norm_cache<DataType_,QKV_TYPE,64>(
          reinterpret_cast<const QKV_TYPE*>(qkv_ptr),
          reinterpret_cast<DataType_*>(key_cache_out->data<T>()),
          reinterpret_cast<DataType_*>(qkv_out->data<T>()),
          block_tables.data<int>(),
          cu_seqlens_q.data<int>(),
          seq_lens.data<int>(),
          seq_lens_encoder.data<int>(),
          cos_emb,
          sin_emb,
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? reinterpret_cast<DataType_*>(
                           const_cast<T*>(qkv_biases.get().data<T>()))
                     : nullptr,
          max_seq_len,
          max_blocks_per_seq,
          num_heads,
          kv_num_heads,
          dim_head,
          block_size,
          bsz,
          stream,
          use_neox_rotary_style,
          rope_3d,
          q_norm_weight ? q_norm_weight.get().data<float>() : nullptr,
          k_norm_weight ? k_norm_weight.get().data<float>() : nullptr,
          rms_norm_eps);
      }else{
        PADDLE_THROW(phi::errors::Fatal(
            "only support dim_head=64 or 128"));
      }
    } else {
      PADDLE_THROW(phi::errors::Fatal(
          "only support cache_quant_type_str=None"));
    }
  }
}
template void
IndexerDecoderRoPENormWriteCacheKernel<paddle::bfloat16, paddle::bfloat16>(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_seq_len,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const float rms_norm_eps);
