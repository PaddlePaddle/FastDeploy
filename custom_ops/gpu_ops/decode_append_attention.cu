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

#include "append_attention/decode_append_attention_c8_impl.cuh"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T>
class type2value;

template <>
class type2value<phi::dtype::bfloat16> {
 public:
  static constexpr paddle::DataType value = paddle::DataType::BFLOAT16;
};

template <>
class type2value<phi::dtype::float16> {
 public:
  static constexpr paddle::DataType value = paddle::DataType::FLOAT16;
};

std::vector<paddle::Tensor> DecodeAppendAttention(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& tmp_workspace,
    const paddle::Tensor& tmp_m,
    const paddle::Tensor& tmp_d,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& block_indices,
    const paddle::Tensor& num_blocks,
    const paddle::Tensor& chunk_size,
    const paddle::Tensor& set_max_lengths,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& mask_offset,
    const paddle::optional<paddle::Tensor>& sinks,
    const std::string& cache_quant_type_str,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const int max_tokens_per_batch,
    const bool causal,
    const int sliding_window) {
  AppendAttnMetaData meta_data;

  const auto& qkv_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  meta_data.token_num = qkv_dims[0];
  meta_data.kv_num_heads = key_cache_dims[1];
  meta_data.head_dims = key_cache_dims[3];
  // TODO: trick method support c4, add attr head_dims in the future
  if (cache_quant_type_str == "cache_int4_zp") {
    meta_data.head_dims *= 2;
  }
  const int total_num_head =
      qkv_dims[qkv_dims.size() - 1] / meta_data.head_dims;
  meta_data.q_num_heads = total_num_head - 2 * meta_data.kv_num_heads;
  const auto group_size = meta_data.q_num_heads / meta_data.kv_num_heads;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = key_cache.dims()[2];
  meta_data.batch_size = seq_lens_this_time.dims()[0];

  // fmha_out generation, rewrite from AppendAttentionKernel
  paddle::Tensor fmha_out = paddle::zeros(
      {meta_data.token_num, meta_data.q_num_heads * meta_data.head_dims},
      qkv.dtype(),
      qkv.place());

  if (mask_offset) {
    meta_data.mask_offset = mask_offset.get().data<int>();
  }

  const int max_just_dec_len_this_time = set_max_lengths.data<int>()[4];
  const int max_kv_len_this_time = set_max_lengths.data<int>()[5];

  auto stream = qkv.stream();
  bool is_fp8 = cache_quant_type_str == "cache_fp8" or
                cache_quant_type_str == "block_wise_fp8";
  bool is_dynamic_cfp8 = cache_quant_type_str == "block_wise_fp8";

  if (max_just_dec_len_this_time > 0) {
    DISPATCH_CAUSAL(
        causal,
        CAUSAL,
        {DISPATCH_GQA_GROUP_SIZE(
            group_size,
            GROUP_SIZE,
            {DISPATCH_HEAD_DIM(
                meta_data.head_dims,
                HEAD_DIM,
                {DISPATCH_BLOCK_SIZE(
                    meta_data.block_size,
                    BLOCK_SIZE,
                    {DISPATCH_Q_TILE_SIZE(
                        group_size,
                        max_tokens_per_batch,
                        Q_TILE_SIZE,
                        {DISPATCH_DyCfp8(
                            is_dynamic_cfp8,
                            IsDynamicC8,
                            {DISPATCH_IS_FP8(is_fp8, IsFP8, {
                              switch (qkv.dtype()) {
                                case paddle::DataType::BFLOAT16: {
                                  DecodeAppendC8Attention<paddle::bfloat16,
                                                          GROUP_SIZE,
                                                          HEAD_DIM,
                                                          BLOCK_SIZE,
                                                          CAUSAL,
                                                          Q_TILE_SIZE,
                                                          IsFP8,
                                                          IsDynamicC8>(
                                      meta_data,
                                      qkv,
                                      key_cache,
                                      value_cache,
                                      tmp_workspace,
                                      tmp_m,
                                      tmp_d,
                                      attn_mask,
                                      cache_quant_type_str == "block_wise_fp8"
                                          ? cache_k_quant_scales.get()
                                          : cache_k_dequant_scales.get(),
                                      cache_quant_type_str == "block_wise_fp8"
                                          ? cache_v_quant_scales.get()
                                          : cache_v_dequant_scales.get(),
                                      sinks,
                                      seq_lens_this_time,
                                      seq_lens_decoder,
                                      seq_lens_encoder,
                                      batch_id_per_token,
                                      cu_seqlens_q,
                                      block_tables,
                                      block_indices,
                                      num_blocks,
                                      chunk_size,
                                      max_input_length,
                                      max_kv_len_this_time,
                                      quant_max_bound,
                                      quant_min_bound,
                                      max_tokens_per_batch,
                                      stream,
                                      &fmha_out,
                                      sliding_window);
                                  break;
                                }
                                case paddle::DataType::FLOAT16: {
                                  DecodeAppendC8Attention<paddle::float16,
                                                          GROUP_SIZE,
                                                          HEAD_DIM,
                                                          BLOCK_SIZE,
                                                          CAUSAL,
                                                          Q_TILE_SIZE,
                                                          IsFP8,
                                                          IsDynamicC8>(
                                      meta_data,
                                      qkv,
                                      key_cache,
                                      value_cache,
                                      tmp_workspace,
                                      tmp_m,
                                      tmp_d,
                                      attn_mask,
                                      cache_quant_type_str == "block_wise_fp8"
                                          ? cache_k_quant_scales.get()
                                          : cache_k_dequant_scales.get(),
                                      cache_quant_type_str == "block_wise_fp8"
                                          ? cache_v_quant_scales.get()
                                          : cache_v_dequant_scales.get(),
                                      sinks,
                                      seq_lens_this_time,
                                      seq_lens_decoder,
                                      seq_lens_encoder,
                                      batch_id_per_token,
                                      cu_seqlens_q,
                                      block_tables,
                                      block_indices,
                                      num_blocks,
                                      chunk_size,
                                      max_input_length,
                                      max_kv_len_this_time,
                                      quant_max_bound,
                                      quant_min_bound,
                                      max_tokens_per_batch,
                                      stream,
                                      &fmha_out,
                                      sliding_window);
                                  break;
                                }
                                default:
                                  PD_THROW(
                                      "NOT supported data type. "
                                      "Only bfloat16 and float16 are "
                                      "supported. ");
                              }
                            })})})})})})})
  }
  return {fmha_out};
}

std::vector<std::vector<int64_t>> DecodeAppendAttentionInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& tmp_workspace_shape,
    const std::vector<int64_t>& tmp_m_shape,
    const std::vector<int64_t>& tmp_d_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& block_indices_shape,
    const std::vector<int64_t>& num_blocks_shape,
    const std::vector<int64_t>& chunk_size_shape,
    const std::vector<int64_t>& set_max_lengths_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& mask_offset_shape,
    const paddle::optional<std::vector<int64_t>>& sinks_shape,
    const std::string& cache_quant_type_str,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const int max_tokens_per_batch,
    const bool causal,
    const int sliding_window) {
  const int token_num = qkv_shape[0];
  const int kv_num_heads = key_cache_shape[1];
  int head_dim = key_cache_shape[3];
  if (cache_quant_type_str == "cache_int4_zp") {
    head_dim *= 2;
  }
  const int total_num_head = qkv_shape[qkv_shape.size() - 1] / head_dim;
  const int num_heads = total_num_head - 2 * kv_num_heads;
  return {{token_num, num_heads * head_dim}};
}

std::vector<paddle::DataType> DecodeAppendAttentionInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& tmp_workspace_dtype,
    const paddle::DataType& tmp_m_dtype,
    const paddle::DataType& tmp_d_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& block_indices_dtype,
    const paddle::DataType& num_blocks_dtype,
    const paddle::DataType& chunk_size_dtype,
    const paddle::DataType& set_max_lengths_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& cache_k_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
    const paddle::optional<paddle::DataType>& cache_v_zp_dtype,
    const paddle::optional<paddle::DataType>& mask_offset_dtype,
    const paddle::optional<paddle::DataType>& sinks_dtype,
    const std::string& cache_quant_type_str,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const int max_tokens_per_batch,
    const bool causal,
    const int sliding_window) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(decode_append_attention)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "tmp_workspace",
             "tmp_m",
             "tmp_d",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "block_indices",
             "num_blocks",
             "chunk_size",
             "set_max_lengths",
             paddle::Optional("attn_mask"),
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("mask_offset"),
             paddle::Optional("sinks")})
    .Outputs({"fmha_out"})
    .Attrs({
        "cache_quant_type: std::string",
        "max_input_length: int",
        "quant_max_bound: float",
        "quant_min_bound: float",
        "max_tokens_per_batch: int",
        "causal: bool",
        "sliding_window: int",
    })
    .SetKernelFn(PD_KERNEL(DecodeAppendAttention))
    .SetInferShapeFn(PD_INFER_SHAPE(DecodeAppendAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DecodeAppendAttentionInferDtype));
