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

#include "append_attn/decoder_write_cache_with_rope_kernel.h"
#include "append_attn/speculate_write_cache_with_rope_kernel.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

template <typename T>
class type2value;

template <>
class type2value<paddle::bfloat16> {
 public:
  static constexpr paddle::DataType value = paddle::DataType::BFLOAT16;
};

template <>
class type2value<paddle::float16> {
 public:
  static constexpr paddle::DataType value = paddle::DataType::FLOAT16;
};

std::vector<paddle::Tensor> DecoderWriteCacheWithRoPE(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& set_max_lengths,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const float rms_norm_eps,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const bool speculate_decoder) {
  auto stream = qkv.stream();

  AppendAttnMetaData meta_data;

  const auto& qkv_dims = qkv.dims();
  const auto& key_cache_dims = key_cache.dims();
  meta_data.token_nums = qkv_dims[0];
  meta_data.kv_num_heads = key_cache_dims[1];
  meta_data.head_dims = key_cache_dims[3];
  // TODO: trick method support c4, add attr head_dims in the future
  if (cache_quant_type_str == "cache_int4_zp") {
    meta_data.head_dims *= 2;
  }
  const int total_num_head =
      qkv_dims[qkv_dims.size() - 1] / meta_data.head_dims;
  meta_data.q_num_heads = total_num_head - 2 * meta_data.kv_num_heads;

  meta_data.max_blocks_per_seq = block_tables.dims()[1];
  meta_data.block_size = key_cache.dims()[2];
  meta_data.batch_size = seq_lens_this_time.dims()[0];

  const int max_just_dec_len_this_time = set_max_lengths.data<int>()[4];

  if (max_just_dec_len_this_time > 0) {
    if (speculate_decoder) {
      switch (qkv.dtype()) {
        case paddle::DataType::BFLOAT16: {
          SpeculateWriteCacheWithRoPEKernel<paddle::bfloat16, paddle::bfloat16>(
              meta_data,
              qkv,  // [token_num, num_heads, head_dim]
              seq_lens_decoder,
              seq_lens_encoder,
              batch_id_per_token,
              cu_seqlens_q,
              block_tables,
              rotary_embs,
              NULL,
              qkv_bias,
              cache_k_quant_scales,
              cache_v_quant_scales,
              cache_k_zp,
              cache_v_zp,
              cache_quant_type_str,
              use_neox_rotary_style,
              rope_3d,
              max_input_length,
              stream,
              const_cast<paddle::Tensor*>(&qkv),
              const_cast<paddle::Tensor*>(&key_cache),
              const_cast<paddle::Tensor*>(&value_cache),
              q_norm_weight,
              k_norm_weight,
              rms_norm_eps);
          break;
        }
        case paddle::DataType::FLOAT16: {
          SpeculateWriteCacheWithRoPEKernel<paddle::float16, paddle::float16>(
              meta_data,
              qkv,  // [token_num, num_heads, head_dim]
              seq_lens_decoder,
              seq_lens_encoder,
              batch_id_per_token,
              cu_seqlens_q,
              block_tables,
              rotary_embs,
              NULL,
              qkv_bias,
              cache_k_quant_scales,
              cache_v_quant_scales,
              cache_k_zp,
              cache_v_zp,
              cache_quant_type_str,
              use_neox_rotary_style,
              rope_3d,
              max_input_length,
              stream,
              const_cast<paddle::Tensor*>(&qkv),
              const_cast<paddle::Tensor*>(&key_cache),
              const_cast<paddle::Tensor*>(&value_cache),
              q_norm_weight,
              k_norm_weight,
              rms_norm_eps);
          break;
        }
        default:
          PD_THROW(
              "NOT supported data type. "
              "Only bfloat16 and float16 are supported. ");
      }
    } else {
      switch (qkv.dtype()) {
        case paddle::DataType::BFLOAT16: {
          DecoderWriteCacheWithRoPEKernel<paddle::bfloat16, paddle::bfloat16>(
              meta_data,
              qkv,  // [token_num, num_heads, head_dim]
              seq_lens_decoder,
              seq_lens_encoder,
              cu_seqlens_q,
              block_tables,
              rotary_embs,
              NULL,
              qkv_bias,
              cache_k_quant_scales,
              cache_v_quant_scales,
              cache_k_zp,
              cache_v_zp,
              cache_quant_type_str,
              use_neox_rotary_style,
              rope_3d,
              max_input_length,
              stream,
              const_cast<paddle::Tensor*>(&qkv),
              const_cast<paddle::Tensor*>(&key_cache),
              const_cast<paddle::Tensor*>(&value_cache),
              q_norm_weight,
              k_norm_weight,
              rms_norm_eps);
          break;
        }
        case paddle::DataType::FLOAT16: {
          DecoderWriteCacheWithRoPEKernel<paddle::float16, paddle::float16>(
              meta_data,
              qkv,  // [token_num, num_heads, head_dim]
              seq_lens_decoder,
              seq_lens_encoder,
              cu_seqlens_q,
              block_tables,
              rotary_embs,
              NULL,
              qkv_bias,
              cache_k_quant_scales,
              cache_v_quant_scales,
              cache_k_zp,
              cache_v_zp,
              cache_quant_type_str,
              use_neox_rotary_style,
              rope_3d,
              max_input_length,
              stream,
              const_cast<paddle::Tensor*>(&qkv),
              const_cast<paddle::Tensor*>(&key_cache),
              const_cast<paddle::Tensor*>(&value_cache),
              q_norm_weight,
              k_norm_weight,
              rms_norm_eps);
          break;
        }
        default:
          PD_THROW(
              "NOT supported data type. "
              "Only bfloat16 and float16 are supported. ");
      }
    }
  }
  return {qkv};
}

std::vector<std::vector<int64_t>> DecoderWriteCacheWithRoPEInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& set_max_lengths_shape,
    const paddle::optional<std::vector<int64_t>>& rotary_embs_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_bias_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& kv_signal_data_shape,
    const paddle::optional<std::vector<int64_t>>& q_norm_weight_shape,
    const paddle::optional<std::vector<int64_t>>& k_norm_weight_shape,
    const float rms_norm_eps,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const bool speculate_decoder) {
  return {qkv_shape};
}

std::vector<paddle::DataType> DecoderWriteCacheWithRoPEInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& set_max_lengths_dtype,
    const paddle::optional<paddle::DataType>& rotary_embs_dtype,
    const paddle::optional<paddle::DataType>& qkv_bias_dtype,
    const paddle::optional<paddle::DataType>& cache_k_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
    const paddle::optional<paddle::DataType>& cache_v_zp_dtype,
    const paddle::optional<paddle::DataType>& kv_signal_data_dtype,
    const paddle::optional<paddle::DataType>& q_norm_weight_dtype,
    const paddle::optional<paddle::DataType>& k_norm_weight_dtype,
    const float rms_norm_eps,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const bool speculate_decoder) {
  return {qkv_dtype};
}

PD_BUILD_STATIC_OP(decoder_write_cache_with_rope)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "set_max_lengths",
             paddle::Optional("rotary_embs"),
             paddle::Optional("qkv_bias"),
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("kv_signal_data"),
             paddle::Optional("q_norm_weight"),
             paddle::Optional("k_norm_weight")})
    .Outputs({"qkv_out"})
    .SetInplaceMap({{"qkv", "qkv_out"}})
    .Attrs({
        "rms_norm_eps: float",
        "cache_quant_type: std::string",
        "use_neox_rotary_style: bool",
        "rope_3d: bool",
        "max_input_length: int",
        "quant_max_bound: float",
        "quant_min_bound: float",
        "speculate_decoder: bool",
    })
    .SetKernelFn(PD_KERNEL(DecoderWriteCacheWithRoPE))
    .SetInferShapeFn(PD_INFER_SHAPE(DecoderWriteCacheWithRoPEInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DecoderWriteCacheWithRoPEInferDtype));
