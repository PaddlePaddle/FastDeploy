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

#include "append_attn/append_attention_kernel.h"
#include "append_attn/decoder_write_cache_with_rope_kernel.h"
#include "append_attn/encoder_write_cache_with_rope_kernel.h"
#include "append_attn/speculate_write_cache_with_rope_kernel.h"

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

template <paddle::DataType D>
void AppendAttentionKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& set_max_lengths,
    paddle::Tensor& fmha_out,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& sinks,
    const float rms_norm_eps,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
    const int sliding_window) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  const int max_len_this_time = set_max_lengths.data<int>()[0];
  const int max_enc_len_this_time = set_max_lengths.data<int>()[1];
  const int max_dec_len_this_time = set_max_lengths.data<int>()[2];
  const int max_enc_dec_len_this_time = set_max_lengths.data<int>()[3];
  const int max_just_dec_len_this_time = set_max_lengths.data<int>()[4];
  const int max_kv_len_this_time = set_max_lengths.data<int>()[5];

  auto main_stream = qkv.stream();
  static cudaEvent_t main_event;
  static cudaEvent_t decoder_event;
  static cudaStream_t decoder_stream;
  static bool init_flag = false;
  if (max_just_dec_len_this_time > 0 && max_enc_len_this_time > 0 &&
      !init_flag) {
    cudaEventCreateWithFlags(&main_event, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&decoder_event, cudaEventDisableTiming);
    cudaStreamCreateWithFlags(&decoder_stream, cudaStreamNonBlocking);
    init_flag = true;
  }

  paddle::Tensor qkv_out;
  if (qkv_out_scales) {
    qkv_out = GetEmptyTensor(qkv.dims(), D, qkv.place());
  } else {
    qkv_out = qkv;
  }

  auto dispatch_CascadeAppendAttentionKernel =
      [&](auto temp_args,
          const paddle::Tensor& lambda_batch_ids,
          const paddle::Tensor& lambda_tile_ids_per_batch,
          const int lambda_num_blocks_data,
          const int lambda_block_shape_q,
          const int lambda_max_dec_len,
          const bool lambda_is_decoder,
          const bool lambda_enable_prefill,
          cudaStream_t& lambda_stream) -> void {
    CascadeAppendAttentionKernel<data_t, decltype(temp_args)>(
        meta_data,
        qkv_out,
        key_cache,
        value_cache,
        attn_mask,
        cache_quant_type_str == "block_wise_fp8" ? cache_k_quant_scales
                                                 : cache_k_dequant_scales,
        cache_quant_type_str == "block_wise_fp8" ? cache_v_quant_scales
                                                 : cache_v_dequant_scales,
        cache_k_zp,
        cache_v_zp,
        out_linear_shifts,
        out_linear_smooths,
        sinks,
        seq_lens_this_time,
        seq_lens_decoder,
        seq_lens_encoder,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        lambda_batch_ids,
        lambda_tile_ids_per_batch,
        cache_quant_type_str,
        lambda_num_blocks_data,
        lambda_block_shape_q,
        max_input_length,
        lambda_max_dec_len,
        quant_max_bound,
        quant_min_bound,
        out_linear_in_scale,
        max_partition_size,
        encoder_max_partition_size,
        speculate_max_draft_token_num,
        causal,
        lambda_is_decoder,
        lambda_enable_prefill,
        lambda_stream,
        &fmha_out,
        sliding_window);
  };

  if (max_enc_len_this_time > 0) {
    if (max_just_dec_len_this_time > 0) {
      cudaEventRecord(main_event, main_stream);
    }
    int encoder_num_blocks_data = encoder_num_blocks.data<int>()[0];
    int kv_num_blocks_data = kv_num_blocks.data<int>()[0];

    auto dispatch_EncoderWriteCacheWithRopeKernel =
        [&](auto temp_args) -> void {
      EncoderWriteCacheWithRopeKernel<data_t, decltype(temp_args)>(
          meta_data,
          qkv,
          seq_lens_this_time,
          seq_lens_encoder,
          seq_lens_decoder,
          batch_id_per_token,
          cu_seqlens_q,
          block_tables,
          kv_batch_ids,
          kv_tile_ids_per_batch,
          rotary_embs,
          qkv_out_scales,
          qkv_bias,
          cache_k_quant_scales,
          cache_v_quant_scales,
          cache_k_zp,
          cache_v_zp,
          kv_signal_data,
          cache_quant_type_str,
          kv_num_blocks_data,
          max_input_length,
          use_neox_rotary_style,
          rope_3d,
          main_stream,
          &qkv_out,
          const_cast<paddle::Tensor*>(&key_cache),
          const_cast<paddle::Tensor*>(&value_cache),
          q_norm_weight,
          k_norm_weight,
          rms_norm_eps);
    };

    if (qkv_out_scales) {
      int tmp;
      dispatch_EncoderWriteCacheWithRopeKernel(tmp);
    } else {
      data_t tmp;
      dispatch_EncoderWriteCacheWithRopeKernel(tmp);
    }
    if (out_linear_in_scale > 0.0) {
      switch (fmha_out.dtype()) {
        case paddle::DataType::INT8: {
          int8_t tmp;
          dispatch_CascadeAppendAttentionKernel(tmp,
                                                encoder_batch_ids,
                                                encoder_tile_ids_per_batch,
                                                encoder_num_blocks_data,
                                                encoder_block_shape_q,
                                                max_enc_dec_len_this_time,
                                                false,
                                                true,
                                                main_stream);
          break;
        }
        case paddle::DataType::FLOAT8_E4M3FN: {
          phi::dtype::float8_e4m3fn tmp;
          dispatch_CascadeAppendAttentionKernel(tmp,
                                                encoder_batch_ids,
                                                encoder_tile_ids_per_batch,
                                                encoder_num_blocks_data,
                                                encoder_block_shape_q,
                                                max_enc_dec_len_this_time,
                                                false,
                                                true,
                                                main_stream);
          break;
        }
        default: {
          PD_THROW(
              "Only supported output fmha_out of quant dtype in ['int8', "
              "'FLOAT8_E4M3FN'].");
          break;
        }
      }
    } else {
      data_t tmp;
      dispatch_CascadeAppendAttentionKernel(tmp,
                                            encoder_batch_ids,
                                            encoder_tile_ids_per_batch,
                                            encoder_num_blocks_data,
                                            encoder_block_shape_q,
                                            max_enc_dec_len_this_time,
                                            false,
                                            true,
                                            main_stream);
    }
  }

  if (max_just_dec_len_this_time > 0) {
    int decoder_num_blocks_data = decoder_num_blocks.data<int>()[0];

    cudaStream_t exec_stream;
    if (max_enc_len_this_time > 0) {
      cudaStreamWaitEvent(decoder_stream, main_event);
      exec_stream = decoder_stream;
    } else {
      exec_stream = main_stream;
    }
    if (speculate_decoder) {
      if (qkv_out_scales) {
        SpeculateWriteCacheWithRoPEKernel<data_t, int>(
            meta_data,
            qkv,  // [token_num, num_heads, head_dim]
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            rotary_embs,
            qkv_out_scales,
            qkv_bias,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_zp,
            cache_v_zp,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            exec_stream,
            &qkv_out,
            const_cast<paddle::Tensor*>(&key_cache),
            const_cast<paddle::Tensor*>(&value_cache),
            q_norm_weight,
            k_norm_weight,
            rms_norm_eps);
      } else {
        SpeculateWriteCacheWithRoPEKernel<data_t, data_t>(
            meta_data,
            qkv_out,  // [token_num, num_heads, head_dim]
            seq_lens_decoder,
            seq_lens_encoder,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            rotary_embs,
            qkv_out_scales,
            qkv_bias,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_zp,
            cache_v_zp,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            exec_stream,
            &qkv_out,
            const_cast<paddle::Tensor*>(&key_cache),
            const_cast<paddle::Tensor*>(&value_cache),
            q_norm_weight,
            k_norm_weight,
            rms_norm_eps);
      }
    } else {
      if (qkv_out_scales) {
        DecoderWriteCacheWithRoPEKernel<data_t, int>(
            meta_data,
            qkv,  // [token_num, num_heads, head_dim]
            seq_lens_decoder,
            seq_lens_encoder,
            cu_seqlens_q,
            block_tables,
            rotary_embs,
            qkv_out_scales,
            qkv_bias,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_zp,
            cache_v_zp,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            exec_stream,
            &qkv_out,
            const_cast<paddle::Tensor*>(&key_cache),
            const_cast<paddle::Tensor*>(&value_cache),
            q_norm_weight,
            k_norm_weight,
            rms_norm_eps);
      } else {
        DecoderWriteCacheWithRoPEKernel<data_t, data_t>(
            meta_data,
            qkv_out,  // [token_num, num_heads, head_dim]
            seq_lens_decoder,
            seq_lens_encoder,
            cu_seqlens_q,
            block_tables,
            rotary_embs,
            qkv_out_scales,
            qkv_bias,
            cache_k_quant_scales,
            cache_v_quant_scales,
            cache_k_zp,
            cache_v_zp,
            cache_quant_type_str,
            use_neox_rotary_style,
            rope_3d,
            max_input_length,
            exec_stream,
            &qkv_out,
            const_cast<paddle::Tensor*>(&key_cache),
            const_cast<paddle::Tensor*>(&value_cache),
            q_norm_weight,
            k_norm_weight,
            rms_norm_eps);
      }
    }

    if (out_linear_in_scale > 0.0) {
      switch (fmha_out.dtype()) {
        case paddle::DataType::INT8: {
          int8_t tmp;
          dispatch_CascadeAppendAttentionKernel(tmp,
                                                decoder_batch_ids,
                                                decoder_tile_ids_per_batch,
                                                decoder_num_blocks_data,
                                                decoder_block_shape_q,
                                                max_kv_len_this_time,
                                                !speculate_decoder,
                                                !speculate_decoder,
                                                exec_stream);
          break;
        }
        case paddle::DataType::FLOAT8_E4M3FN: {
          phi::dtype::float8_e4m3fn tmp;
          dispatch_CascadeAppendAttentionKernel(tmp,
                                                decoder_batch_ids,
                                                decoder_tile_ids_per_batch,
                                                decoder_num_blocks_data,
                                                decoder_block_shape_q,
                                                max_kv_len_this_time,
                                                !speculate_decoder,
                                                !speculate_decoder,
                                                exec_stream);
          break;
        }
      }
    } else {
      data_t tmp;
      dispatch_CascadeAppendAttentionKernel(tmp,
                                            decoder_batch_ids,
                                            decoder_tile_ids_per_batch,
                                            decoder_num_blocks_data,
                                            decoder_block_shape_q,
                                            max_kv_len_this_time,
                                            !speculate_decoder,
                                            !speculate_decoder,
                                            exec_stream);
    }
    if (max_enc_len_this_time > 0) {
      cudaEventRecord(decoder_event, exec_stream);
      cudaStreamWaitEvent(main_stream, decoder_event);
    }
  }
}

std::vector<paddle::Tensor> AppendAttention(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& set_max_lengths,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const paddle::optional<paddle::Tensor>& mask_offset,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& sinks,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
    const int sliding_window) {
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

  // template dtype generation
  phi::DataType dtype_id;
  switch (qkv.dtype()) {
    case paddle::DataType::FLOAT16: {
      dtype_id = phi::DataType::FLOAT16;
      break;
    }
    case paddle::DataType::BFLOAT16: {
      dtype_id = phi::DataType::BFLOAT16;
      break;
    }
    case paddle::DataType::INT32: {
      if (compute_dtype == "bf16") {
        dtype_id = phi::DataType::BFLOAT16;
        break;
      } else if (compute_dtype == "fp16") {
        dtype_id = phi::DataType::FLOAT16;
        break;
      } else {
        PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
        break;
      }
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and bfloat16 are supported. ");
      break;
    }
  }

  // fmha_out generation, rewrite from AppendAttentionKernel
  paddle::Tensor fmha_out;
  if (out_linear_in_scale > 0.0) {
    if (fabs(quant_max_bound - 127.0f) < 0.000001) {
      fmha_out = paddle::zeros(
          {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims},
          paddle::DataType::INT8,
          qkv.place());
    } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
      fmha_out = paddle::zeros(
          {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims},
          paddle::DataType::FLOAT8_E4M3FN,
          qkv.place());
    } else {
      PD_THROW("Only supported attr of quant_max_bound in ['127', '448'].");
    }
  } else {
    fmha_out = paddle::zeros(
        {meta_data.token_nums, meta_data.q_num_heads * meta_data.head_dims},
        dtype_id,
        qkv.place());
  }

  if (mask_offset) {
    meta_data.mask_offset = mask_offset.get().data<int>();
  }

  auto dispatch_by_template = [&](auto temp_args) -> void {
    AppendAttentionKernel<type2value<decltype(temp_args)>::value>(
        meta_data,
        qkv,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        kv_batch_ids,
        kv_tile_ids_per_batch,
        kv_num_blocks,
        decoder_batch_ids,
        decoder_tile_ids_per_batch,
        decoder_num_blocks,
        set_max_lengths,
        fmha_out,
        rotary_embs,
        attn_mask,
        qkv_bias,
        qkv_out_scales,
        cache_k_quant_scales,
        cache_v_quant_scales,
        cache_k_dequant_scales,
        cache_v_dequant_scales,
        cache_k_zp,
        cache_v_zp,
        out_linear_shifts,
        out_linear_smooths,
        kv_signal_data,
        q_norm_weight,
        k_norm_weight,
        sinks,
        rms_norm_eps,
        cache_quant_type_str,
        use_neox_rotary_style,
        rope_3d,
        max_input_length,
        quant_max_bound,
        quant_min_bound,
        out_linear_in_scale,
        encoder_block_shape_q,
        decoder_block_shape_q,
        max_partition_size,
        encoder_max_partition_size,
        speculate_max_draft_token_num,
        causal,
        speculate_decoder,
        sliding_window);
  };

  phi::dtype::float16 fp16_dtype;
  phi::dtype::bfloat16 bp16_dtype;
  switch (dtype_id) {
    case phi::DataType::FLOAT16: {
      dispatch_by_template(fp16_dtype);
      return {fmha_out};
    }
    case phi::DataType::BFLOAT16: {
      dispatch_by_template(bp16_dtype);
      return {fmha_out};
    }
    default:
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and bfloat16 are supported. ");
      break;
  }

  return {paddle::Tensor{}};
}

std::vector<paddle::Tensor> AppendAttentionWithOutput(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& encoder_batch_ids,
    const paddle::Tensor& encoder_tile_ids_per_batch,
    const paddle::Tensor& encoder_num_blocks,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks,
    const paddle::Tensor& set_max_lengths,
    paddle::Tensor& fmha_out,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const paddle::optional<paddle::Tensor>& mask_offset,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const paddle::optional<paddle::Tensor>& sinks,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
    const int sliding_window) {
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

  if (mask_offset) {
    meta_data.mask_offset = mask_offset.get().data<int>();
  }

  auto dispatch_by_template = [&](auto temp_args) -> void {
    AppendAttentionKernel<type2value<decltype(temp_args)>::value>(
        meta_data,
        qkv,
        key_cache,
        value_cache,
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        batch_id_per_token,
        cu_seqlens_q,
        block_tables,
        encoder_batch_ids,
        encoder_tile_ids_per_batch,
        encoder_num_blocks,
        kv_batch_ids,
        kv_tile_ids_per_batch,
        kv_num_blocks,
        decoder_batch_ids,
        decoder_tile_ids_per_batch,
        decoder_num_blocks,
        set_max_lengths,
        fmha_out,
        rotary_embs,
        attn_mask,
        qkv_bias,
        qkv_out_scales,
        cache_k_quant_scales,
        cache_v_quant_scales,
        cache_k_dequant_scales,
        cache_v_dequant_scales,
        cache_k_zp,
        cache_v_zp,
        out_linear_shifts,
        out_linear_smooths,
        kv_signal_data,
        q_norm_weight,
        k_norm_weight,
        sinks,
        rms_norm_eps,
        cache_quant_type_str,
        use_neox_rotary_style,
        rope_3d,
        max_input_length,
        quant_max_bound,
        quant_min_bound,
        out_linear_in_scale,
        encoder_block_shape_q,
        decoder_block_shape_q,
        max_partition_size,
        encoder_max_partition_size,
        speculate_max_draft_token_num,
        causal,
        speculate_decoder,
        sliding_window);
  };

  phi::dtype::float16 fp16_dtype;
  phi::dtype::bfloat16 bp16_dtype;

  switch (qkv.dtype()) {
    case paddle::DataType::FLOAT16: {
      dispatch_by_template(fp16_dtype);
      break;
    }
    case paddle::DataType::BFLOAT16: {
      dispatch_by_template(bp16_dtype);
      break;
    }
    case paddle::DataType::INT32: {
      if (compute_dtype == "bf16") {
        dispatch_by_template(bp16_dtype);
        break;
      } else if (compute_dtype == "fp16") {
        dispatch_by_template(fp16_dtype);
        break;
      } else {
        PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
        break;
      }
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16 and bfloat16 are supported. ");
      break;
    }
  }

  return {fmha_out};
}

std::vector<std::vector<int64_t>> AppendAttentionInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& encoder_batch_ids_shape,
    const std::vector<int64_t>& encoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& encoder_num_blocks_shape,
    const std::vector<int64_t>& kv_batch_ids_shape,
    const std::vector<int64_t>& kv_tile_ids_per_batch_shape,
    const std::vector<int64_t>& kv_num_blocks_shape,
    const std::vector<int64_t>& decoder_batch_ids_shape,
    const std::vector<int64_t>& decoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& decoder_num_blocks_shape,
    const std::vector<int64_t>& set_max_lengths_shape,
    const paddle::optional<std::vector<int64_t>>& rotary_embs_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_bias_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_out_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_shifts_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_smooths_shape,
    const paddle::optional<std::vector<int64_t>>& mask_offset_shape,
    const paddle::optional<std::vector<int64_t>>& kv_signal_data_shape,
    const paddle::optional<std::vector<int64_t>>& q_norm_weight_shape,
    const paddle::optional<std::vector<int64_t>>& k_norm_weight_shape,
    const paddle::optional<std::vector<int64_t>>& sinks_shape,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
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

std::vector<paddle::DataType> AppendAttentionInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& encoder_batch_ids_dtype,
    const paddle::DataType& encoder_tile_ids_per_batch_dtype,
    const paddle::DataType& encoder_num_blocks_dtype,
    const paddle::DataType& kv_batch_ids_dtype,
    const paddle::DataType& kv_tile_ids_per_batch_dtype,
    const paddle::DataType& kv_num_blocks_dtype,
    const paddle::DataType& decoder_batch_ids_dtype,
    const paddle::DataType& decoder_tile_ids_per_batch_dtype,
    const paddle::DataType& decoder_num_blocks_dtype,
    const paddle::DataType& set_max_lengths_dtype,
    const paddle::optional<paddle::DataType>& rotary_embs_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& qkv_bias_dtype,
    const paddle::optional<paddle::DataType>& qkv_out_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
    const paddle::optional<paddle::DataType>& cache_v_zp_dtype,
    const paddle::optional<paddle::DataType>& out_linear_shifts_dtype,
    const paddle::optional<paddle::DataType>& out_linear_smooths_dtype,
    const paddle::optional<paddle::DataType>& mask_offset_dtype,
    const paddle::optional<paddle::DataType>& kv_signal_data_dtype,
    const paddle::optional<paddle::DataType>& q_norm_weight_dtype,
    const paddle::optional<paddle::DataType>& k_norm_weight_dtype,
    const paddle::optional<paddle::DataType>& sinks_dtype,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
    const int sliding_window) {
  if (compute_dtype == "bf16") {
    if (out_linear_in_scale > 0.0) {
      if (fabs(quant_max_bound - 127.0f) < 0.000001) {
        return {paddle::DataType::INT8};
      } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
        return {paddle::DataType::FLOAT8_E4M3FN};
      } else {
        PD_THROW(
            "Only supported attr of quant_max_bound in ['127.0', '448.0'].");
      }
    } else {
      return {paddle::DataType::BFLOAT16};
    }
  } else if (compute_dtype == "fp16") {
    if (out_linear_in_scale > 0.0) {
      if (fabs(quant_max_bound - 127.0f) < 0.000001) {
        return {paddle::DataType::INT8};
      } else if (fabs(quant_max_bound - 448.0f) < 0.000001) {
        return {paddle::DataType::FLOAT8_E4M3FN};
      } else {
        PD_THROW(
            "Only supported attr of quant_max_bound in ['127.0', '448.0'].");
      }
    } else {
      return {paddle::DataType::FLOAT16};
    }
  } else {
    PD_THROW("Only supported attr of compute_dtype in ['fp16', 'bf16'].");
  }
}

std::vector<std::vector<int64_t>> AppendAttentionWithOutputInferShape(
    const std::vector<int64_t>& qkv_shape,
    const std::vector<int64_t>& key_cache_shape,
    const std::vector<int64_t>& value_cache_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& batch_id_per_token_shape,
    const std::vector<int64_t>& cu_seqlens_q_shape,
    const std::vector<int64_t>& block_tables_shape,
    const std::vector<int64_t>& encoder_batch_ids_shape,
    const std::vector<int64_t>& encoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& encoder_num_blocks_shape,
    const std::vector<int64_t>& kv_batch_ids_shape,
    const std::vector<int64_t>& kv_tile_ids_per_batch_shape,
    const std::vector<int64_t>& kv_num_blocks_shape,
    const std::vector<int64_t>& decoder_batch_ids_shape,
    const std::vector<int64_t>& decoder_tile_ids_per_batch_shape,
    const std::vector<int64_t>& decoder_num_blocks_shape,
    const std::vector<int64_t>& set_max_lengths_shape,
    const std::vector<int64_t>& fmha_out_shape,
    const paddle::optional<std::vector<int64_t>>& rotary_embs_shape,
    const paddle::optional<std::vector<int64_t>>& attn_mask_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_bias_shape,
    const paddle::optional<std::vector<int64_t>>& qkv_out_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_quant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_dequant_scales_shape,
    const paddle::optional<std::vector<int64_t>>& cache_k_zp_shape,
    const paddle::optional<std::vector<int64_t>>& cache_v_zp_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_shifts_shape,
    const paddle::optional<std::vector<int64_t>>& out_linear_smooths_shape,
    const paddle::optional<std::vector<int64_t>>& mask_offset_shape,
    const paddle::optional<std::vector<int64_t>>& kv_signal_data_shape,
    const paddle::optional<std::vector<int64_t>>& q_norm_weight_shape,
    const paddle::optional<std::vector<int64_t>>& k_norm_weight_shape,
    const paddle::optional<std::vector<int64_t>>& sinks_shape,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
    const int sliding_window) {
  return {fmha_out_shape};
}

std::vector<paddle::DataType> AppendAttentionWithOutputInferDtype(
    const paddle::DataType& qkv_dtype,
    const paddle::DataType& key_cache_dtype,
    const paddle::DataType& value_cache_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& batch_id_per_token_dtype,
    const paddle::DataType& cu_seqlens_q_dtype,
    const paddle::DataType& block_tables_dtype,
    const paddle::DataType& encoder_batch_ids_dtype,
    const paddle::DataType& encoder_tile_ids_per_batch_dtype,
    const paddle::DataType& encoder_num_blocks_dtype,
    const paddle::DataType& kv_batch_ids_dtype,
    const paddle::DataType& kv_tile_ids_per_batch_dtype,
    const paddle::DataType& kv_num_blocks_dtype,
    const paddle::DataType& decoder_batch_ids_dtype,
    const paddle::DataType& decoder_tile_ids_per_batch_dtype,
    const paddle::DataType& decoder_num_blocks_dtype,
    const paddle::DataType& set_max_lengths_dtype,
    const paddle::DataType& fmha_out_dtype,
    const paddle::optional<paddle::DataType>& rotary_embs_dtype,
    const paddle::optional<paddle::DataType>& attn_mask_dtype,
    const paddle::optional<paddle::DataType>& qkv_bias_dtype,
    const paddle::optional<paddle::DataType>& qkv_out_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_quant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_v_dequant_scales_dtype,
    const paddle::optional<paddle::DataType>& cache_k_zp_dtype,
    const paddle::optional<paddle::DataType>& cache_v_zp_dtype,
    const paddle::optional<paddle::DataType>& out_linear_shifts_dtype,
    const paddle::optional<paddle::DataType>& out_linear_smooths_dtype,
    const paddle::optional<paddle::DataType>& mask_offset_dtype,
    const paddle::optional<paddle::DataType>& kv_signal_data_dtype,
    const paddle::optional<paddle::DataType>& q_norm_weight_dtype,
    const paddle::optional<paddle::DataType>& k_norm_weight_dtype,
    const paddle::optional<paddle::DataType>& sinks_dtype,
    const float rms_norm_eps,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const bool use_neox_rotary_style,
    const bool rope_3d,
    const int max_input_length,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int max_partition_size,
    const int encoder_max_partition_size,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder,
    const int sliding_window) {
  return {fmha_out_dtype};
}

PD_BUILD_STATIC_OP(append_attention)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "encoder_batch_ids",
             "encoder_tile_ids_per_batch",
             "encoder_num_blocks",
             "kv_batch_ids",
             "kv_tile_ids_per_batch",
             "kv_num_blocks",
             "decoder_batch_ids",
             "decoder_tile_ids_per_batch",
             "decoder_num_blocks",
             "set_max_lengths",
             paddle::Optional("rotary_embs"),
             paddle::Optional("attn_mask"),
             paddle::Optional("qkv_bias"),
             paddle::Optional("qkv_out_scales"),
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("out_linear_shifts"),
             paddle::Optional("out_linear_smooths"),
             paddle::Optional("mask_offset"),
             paddle::Optional("kv_signal_data"),
             paddle::Optional("q_norm_weight"),
             paddle::Optional("k_norm_weight"),
             paddle::Optional("sinks")})
    .Outputs({"fmha_out"})
    .Attrs({
        "rms_norm_eps: float",
        "compute_type: std::string",
        "cache_quant_type: std::string",
        "use_neox_rotary_style: bool",
        "rope_3d: bool",
        "max_input_length: int",
        "quant_max_bound: float",
        "quant_min_bound: float",
        "out_linear_in_scale: float",
        "encoder_block_shape_q: int",
        "decoder_block_shape_q: int",
        "max_partition_size: int",
        "encoder_max_partition_size: int",
        "speculate_max_draft_token_num: int",
        "causal: bool",
        "speculate_decoder: bool",
        "sliding_window: int",
    })
    .SetKernelFn(PD_KERNEL(AppendAttention))
    .SetInferShapeFn(PD_INFER_SHAPE(AppendAttentionInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AppendAttentionInferDtype));

PD_BUILD_STATIC_OP(append_attention_with_output)
    .Inputs({"qkv",
             "key_cache",
             "value_cache",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "seq_lens_this_time",
             "batch_id_per_token",
             "cu_seqlens_q",
             "block_tables",
             "encoder_batch_ids",
             "encoder_tile_ids_per_batch",
             "encoder_num_blocks",
             "kv_batch_ids",
             "kv_tile_ids_per_batch",
             "kv_num_blocks",
             "decoder_batch_ids",
             "decoder_tile_ids_per_batch",
             "decoder_num_blocks",
             "set_max_lengths",
             "fmha_out",
             paddle::Optional("rotary_embs"),
             paddle::Optional("attn_mask"),
             paddle::Optional("qkv_bias"),
             paddle::Optional("qkv_out_scales"),
             paddle::Optional("cache_k_quant_scales"),
             paddle::Optional("cache_v_quant_scales"),
             paddle::Optional("cache_k_dequant_scales"),
             paddle::Optional("cache_v_dequant_scales"),
             paddle::Optional("cache_k_zp"),
             paddle::Optional("cache_v_zp"),
             paddle::Optional("out_linear_shifts"),
             paddle::Optional("out_linear_smooths"),
             paddle::Optional("mask_offset"),
             paddle::Optional("kv_signal_data"),
             paddle::Optional("q_norm_weight"),
             paddle::Optional("k_norm_weight"),
             paddle::Optional("sinks")})
    .Outputs({"fmha_out_out"})
    .SetInplaceMap({{"fmha_out", "fmha_out_out"}})
    .Attrs({
        "rms_norm_eps: float",
        "compute_type: std::string",
        "cache_quant_type: std::string",
        "use_neox_rotary_style: bool",
        "rope_3d: bool",
        "max_input_length: int",
        "quant_max_bound: float",
        "quant_min_bound: float",
        "out_linear_in_scale: float",
        "encoder_block_shape_q: int",
        "decoder_block_shape_q: int",
        "max_partition_size: int",
        "encoder_max_partition_size: int",
        "speculate_max_draft_token_num: int",
        "causal: bool",
        "speculate_decoder: bool",
        "sliding_window: int",
    })
    .SetKernelFn(PD_KERNEL(AppendAttentionWithOutput))
    .SetInferShapeFn(PD_INFER_SHAPE(AppendAttentionWithOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AppendAttentionWithOutputInferDtype));
