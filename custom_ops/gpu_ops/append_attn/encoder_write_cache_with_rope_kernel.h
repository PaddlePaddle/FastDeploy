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
#pragma once

#include "encoder_write_cache_with_rope_impl.cuh"
#include "remote_cache_kv_ipc.h"

template <typename T, typename QKV_TYPE>
void EncoderWriteCacheWithRopeKernel(
    const AppendAttnMetaData& meta_data,
    const paddle::Tensor&
        qkv,  // [token_num, 3, num_head, head_dim] ([token_num, num_head + 2 *
              // kv_num_heads, head_dim] if GQA)
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& batch_ids,
    const paddle::Tensor& tile_ids,
    const paddle::optional<paddle::Tensor>& rotary_embs,
    const paddle::optional<paddle::Tensor>& qkv_out_scales,
    const paddle::optional<paddle::Tensor>& qkv_biases,
    const paddle::optional<paddle::Tensor>& cache_k_scale,
    const paddle::optional<paddle::Tensor>& cache_v_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const std::string& cache_quant_type_str,
    const int num_blocks,
    const int max_seq_len,
    const bool use_neox_style,
    const bool rope_3d,
    cudaStream_t& stream,
    paddle::Tensor* qkv_out,
    paddle::Tensor* key_cache_out,
    paddle::Tensor* value_cache_out,
    const paddle::optional<paddle::Tensor>& q_norm_weight,
    const paddle::optional<paddle::Tensor>& k_norm_weight,
    const float rms_norm_eps) {
  auto token_num = meta_data.token_nums;
  auto num_heads = meta_data.q_num_heads;
  auto kv_num_heads = meta_data.kv_num_heads;
  auto head_dim = meta_data.head_dims;
  bool is_scale_channel_wise = false;
  int rotary_dim = head_dim;
  if (cache_k_scale && cache_k_scale.get().dims()[0] == head_dim * kv_num_heads) {
    is_scale_channel_wise = true;
  }
  if (rotary_embs){
    rotary_dim = rotary_embs.get().dims()[rotary_embs.get().dims().size()-1] * 2;
    if(rotary_dim < head_dim){
      if (!use_neox_style || q_norm_weight || k_norm_weight || num_heads == kv_num_heads || is_scale_channel_wise){
        PADDLE_THROW(phi::errors::Fatal(
          "partial_rotary_factor < 1.0 only supports use_neox_rotary_style=True, q_norm_weight/k_norm_weight) is None, GQA and is_scale_channel_wise=false."));
      }
    }
  }

  if (q_norm_weight && k_norm_weight) {
    if (num_heads != kv_num_heads && !is_scale_channel_wise && !use_neox_style) {
      gqa_rotary_qk_norm_variable(
        qkv_out->data<T>(),
        qkv.data<QKV_TYPE>(),
        qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
        qkv_biases ? qkv_biases.get().data<T>() : nullptr,
        rotary_embs.get().data<float>(),
        batch_id_per_token.data<int>(),
        cu_seqlens_q.data<int>(),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        token_num,
        num_heads,
        kv_num_heads,
        max_seq_len,
        rope_3d ? rotary_embs.get().dims()[3] : rotary_embs.get().dims()[2],
        head_dim,
        stream,
        use_neox_style,
        rope_3d,
        q_norm_weight ? q_norm_weight.get().data<float>() : nullptr,
        k_norm_weight ? k_norm_weight.get().data<float>() : nullptr,
        rms_norm_eps);
    } else {
      PD_THROW(
          "gqa_rotary_qk_norm_variable only support gqa mode. channel wise scale and neox style are not supported");
    }
  } else {
    if (num_heads == kv_num_heads) {
      rotary_qk_variable(
          qkv_out->data<T>(),
          qkv.data<QKV_TYPE>(),
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? qkv_biases.get().data<T>() : nullptr,
          rotary_embs.get().data<float>(),
          batch_id_per_token.data<int>(),
          cu_seqlens_q.data<int>(),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          token_num,
          num_heads,
          max_seq_len,
          rotary_embs.get().dims()[2],
          head_dim,
          stream,
          use_neox_style,
          rope_3d);
    } else {
      if (!is_scale_channel_wise) {
        gqa_rotary_qk_variable(
          qkv_out->data<T>(),
          qkv.data<QKV_TYPE>(),
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? qkv_biases.get().data<T>() : nullptr,
          rotary_embs.get().data<float>(),
          batch_id_per_token.data<int>(),
          cu_seqlens_q.data<int>(),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          token_num,
          num_heads,
          kv_num_heads,
          max_seq_len,
          rope_3d ? rotary_embs.get().dims()[3] : rotary_embs.get().dims()[2],
          head_dim,
          rotary_dim,
          stream,
          use_neox_style,
          rope_3d);
      } else {
        gqa_rotary_qk_quant_variable(
          qkv_out->data<T>(),
          qkv.data<QKV_TYPE>(),
          qkv_out_scales ? qkv_out_scales.get().data<float>() : nullptr,
          qkv_biases ? qkv_biases.get().data<T>() : nullptr,
          cache_k_scale ? cache_k_scale.get().data<T>() : nullptr,
          cache_v_scale ? cache_v_scale.get().data<T>() : nullptr,
          rotary_embs.get().data<float>(),
          batch_id_per_token.data<int>(),
          cu_seqlens_q.data<int>(),
          seq_lens_encoder.data<int>(),
          seq_lens_decoder.data<int>(),
          token_num,
          num_heads,
          kv_num_heads,
          max_seq_len,
          rotary_embs.get().dims()[2],
          head_dim,
          stream,
          use_neox_style,
          rope_3d);
      }

    }
  }
  const uint32_t block_size = meta_data.block_size;
  if (cache_quant_type_str == "none") {
    CascadeAppendWriteCacheKVQKV<T>(meta_data,
                                    *qkv_out,
                                    block_tables,
                                    batch_id_per_token,
                                    cu_seqlens_q,
                                    seq_lens_encoder,
                                    seq_lens_decoder,
                                    max_seq_len,
                                    stream,
                                    key_cache_out,
                                    value_cache_out);
  } else if (cache_quant_type_str == "cache_int8" or cache_quant_type_str == "cache_fp8" or cache_quant_type_str == "block_wise_fp8") {
    DISPATCH_HEAD_DIM(
        head_dim, HEAD_DIM, {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
          CascadeAppendWriteCacheKVC8QKV<T, HEAD_DIM, BLOCK_SIZE>(
              meta_data,
              *key_cache_out,
              *value_cache_out,
              *qkv_out,
              cache_k_scale.get(),
              cache_v_scale.get(),
              seq_lens_this_time,
              seq_lens_decoder,
              batch_id_per_token,
              cu_seqlens_q,
              block_tables,
              batch_ids,
              tile_ids,
              num_blocks,
              max_seq_len,
              is_scale_channel_wise,
              cache_quant_type_str,
              stream,
              key_cache_out,
              value_cache_out);
        })})
  } else if (cache_quant_type_str == "cache_int4_zp") {
    DISPATCH_HEAD_DIM(
        head_dim, HEAD_DIM, {DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
          CascadeAppendWriteCacheKVC4QKV<T, HEAD_DIM, BLOCK_SIZE>(
              meta_data,
              *key_cache_out,
              *value_cache_out,
              *qkv_out,
              cache_k_scale.get(),
              cache_v_scale.get(),
              cache_k_zp.get(),
              cache_v_zp.get(),
              seq_lens_this_time,
              seq_lens_decoder,
              batch_id_per_token,
              cu_seqlens_q,
              block_tables,
              batch_ids,
              tile_ids,
              num_blocks,
              max_seq_len,
              stream,
              key_cache_out,
              value_cache_out);
        })})
  } else {
    PD_THROW(
        "cache_quant_type_str should be one of [none, cache_int8, "
        "cache_int4_zp]");
  }

  const char* fmt_write_cache_completed_signal_str = std::getenv("FLAGS_fmt_write_cache_completed_signal");
  const char* FLAGS_use_pd_disaggregation_per_chunk = std::getenv("FLAGS_use_pd_disaggregation_per_chunk");
  if (fmt_write_cache_completed_signal_str &&
      (std::strcmp(fmt_write_cache_completed_signal_str, "true") == 0 ||
       std::strcmp(fmt_write_cache_completed_signal_str, "1") == 0)) {
      if (FLAGS_use_pd_disaggregation_per_chunk &&
          (std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "true") == 0 ||
           std::strcmp(FLAGS_use_pd_disaggregation_per_chunk, "1") == 0)) {
        cudaLaunchHostFunc(qkv.stream(),
                           &(RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise_per_query),
                           (void*)nullptr);
      } else {
        if (kv_signal_data) {
          cudaLaunchHostFunc(qkv.stream(),
                            &RemoteCacheKvIpc::save_cache_kv_complete_signal_layerwise,
                            (void*)(const_cast<int64_t*>(kv_signal_data.get().data<int64_t>())));
        }
      }
  }
}
