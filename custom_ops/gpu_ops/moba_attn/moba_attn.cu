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

#include "paddle/extension.h"
#include "moba_attn.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

std::vector<paddle::Tensor> MobaAttention(
        const paddle::Tensor& qkv,
        const paddle::Tensor& q_input,
        const paddle::Tensor& k_input,
        const paddle::Tensor& v_input,
        const paddle::Tensor& cu_seq_q,
        const paddle::Tensor& cu_seq_k,
        const paddle::Tensor& cu_seq_q_pack,
        const paddle::Tensor& q_pack_tokens,
        const paddle::Tensor& seq_len_encoder,
        const paddle::Tensor& seq_len_decoder,
        const paddle::Tensor& key_cache,
        const paddle::Tensor& value_cache,
        const paddle::Tensor& block_tables,
        const paddle::Tensor& rope_sin_cos,
        const paddle::Tensor& k_block_means,
        const paddle::optional<paddle::Tensor>& attn_gate_weight,
        const paddle::optional<paddle::Tensor>& qkv_bias,
        const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
        const paddle::optional<paddle::Tensor>& cache_k_zero_points,
        const paddle::optional<paddle::Tensor>& cache_v_zero_points,
        const int head_num,
        const int kv_head_num,
        const int head_dim,
        const int max_seq_len,
        const int max_enc_len_this_time,
        const int max_dec_len_this_time,
        const int moba_encoder_top_k_left,
        const int moba_encoder_top_k_right,
        const int moba_use_encoder_seq_limit,
        const int moba_decoder_top_k_left,
        const int moba_decoder_top_k_right,
        const int moba_use_decoder_seq_limit,
        const bool moba_use_mlp,
        const std::string &cache_quant_type_str) {

    paddle::Tensor out = paddle::empty({qkv.dims()[0], head_num * head_dim}, qkv.dtype(), qkv.place());
    if (max_dec_len_this_time > 0) {
        MobaDecoderAttnWriteCacheKv(
            qkv,
            q_input,
            cu_seq_q,
            cu_seq_k,
            seq_len_encoder,
            seq_len_decoder,
            key_cache,
            value_cache,
            block_tables,
            rope_sin_cos,
            k_block_means,
            qkv_bias,
            cache_k_quant_scale,
            cache_v_quant_scale,
            cache_k_dequant_scale,
            cache_v_dequant_scale,
            cache_k_zero_points,
            cache_v_zero_points,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_len,
            cache_quant_type_str);

        auto qk_gate_weight = MobaQKGemm(
            q_input,
            k_block_means,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            max_dec_len_this_time,
            max_dec_len_this_time,
            head_num,
            kv_head_num,
            true,
            moba_use_decoder_seq_limit
        )[0];

        auto qk_gate_topk_idx = QkSortDecoder(
            qk_gate_weight,
            seq_len_encoder,
            seq_len_decoder,
            head_num,
            kv_head_num,
            moba_decoder_top_k_left,
            moba_decoder_top_k_right,
            moba_use_decoder_seq_limit
        )[0];

        MobaDecoderAttn(
            q_input,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            key_cache,
            value_cache,
            block_tables,
            k_block_means,
            out,
            qk_gate_topk_idx,
            cache_k_quant_scale,
            cache_v_quant_scale,
            cache_k_dequant_scale,
            cache_v_dequant_scale,
            cache_k_zero_points,
            cache_v_zero_points,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_len,
            moba_use_decoder_seq_limit,
            max_dec_len_this_time,
            max_dec_len_this_time,
            cache_quant_type_str
        );
    }

    if (max_enc_len_this_time > 0) {
        FusedBlockMeanAndRope(
            qkv,
            k_block_means,
            q_input,
            k_input,
            v_input,
            rope_sin_cos,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            qkv_bias,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_len,
            max_enc_len_this_time,
            max_enc_len_this_time,
            cache_quant_type_str
        );

        MobaEncoderAttnWriteCacheKv(
            k_input,
            v_input,
            cu_seq_k,
            seq_len_encoder,
            seq_len_decoder,
            key_cache,
            value_cache,
            block_tables,
            cache_k_quant_scale,
            cache_v_quant_scale,
            cache_k_dequant_scale,
            cache_v_dequant_scale,
            cache_k_zero_points,
            cache_v_zero_points,
            head_num,
            kv_head_num,
            head_dim,
            max_enc_len_this_time,
            cache_quant_type_str
        );

        GetKVFromCache(
            k_input,
            v_input,
            cu_seq_k,
            seq_len_encoder,
            seq_len_decoder,
            key_cache,
            value_cache,
            block_tables,
            cache_k_dequant_scale,
            cache_v_dequant_scale,
            cache_k_zero_points,
            cache_v_zero_points,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_len,
            max_enc_len_this_time + max_dec_len_this_time,
            cache_quant_type_str
        );

        paddle::Tensor *k_gate_weight = const_cast<paddle::Tensor*>(&k_block_means);

        if (moba_use_mlp && attn_gate_weight) {
            paddle::Tensor k_gate_mlp = MobaMlpEinsum(
                k_input,
                attn_gate_weight.get(),
                seq_len_encoder,
                seq_len_decoder,
                cu_seq_k,
                max_seq_len,
                kv_head_num
            )[0];
            k_gate_weight = &k_gate_mlp;
        }

        auto qk_gate_weight = MobaQKGemm(
            q_input,
            *k_gate_weight,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            max_enc_len_this_time,
            max_enc_len_this_time + max_dec_len_this_time,
            head_num,
            kv_head_num,
            false,
            moba_use_encoder_seq_limit
        )[0];


        auto qk_gate_topk_idx = QkSortEncoder(
            qk_gate_weight,
            seq_len_encoder,
            seq_len_decoder,
            cu_seq_q,
            cu_seq_k,
            cu_seq_q_pack,
            q_pack_tokens,
            max_enc_len_this_time,
            max_enc_len_this_time + max_dec_len_this_time,
            head_num,
            kv_head_num,
            moba_encoder_top_k_left,
            moba_encoder_top_k_right,
            moba_use_mlp && !attn_gate_weight ? max_seq_len : moba_use_encoder_seq_limit)[0];

        MobaEncoderAttn(
            q_input,
            k_input,
            v_input,
            qk_gate_topk_idx,
            cu_seq_q,
            cu_seq_k,
            cu_seq_q_pack,
            seq_len_encoder,
            seq_len_decoder,
            out,
            max_enc_len_this_time,
            max_enc_len_this_time + max_dec_len_this_time,
            head_num,
            kv_head_num,
            head_dim,
            max_seq_len
        );
    }

    return {out};
}


PD_BUILD_STATIC_OP(moba_attention)
    .Inputs({
        "qkv",
        "q_input",
        "k_input",
        "v_input",
        "cu_seq_q",
        "cu_seq_k",
        "cu_seq_q_pack",
        "q_pack_tokens",
        "seq_len_encoder",
        "seq_len_decoder",
        "key_cache",
        "value_cache",
        "block_tables",
        "rope_sin_cos",
        "k_block_means",
        paddle::Optional("attn_gate_weight"),
        paddle::Optional("qkv_bias"),
        paddle::Optional("cache_k_quant_scale"),
        paddle::Optional("cache_v_quant_scale"),
        paddle::Optional("cache_k_dequant_scale"),
        paddle::Optional("cache_v_dequant_scale"),
        paddle::Optional("cache_k_zero_points"),
        paddle::Optional("cache_v_zero_points")})
    .Attrs({
        "head_num: int",
        "kv_head_num: int",
        "head_dim: int",
        "max_seq_len: int",
        "max_enc_len_this_time: int",
        "max_dec_len_this_time: int",
        "moba_encoder_top_k_left: int",
        "moba_encoder_top_k_right: int",
        "moba_use_encoder_seq_limit: int",
        "moba_decoder_top_k_left: int",
        "moba_decoder_top_k_right: int",
        "moba_use_decoder_seq_limit: int",
        "moba_use_mlp: bool",
        "cache_quant_type_str: std::string"})
    .Outputs({
        "out",
        "q_input_out",
        "k_input_out",
        "v_input_out",
        "key_cache_out",
        "value_cache_out",
        "k_block_means_out"})
    .SetInplaceMap({{
        "q_input", "q_input_out"},
        {"k_input", "k_input_out"},
        {"v_input", "v_input_out"},
        {"key_cache", "key_cache_out"},
        {"value_cache", "value_cache_out"},
        {"k_block_means", "k_block_means_out"}})
    .SetKernelFn(PD_KERNEL(MobaAttention));
