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

#include "paddle/extension.h"

void MobaDecoderAttnWriteCacheKv(
    const paddle::Tensor& qkv_out,
    const paddle::Tensor& q_input,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& cu_seq_k,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& rope_sin_cos,
    const paddle::Tensor& k_block_means,
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
    const int max_input_length,
    const std::string &cache_quant_type_str);

void MobaEncoderAttnWriteCacheKv(
    const paddle::Tensor& k_input,
    const paddle::Tensor& v_input,
    const paddle::Tensor& cu_seq_k,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zero_points,
    const paddle::optional<paddle::Tensor>& cache_v_zero_points,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int max_seq_q,
    const std::string &cache_quant_type_str);

void MobaDecoderAttn(
    const paddle::Tensor& q_input,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& k_block_means,
    const paddle::Tensor& out,
    const paddle::Tensor& qk_gate_topk_idx,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scale,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scale,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zero_points,
    const paddle::optional<paddle::Tensor>& cache_v_zero_points,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int max_input_length,
    const int use_moba_seq_limit,
    const int max_seq_q,
    const int max_seq_k,
    const std::string &cache_quant_type_str);


void FusedBlockMeanAndRope(
    const paddle::Tensor& qkv_out,
    const paddle::Tensor& k_block_means,
    const paddle::Tensor& q_input,
    const paddle::Tensor& k_input,
    const paddle::Tensor& v_input,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& cu_seq_k,
    const paddle::optional<paddle::Tensor>& qkv_bias,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int max_input_length,
    const int max_seq_q,
    const int max_seq_k,
    const std::string &cache_quant_type_str);

std::vector<paddle::Tensor> GetCurCuSeqLenk(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const int pack_size);

std::vector<paddle::Tensor> MobaQKGemm(
    const paddle::Tensor& q_input,
    const paddle::Tensor& k_block_means,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& cu_seq_k,
    const int max_seq_q,
    const int max_seq_k,
    const int head_num,
    const int kv_head_num,
    const bool is_split_kv,
    const int use_moba_seq_limit);

std::vector<paddle::Tensor> QkSortDecoder(
    const paddle::Tensor& qk_gate_weight,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const int head_num,
    const int kv_head_num,
    const int top_k_left,
    const int top_k_right,
    const int use_moba_seq_limit);

void GetKVFromCache(
    const paddle::Tensor& k_input,
    const paddle::Tensor& v_input,
    const paddle::Tensor& cu_seq_k,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cache_k,
    const paddle::Tensor& cache_v,
    const paddle::Tensor& block_tables,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scale,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scale,
    const paddle::optional<paddle::Tensor>& cache_k_zero_points,
    const paddle::optional<paddle::Tensor>& cache_v_zero_points,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int max_input_length,
    const int max_seq_k,
    const std::string &cache_quant_type_str);


void MobaEncoderAttn(
    const paddle::Tensor& q_input,
    const paddle::Tensor& k_input,
    const paddle::Tensor& v_input,
    const paddle::Tensor& qk_gate_topk_idx,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& cu_seq_k,
    const paddle::Tensor& cu_seq_q_pack,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& out,
    const int max_seq_q,
    const int max_seq_k,
    const int head_num,
    const int kv_head_num,
    const int head_dim,
    const int max_input_length);

std::vector<paddle::Tensor> QkSortEncoder(
    const paddle::Tensor& qk_gate_weight,
    const paddle::Tensor& seq_len_encoder,
    const paddle::Tensor& seq_len_decoder,
    const paddle::Tensor& cu_seq_q,
    const paddle::Tensor& cu_seq_k,
    const paddle::Tensor& cu_seq_q_pack,
    const paddle::Tensor& q_pack_tokens,
    const int max_seq_q,
    const int max_seq_k,
    const int head_num,
    const int kv_head_num,
    const int top_k_left,
    const int top_k_right,
    const int use_moba_seq_limit);

std::vector<paddle::Tensor> MobaMlpEinsum(
    const paddle::Tensor& k_input,
    const paddle::Tensor& attn_gate_weight,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& cu_seq_k,
    const int max_seq_len,
    const int kv_head_num);
