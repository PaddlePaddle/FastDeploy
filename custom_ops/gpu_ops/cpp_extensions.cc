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

#include "paddle/extension.h"
#include "pybind11/pybind11.h"
namespace py = pybind11;

// 自定义异常类，用于处理CUDA错误
class CudaError : public std::exception {
 public:
  explicit CudaError(cudaError_t error) : error_(error) {}

  const char* what() const noexcept override {
    return cudaGetErrorString(error_);
  }

 private:
  cudaError_t error_;
};

// 检查CUDA错误并抛出异常
void check_cuda_error(cudaError_t error) {
  if (error != cudaSuccess) {
    throw CudaError(error);
  }
}

// 封装cudaHostAlloc的Python函数
uintptr_t cuda_host_alloc(size_t size,
                          unsigned int flags = cudaHostAllocDefault) {
  void* ptr = nullptr;
  check_cuda_error(cudaHostAlloc(&ptr, size, flags));
  return reinterpret_cast<uintptr_t>(ptr);
}

// 封装cudaFreeHost的Python函数
void cuda_host_free(uintptr_t ptr) {
  check_cuda_error(cudaFreeHost(reinterpret_cast<void*>(ptr)));
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
    const paddle::Tensor& decoder_num_blocks_cpu,
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
    const int sliding_window);

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
    const paddle::Tensor& decoder_num_blocks_cpu,
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
    const int sliding_window);

std::vector<paddle::Tensor> GQARopeWriteCacheKernel(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& cu_seqlens_k,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& cache_batch_ids,
    const paddle::Tensor& cache_tile_ids,
    const paddle::Tensor& cache_num_blocks,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& kv_signal_data,
    const int kv_token_num,
    const int max_seq_len,
    const std::string& cache_quant_type,
    const bool rope_3d);

std::vector<paddle::Tensor> PreCacheLenConcat(
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const int max_dec_len,
    const int block_size);

paddle::Tensor FusedExpertMoeFunc(
    const paddle::Tensor& input,
    const paddle::Tensor& gate_weight,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_bias,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const std::string& quant_method,
    const int moe_topk,
    const bool norm_topk_prob,
    const bool group_moe);

std::vector<paddle::Tensor> MacheteMMKernel(
    paddle::Tensor const& A,
    paddle::Tensor const& B,
    paddle::optional<paddle::Tensor> const& maybe_group_scales,
    paddle::optional<paddle::Tensor> const& maybe_group_zeros,
    paddle::optional<paddle::Tensor> const& maybe_channel_scales,
    paddle::optional<paddle::Tensor> const& maybe_token_scales,
    std::string const& b_type_str,
    std::string const& maybe_out_type_str,
    int64_t const& maybe_group_size,
    std::string const& maybe_schedule);

std::vector<paddle::Tensor> MachetePrepackBKernel(
    paddle::Tensor const& B,
    std::string const& a_type_str,
    std::string const& b_type_str,
    std::string const& maybe_group_scales_type_str);

std::vector<std::string> MacheteSupportedSchedules(
    std::string const& a_type_str, std::string const& b_type_str);

std::vector<paddle::Tensor> MoeExpertDispatch(
    const paddle::Tensor& input,
    const paddle::Tensor& gating_output,
    const paddle::optional<paddle::Tensor>& gating_correction_bias,
    const paddle::optional<paddle::Tensor>& w4a8_in_scale,
    const int moe_topk,
    const bool group_moe,
    const std::string& moe_quant_type,
    const bool topk_only_mode);

std::vector<paddle::Tensor> MoETopKSelectKernel(
    const paddle::Tensor& gating_logits,
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused);

std::vector<paddle::Tensor> MoERedundantTopKSelectKernel(
    const paddle::Tensor& gating_logits,
    const paddle::Tensor& expert_id_to_ep_rank_array,
    const paddle::Tensor& expert_in_rank_num_list,
    paddle::Tensor& tokens_per_expert_stats_list,
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one);

std::vector<paddle::Tensor> EPMoeExpertDispatch(
    const paddle::Tensor& input,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::optional<paddle::Tensor>& up_gate_proj_in_scale,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string& moe_quant_type);

std::vector<paddle::Tensor> EPMoeExpertDispatchFP8(
    const paddle::Tensor& input,
    const paddle::Tensor& scale,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::Tensor& token_nums_per_expert,
    const paddle::Tensor& token_nums_per_expert_padded,
    const bool use_in_ep,
    const int token_nums_this_rank_padded);

std::vector<paddle::Tensor> PerTokenQuant(paddle::Tensor& input,
                                          const int block_size);
std::vector<paddle::Tensor> PerTokenQuantPadding(paddle::Tensor& input,
                                                 const int block_size);
std::vector<paddle::Tensor> MaskedPerTokenQuant(
    paddle::Tensor& input,
    paddle::Tensor& recv_expert_count,
    const int block_size);

std::vector<paddle::Tensor> EPMoeExpertCombine(
    const paddle::Tensor& ffn_out,
    const paddle::Tensor& expert_scales_float,
    const paddle::Tensor& permute_indices_per_token,
    const paddle::Tensor& top_k_indices,
    const paddle::optional<paddle::Tensor>& down_proj_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor);

std::vector<std::vector<int>> GetExpertTokenNum(const paddle::Tensor& topk_ids,
                                                const int num_experts);

paddle::Tensor MoeExpertFFNFunc(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_proj_in_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method,
    const bool used_in_ep_low_latency,
    const int estimate_total_token_nums,
    const int hadamard_block_size,
    const std::string& activation);

paddle::Tensor MoeExpertFFNWint2Func(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_scale,
    const paddle::optional<paddle::Tensor>& down_proj_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_local_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_scale,
    const paddle::optional<paddle::Tensor>& up_gate_proj_code_zp,
    const paddle::optional<paddle::Tensor>& down_proj_local_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_scale,
    const paddle::optional<paddle::Tensor>& down_proj_code_zp,
    const bool used_in_ep_low_latency);

paddle::Tensor MoeExpertReduceFunc(
    const paddle::Tensor& ffn_out,
    const paddle::Tensor& top_k_weight,
    const paddle::Tensor& permute_indices_per_token,
    const paddle::Tensor& top_k_indices,
    const paddle::optional<paddle::Tensor>& down_proj_bias,
    const bool norm_topk_prob,
    const float routed_scaling_factor);

void InitKVSignalPerQuery(const paddle::Tensor& seq_lens_encoder_tensor,
                          const paddle::Tensor& seq_lens_this_time_tensor,
                          const paddle::Tensor& seq_lens_decoder_tensor,
                          const int rank,
                          const int num_layers);

void GetOutputKVSignal(const paddle::Tensor& x,
                       int64_t rank_id,
                       bool wait_flag);

paddle::Tensor DequantInt8Func(const paddle::Tensor& input,
                               const paddle::Tensor& out_scale,
                               std::string dtype);

paddle::Tensor OpenShmAndGetMetaSignalFunc(const int rank,
                                           const int device_id,
                                           const bool keep_pd_step_flag);

paddle::Tensor InitSignalLayerwiseFunc(const paddle::Tensor& kv_signal_metadata,
                                       const int layer_id);

void GetBlockShapeAndSplitKVBlock(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    paddle::Tensor& decoder_batch_ids,           // Inplace
    paddle::Tensor& decoder_tile_ids_per_batch,  // Inplace
    paddle::Tensor& decoder_num_blocks_cpu,      // Inplace, Pinned Memory
    paddle::Tensor& decoder_num_blocks_device,   // Inplace
    paddle::Tensor& decoder_chunk_size_device,   // Inplace
    paddle::Tensor& max_len_tensor_cpu,          // Inplace, Pinned Memory
    paddle::Tensor& encoder_batch_ids,           // Inplace
    paddle::Tensor& encoder_tile_ids_per_batch,  // Inplace
    paddle::Tensor& encoder_num_blocks_x_cpu,    // Inplace, Pinned Memory
    paddle::Tensor& kv_batch_ids,                // Inplace
    paddle::Tensor& kv_tile_ids_per_batch,       // Inplace
    paddle::Tensor& kv_num_blocks_x_cpu,         // Inplace, Pinned Memory
    const int encoder_block_shape_q,
    const int decoder_block_shape_q,
    const int group_size,
    const int block_size);

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len);

void SetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all,
                           const paddle::Tensor& input_ids,
                           const paddle::Tensor& seq_lens_this_time,
                           const paddle::Tensor& seq_lens_encoder,
                           const paddle::Tensor& seq_lens_decoder,
                           const paddle::Tensor& step_idx,
                           const paddle::Tensor& stop_flags);

paddle::Tensor RebuildPaddingFunc(
    const paddle::Tensor& tmp_out,      // [token_num, dim_embed]
    const paddle::Tensor& cum_offsets,  // [bsz, 1]
    const paddle::Tensor& seq_len_this_time,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::optional<paddle::Tensor>& output_padding_offset,
    const paddle::optional<paddle::Tensor>& first_token_out,
    int max_input_length,
    bool enable_logprob);

void GetStopFlagsMulti(const paddle::Tensor& topk_ids,
                       const paddle::Tensor& stop_flags,
                       const paddle::Tensor& seq_lens,
                       const paddle::Tensor& end_ids,
                       const paddle::Tensor& next_tokens,
                       const paddle::Tensor& pre_ids,
                       const paddle::Tensor& step_idx,
                       const paddle::Tensor& stop_seqs,
                       const paddle::Tensor& stop_seqs_len,
                       const bool beam_search);

void UpdateInputs(const paddle::Tensor& stop_flags,
                  const paddle::Tensor& not_need_stop,  // only on cpu
                  const paddle::Tensor& seq_lens_this_time,
                  const paddle::Tensor& seq_lens_encoder,
                  const paddle::Tensor& seq_lens_decoder,
                  const paddle::Tensor& input_ids,
                  const paddle::Tensor& stop_nums,
                  const paddle::Tensor& next_tokens,
                  const paddle::Tensor& is_block_step);

void UpdateInputsV1(const paddle::Tensor& stop_flags,
                    const paddle::Tensor& not_need_stop,  // only on cpu
                    const paddle::Tensor& seq_lens_this_time,
                    const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& step_seq_lens_decoder,
                    const paddle::Tensor& prompt_lens,
                    const paddle::Tensor& topk_ids,
                    const paddle::Tensor& input_ids,
                    const paddle::Tensor& block_tables,
                    const paddle::Tensor& stop_nums,
                    const paddle::Tensor& next_tokens,
                    const paddle::Tensor& is_block_step,
                    const int block_size);

void RecoverDecodeTask(
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& step_seq_lens_decoder,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& is_block_step,
    const paddle::optional<paddle::Tensor>& draft_tokens,
    const paddle::optional<paddle::Tensor>& step_draft_tokens,
    const paddle::optional<paddle::Tensor>& step_seq_lens_this_time,
    const int block_size,
    const int max_draft_tokens);

paddle::Tensor GroupSwigluWithMasked(
    const paddle::Tensor& fc1_out_tensor,
    const paddle::Tensor& token_nums_per_expert);

std::vector<paddle::Tensor> ExtractTextTokenOutput(
    const paddle::Tensor& max_seq_len,
    const paddle::Tensor& max_seq_len_index,
    const paddle::Tensor& mm_token_num_len,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& hidden_states);

std::vector<paddle::Tensor> MoEDeepGEMMPermute(const paddle::Tensor& x,
                                               const paddle::Tensor& topk_idx,
                                               const int num_experts,
                                               const int max_tokens_per_expert);

std::vector<paddle::Tensor> MoEDeepGEMMDePermute(
    const paddle::Tensor&
        ffn_out,  // [num_experts, max_tokens_per_expert, hidden]
    const paddle::Tensor& permute_indices_per_token,  // [token_num, topk}]
    const paddle::Tensor& topk_idx,
    const paddle::Tensor& topk_weights);

void TextImageIndexOut(const paddle::Tensor& token_type_ids,
                       paddle::Tensor& text_input,
                       paddle::Tensor& image_input);

std::vector<paddle::Tensor> TextImageGatherScatter(
    paddle::Tensor& input,
    paddle::Tensor& text_input,
    paddle::Tensor& image_input,
    paddle::Tensor& token_type_ids,
    paddle::Tensor& text_index,
    paddle::Tensor& image_index,
    const bool is_scatter);

paddle::Tensor count_tokens_per_expert_func(const paddle::Tensor& topk_ids,
                                            int64_t num_experts);
void GetPositionIdsAndMaskEncoderBatch(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& position_ids,
    const paddle::Tensor& mask_encoder_batch);

std::vector<paddle::Tensor> DecodeMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len,
    const bool speculate_decoder);

std::vector<paddle::Tensor> PrefillMLAWriteCacheKernel(
    const paddle::Tensor& kv_nope,
    const paddle::Tensor& kv_pe,
    const paddle::Tensor& kv_cache,
    const paddle::Tensor& seq_lens,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& block_tables,
    const std::string& cache_quant_type_str,
    const int max_seq_len);

void FusedRotaryPositionEncoding(
    paddle::Tensor& query,  // [num_tokens, num_heads, head_size] or
                            // [num_tokens, num_heads * head_size]
    paddle::Tensor& key,
    // [num_tokens, num_kv_heads, head_size] or [num_tokens, num_kv_heads *
    // head_size]
    const paddle::Tensor& position_ids,   // [num_tokens]
    const paddle::Tensor& cos_sin_cache,  // [max_position, rot_dim]
    int head_size,
    bool is_neox);

std::vector<paddle::Tensor> MultiHeadLatentAttention(
    const paddle::Tensor& query,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& batch_id_per_token,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& kv_batch_ids,
    const paddle::Tensor& kv_tile_ids_per_batch,
    const paddle::Tensor& kv_num_blocks,
    const paddle::Tensor& decoder_batch_ids,
    const paddle::Tensor& decoder_tile_ids_per_batch,
    const paddle::Tensor& decoder_num_blocks_device,
    const paddle::Tensor& decoder_chunk_size_device,
    const paddle::Tensor& max_dec_len_this_time,
    const paddle::Tensor& max_len_kv,
    const paddle::optional<paddle::Tensor>& attn_mask,
    const paddle::optional<paddle::Tensor>& query_bias,
    const paddle::optional<paddle::Tensor>& query_out_scales,
    const paddle::optional<paddle::Tensor>& cache_k_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_quant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor>& cache_k_zp,
    const paddle::optional<paddle::Tensor>& cache_v_zp,
    const paddle::optional<paddle::Tensor>& out_linear_shifts,
    const paddle::optional<paddle::Tensor>& out_linear_smooths,
    const std::string& compute_dtype,
    const std::string& cache_quant_type_str,
    const int nope_size,
    const int max_input_length,
    const float softmax_scale,
    const float quant_max_bound,
    const float quant_min_bound,
    const float out_linear_in_scale,
    const int speculate_max_draft_token_num,
    const bool causal,
    const bool speculate_decoder);

std::vector<paddle::Tensor> tritonmoe_preprocess_kernel(
    const paddle::Tensor& topk_ids,
    int64_t num_experts,
    int64_t GEMM_BLOCK_SIZE_M);

std::vector<paddle::Tensor> MoeWna16MarlinGemmApi(
    const paddle::Tensor& a,
    const paddle::optional<paddle::Tensor>& c_or_none,
    const paddle::Tensor& b_q_weight,
    const paddle::Tensor& b_scales,
    const paddle::optional<paddle::Tensor>& global_scale_or_none,
    const paddle::optional<paddle::Tensor>& b_zeros_or_none,
    const paddle::optional<paddle::Tensor>& g_idx_or_none,
    const paddle::optional<paddle::Tensor>& perm_or_none,
    const paddle::Tensor& workspace,
    const paddle::Tensor& sorted_token_ids,
    const paddle::Tensor& expert_ids,
    const paddle::Tensor& num_tokens_post_padded,
    const paddle::Tensor& topk_weights,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    bool is_ep,
    const std::string& b_q_type_str,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool is_k_full,
    bool use_atomic_add,
    bool use_fp32_reduce,
    bool is_zp_float);
void CutlassScaledMm(paddle::Tensor& c,
                     paddle::Tensor const& a,
                     paddle::Tensor const& b,
                     paddle::Tensor const& a_scales,
                     paddle::Tensor const& b_scales,
                     paddle::optional<paddle::Tensor> const& bias);

void CutlassScaledMmAzp(paddle::Tensor& c,
                        paddle::Tensor const& a,
                        paddle::Tensor const& b,
                        paddle::Tensor const& a_scales,
                        paddle::Tensor const& b_scales,
                        paddle::Tensor const& azp_adj,
                        paddle::optional<paddle::Tensor> const& azp,
                        paddle::optional<paddle::Tensor> const& bias);

void StaticScaledFp8Quant(paddle::Tensor& out,
                          paddle::Tensor const& input,
                          paddle::Tensor const& scale);

void DynamicScaledFp8Quant(paddle::Tensor& out,
                           paddle::Tensor const& input,
                           paddle::Tensor& scale);

void DynamicPerTokenScaledFp8Quant(paddle::Tensor& out,
                                   paddle::Tensor const& input,
                                   paddle::Tensor& scales,
                                   float scale_ub);

std::vector<paddle::Tensor> NoauxTc(paddle::Tensor& scores,
                                    paddle::Tensor& scores_with_bias,
                                    int n_group,
                                    int topk_group,
                                    int topk,
                                    bool renormalize,
                                    float routed_scaling_factor);

std::vector<paddle::Tensor> NoauxTcRedundant(
    paddle::Tensor& scores,
    paddle::Tensor& scores_with_bias,
    paddle::Tensor& expert_id_to_ep_rank_array,
    paddle::Tensor& expert_in_rank_num_list,
    paddle::Tensor& tokens_per_expert_stats_list,
    int n_group,
    int topk_group,
    int topk,
    bool renormalize,
    float routed_scaling_factor,
    int redundant_ep_rank_num_plus_one);

#ifdef ENABLE_FP8
paddle::Tensor cutlass_fp8_fp8_half_gemm_func(
    const paddle::Tensor& x,
    const paddle::Tensor& y,
    const paddle::optional<paddle::Tensor>& bias,
    bool trans_x,
    bool trans_y,
    float scale,  // only support per-tensor quantization
    std::string output_dtype,
    std::string activation_type);

paddle::Tensor MoeFusedHadamardQuantFp8Func(const paddle::Tensor& input,
                                            const paddle::Tensor& scale,
                                            const paddle::Tensor& topk_ids,
                                            const int top_k,
                                            const int intermediate_size,
                                            const bool tiled);

paddle::Tensor FusedHadamardQuantFp8Func(const paddle::Tensor& input,
                                         const float scale);
#endif

int64_t init_custom_all_reduce(const std::vector<int64_t>& fake_ipc_ptrs,
                               paddle::Tensor& rank_data,
                               int64_t rank,
                               bool full_nvlink);

void all_reduce(paddle::Tensor& inp,
                paddle::Tensor& out,
                int64_t _fa,
                int64_t reg_buffer,
                int64_t reg_buffer_sz_bytes);

void dispose(int64_t _fa);

int64_t meta_size();

void register_buffer(int64_t _fa, const std::vector<int64_t>& fake_ipc_ptrs);

std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(int64_t _fa);

void register_graph_buffers(int64_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets);

std::tuple<int64_t, paddle::Tensor> allocate_shared_buffer_and_handle(
    int64_t size);

int64_t open_mem_handle(paddle::Tensor& mem_handle);

void free_shared_buffer(int64_t buffer);

void clear_ipc_handles(int64_t _fa);

// speculative decoding Kernel
std::vector<paddle::Tensor> SpeculateGetPaddingOffset(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& draft_tokens,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& token_num,
    const paddle::Tensor& seq_len,
    const paddle::Tensor& seq_lens_encoder);

std::vector<paddle::Tensor> SpeculateGetSeqLensOutput(
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder);

std::vector<paddle::Tensor> SpeculateGetOutputPaddingOffset(
    const paddle::Tensor& output_cum_offsets_tmp,
    const paddle::Tensor& out_token_num,
    const paddle::Tensor& seq_lens_output,
    const int max_seq_len);

void SpecTokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                                 const paddle::Tensor& logits,
                                 const paddle::Tensor& penalty_scores,
                                 const paddle::Tensor& frequency_scores,
                                 const paddle::Tensor& presence_scores,
                                 const paddle::Tensor& temperatures,
                                 const paddle::Tensor& bad_tokens,
                                 const paddle::Tensor& cur_len,
                                 const paddle::Tensor& min_len,
                                 const paddle::Tensor& eos_token_id,
                                 const paddle::Tensor& seq_lens_this_time,
                                 const paddle::Tensor& output_padding_offset,
                                 const paddle::Tensor& output_cum_offsets,
                                 const int max_seq_len);

void SpecGetStopFlagsMultiSeqs(const paddle::Tensor& accept_tokens,
                               const paddle::Tensor& accept_num,
                               const paddle::Tensor& pre_ids,
                               const paddle::Tensor& step_idx,
                               const paddle::Tensor& stop_flags,
                               const paddle::Tensor& seq_lens,
                               const paddle::Tensor& stop_seqs,
                               const paddle::Tensor& stop_seqs_len,
                               const paddle::Tensor& end_ids);

void SpeculateVerify(const paddle::Tensor& sampled_token_ids,
                     const paddle::Tensor& accept_tokens,
                     const paddle::Tensor& accept_num,
                     const paddle::Tensor& step_idx,
                     const paddle::Tensor& stop_flags,
                     const paddle::Tensor& seq_lens_encoder,
                     const paddle::Tensor& seq_lens_decoder,
                     const paddle::Tensor& draft_tokens,
                     const paddle::Tensor& seq_lens_this_time,
                     const paddle::Tensor& verify_tokens,
                     const paddle::Tensor& verify_scores,
                     const paddle::Tensor& max_dec_len,
                     const paddle::Tensor& end_tokens,
                     const paddle::Tensor& is_block_step,
                     const paddle::Tensor& output_cum_offsets,
                     const paddle::Tensor& actual_candidate_len,
                     const paddle::Tensor& actual_draft_token_nums,
                     const paddle::Tensor& topp,
                     int max_seq_len,
                     int verify_window,
                     bool enable_topp,
                     bool benchmark_mode,
                     bool accept_all_drafts);

void SpeculateUpdate(const paddle::Tensor& seq_lens_encoder,
                     const paddle::Tensor& seq_lens_decoder,
                     const paddle::Tensor& not_need_stop,
                     const paddle::Tensor& draft_tokens,
                     const paddle::Tensor& actual_draft_token_nums,
                     const paddle::Tensor& accept_tokens,
                     const paddle::Tensor& accept_num,
                     const paddle::Tensor& stop_flags,
                     const paddle::Tensor& seq_lens_this_time,
                     const paddle::Tensor& is_block_step,
                     const paddle::Tensor& stop_nums,
                     const paddle::Tensor& mask_rollback);

void SpeculateSetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all,
                                    const paddle::Tensor& accept_tokens,
                                    const paddle::Tensor& accept_num,
                                    const paddle::Tensor& stop_flags,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& step_idx);

void SpeculateSaveWithOutputMsgStatic(const paddle::Tensor& accept_tokens,
                                      const paddle::Tensor& accept_num,
                                      const paddle::Tensor& not_need_stop,
                                      const paddle::Tensor& seq_lens_decoder,
                                      const paddle::Tensor& prompt_lens,
                                      int64_t rank_id,
                                      bool save_each_rank,
                                      bool skip_prefill);

void SpeculateClearAcceptNums(const paddle::Tensor& accept_num,
                              const paddle::Tensor& seq_lens_decoder);

void SpeculateScheduleCache(const paddle::Tensor& draft_tokens,
                            const paddle::Tensor& block_tables,
                            const paddle::Tensor& stop_flags,
                            const paddle::Tensor& prompt_lens,
                            const paddle::Tensor& seq_lens_this_time,
                            const paddle::Tensor& seq_lens_encoder,
                            const paddle::Tensor& seq_lens_decoder,
                            const paddle::Tensor& step_seq_lens_decoder,
                            const paddle::Tensor& step_draft_tokens,
                            const paddle::Tensor& step_seq_lens_this_time,
                            const paddle::Tensor& accept_num,
                            const paddle::Tensor& accept_tokens,
                            const paddle::Tensor& is_block_step,
                            const paddle::Tensor& not_need_stop,
                            const paddle::Tensor& stop_nums,
                            const int block_size,
                            const int max_draft_tokens);

void NgramMatch(const paddle::Tensor& input_ids,
                const paddle::Tensor& input_ids_len,
                const paddle::Tensor& pre_ids,
                const paddle::Tensor& step_idx,
                const paddle::Tensor& draft_token_num,
                const paddle::Tensor& draft_tokens,
                const paddle::Tensor& seq_lens_this_time,
                const paddle::Tensor& seq_lens_encoder,
                const paddle::Tensor& seq_lens_decoder,
                const paddle::Tensor& max_dec_len,
                const int max_ngram_size,
                const int max_draft_tokens);

void HybridMtpNgram(const paddle::Tensor& input_ids,
                    const paddle::Tensor& input_ids_len,
                    const paddle::Tensor& pre_ids,
                    const paddle::Tensor& step_idx,
                    const paddle::Tensor& draft_token_num,
                    const paddle::Tensor& draft_tokens,
                    const paddle::Tensor& seq_lens_this_time,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& max_dec_len,
                    const int max_ngram_size,
                    const int min_ngram_size,
                    const int max_draft_tokens);

// MTP
void DraftModelPostprocess(const paddle::Tensor& base_model_draft_tokens,
                           const paddle::Tensor& base_model_seq_lens_this_time,
                           const paddle::Tensor& base_model_seq_lens_encoder,
                           const paddle::Tensor& base_model_stop_flags);

void DraftModelPreprocess(const paddle::Tensor& draft_tokens,
                          const paddle::Tensor& input_ids,
                          const paddle::Tensor& stop_flags,
                          const paddle::Tensor& seq_lens_this_time,
                          const paddle::Tensor& seq_lens_encoder,
                          const paddle::Tensor& seq_lens_decoder,
                          const paddle::Tensor& step_idx,
                          const paddle::Tensor& not_need_stop,
                          const paddle::Tensor& is_block_step,
                          const paddle::Tensor& batch_drop,
                          const paddle::Tensor& pre_ids,
                          const paddle::Tensor& accept_tokens,
                          const paddle::Tensor& accept_num,
                          const paddle::Tensor& base_model_seq_lens_this_time,
                          const paddle::Tensor& base_model_seq_lens_encoder,
                          const paddle::Tensor& base_model_seq_lens_decoder,
                          const paddle::Tensor& base_model_step_idx,
                          const paddle::Tensor& base_model_stop_flags,
                          const paddle::Tensor& base_model_is_block_step,
                          const paddle::Tensor& base_model_draft_tokens,
                          const int max_draft_token,
                          const bool truncate_first_token,
                          const bool splitwise_prefill,
                          const bool kvcache_scheduler_v1);

void DraftModelUpdate(const paddle::Tensor& inter_next_tokens,
                      const paddle::Tensor& draft_tokens,
                      const paddle::Tensor& pre_ids,
                      const paddle::Tensor& seq_lens_this_time,
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& seq_lens_decoder,
                      const paddle::Tensor& step_idx,
                      const paddle::Tensor& output_cum_offsets,
                      const paddle::Tensor& stop_flags,
                      const paddle::Tensor& not_need_stop,
                      const paddle::Tensor& max_dec_len,
                      const paddle::Tensor& end_ids,
                      const paddle::Tensor& base_model_draft_tokens,
                      const int max_seq_len,
                      const int substep);

std::vector<paddle::Tensor> EagleGetHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& accept_nums,
    const paddle::Tensor& base_model_seq_lens_this_time,
    const paddle::Tensor& base_model_seq_lens_encoder,
    const int actual_draft_token_num);

std::vector<paddle::Tensor> EagleGetSelfHiddenStates(
    const paddle::Tensor& input,
    const paddle::Tensor& last_seq_lens_this_time,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& step_idx);

void MTPStepPaddle(
    const paddle::Tensor& base_model_stop_flags,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& batch_drop,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& block_tables,  // [bsz, block_num_per_seq]
    const paddle::Tensor& encoder_block_lens,
    const paddle::Tensor& used_list_len,
    const paddle::Tensor& free_list,
    const paddle::Tensor& free_list_len,
    const int block_size,
    const int max_draft_tokens);

void SpeculateStepPaddle(
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& ori_seq_lens_encoder,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& block_tables,  // [bsz, block_num_per_seq]
    const paddle::Tensor& encoder_block_lens,
    const paddle::Tensor& is_block_step,
    const paddle::Tensor& step_block_list,
    const paddle::Tensor& step_lens,
    const paddle::Tensor& recover_block_list,
    const paddle::Tensor& recover_lens,
    const paddle::Tensor& need_block_list,
    const paddle::Tensor& need_block_len,
    const paddle::Tensor& used_list_len,
    const paddle::Tensor& free_list,
    const paddle::Tensor& free_list_len,
    const paddle::Tensor& input_ids,
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& next_tokens,
    const paddle::Tensor& first_token_ids,
    const paddle::Tensor& accept_num,
    const int block_size,
    const int encoder_decoder_block_num,
    const int max_draft_tokens);

void MergePrefillDecodeOutput(const paddle::Tensor& encoder_res,
                              const paddle::Tensor& decoder_res,
                              const paddle::Tensor& seq_lens_encoder,
                              const paddle::Tensor& seq_lens_decoder,
                              const paddle::Tensor& seq_lens_this_time,
                              const paddle::Tensor& cu_seq_q,
                              const int head_num,
                              const int head_dim,
                              const int max_token);

std::vector<paddle::Tensor> TopPSamplingReject(
    const paddle::Tensor& probs,
    const paddle::Tensor& top_p,
    const paddle::optional<paddle::Tensor>& top_k,
    int64_t seed);

std::vector<paddle::Tensor> TopKRenorm(const paddle::Tensor& probs,
                                       const paddle::Tensor& top_k);

std::vector<paddle::Tensor> MinPSamplingFromProbs(const paddle::Tensor& probs,
                                                  const paddle::Tensor& min_p);

void SaveOutMmsgStatic(const paddle::Tensor& x,
                       const paddle::Tensor& not_need_stop,
                       int64_t rank_id,
                       bool save_each_rank);

void LimitThinkingContentLengthV1(const paddle::Tensor& next_tokens,
                                  const paddle::Tensor& max_think_lens,
                                  const paddle::Tensor& step_idx,
                                  const paddle::Tensor& limit_think_status,
                                  const paddle::Tensor& stop_flags,
                                  const paddle::Tensor& eos_token_ids,
                                  const int64_t think_end_id);

void LimitThinkingContentLengthV2(const paddle::Tensor& next_tokens,
                                  const paddle::Tensor& max_think_lens,
                                  const paddle::Tensor& step_idx,
                                  const paddle::Tensor& limit_think_status,
                                  const paddle::Tensor& stop_flags,
                                  const int64_t think_end_id,
                                  const int64_t line_break_id);

void SpeculateLimitThinkingContentLengthV1(
    const paddle::Tensor& next_tokens,
    const paddle::Tensor& max_think_lens,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& limit_think_status,
    const paddle::Tensor& accept_num,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& eos_token_ids,
    const int64_t think_end_id);

void SpeculateLimitThinkingContentLengthV2(
    const paddle::Tensor& next_tokens,
    const paddle::Tensor& max_think_lens,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& limit_think_status,
    const paddle::Tensor& accept_num,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& stop_flags,
    const int64_t think_end_id,
    const int64_t line_break_id);

void SpeculateGetLogits(const paddle::Tensor& draft_logits,
                        const paddle::Tensor& next_token_num,
                        const paddle::Tensor& batch_token_num,
                        const paddle::Tensor& cu_next_token_offset,
                        const paddle::Tensor& cu_batch_token_offset,
                        const paddle::Tensor& logits,
                        const paddle::Tensor& first_token_logits,
                        const paddle::Tensor& seq_lens_this_time,
                        const paddle::Tensor& seq_lens_encoder);

void SpeculateInsertFirstToken(const paddle::Tensor& token_ids,
                               const paddle::Tensor& accept_tokens,
                               const paddle::Tensor& next_tokens,
                               const paddle::Tensor& cu_next_token_offset,
                               const paddle::Tensor& cu_batch_token_offset,
                               const paddle::Tensor& seq_lens_this_time,
                               const paddle::Tensor& seq_lens_encoder);

void SpeculateGetTargetLogits(const paddle::Tensor& target_logits,
                              const paddle::Tensor& logits,
                              const paddle::Tensor& cu_batch_token_offset,
                              const paddle::Tensor& ori_cu_batch_token_offset,
                              const paddle::Tensor& seq_lens_this_time,
                              const paddle::Tensor& seq_lens_encoder,
                              const paddle::Tensor& accept_num);

std::vector<paddle::Tensor> UpdateAttnMaskOffsets(
    const paddle::Tensor& ids_remove_padding,
    const paddle::Tensor& seq_lens_this_time,  // only on cpu
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& attn_mask_offsets_full,
    const paddle::Tensor& attn_mask_offsets_decoder,
    const paddle::Tensor& is_block_step,
    const paddle::Tensor& decode_states,
    const paddle::Tensor& mask_rollback);

std::vector<paddle::Tensor> FusedNeoxRopeEmbedding(
    const paddle::Tensor& qkv,
    const paddle::Tensor& cos_emb,
    const paddle::Tensor& sin_emb,
    const int num_heads,
    const int head_dim);

std::vector<paddle::Tensor> GeluTanh(paddle::Tensor& input);

PYBIND11_MODULE(fastdeploy_ops, m) {
  m.def("get_expert_token_num",
        &GetExpertTokenNum,
        py::arg("topk_ids"),
        py::arg("num_experts"),
        "get expert token num");

  /**
   * moe/fused_moe/moe_redundant_topk_select.cu
   * moe_redundant_topk_select
   */
  m.def("moe_redundant_topk_select",
        &MoERedundantTopKSelectKernel,
        py::arg("gating_logits"),
        py::arg("expert_id_to_ep_rank_array"),
        py::arg("expert_in_rank_num_list"),
        py::arg("tokens_per_expert_stats_list"),
        py::arg("bias"),
        py::arg("moe_topk"),
        py::arg("apply_norm_weight"),
        py::arg("enable_softmax_top_k_fused"),
        py::arg("redundant_ep_rank_num_plus_one"),
        "moe export RedundantTopKSelect function");

  /**
   * open_shm_and_get_meta_signal.cc
   * InitKVSignalPerQuery
   */
  m.def("init_kv_signal_per_query",
        &InitKVSignalPerQuery,
        py::arg("seq_lens_encoder_tensor"),
        py::arg("seq_lens_this_time_tensor"),
        py::arg("seq_lens_decoder_tensor"),
        py::arg("rank"),
        py::arg("num_layers"),
        "init_kv_signal_per_query function");

  /**
   * GetOutputKVSignal
   */
  m.def("get_output_kv_signal",
        &GetOutputKVSignal,
        py::arg("x"),
        py::arg("rank_id"),
        py::arg("wait_flag"),
        "get_output_kv_signal function");

  m.def("moe_deepgemm_permute", &MoEDeepGEMMPermute, "MoEDeepGEMMPermute");
  m.def(
      "moe_deepgemm_depermute", &MoEDeepGEMMDePermute, "MoEDeepGEMMDePermute");
  /**
   * alloc_cache_pinned.cc
   * cuda_host_alloc
   * cuda_host_free
   */
  m.def("cuda_host_alloc",
        &cuda_host_alloc,
        "Allocate pinned memory",
        py::arg("size"),
        py::arg("flags") = cudaHostAllocDefault);
  m.def(
      "cuda_host_free", &cuda_host_free, "Free pinned memory", py::arg("ptr"));
  py::register_exception<CudaError>(m, "CudaError");
  /**
   * append_attention.cu
   * append_attention
   */
  m.def("append_attention", &AppendAttention, "append attention function");
  m.def("append_attention_with_output",
        &AppendAttentionWithOutput,
        "append attention with output function");
  /**
   * gqa_rope_write_cache.cu
   * gqa_rope_write_cache
   */
  m.def("gqa_rope_write_cache",
        &GQARopeWriteCacheKernel,
        "gqa rope write cache function");
  /**
   * pre_cache_len_concat.cu
   * pre_cache_len_concat
   */
  m.def("pre_cache_len_concat",
        &PreCacheLenConcat,
        "pre_cache len concat function");
  /**
   * moe/fused_moe/fused_moe.cu
   * fused_moe
   */
  m.def("fused_moe", &FusedExpertMoeFunc, "fused moe function");

  /**
   * moe/fused_moe/fused_moe.cu
   * fused_expert_moe
   */
  m.def("fused_expert_moe", &FusedExpertMoeFunc, "fused moe function");

  /**
   * moe/fused_moe/moe_dispatch.cu
   * moe_expert_dispatch
   */
  m.def("moe_expert_dispatch",
        &MoeExpertDispatch,
        py::arg("input"),
        py::arg("gating_output"),
        py::arg("gating_correction_bias"),
        py::arg("w4a8_in_scale"),
        py::arg("moe_topk"),
        py::arg("group_moe"),
        py::arg("moe_quant_type"),
        py::arg("topk_only_mode"),
        "moe export dispatch function");

  /**
   * moe/fused_moe/ep_moe_prefill_func.cu
   * ep_moe_dispatch
   */
  m.def("ep_moe_expert_dispatch",
        &EPMoeExpertDispatch,
        py::arg("input"),
        py::arg("topk_ids"),
        py::arg("topk_weights"),
        py::arg("up_gate_proj_in_scale"),
        py::arg("token_nums_per_expert"),
        py::arg("token_nums_this_rank"),
        py::arg("moe_quant_type"),
        "ep moe export dispatch function");

  m.def("ep_moe_expert_dispatch_fp8", &EPMoeExpertDispatchFP8);

  m.def("ep_moe_expert_combine",
        &EPMoeExpertCombine,
        py::arg("ffn_out"),
        py::arg("expert_scales_float"),
        py::arg("permute_indices_per_token"),
        py::arg("top_k_indices"),
        py::arg("down_proj_bias"),
        py::arg("norm_topk_prob"),
        py::arg("routed_scaling_factor"),
        "ep moe export combine function");

  m.def("per_token_quant",
        &PerTokenQuant,
        py::arg("input"),
        py::arg("block_size"),
        "per token per block quant");

  m.def("per_token_quant_padding",
        &PerTokenQuantPadding,
        py::arg("input"),
        py::arg("block_size"),
        "per token per block quant and padding transpose scale");

  m.def("masked_per_token_quant",
        &MaskedPerTokenQuant,
        py::arg("input"),
        py::arg("recv_expert_count"),
        py::arg("block_size"),
        "per token per block quant");

#ifdef ENABLE_MACHETE
  /*machete/machete_mm.cu
   * machete_mm
   */
  m.def("machete_mm",
        &MacheteMMKernel,
        py::arg("A"),
        py::arg("B"),
        py::arg("maybe_group_scale"),
        py::arg("maybe_group_zeros"),
        py::arg("maybe_channel_scales"),
        py::arg("maybe_token_scales"),
        py::arg("b_type_str"),
        py::arg("maybe_out_type_str"),
        py::arg("maybe_group_size"),
        py::arg("maybe_schedule"),
        "machete mm function");

  /*machete/machete_prepack_B.cu
   * machete_prepack_B
   */
  m.def("machete_prepack_B",
        &MachetePrepackBKernel,
        "machete prepacked B function");

  /*machete/machete_supported_schedules.cu
   * machete_supported_schedules
   */
  m.def("machete_supported_schedules",
        &MacheteSupportedSchedules,
        "machete supported schedules function");
#endif

  /**
   * moe/fused_moe/moe_topk_select.cu
   * moe_topk_select
   */
  m.def("moe_topk_select",
        &MoETopKSelectKernel,
        py::arg("gating_logits"),
        py::arg("bias"),
        py::arg("moe_topk"),
        py::arg("apply_norm_weight"),
        py::arg("enable_softmax_top_k_fused"),
        "moe export TopKSelect function");

  /**
   * moe/fused_moe/moe_ffn.cu
   * moe_expert_ffn
   */
  m.def("moe_expert_ffn", &MoeExpertFFNFunc, "moe export ffn function");

  /**
   * moe/fused_moe/moe_expert_ffn_wint2.cu
   * moe_expert_ffn_wint2
   */
  m.def("moe_expert_ffn_wint2",
        &MoeExpertFFNWint2Func,
        "moe export ffn wint2 function");

  /**
   * moe/fused_moe/moe_expert_reduce.cu
   * moe_expert_reduce
   */
  m.def("moe_expert_reduce",
        &MoeExpertReduceFunc,
        py::arg("ffn_out"),
        py::arg("top_k_weight"),
        py::arg("permute_indices_per_token"),
        py::arg("top_k_indices"),
        py::arg("down_proj_bias"),
        py::arg("norm_topk_prob"),
        py::arg("routed_scaling_factor"),
        "moe export reduce function");

  /**
   * dequant_int8.cu
   * dequant_int8
   */
  m.def("dequant_int8", &DequantInt8Func, "dequant int8 function");

  /**
   * init_signal_layerwise.cc
   * init_signal_layerwise
   */
  m.def("init_signal_layerwise",
        &InitSignalLayerwiseFunc,
        "init_signal_layerwise function");

  /**
   * open_shm_and_get_meta_signal.cc
   * open_shm_and_get_meta_signal
   */
  m.def("open_shm_and_get_meta_signal",
        &OpenShmAndGetMetaSignalFunc,
        "open_shm_and_get_meta_signal function");

  /**
   * append_attn/get_block_shape_and_split_kv_block.cu
   * get_block_shape_and_split_kv_block
   */
  m.def("get_block_shape_and_split_kv_block",
        &GetBlockShapeAndSplitKVBlock,
        "get_block_shape_and_split_kv_block function");

  /**
   * get_padding_offset.cu
   * get_padding_offset
   */
  m.def("get_padding_offset", &GetPaddingOffset, "get_padding_offset function");

  /**
   * get_padding_offset.cu
   * get_padding_offset
   */
  m.def("set_value_by_flags_and_idx",
        &SetValueByFlagsAndIdx,
        "SetValueByFlagsAndIdx");

  /**
   * get_padding_offset.cu
   * get_padding_offset
   */
  m.def("rebuild_padding", &RebuildPaddingFunc, "update_inputs function");

  /**
   * stop_generation_multi_ends.cu
   * set_stop_value_multi_ends
   */
  m.def("set_stop_value_multi_ends",
        &GetStopFlagsMulti,
        "update_inputs function");

  /**
   * update_inputs.cu
   * update_inputs
   */
  m.def("update_inputs", &UpdateInputs, "update_inputs function");

  /**
   * update_inputs_v1.cu
   * update_inputs_v1
   */
  m.def("update_inputs_v1",
        &UpdateInputsV1,
        "update inputs for scheduler v1 function");

  /**
   * recover_decode_task.cu
   * recover_decode_task
   */
  m.def("recover_decode_task",
        &RecoverDecodeTask,
        "recover decode task for scheduler v1 function");

  m.def("group_swiglu_with_masked",
        &GroupSwigluWithMasked,
        "group_swiglu_with_masked function");

  m.def("text_image_index_out",
        &TextImageIndexOut,
        "text_image_index_out function");

  m.def("text_image_gather_scatter",
        &TextImageGatherScatter,
        "text_image_gather_scatter function");

  m.def("count_tokens_per_expert_func", &count_tokens_per_expert_func);
  m.def("tritonmoe_preprocess_func", &tritonmoe_preprocess_kernel);

  m.def("MoeWna16MarlinGemmApi",
        &MoeWna16MarlinGemmApi,
        py::arg("a"),
        py::arg("c_or_none"),
        py::arg("b_q_weight"),
        py::arg("b_scales"),
        py::arg("global_scale_or_none"),
        py::arg("b_zeros_or_none"),
        py::arg("g_idx_or_none"),
        py::arg("perm_or_none"),
        py::arg("workspace"),
        py::arg("sorted_token_ids"),
        py::arg("expert_ids"),
        py::arg("num_tokens_post_padded"),
        py::arg("topk_weights"),
        py::arg("moe_block_size"),
        py::arg("top_k"),
        py::arg("mul_topk_weights"),
        py::arg("is_ep"),
        py::arg("b_q_type_str"),
        py::arg("size_m"),
        py::arg("size_n"),
        py::arg("size_k"),
        py::arg("is_k_full"),
        py::arg("use_atomic_add"),
        py::arg("use_fp32_reduce"),
        py::arg("is_zp_float"));

  m.def("get_position_ids_and_mask_encoder_batch",
        &GetPositionIdsAndMaskEncoderBatch,
        "get_position_ids_and_mask_encoder_batch function");

  /**
   * cutlass_scaled_mm.cu
   * cutlass_scaled_mm
   * cutlass_scaled_mm_azp
   */
  m.def("cutlass_scaled_mm", &CutlassScaledMm, "cutlass_scaled_mm function");
  m.def("cutlass_scaled_mm_azp",
        &CutlassScaledMmAzp,
        "cutlass_scaled_mm_azp function");

  /**
   * quantization/common.cu
   * static_scaled_fp8_quant
   * dynamic_scaled_fp8_quant
   * dynamic_per_token_scaled_fp8_quant
   */
  m.def("static_scaled_fp8_quant",
        &StaticScaledFp8Quant,
        "static_scaled_fp8_quant function",
        py::arg("out"),
        py::arg("input"),
        py::arg("scale"));

  m.def("dynamic_scaled_fp8_quant",
        &DynamicScaledFp8Quant,
        "dynamic_scaled_fp8_quant function",
        py::arg("out"),
        py::arg("input"),
        py::arg("scale"));

  m.def("dynamic_per_token_scaled_fp8_quant",
        &DynamicPerTokenScaledFp8Quant,
        "dynamic_per_token_scaled_fp8_quant function",
        py::arg("out"),
        py::arg("input"),
        py::arg("scales"),
        py::arg("scale_ub"));
  m.def("decode_mla_write_cache",
        &DecodeMLAWriteCacheKernel,
        "decode_mla_write_cache function");

  m.def("prefill_mla_write_cache",
        &PrefillMLAWriteCacheKernel,
        "prefill_mla_write_cache function");

  m.def("fused_rotary_position_encoding",
        &FusedRotaryPositionEncoding,
        "fused_rotary_position_encoding function");

  m.def("multi_head_latent_attention",
        &MultiHeadLatentAttention,
        "multi_head_latent_attention function");

  m.def("noaux_tc", &NoauxTc, "noaux_tc for Deepseekv3 MoE compute");

  m.def("noaux_tc_redundant",
        &NoauxTcRedundant,
        "noaux_tc_redundant for MoE compute");

#ifdef ENABLE_FP8
  m.def("cutlass_fp8_fp8_half_gemm_fused",
        &cutlass_fp8_fp8_half_gemm_func,
        py::arg("x"),
        py::arg("y"),
        py::arg("bias"),
        py::arg("transpose_x"),
        py::arg("transpose_y"),
        py::arg("scale"),
        py::arg("output_dtype"),
        py::arg("activation_type"),
        "cutlass_fp8_fp8_half_gemm_fused function");
  m.def("moe_fused_hadamard_quant_fp8",
        &MoeFusedHadamardQuantFp8Func,
        py::arg("input"),
        py::arg("scale"),
        py::arg("topk_ids"),
        py::arg("top_k"),
        py::arg("intermediate_size"),
        py::arg("tiled"),
        "moe_fused_hadamard_quant_fp8 function");
  m.def("fused_hadamard_quant_fp8",
        &FusedHadamardQuantFp8Func,
        py::arg("input"),
        py::arg("scale"),
        "fused_hadamard_quant_fp8 function");
#endif

  m.def("init_custom_all_reduce",
        &init_custom_all_reduce,
        "init all reduce class function");

  m.def("all_reduce", &all_reduce, "all reduce function");

  m.def("dispose", &dispose, "del function for python");

  m.def("meta_size", &meta_size, "meta_size function for Signal struct");

  m.def("register_buffer", &register_buffer, "register ipc buffer");

  m.def("register_graph_buffers",
        &register_graph_buffers,
        "register_graph_buffers");

  m.def("allocate_shared_buffer_and_handle",
        &allocate_shared_buffer_and_handle,
        "allocate_shared_buffer_and_handle");

  m.def("free_shared_buffer", &free_shared_buffer, "free_shared_buffer");

  m.def("clear_ipc_handles", &clear_ipc_handles, "clear_ipc_handles");

  m.def("open_mem_handle", &open_mem_handle, "open_mem_handle");

  m.def("get_graph_buffer_ipc_meta",
        &get_graph_buffer_ipc_meta,
        "get_graph_buffer_ipc_meta");

  // speculative decoding Kernel
  m.def("speculate_get_padding_offset",
        &SpeculateGetPaddingOffset,
        "speculate_get_padding_offset function");

  m.def("speculate_get_seq_lens_output",
        &SpeculateGetSeqLensOutput,
        "speculate_get_seq_lens_output function");

  m.def("speculate_get_output_padding_offset",
        &SpeculateGetOutputPaddingOffset,
        "speculate_get_output_padding_offset function");

  m.def("speculate_get_token_penalty_multi_scores",
        &SpecTokenPenaltyMultiScores,
        "speculate_get_token_penalty_multi_scores function");

  m.def("speculate_set_stop_value_multi_seqs",
        &SpecGetStopFlagsMultiSeqs,
        "speculate_set_stop_value_multi_seqs function");

  m.def("speculate_verify", &SpeculateVerify, "speculate_verify function");

  m.def("speculate_update", &SpeculateUpdate, "Speculate Update Kernel");

  m.def("speculate_set_value_by_flags_and_idx",
        &SpeculateSetValueByFlagsAndIdx,
        "speculate_set_value_by_flags_and_idx function");

  m.def("speculate_save_output",
        &SpeculateSaveWithOutputMsgStatic,
        "speculate_save_output function");

  m.def("speculate_clear_accept_nums",
        &SpeculateClearAcceptNums,
        "speculate_clear_accept_nums function");

  m.def("speculate_schedule_cache",
        &SpeculateScheduleCache,
        "SpeculateScheduleCache function");

  m.def("ngram_match", &NgramMatch, "ngram_match function");

  m.def("hybird_mtp_ngram", &HybridMtpNgram, "ngram_match_mixed function");

  m.def("draft_model_postprocess",
        &DraftModelPostprocess,
        "draft_model_postprocess function");

  m.def("draft_model_preprocess",
        &DraftModelPreprocess,
        "draft_model_preprocess function");

  m.def("draft_model_update", &DraftModelUpdate, "draft_model_update function");

  m.def("eagle_get_hidden_states",
        &EagleGetHiddenStates,
        "eagle_get_hidden_states function");

  m.def("eagle_get_self_hidden_states",
        &EagleGetSelfHiddenStates,
        "eagle_get_self_hidden_states function");

  m.def("mtp_step_paddle", &MTPStepPaddle, "mtp_step_paddle function");

  m.def("speculate_step_paddle",
        &SpeculateStepPaddle,
        "speculate_step_paddle function");

  m.def("merge_prefill_decode_output",
        &MergePrefillDecodeOutput,
        "merge_prefill_decode_output function");

  m.def("rejection_top_p_sampling",
        &TopPSamplingReject,
        "rejection_top_p_sampling function");

  m.def("top_k_renorm_probs", &TopKRenorm, "top_k_renorm_probs function");

  m.def("min_p_sampling", &MinPSamplingFromProbs, "min_p_sampling function");

  m.def("save_output", &SaveOutMmsgStatic, "save_output function");

  m.def("limit_thinking_content_length_v1",
        &LimitThinkingContentLengthV1,
        "limit_thinking_content_length_v1 function");

  m.def("limit_thinking_content_length_v2",
        &LimitThinkingContentLengthV2,
        "limit_thinking_content_length_v2 function");

  m.def("speculate_limit_thinking_content_length_v1",
        &SpeculateLimitThinkingContentLengthV1,
        "speculate limit thinking content length function");

  m.def("speculate_limit_thinking_content_length_v2",
        &SpeculateLimitThinkingContentLengthV2,
        "speculate limit thinking content length function");

  m.def("speculate_get_logits",
        &SpeculateGetLogits,
        "speculate_get_logits function");

  m.def("speculate_insert_first_token",
        &SpeculateInsertFirstToken,
        "speculate_insert_first_token function");

  m.def("speculate_get_target_logits",
        &SpeculateGetTargetLogits,
        "speculate_get_target_logits function");

  m.def("update_attn_mask_offsets",
        &UpdateAttnMaskOffsets,
        "update attention mask");

  m.def("fused_neox_rope_embedding",
        &FusedNeoxRopeEmbedding,
        "fused_neox_rope_embedding function");

  m.def("gelu_tanh", &GeluTanh, "gelu_tanh function");
}
