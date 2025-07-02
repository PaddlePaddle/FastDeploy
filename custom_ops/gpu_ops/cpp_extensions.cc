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

  const char *what() const noexcept override {
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
  void *ptr = nullptr;
  check_cuda_error(cudaHostAlloc(&ptr, size, flags));
  return reinterpret_cast<uintptr_t>(ptr);
}

// 封装cudaFreeHost的Python函数
void cuda_host_free(uintptr_t ptr) {
  check_cuda_error(cudaFreeHost(reinterpret_cast<void *>(ptr)));
}

std::vector<paddle::Tensor> AppendAttention(
    const paddle::Tensor &qkv, const paddle::Tensor &key_cache,
    const paddle::Tensor &value_cache, const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &padding_offsets, const paddle::Tensor &cum_offsets,
    const paddle::Tensor &block_tables, const paddle::Tensor &encoder_batch_ids,
    const paddle::Tensor &encoder_tile_ids_per_batch,
    const paddle::Tensor &encoder_num_blocks,
    const paddle::Tensor &kv_batch_ids,
    const paddle::Tensor &kv_tile_ids_per_batch,
    const paddle::Tensor &kv_num_blocks,
    const paddle::Tensor &decoder_batch_ids,
    const paddle::Tensor &decoder_tile_ids_per_batch,
    const paddle::Tensor &decoder_num_blocks,
    const paddle::Tensor &set_max_lengths, const paddle::Tensor &max_len_kv,
    const paddle::optional<paddle::Tensor> &rotary_embs,
    const paddle::optional<paddle::Tensor> &attn_mask,
    const paddle::optional<paddle::Tensor> &qkv_bias,
    const paddle::optional<paddle::Tensor> &qkv_out_scales,
    const paddle::optional<paddle::Tensor> &cache_k_quant_scales,
    const paddle::optional<paddle::Tensor> &cache_v_quant_scales,
    const paddle::optional<paddle::Tensor> &cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor> &cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor> &cache_k_zp,
    const paddle::optional<paddle::Tensor> &cache_v_zp,
    const paddle::optional<paddle::Tensor> &out_linear_shifts,
    const paddle::optional<paddle::Tensor> &out_linear_smooths,
    const paddle::optional<paddle::Tensor> &kv_signal_data,
    const std::string &compute_dtype, const std::string &cache_quant_type_str,
    const bool use_neox_rotary_style, const bool rope_3d,
    const int max_input_length, const float quant_max_bound,
    const float quant_min_bound, const float out_linear_in_scale,
    const int encoder_block_shape_q, const int decoder_block_shape_q,
    const int max_partition_size, const int encoder_max_partition_size,
    const int speculate_max_draft_token_num, const bool causal,
    const bool speculate_decoder);

std::vector<paddle::Tensor> GQARopeWriteCacheKernel(
    const paddle::Tensor &qkv, const paddle::Tensor &key_cache,
    const paddle::Tensor &value_cache, const paddle::Tensor &cu_seqlens_q,
    const paddle::Tensor &cu_seqlens_k, const paddle::Tensor &rotary_embs,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &padding_offsets, const paddle::Tensor &cum_offsets,
    const paddle::Tensor &block_tables, const paddle::Tensor &kv_batch_ids,
    const paddle::Tensor &kv_tile_ids, const paddle::Tensor &kv_num_blocks,
    const paddle::Tensor &cache_batch_ids, const paddle::Tensor &cache_tile_ids,
    const paddle::Tensor &cache_num_blocks,
    const paddle::optional<paddle::Tensor> &cache_k_quant_scales,
    const paddle::optional<paddle::Tensor> &cache_v_quant_scales,
    const paddle::optional<paddle::Tensor> &cache_k_dequant_scales,
    const paddle::optional<paddle::Tensor> &cache_v_dequant_scales,
    const paddle::optional<paddle::Tensor> &cache_k_zp,
    const paddle::optional<paddle::Tensor> &cache_v_zp,
    const paddle::optional<paddle::Tensor> &kv_signal_data,
    const int kv_token_num, const int max_seq_len,
    const std::string &cache_quant_type);

std::vector<paddle::Tensor>
PreCacheLenConcat(const paddle::Tensor &seq_lens_decoder,
                  const paddle::Tensor &seq_lens_this_time,
                  const int max_dec_len, const int block_size);

paddle::Tensor FusedExpertMoeFunc(
    const paddle::Tensor &input, const paddle::Tensor &gate_weight,
    const paddle::Tensor &ffn1_weight, const paddle::Tensor &ffn2_weight,
    const paddle::optional<paddle::Tensor> &ffn1_bias,
    const paddle::optional<paddle::Tensor> &ffn1_scale,
    const paddle::optional<paddle::Tensor> &ffn2_bias,
    const paddle::optional<paddle::Tensor> &ffn2_scale,
    const std::string &quant_method, const int moe_topk,
    const bool norm_topk_prob, const bool group_moe);

std::vector<paddle::Tensor> MoeExpertDispatch(
    const paddle::Tensor &input, const paddle::Tensor &gating_output,
    const paddle::optional<paddle::Tensor> &gating_correction_bias,
    const paddle::optional<paddle::Tensor> &w4a8_in_scale, const int moe_topk,
    const bool group_moe, const bool topk_only_mode);

std::vector<paddle::Tensor>
MoETopKSelectKernel(const paddle::Tensor &gating_logits,
                    const paddle::optional<paddle::Tensor> &bias,
                    const int moe_topk, const bool apply_norm_weight,
                    const bool enable_softmax_top_k_fused);

std::vector<paddle::Tensor>
MoERedundantTopKSelectKernel(const paddle::Tensor &gating_logits,
                             const paddle::Tensor &expert_id_to_ep_rank_array,
                             const paddle::Tensor &expert_in_rank_num_list,
                             paddle::Tensor &tokens_per_expert_stats_list,
                             const paddle::optional<paddle::Tensor> &bias,
                             const int moe_topk, const bool apply_norm_weight,
                             const bool enable_softmax_top_k_fused,
                             const int redundant_ep_rank_num_plus_one);

std::vector<paddle::Tensor>
EPMoeExpertDispatch(const paddle::Tensor &input, const paddle::Tensor &topk_ids,
                    const paddle::Tensor &topk_weights,
                    const paddle::optional<paddle::Tensor> &ffn1_in_scale,
                    const std::vector<int> &token_nums_per_expert,
                    const int token_nums_this_rank,
                    const std::string &moe_quant_type);

std::vector<paddle::Tensor> EPMoeExpertDispatchFP8(
    const paddle::Tensor &input, const paddle::Tensor &scale,
    const paddle::Tensor &topk_ids, const paddle::Tensor &topk_weights,
    const paddle::Tensor &token_nums_per_expert,
    const paddle::Tensor &token_nums_per_expert_padded);

std::vector<paddle::Tensor> PerTokenQuant(paddle::Tensor &input,
                                          const int block_size);
std::vector<paddle::Tensor> PerTokenQuantPadding(paddle::Tensor &input,
                                                 const int block_size);
std::vector<paddle::Tensor>
MaskedPerTokenQuant(paddle::Tensor &input, paddle::Tensor &recv_expert_count,
                    const int block_size);

std::vector<paddle::Tensor> EPMoeExpertCombine(
    const paddle::Tensor &ffn_out, const paddle::Tensor &expert_scales_float,
    const paddle::Tensor &permute_indices_per_token,
    const paddle::Tensor &top_k_indices,
    const paddle::optional<paddle::Tensor> &ffn2_bias,
    const bool norm_topk_prob, const float routed_scaling_factor);

std::vector<std::vector<int>> GetExpertTokenNum(const paddle::Tensor &topk_ids,
                                                const int num_experts);

paddle::Tensor MoeExpertFFNFunc(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight, const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn2_in_scale,
    const paddle::optional<paddle::Tensor>& expert_idx_per_token,
    const std::string& quant_method, const bool used_in_ep_low_latency);

paddle::Tensor MoeExpertFFNWint2Func(
    const paddle::Tensor& permute_input,
    const paddle::Tensor& tokens_expert_prefix_sum,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn1_scale,
    const paddle::optional<paddle::Tensor>& ffn2_scale,
    const paddle::optional<paddle::Tensor>& ffn1_local_scale,
    const paddle::optional<paddle::Tensor>& ffn1_code_scale,
    const paddle::optional<paddle::Tensor>& ffn1_code_zp,
    const paddle::optional<paddle::Tensor>& ffn2_local_scale,
    const paddle::optional<paddle::Tensor>& ffn2_code_scale,
    const paddle::optional<paddle::Tensor>& ffn2_code_zp,
    const bool used_in_ep_low_latency);

paddle::Tensor MoeExpertReduceFunc(
    const paddle::Tensor &ffn_out, const paddle::Tensor &top_k_weight,
    const paddle::Tensor &permute_indices_per_token,
    const paddle::Tensor &top_k_indices,
    const paddle::optional<paddle::Tensor> &ffn2_bias,
    const bool norm_topk_prob, const float routed_scaling_factor);

void InitKVSignalPerQuery(const paddle::Tensor &seq_lens_encoder_tensor,
                          const paddle::Tensor &seq_lens_this_time_tensor,
                          const paddle::Tensor &seq_lens_decoder_tensor,
                          const int rank, const int num_layers);

void GetOutputKVSignal(const paddle::Tensor &x, int64_t rank_id,
                       bool wait_flag);

paddle::Tensor DequantInt8Func(const paddle::Tensor &input,
                               const paddle::Tensor &out_scale,
                               std::string dtype);

paddle::Tensor OpenShmAndGetMetaSignalFunc(const int rank, const int device_id,
                                           const bool keep_pd_step_flag);

paddle::Tensor InitSignalLayerwiseFunc(const paddle::Tensor &kv_signal_metadata,
                                       const int layer_id);

std::vector<paddle::Tensor> GetBlockShapeAndSplitKVBlock(
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_this_time, const paddle::Tensor &cum_offsets,
    const int encoder_block_shape_q, const int decoder_block_shape_q,
    const int group_size, const int block_size,
    const int decoder_step_token_num);

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor &input_ids,
                                             const paddle::Tensor &cum_offsets,
                                             const paddle::Tensor &token_num,
                                             const paddle::Tensor &seq_len);

void SetValueByFlagsAndIdx(const paddle::Tensor &pre_ids_all,
                           const paddle::Tensor &input_ids,
                           const paddle::Tensor &seq_lens_this_time,
                           const paddle::Tensor &seq_lens_encoder,
                           const paddle::Tensor &seq_lens_decoder,
                           const paddle::Tensor &step_idx,
                           const paddle::Tensor &stop_flags);

paddle::Tensor RebuildPaddingFunc(
    const paddle::Tensor &tmp_out,     // [token_num, dim_embed]
    const paddle::Tensor &cum_offsets, // [bsz, 1]
    const paddle::Tensor &seq_len_this_time,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::optional<paddle::Tensor> &output_padding_offset,
    int max_input_length);

void GetStopFlagsMulti(const paddle::Tensor &topk_ids,
                       const paddle::Tensor &stop_flags,
                       const paddle::Tensor &seq_lens,
                       const paddle::Tensor &end_ids,
                       const paddle::Tensor &next_tokens,
                       const bool beam_search);

void GetStopFlagsMultiSeqs(
    const paddle::Tensor &topk_ids, const paddle::Tensor &pre_ids,
    const paddle::Tensor &step_idx, const paddle::Tensor &stop_flags,
    const paddle::Tensor &seq_lens, const paddle::Tensor &stop_seqs,
    const paddle::Tensor &stop_seqs_len, const paddle::Tensor &end_ids);

void UpdateInputes(const paddle::Tensor &stop_flags,
                   const paddle::Tensor &not_need_stop, // only on cpu
                   const paddle::Tensor &seq_lens_this_time,
                   const paddle::Tensor &seq_lens_encoder,
                   const paddle::Tensor &seq_lens_decoder,
                   const paddle::Tensor &input_ids,
                   const paddle::Tensor &stop_nums,
                   const paddle::Tensor &next_tokens,
                   const paddle::Tensor &is_block_step);

paddle::Tensor
GroupSwigluWithMasked(const paddle::Tensor &fc1_out_tensor,
                      const paddle::Tensor &token_nums_per_expert);

std::vector<paddle::Tensor> ExtractTextTokenOutput(
    const paddle::Tensor &max_seq_len, const paddle::Tensor &max_seq_len_index,
    const paddle::Tensor &mm_token_num_len,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &cu_seqlens_q, const paddle::Tensor &score_text);

std::vector<paddle::Tensor> MoEDeepGEMMPermute(const paddle::Tensor &x,
                                               const paddle::Tensor &topk_idx,
                                               const int num_experts,
                                               const int max_tokens_per_expert);

std::vector<paddle::Tensor> MoEDeepGEMMDePermute(
    const paddle::Tensor
        &ffn_out, // [num_experts, max_tokens_per_expert, hidden]
    const paddle::Tensor &permute_indices_per_token, // [token_num, topk}]
    const paddle::Tensor &topk_idx, const paddle::Tensor &topk_weights);

void TextImageIndexOut(const paddle::Tensor &token_type_ids,
                       const paddle::Tensor &text_input,
                       const paddle::Tensor &image_input);

void TextImageGatherScatter(paddle::Tensor &input, paddle::Tensor &text_input,
                            paddle::Tensor &image_input,
                            paddle::Tensor &token_type_ids,
                            paddle::Tensor &text_index,
                            paddle::Tensor &image_index, const bool is_scatter);

paddle::Tensor count_tokens_per_expert_func(const paddle::Tensor &topk_ids,
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
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
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
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
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
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& padding_offsets,
    const paddle::Tensor& cum_offsets,
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
    const paddle::Tensor& decoder_num_blocks_cpu,
    const paddle::Tensor& max_enc_len_this_time,
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


std::vector<paddle::Tensor> tritonmoe_preprocess_kernel(const paddle::Tensor& topk_ids, int64_t num_experts, int64_t GEMM_BLOCK_SIZE_M);


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
void CutlassScaledMm(paddle::Tensor &c, paddle::Tensor const &a,
                     paddle::Tensor const &b, paddle::Tensor const &a_scales,
                     paddle::Tensor const &b_scales,
                     paddle::optional<paddle::Tensor> const &bias);

void CutlassScaledMmAzp(paddle::Tensor& c, paddle::Tensor const& a,
                           paddle::Tensor const& b,
                           paddle::Tensor const& a_scales,
                           paddle::Tensor const& b_scales,
                           paddle::Tensor const& azp_adj,
                           paddle::optional<paddle::Tensor> const& azp,
                           paddle::optional<paddle::Tensor> const& bias);

void StaticScaledFp8Quant(paddle::Tensor &out, paddle::Tensor const &input,
                          paddle::Tensor const &scale);

void DynamicScaledFp8Quant(paddle::Tensor &out, paddle::Tensor const &input,
                           paddle::Tensor &scale);

void DynamicPerTokenScaledFp8Quant(paddle::Tensor &out,
                                   paddle::Tensor const &input,
                                   paddle::Tensor &scales, float scale_ub);

std::vector<paddle::Tensor> NoauxTc(
      paddle::Tensor& scores,
      paddle::Tensor& scores_with_bias,
      int n_group,
      int topk_group,
      int topk,
      float routed_scaling_factor);

PYBIND11_MODULE(fastdeploy_ops, m) {

  m.def("get_expert_token_num", &GetExpertTokenNum, py::arg("topk_ids"),
        py::arg("num_experts"), "get expert token num");

  /**
   * moe/fused_moe/moe_redundant_topk_select.cu
   * moe_redundant_topk_select
   */
  m.def("f_moe_redundant_topk_select", &MoERedundantTopKSelectKernel,
        py::arg("gating_logits"), py::arg("expert_id_to_ep_rank_array"),
        py::arg("expert_in_rank_num_list"),
        py::arg("tokens_per_expert_stats_list"), py::arg("bias"),
        py::arg("moe_topk"), py::arg("apply_norm_weight"),
        py::arg("enable_softmax_top_k_fused"),
        py::arg("redundant_ep_rank_num_plus_one"),
        "moe export RedundantTopKSelect function");

  /**
   * open_shm_and_get_meta_signal.cc
   * InitKVSignalPerQuery
   */
  m.def("init_kv_signal_per_query", &InitKVSignalPerQuery,
        py::arg("seq_lens_encoder_tensor"),
        py::arg("seq_lens_this_time_tensor"),
        py::arg("seq_lens_decoder_tensor"), py::arg("rank"),
        py::arg("num_layers"), "init_kv_signal_per_query function");

  /**
   * GetOutputKVSignal
   */
  m.def("get_output_kv_signal", &GetOutputKVSignal, py::arg("x"),
        py::arg("rank_id"), py::arg("wait_flag"),
        "get_output_kv_signal function");

  m.def("moe_deepgemm_permute", &MoEDeepGEMMPermute, "MoEDeepGEMMPermute");
  m.def("moe_deepgemm_depermute", &MoEDeepGEMMDePermute,
        "MoEDeepGEMMDePermute");
  /**
   * alloc_cache_pinned.cc
   * cuda_host_alloc
   * cuda_host_free
   */
  m.def("cuda_host_alloc", &cuda_host_alloc, "Allocate pinned memory",
        py::arg("size"), py::arg("flags") = cudaHostAllocDefault);
  m.def("cuda_host_free", &cuda_host_free, "Free pinned memory",
        py::arg("ptr"));
  py::register_exception<CudaError>(m, "CudaError");
  /**
   * append_attention.cu
   * append_attention
   */
  m.def("append_attention", &AppendAttention, "append attention function");
  /**
   * gqa_rope_write_cache.cu
   * gqa_rope_write_cache
   */
  m.def("gqa_rope_write_cache", &GQARopeWriteCacheKernel,
        "gqa rope write cache function");
  /**
   * pre_cache_len_concat.cu
   * pre_cache_len_concat
   */
  m.def("pre_cache_len_concat", &PreCacheLenConcat,
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
  m.def("moe_expert_dispatch", &MoeExpertDispatch, py::arg("input"),
        py::arg("gating_output"), py::arg("gating_correction_bias"),
        py::arg("w4a8_in_scale"), py::arg("moe_topk"), py::arg("group_moe"),
        py::arg("topk_only_mode"), "moe export dispatch function");

  /**
   * moe/fused_moe/ep_moe_prefill_func.cu
   * ep_moe_dispatch
   */
  m.def("ep_moe_expert_dispatch", &EPMoeExpertDispatch, py::arg("input"),
        py::arg("topk_ids"), py::arg("topk_weights"), py::arg("ffn1_in_scale"),
        py::arg("token_nums_per_expert"), py::arg("token_nums_this_rank"),
        py::arg("moe_quant_type"), "ep moe export dispatch function");

  m.def("ep_moe_expert_dispatch_fp8", &EPMoeExpertDispatchFP8);

  m.def("ep_moe_expert_combine", &EPMoeExpertCombine, py::arg("ffn_out"),
        py::arg("expert_scales_float"), py::arg("permute_indices_per_token"),
        py::arg("top_k_indices"), py::arg("ffn2_bias"),
        py::arg("norm_topk_prob"), py::arg("routed_scaling_factor"),
        "ep moe export combine function");

  m.def("per_token_quant", &PerTokenQuant, py::arg("input"),
        py::arg("block_size"), "per token per block quant");

  m.def("per_token_quant_padding", &PerTokenQuantPadding, py::arg("input"),
        py::arg("block_size"),
        "per token per block quant and padding tranpose scale");

  m.def("masked_per_token_quant", &MaskedPerTokenQuant, py::arg("input"),
        py::arg("recv_expert_count"), py::arg("block_size"),
        "per token per block quant");

  /**
   * moe/fused_moe/moe_topk_select.cu
   * moe_topk_select
   */
  m.def("moe_topk_select", &MoETopKSelectKernel, py::arg("gating_logits"),
        py::arg("bias"), py::arg("moe_topk"), py::arg("apply_norm_weight"),
        py::arg("enable_softmax_top_k_fused"),
        "moe export TopKSelect function");

  /**
   * moe/fused_moe/moe_ffn.cu
   * moe_expert_ffn
   */
  m.def("moe_expert_ffn", &MoeExpertFFNFunc, "moe export ffn function");

  /**
   * moe/fused_moe/moe_ffn_wint2.cu
   * moe_expert_ffn_wint2
   */
  m.def("moe_expert_ffn_wint2", &MoeExpertFFNWint2Func, "moe export ffn wint2 function");

  /**
   * moe/fused_moe/moe_expert_reduce.cu
   * moe_expert_reduce
   */
  m.def("moe_expert_reduce", &MoeExpertReduceFunc, py::arg("ffn_out"),
        py::arg("top_k_weight"), py::arg("permute_indices_per_token"),
        py::arg("top_k_indices"), py::arg("ffn2_bias"),
        py::arg("norm_topk_prob"), py::arg("routed_scaling_factor"),
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
  m.def("init_signal_layerwise", &InitSignalLayerwiseFunc,
        "init_signal_layerwise function");

  /**
   * open_shm_and_get_meta_signal.cc
   * open_shm_and_get_meta_signal
   */
  m.def("open_shm_and_get_meta_signal", &OpenShmAndGetMetaSignalFunc,
        "open_shm_and_get_meta_signal function");

  /**
   * append_attn/get_block_shape_and_split_kv_block.cu
   * get_block_shape_and_split_kv_block
   */
  // m.def("f_get_block_shape_and_split_kv_block",
  // &GetBlockShapeAndSplitKVBlock, "get_block_shape_and_split_kv_block
  // function");

  /**
   * get_padding_offset.cu
   * get_padding_offset
   */
  m.def("get_padding_offset", &GetPaddingOffset, "get_padding_offset function");

  /**
   * get_padding_offset.cu
   * get_padding_offset
   */
  m.def("set_value_by_flags_and_idx", &SetValueByFlagsAndIdx,
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
  m.def("set_stop_value_multi_ends", &GetStopFlagsMulti,
        "update_inputs function");

  /**
   * stop_generation_multi_stop_seqs.cu
   * set_stop_value_multi_seqs
   */
  m.def("set_stop_value_multi_seqs", &GetStopFlagsMultiSeqs,
        "update_inputs function");

  /**
   * update_inputs.cu
   * update_inputs
   */
  m.def("update_inputs", &UpdateInputes, "update_inputs function");

  /**
   * extract_text_token_output.cu
   * extract_text_token_output
   */
  m.def("extract_text_token_output", &ExtractTextTokenOutput,
        "extract_text_token_output function");

  m.def("group_swiglu_with_masked", &GroupSwigluWithMasked,
        "group_swiglu_with_masked function");

  m.def("text_image_index_out", &TextImageIndexOut,
        "text_image_index_out function");

  m.def("text_image_gather_scatter", &TextImageGatherScatter,
        "text_image_gather_scatter function");

  m.def("count_tokens_per_expert_func", &count_tokens_per_expert_func);
  m.def("tritonmoe_preprocess_func", &tritonmoe_preprocess_kernel);

  m.def("MoeWna16MarlinGemmApi", &MoeWna16MarlinGemmApi,
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
  m.def("get_position_ids_and_mask_encoder_batch", &GetPositionIdsAndMaskEncoderBatch,
        "get_position_ids_and_mask_encoder_batch function");


  /**
   * cutlass_scaled_mm.cu
   * cutlass_scaled_mm
   * cutlass_scaled_mm_azp
   */
  m.def("cutlass_scaled_mm", &CutlassScaledMm, "cutlass_scaled_mm function");
  m.def("cutlass_scaled_mm_azp", &CutlassScaledMmAzp, "cutlass_scaled_mm_azp function");

  /**
   * quantization/common.cu
   * static_scaled_fp8_quant
   * dynamic_scaled_fp8_quant
   * dynamic_per_token_scaled_fp8_quant
   */
  m.def("static_scaled_fp8_quant", &StaticScaledFp8Quant, "static_scaled_fp8_quant function",
      py::arg("out"), py::arg("input"), py::arg("scale"));

  m.def("dynamic_scaled_fp8_quant", &DynamicScaledFp8Quant,
        "dynamic_scaled_fp8_quant function",
        py::arg("out"), py::arg("input"), py::arg("scale"));

  m.def("dynamic_per_token_scaled_fp8_quant", &DynamicPerTokenScaledFp8Quant,
        "dynamic_per_token_scaled_fp8_quant function",
         py::arg("out"), py::arg("input"), py::arg("scales"), py::arg("scale_ub"));
  m.def("decode_mla_write_cache", &DecodeMLAWriteCacheKernel, "decode_mla_write_cache function");

  m.def("prefill_mla_write_cache", &PrefillMLAWriteCacheKernel, "prefill_mla_write_cache function");

  m.def("fused_rotary_position_encoding", &FusedRotaryPositionEncoding, "fused_rotary_position_encoding function");

  m.def("multi_head_latent_attention", &MultiHeadLatentAttention, "multi_head_latent_attention function");

  m.def("noaux_tc",&NoauxTc, "noaux_tc for Deepseekv3 MoE compute");
}
