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

#include "ops/pybind/pybind.h"
#include <paddle/phi/backends/xpu/xpu_context.h>
#include "cuda_runtime_api.h"  // NOLINT
#include "paddle/extension.h"

namespace py = pybind11;

uintptr_t custom_xpu_host_alloc(size_t size, unsigned int flags);

void custom_xpu_host_free(uintptr_t ptr);

uintptr_t xpu_get_peer_mem_addr(uintptr_t ptr);

void xpu_cuda_host_register(uintptr_t ptr,
                            size_t size,
                            unsigned int flags = cudaHostRegisterDefault);

void prof_start();

void prof_stop();

std::vector<paddle::Tensor> AdjustBatch(
    const paddle::Tensor& x,            // [token_num, dim_embed]
    const paddle::Tensor& cum_offsets,  // [bsz, 1]
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_batch_idx,
    const paddle::Tensor& decoder_batch_idx,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_idx_cpu,
    const paddle::Tensor& decoder_batch_idx_cpu,
    const paddle::Tensor& len_info_cpu,
    const paddle::optional<paddle::Tensor>& output_padding_offset,
    int max_input_length);

void InitKVSignalPerQuery(const paddle::Tensor& seq_lens_encoder_tensor,
                          const paddle::Tensor& seq_lens_this_time_tensor,
                          const paddle::Tensor& seq_lens_decoder_tensor,
                          const int rank,
                          const int num_layers);

void GetOutputKVSignal(const paddle::Tensor& x,
                       int64_t rank_id,
                       bool wait_flag);

std::vector<paddle::Tensor> BlockAttn(
    const paddle::Tensor& qkv,
    const paddle::Tensor& key_cache,
    const paddle::Tensor& value_cache,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& rotary_embs,
    const paddle::Tensor& block_tables,
    const paddle::Tensor& prefix_block_tables,
    const paddle::Tensor& len_info_cpu,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_kv_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_context_len_cpu,
    const paddle::Tensor& decoder_context_len_cache_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& prefix_len_cpu,
    const paddle::optional<paddle::Tensor>& k_scales,
    const paddle::optional<paddle::Tensor>& v_scales,
    const paddle::optional<paddle::Tensor>& k_scales_inv,
    const paddle::optional<paddle::Tensor>& v_scales_inv,
    const paddle::optional<paddle::Tensor>& k_zeros,
    const paddle::optional<paddle::Tensor>& v_zeros,
    const paddle::optional<paddle::Tensor>& shift,
    const paddle::optional<paddle::Tensor>& smooth,
    const paddle::optional<paddle::Tensor>& kv_signal_data_cpu,
    const paddle::optional<paddle::Tensor>& cachekv_signal_thread_cpu,
    const std::string& pos_emb_type = "NORMAL",
    bool rope_3d = false);

std::vector<paddle::Tensor> MoeLayer(
    const paddle::Tensor& x,
    const paddle::Tensor& gate_weight,
    const paddle::optional<paddle::Tensor>& gate_correction_bias,
    const paddle::Tensor& up_gate_proj_weight,
    const paddle::Tensor& down_proj_weight,
    const paddle::optional<paddle::Tensor>& up_gate_proj_bias,
    const paddle::optional<paddle::Tensor>& down_proj_bias,
    const paddle::optional<paddle::Tensor>& up_gate_proj_weight_scale,
    const paddle::optional<paddle::Tensor>& down_proj_weight_scale,
    const paddle::optional<paddle::Tensor>& down_proj_in_scale,
    const std::string& quant_method,
    const int moe_top_k,
    const bool moe_group);

std::vector<paddle::Tensor> MoERedundantTopKSelect(
    const paddle::Tensor& gating_logits,
    const paddle::Tensor& expert_id_to_ep_rank_array,
    const paddle::Tensor& expert_in_rank_num_list,
    paddle::Tensor& tokens_per_expert_stats_list,  // NOLINT
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight,
    const bool enable_softmax_top_k_fused,
    const int redundant_ep_rank_num_plus_one);

void set_ncluster(int num) {
  phi::XPUPlace place(phi::backends::xpu::GetXPUCurrentDeviceId());
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(place);
  auto xpu_ctx = static_cast<const phi::XPUContext*>(dev_ctx);
  xpu_ctx->x_context()->set_ncluster(num);
}

std::vector<paddle::Tensor> RmsNorm(
    const paddle::Tensor& x,
    const paddle::optional<paddle::Tensor>& bias,
    const paddle::optional<paddle::Tensor>& residual,
    const paddle::Tensor& norm_weight,
    const paddle::optional<paddle::Tensor>& norm_bias,
    const float epsilon,
    const int begin_norm_axis,
    const float quant_scale,
    const int quant_round_type,
    const float quant_max_bound,
    const float quant_min_bound);

std::vector<paddle::Tensor> WeightOnlyLinear(
    const paddle::Tensor& x,
    const paddle::Tensor& weight,
    const paddle::Tensor& weight_scale,
    const paddle::optional<paddle::Tensor>& bias,
    const std::string& weight_dtype,
    const int arch,
    const int group_size);

std::vector<paddle::Tensor> MoeEPCombine(const paddle::Tensor& ffn_out,
                                         const paddle::Tensor& moe_index,
                                         const paddle::Tensor& weights,
                                         const int recv_token_num,
                                         const int expand_token_num,
                                         const int hidden_dim,
                                         const int topk);

std::vector<paddle::Tensor> EPMoeExpertDispatch(
    const paddle::Tensor& input,
    const paddle::Tensor& topk_ids,
    const paddle::Tensor& topk_weights,
    const paddle::optional<paddle::Tensor>& input_scales,
    const std::vector<int>& token_nums_per_expert,
    const int token_nums_this_rank,
    const std::string quant_method);

std::vector<paddle::Tensor> MoeExpertFFN(
    const paddle::Tensor& ffn_in,
    const paddle::Tensor& token_num_info,
    const paddle::Tensor& ffn1_weight,
    const paddle::Tensor& ffn2_weight,
    const paddle::optional<paddle::Tensor>& ffn1_bias,
    const paddle::optional<paddle::Tensor>& ffn2_bias,
    const paddle::optional<paddle::Tensor>& ffn1_act_scale,
    const paddle::optional<paddle::Tensor>& ffn2_act_scale,
    const paddle::optional<paddle::Tensor>& ffn1_weight_scale,
    const paddle::optional<paddle::Tensor>& ffn2_weight_scale,
    const paddle::optional<paddle::Tensor>& ffn2_shift,
    const paddle::optional<paddle::Tensor>& ffn2_smooth,
    const std::string& quant_method,
    const int hadamard_blocksize,
    const int valid_token_num);

std::vector<paddle::Tensor> MoeTopkSelect(
    const paddle::Tensor& gating_logits,
    const paddle::optional<paddle::Tensor>& bias,
    const int moe_topk,
    const bool apply_norm_weight);

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

void SpeculateUpdateV3(const paddle::Tensor& seq_lens_encoder,
                       const paddle::Tensor& seq_lens_decoder,
                       const paddle::Tensor& not_need_stop,
                       const paddle::Tensor& draft_tokens,
                       const paddle::Tensor& actual_draft_token_nums,
                       const paddle::Tensor& accept_tokens,
                       const paddle::Tensor& accept_num,
                       const paddle::Tensor& stop_flags,
                       const paddle::Tensor& seq_lens_this_time,
                       const paddle::Tensor& is_block_step,
                       const paddle::Tensor& stop_nums);

void SpeculateTokenPenaltyMultiScores(
    const paddle::Tensor& pre_ids,
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

void SpeculateUpdateV3(const paddle::Tensor& seq_lens_encoder,
                       const paddle::Tensor& seq_lens_decoder,
                       const paddle::Tensor& not_need_stop,
                       const paddle::Tensor& draft_tokens,
                       const paddle::Tensor& actual_draft_token_nums,
                       const paddle::Tensor& accept_tokens,
                       const paddle::Tensor& accept_num,
                       const paddle::Tensor& stop_flags,
                       const paddle::Tensor& seq_lens_this_time,
                       const paddle::Tensor& is_block_step,
                       const paddle::Tensor& stop_nums);

std::vector<paddle::Tensor> TopPCandidates(
    const paddle::Tensor& probs,
    const paddle::Tensor& top_p,
    const paddle::Tensor& output_padding_offset,
    int candidates_len,
    int max_seq_len);

void SpeculateVerify(const paddle::Tensor& accept_tokens,
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

void SpeculateClearAcceptNums(const paddle::Tensor& accept_num,
                              const paddle::Tensor& seq_lens_decoder);

void SpeculateSetValueByFlagsAndIdx(const paddle::Tensor& pre_ids_all,
                                    const paddle::Tensor& accept_tokens,
                                    const paddle::Tensor& accept_num,
                                    const paddle::Tensor& stop_flags,
                                    const paddle::Tensor& seq_lens_this_time,
                                    const paddle::Tensor& seq_lens_encoder,
                                    const paddle::Tensor& seq_lens_decoder,
                                    const paddle::Tensor& step_idx);

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
                          const int num_model_step,
                          const bool truncate_first_token,
                          const bool splitwise_prefill,
                          const bool kvcache_scheduler_v1);

void DraftModelPostprocess(const paddle::Tensor& base_model_draft_tokens,
                           const paddle::Tensor& base_model_seq_lens_this_time,
                           const paddle::Tensor& base_model_seq_lens_encoder,
                           const paddle::Tensor& base_model_stop_flags);

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

std::vector<paddle::Tensor> GatherNextToken(
    const paddle::Tensor& x,            // [token_num, dim_embed]
    const paddle::Tensor& cum_offsets,  // [bsz, 1]
    const paddle::Tensor& encoder_seq_lod,
    const paddle::Tensor& decoder_seq_lod,
    const paddle::Tensor& encoder_batch_map,
    const paddle::Tensor& decoder_batch_map,
    const paddle::Tensor& encoder_seq_lod_cpu,
    const paddle::Tensor& decoder_seq_lod_cpu,
    const paddle::Tensor& encoder_batch_map_cpu,
    const paddle::Tensor& decoder_batch_map_cpu,
    const paddle::Tensor& len_info_cpu,
    const paddle::optional<paddle::Tensor>& output_padding_offset,
    int max_bsz);

std::vector<paddle::Tensor> GetImgBoundaries(
    const paddle::Tensor& task_input_ids,
    const paddle::Tensor& grid_thw,
    const int64_t image_patch_id);

std::vector<paddle::Tensor> GetInferParam(
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& block_tables,
    int block_size);

void GetOutputStatic(const paddle::Tensor& x, int64_t rank_id, bool wait_flag);

void GetOutputDynamic(const paddle::Tensor& x,
                      int64_t rank_id,
                      bool wait_flag,
                      int msg_queue_id);

std::vector<paddle::Tensor> GetPaddingOffset(const paddle::Tensor& input_ids,
                                             const paddle::Tensor& cum_offsets,
                                             const paddle::Tensor& token_num,
                                             const paddle::Tensor& seq_len);

void GetStopFlagsMulti(const paddle::Tensor& topk_ids,
                       const paddle::Tensor& stop_flags,
                       const paddle::Tensor& seq_lens,
                       const paddle::Tensor& end_ids,
                       const paddle::Tensor& next_tokens,
                       const bool beam_search);

void RecoverDecodeTask(const paddle::Tensor& stop_flags,
                       const paddle::Tensor& seq_lens_this_time,
                       const paddle::Tensor& seq_lens_encoder,
                       const paddle::Tensor& seq_lens_decoder,
                       const paddle::Tensor& step_seq_lens_decoder,
                       const paddle::Tensor& block_tables,
                       const paddle::Tensor& is_block_step,
                       const int block_size);

std::vector<paddle::Tensor> ShareExternalData(const paddle::Tensor& input,
                                              const std::string shm_name,
                                              const std::vector<int>& shape,
                                              bool use_ipc);

std::vector<paddle::Tensor> SpeculateGetOutputPaddingOffset(
    const paddle::Tensor& output_cum_offsets_tmp,
    const paddle::Tensor& out_token_num,
    const paddle::Tensor& seq_lens_output,
    const int max_seq_len);

std::vector<paddle::Tensor> SpeculateGetPaddingOffset(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& draft_tokens,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& token_num,
    const paddle::Tensor& seq_len,
    const paddle::Tensor& seq_lens_encoder);

void StepPaddle(const paddle::Tensor& stop_flags,
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
                const int block_size,
                const int encoder_decoder_block_num);

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

void SaveOutMmsgStatic(const paddle::Tensor& x,
                       const paddle::Tensor& not_need_stop,
                       int64_t rank_id,
                       bool save_each_rank);

void SaveOutMmsgDynamic(const paddle::Tensor& x,
                        const paddle::Tensor& not_need_stop,
                        int64_t rank_id,
                        int msg_queue_id,
                        bool save_each_rank);

void SpeculateStepSchedule(
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

std::vector<paddle::Tensor> SpeculateGetSeqLensOutput(
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder);

void SetDataIpc(const paddle::Tensor& tmp_input, const std::string& shm_name);

std::vector<paddle::Tensor> TextImageGatherScatter(
    paddle::Tensor& input,
    paddle::Tensor& text_input,
    paddle::Tensor& image_input,
    paddle::Tensor& token_type_ids,
    paddle::Tensor& text_index,
    paddle::Tensor& image_index,
    const bool is_scatter);

void TextImageIndexOut(const paddle::Tensor& token_type_ids,
                       const paddle::Tensor& text_index,
                       const paddle::Tensor& image_index);

void TokenPenaltyMultiScores(const paddle::Tensor& pre_ids,
                             const paddle::Tensor& logits,
                             const paddle::Tensor& penalty_scores,
                             const paddle::Tensor& frequency_scores,
                             const paddle::Tensor& presence_scores,
                             const paddle::Tensor& temperatures,
                             const paddle::Tensor& bad_tokens,
                             const paddle::Tensor& cur_len,
                             const paddle::Tensor& min_len,
                             const paddle::Tensor& eos_token_id);

void UpdateInputs(const paddle::Tensor& stop_flags,
                  const paddle::Tensor& not_need_stop,  // cpu
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

std::vector<paddle::Tensor> WeightQuantize(const paddle::Tensor& x,
                                           const std::string& algo,
                                           const int32_t arch,
                                           const int32_t group_size);

PYBIND11_MODULE(fastdeploy_ops, m) {
  m.def("adjust_batch",
        &AdjustBatch,
        py::arg("x"),
        py::arg("cum_offsets"),
        py::arg("encoder_seq_lod"),
        py::arg("decoder_seq_lod"),
        py::arg("encoder_batch_idx"),
        py::arg("decoder_batch_idx"),
        py::arg("encoder_seq_lod_cpu"),
        py::arg("decoder_seq_lod_cpu"),
        py::arg("encoder_batch_idx_cpu"),
        py::arg("decoder_batch_idx_cpu"),
        py::arg("len_info_cpu"),
        py::arg("output_padding_offset"),
        py::arg("max_input_length"),
        "adjust batch in XPU");

  m.def("block_attn",
        &BlockAttn,
        py::arg("qkv"),
        py::arg("key_cache"),
        py::arg("value_cache"),
        py::arg("cum_offsets"),
        py::arg("rotary_embs"),
        py::arg("block_tables"),
        py::arg("prefix_block_tables"),
        py::arg("len_info_cpu"),
        py::arg("encoder_seq_lod_cpu"),
        py::arg("decoder_seq_lod_cpu"),
        py::arg("encoder_kv_lod_cpu"),
        py::arg("encoder_batch_map_cpu"),
        py::arg("decoder_context_len_cpu"),
        py::arg("decoder_context_len_cache_cpu"),
        py::arg("decoder_batch_map_cpu"),
        py::arg("prefix_len_cpu"),
        py::arg("k_scales"),
        py::arg("v_scales"),
        py::arg("k_scales_inv"),
        py::arg("v_scales_inv"),
        py::arg("k_zeros"),
        py::arg("v_zeros"),
        py::arg("shift"),
        py::arg("smooth"),
        py::arg("kv_signal_data_cpu"),
        py::arg("cachekv_signal_thread_cpu"),
        py::arg("pos_emb_type") = "NORMAL",
        py::arg("rope_3d") = false,
        "block attention in XPU");

  m.def("create_kv_signal_sender",
        &create_cachekv_signal_thread,
        "init write cache kv signal thread");

  m.def("cuda_host_alloc",
        &custom_xpu_host_alloc,
        "Allocate pinned memory",
        py::arg("size"),
        py::arg("flags") = 0x00);

  m.def("cuda_host_free",
        &custom_xpu_host_free,
        "Free pinned memory",
        py::arg("ptr"));

  m.def("cuda_host_register",
        &xpu_cuda_host_register,
        "Register pinned memory",
        py::arg("ptr"),
        py::arg("size"),
        py::arg("flags") = cudaHostRegisterDefault);

  m.def("destroy_kv_signal_sender",
        &destroy_cachekv_signal_thread,
        "write cache kv signal thread exit");

  m.def("draft_model_preprocess",
        &DraftModelPreprocess,
        py::arg("draft_tokens"),
        py::arg("input_ids"),
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("step_idx"),
        py::arg("not_need_stop"),
        py::arg("is_block_step"),
        py::arg("batch_drop"),
        py::arg("pre_ids"),
        py::arg("accept_tokens"),
        py::arg("accept_num"),
        py::arg("base_model_seq_lens_this_time"),
        py::arg("base_model_seq_lens_encoder"),
        py::arg("base_model_seq_lens_decoder"),
        py::arg("base_model_step_idx"),
        py::arg("base_model_stop_flags"),
        py::arg("base_model_is_block_step"),
        py::arg("base_model_draft_tokens"),
        py::arg("num_model_step"),
        py::arg("truncate_first_token"),
        py::arg("splitwise_prefill"),
        py::arg("kvcache_scheduler_v1"),
        "Preprocess data for draft model in speculative decoding");

  m.def("draft_model_postprocess",
        &DraftModelPostprocess,
        py::arg("base_model_draft_tokens"),
        py::arg("base_model_seq_lens_this_time"),
        py::arg("base_model_seq_lens_encoder"),
        py::arg("base_model_stop_flags"),
        "Postprocess data for draft model in speculative decoding");

  m.def("draft_model_update",
        &DraftModelUpdate,
        "Update draft model states during speculative decoding",
        py::arg("inter_next_tokens"),        // 中间next tokens张量
        py::arg("draft_tokens"),             // 草稿token张量
        py::arg("pre_ids"),                  // 前置ID张量
        py::arg("seq_lens_this_time"),       // 当前步骤序列长度张量
        py::arg("seq_lens_encoder"),         // 编码器序列长度张量
        py::arg("seq_lens_decoder"),         // 解码器序列长度张量
        py::arg("step_idx"),                 // 步骤索引张量
        py::arg("output_cum_offsets"),       // 输出累积偏移量张量
        py::arg("stop_flags"),               // 停止标志张量
        py::arg("not_need_stop"),            // 无需停止标志张量
        py::arg("max_dec_len"),              // 最大解码长度张量
        py::arg("end_ids"),                  // 结束ID张量
        py::arg("base_model_draft_tokens"),  // 基础模型草稿token张量
        py::arg("max_seq_len"),              // 最大序列长度（int）
        py::arg("substep")                   // 子步骤编号（int）
  );

  m.def("eagle_get_hidden_states",
        &EagleGetHiddenStates,
        py::arg("input"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("stop_flags"),
        py::arg("accept_nums"),
        py::arg("base_model_seq_lens_this_time"),
        py::arg("base_model_seq_lens_encoder"),
        py::arg("actual_draft_token_num"),
        "Get draft model hidden states");

  m.def("eagle_get_self_hidden_states",
        &EagleGetSelfHiddenStates,
        py::arg("input"),
        py::arg("last_seq_lens_this_time"),
        py::arg("seq_lens_this_time"),
        py::arg("step_idx"),
        "Rebuild draft model hidden states");

  m.def("ep_moe_expert_combine",
        &MoeEPCombine,
        "MoE (Mixture of Experts) EP combine operation",
        py::arg("ffn_out"),           // FFN输出张量 [token_num, hidden_dim]
        py::arg("moe_index"),         // MoE专家索引张量 [token_num, topk]
        py::arg("weights"),           // 专家权重张量 [token_num, topk]
        py::arg("recv_token_num"),    // 接收的token数量（int）
        py::arg("expand_token_num"),  // 扩展的token数量（int）
        py::arg("hidden_dim"),        // 隐藏层维度（int）
        py::arg("topk")               // 选择的专家数量（int）
  );

  m.def("ep_moe_expert_dispatch",
        &EPMoeExpertDispatch,
        "EP MoE expert dispatch operation",
        py::arg("input"),
        py::arg("topk_ids"),
        py::arg("topk_weights"),
        py::arg("input_scales") = py::none(),
        py::arg("token_nums_per_expert"),
        py::arg("token_nums_this_rank"),
        py::arg("quant_method"));

  m.def("fused_rms_norm_xpu",
        &RmsNorm,
        "Fused RMS normalization for XPU",
        py::arg("x"),                 // 输入张量
        py::arg("bias"),              // 偏置（可选）
        py::arg("residual"),          // 残差连接（可选）
        py::arg("norm_weight"),       // 归一化权重
        py::arg("norm_bias"),         // 归一化偏置（可选）
        py::arg("epsilon"),           // 数值稳定项
        py::arg("begin_norm_axis"),   // 归一化起始维度
        py::arg("quant_scale"),       // 量化缩放因子
        py::arg("quant_round_type"),  // 量化舍入类型
        py::arg("quant_max_bound"),   // 量化最大值边界
        py::arg("quant_min_bound")    // 量化最小值边界
  );

  m.def("gather_next_token",
        &GatherNextToken,
        py::arg("x"),
        py::arg("cum_offsets"),
        py::arg("encoder_seq_lod"),
        py::arg("decoder_seq_lod"),
        py::arg("encoder_batch_map"),
        py::arg("decoder_batch_map"),
        py::arg("encoder_seq_lod_cpu"),
        py::arg("decoder_seq_lod_cpu"),
        py::arg("encoder_batch_map_cpu"),
        py::arg("decoder_batch_map_cpu"),
        py::arg("len_info_cpu"),
        py::arg("output_padding_offset"),
        py::arg("max_bsz"),
        "Gather next token for XPU");

  m.def("get_img_boundaries",
        &GetImgBoundaries,
        py::arg("task_input_ids"),
        py::arg("grid_thw"),
        py::arg("image_patch_id"),
        "Get image boundaries in VL model");

  m.def("get_infer_param",
        &GetInferParam,
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("seq_lens_this_time"),
        py::arg("block_tables"),
        py::arg("block_size"),
        "Get infer parameters for block attention in XPU");

  m.def("get_peer_mem_addr",
        &xpu_get_peer_mem_addr,
        "Get Host memory address of device pointer",
        py::arg("ptr"));

  m.def("get_token_penalty_multi_scores",
        &TokenPenaltyMultiScores,
        py::arg("pre_ids"),
        py::arg("logits"),
        py::arg("penalty_scores"),
        py::arg("frequency_scores"),
        py::arg("presence_scores"),
        py::arg("temperatures"),
        py::arg("bad_tokens"),
        py::arg("cur_len"),
        py::arg("min_len"),
        py::arg("eos_token_id"),
        "get token_penalty_multi_scores function");

  m.def("get_output",
        &GetOutputStatic,
        py::arg("x"),
        py::arg("rank_id"),
        py::arg("wait_flag"),
        "get_output function");

  m.def("get_output_ep",
        &GetOutputStatic,
        py::arg("x"),
        py::arg("rank_id"),
        py::arg("wait_flag"),
        "get_output_ep function");

  m.def("get_output_dynamic",
        &GetOutputDynamic,
        py::arg("x"),
        py::arg("rank_id"),
        py::arg("wait_flag"),
        py::arg("msg_queue_id"),
        "get_output_dynamic function");

  m.def("get_output_ep_dynamic",
        &GetOutputDynamic,
        py::arg("x"),
        py::arg("rank_id"),
        py::arg("wait_flag"),
        py::arg("msg_queue_id"),
        "get_output_ep_dynamic function");

  m.def("get_output_kv_signal",
        &GetOutputKVSignal,
        py::arg("x"),
        py::arg("rank_id"),
        py::arg("wait_flag"),
        "get_output_kv_signal function");

  m.def("get_padding_offset",
        &GetPaddingOffset,
        py::arg("input_ids"),
        py::arg("cum_offsets"),
        py::arg("token_num"),
        py::arg("seq_len"),
        "get padding offset function");

  m.def("init_kv_signal_per_query",
        &InitKVSignalPerQuery,
        py::arg("seq_lens_encoder_tensor"),
        py::arg("seq_lens_this_time_tensor"),
        py::arg("seq_lens_decoder_tensor"),
        py::arg("rank"),
        py::arg("num_layers"),
        "init_kv_signal_per_query function");

  m.def("moe_redundant_topk_select",
        &MoERedundantTopKSelect,
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

  m.def("mtp_step_paddle",
        &MTPStepPaddle,
        py::arg("base_model_stop_flags"),
        py::arg("stop_flags"),
        py::arg("batch_drop"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("block_tables"),  // [bsz, block_num_per_seq]
        py::arg("encoder_block_lens"),
        py::arg("used_list_len"),
        py::arg("free_list"),
        py::arg("free_list_len"),
        py::arg("block_size"),
        py::arg("max_draft_tokens"),
        "MTP step paddle");

  m.def("moe_expert_ffn",
        &MoeExpertFFN,
        "MoE expert feed-forward network with quantization support",
        py::arg("ffn_in"),  // [valid_token_num, hidden_dim]
        py::arg("token_num_info"),
        py::arg("ffn1_weight"),
        py::arg("ffn2_weight"),
        py::arg("ffn1_bias") = py::none(),
        py::arg("ffn2_bias") = py::none(),
        py::arg("ffn1_act_scale") = py::none(),
        py::arg("ffn2_act_scale") = py::none(),
        py::arg("ffn1_weight_scale") = py::none(),
        py::arg("ffn2_weight_scale") = py::none(),
        py::arg("ffn2_shift") = py::none(),
        py::arg("ffn2_smooth") = py::none(),
        py::arg("quant_method"),
        py::arg("hadamard_blocksize"),
        py::arg("valid_token_num"));

  m.def("moe_topk_select",
        &MoeTopkSelect,
        "MoE Top-k selection: selects top-k experts via gating logits",
        py::arg("gating_logits"),
        py::arg("bias") = py::none(),
        py::arg("moe_topk"),
        py::arg("apply_norm_weight"));

  m.def("prof_start", &prof_start, "prof_start");

  m.def("prof_stop", &prof_stop, "prof_stop");

  m.def("recover_decode_task",
        &RecoverDecodeTask,
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("step_seq_lens_decoder"),
        py::arg("block_tables"),
        py::arg("is_block_step"),
        py::arg("block_size"),
        "Recover decode task function");

  m.def("save_output",
        &SaveOutMmsgStatic,
        py::arg("x"),
        py::arg("not_need_stop"),
        py::arg("rank_id"),
        py::arg("save_each_rank"),
        "Save output function");

  m.def("save_output_dynamic",
        &SaveOutMmsgDynamic,
        py::arg("x"),
        py::arg("not_need_stop"),
        py::arg("rank_id"),
        py::arg("msg_queue_id"),
        py::arg("save_each_rank"),
        "Save output dynamic function");

  m.def("share_external_data",
        &ShareExternalData,
        py::arg("input"),
        py::arg("shm_name"),
        py::arg("shape"),
        py::arg("use_ipc"),
        "Share external data function");

  m.def("speculate_get_token_penalty_multi_scores",
        &SpeculateTokenPenaltyMultiScores,
        py::arg("pre_ids"),
        py::arg("logits"),
        py::arg("penalty_scores"),
        py::arg("frequency_scores"),
        py::arg("presence_scores"),
        py::arg("temperatures"),
        py::arg("bad_tokens"),
        py::arg("cur_len"),
        py::arg("min_len"),
        py::arg("eos_token_id"),
        py::arg("seq_lens_this_time"),
        py::arg("output_padding_offset"),
        py::arg("output_cum_offsets"),
        py::arg("max_seq_len"),
        "Applies token penalty with multiple scores");

  m.def("speculate_update_v3",
        &SpeculateUpdateV3,
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("not_need_stop"),
        py::arg("draft_tokens"),
        py::arg("actual_draft_token_nums"),
        py::arg("accept_tokens"),
        py::arg("accept_num"),
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("is_block_step"),
        py::arg("stop_nums"),
        "Update speculative decoding states (V3)");

  m.def("speculate_verify",
        &SpeculateVerify,
        py::arg("accept_tokens"),
        py::arg("accept_num"),
        py::arg("step_idx"),
        py::arg("stop_flags"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("draft_tokens"),
        py::arg("seq_lens_this_time"),
        py::arg("verify_tokens"),
        py::arg("verify_scores"),
        py::arg("max_dec_len"),
        py::arg("end_tokens"),
        py::arg("is_block_step"),
        py::arg("output_cum_offsets"),
        py::arg("actual_candidate_len"),
        py::arg("actual_draft_token_nums"),
        py::arg("topp"),
        py::arg("max_seq_len"),
        py::arg("verify_window"),
        py::arg("enable_topp"),
        py::arg("benchmark_mode"),
        py::arg("accept_all_drafts"),
        "Perform speculative verification for decoding");

  m.def("speculate_clear_accept_nums",
        &SpeculateClearAcceptNums,
        py::arg("accept_num"),
        py::arg("seq_lens_decoder"),
        "Clear accept numbers based on decoder sequence lengths");

  m.def("speculate_set_value_by_flags_and_idx",
        &SpeculateSetValueByFlagsAndIdx,
        py::arg("pre_ids_all"),
        py::arg("accept_tokens"),
        py::arg("accept_num"),
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("step_idx"),
        "Set values based on flags and indices in speculative decoding");

  m.def("speculate_get_output_padding_offset",
        &SpeculateGetOutputPaddingOffset,
        py::arg("output_cum_offsets_tmp"),
        py::arg("out_token_num"),
        py::arg("seq_lens_output"),
        py::arg("max_seq_len"),
        "Get output padding offset");

  m.def("speculate_get_padding_offset",
        &SpeculateGetPaddingOffset,
        py::arg("input_ids"),
        py::arg("draft_tokens"),
        py::arg("cum_offsets"),
        py::arg("token_num"),
        py::arg("seq_len"),
        py::arg("seq_lens_encoder"),
        "Get padding offset");

  m.def("speculate_step_reschedule",
        &SpeculateStepSchedule,
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("ori_seq_lens_encoder"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("block_tables"),
        py::arg("encoder_block_lens"),
        py::arg("is_block_step"),
        py::arg("step_block_list"),
        py::arg("step_lens"),
        py::arg("recover_block_list"),
        py::arg("recover_lens"),
        py::arg("need_block_list"),
        py::arg("need_block_len"),
        py::arg("used_list_len"),
        py::arg("free_list"),
        py::arg("free_list_len"),
        py::arg("input_ids"),
        py::arg("pre_ids"),
        py::arg("step_idx"),
        py::arg("next_tokens"),
        py::arg("first_token_ids"),
        py::arg("accept_num"),
        py::arg("block_size"),
        py::arg("encoder_decoder_block_num"),
        py::arg("max_draft_tokens"),
        "Step reschedule");

  m.def("speculate_get_seq_lens_output",
        &SpeculateGetSeqLensOutput,
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        "Get sequence lengths output");

  m.def("set_data_ipc",
        &SetDataIpc,
        py::arg("tmp_input"),
        py::arg("shm_name"),
        "Set data IPC function");

  m.def("set_ncluster", &set_ncluster, "set ncluster");

  m.def("set_stop_value_multi_ends",
        &GetStopFlagsMulti,
        py::arg("topk_ids"),
        py::arg("stop_flags"),
        py::arg("seq_lens"),
        py::arg("end_ids"),
        py::arg("next_tokens"),
        py::arg("beam_search"),
        "Set stop value multi ends function");

  m.def("step_paddle",
        &StepPaddle,
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("ori_seq_lens_encoder"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("block_tables"),
        py::arg("encoder_block_lens"),
        py::arg("is_block_step"),
        py::arg("step_block_list"),
        py::arg("step_lens"),
        py::arg("recover_block_list"),
        py::arg("recover_lens"),
        py::arg("need_block_list"),
        py::arg("need_block_len"),
        py::arg("used_list_len"),
        py::arg("free_list"),
        py::arg("free_list_len"),
        py::arg("input_ids"),
        py::arg("pre_ids"),
        py::arg("step_idx"),
        py::arg("next_tokens"),
        py::arg("first_token_ids"),
        py::arg("block_size"),
        py::arg("encoder_decoder_block_num"),
        "Step paddle function");

  m.def("speculate_step_paddle",
        &SpeculateStepPaddle,
        py::arg("stop_flags"),
        py::arg("seq_lens_this_time"),
        py::arg("ori_seq_lens_encoder"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("block_tables"),
        py::arg("encoder_block_lens"),
        py::arg("is_block_step"),
        py::arg("step_block_list"),
        py::arg("step_lens"),
        py::arg("recover_block_list"),
        py::arg("recover_lens"),
        py::arg("need_block_list"),
        py::arg("need_block_len"),
        py::arg("used_list_len"),
        py::arg("free_list"),
        py::arg("free_list_len"),
        py::arg("input_ids"),
        py::arg("pre_ids"),
        py::arg("step_idx"),
        py::arg("next_tokens"),
        py::arg("first_token_ids"),
        py::arg("accept_num"),
        py::arg("block_size"),
        py::arg("encoder_decoder_block_num"),
        py::arg("max_draft_tokens"),
        "Step paddle function");

  m.def("text_image_gather_scatter",
        &TextImageGatherScatter,
        py::arg("input"),
        py::arg("text_input"),
        py::arg("image_input"),
        py::arg("token_type_ids"),
        py::arg("text_index"),
        py::arg("image_index"),
        py::arg("is_scatter"),
        "Scatter image and text from hidden states, or gather them to hidden "
        "states");

  m.def("text_image_index_out",
        &TextImageIndexOut,
        py::arg("token_type_ids"),
        py::arg("text_index"),
        py::arg("image_index"),
        "Generate index for text and image");

  m.def("top_p_candidates",
        &TopPCandidates,
        py::arg("probs"),
        py::arg("top_p"),
        py::arg("output_padding_offset"),
        py::arg("candidates_len"),
        py::arg("max_seq_len"),
        "Generate top-p candidates based on probability distributions");

  m.def("update_inputs",
        &UpdateInputs,
        py::arg("stop_flags"),
        py::arg("not_need_stop"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("input_ids"),
        py::arg("stop_nums"),
        py::arg("next_tokens"),
        py::arg("is_block_step"),
        "Update inputs function");

  m.def("update_inputs_v1",
        &UpdateInputsV1,
        py::arg("stop_flags"),
        py::arg("not_need_stop"),
        py::arg("seq_lens_this_time"),
        py::arg("seq_lens_encoder"),
        py::arg("seq_lens_decoder"),
        py::arg("step_seq_lens_decoder"),
        py::arg("prompt_lens"),
        py::arg("topk_ids"),
        py::arg("input_ids"),
        py::arg("block_tables"),
        py::arg("stop_nums"),
        py::arg("next_tokens"),
        py::arg("is_block_step"),
        py::arg("block_size"),
        "Update inputs v1 function");

  m.def("weight_quantize_xpu",
        &WeightQuantize,
        py::arg("x"),
        py::arg("algo"),
        py::arg("arch"),
        py::arg("group_size"),
        "Quantize weights on XPU");

  m.def("weight_only_linear_xpu",
        &WeightOnlyLinear,
        "Weight-only quantized linear layer",
        py::arg("x"),
        py::arg("weight"),
        py::arg("weight_scale"),
        py::arg("bias"),
        py::arg("weight_dtype"),
        py::arg("arch"),
        py::arg("group_size") = -1);

  m.def("xpu_moe_layer",
        &MoeLayer,
        py::arg("x"),
        py::arg("gate_weight"),
        py::arg("gate_correction_bias"),
        py::arg("up_gate_proj_weight"),
        py::arg("down_proj_weight"),
        py::arg("up_gate_proj_bias"),
        py::arg("down_proj_bias"),
        py::arg("up_gate_proj_weight_scale"),
        py::arg("down_proj_weight_scale"),
        py::arg("down_proj_in_scale"),
        py::arg("quant_method"),
        py::arg("moe_top_k"),
        py::arg("moe_group"),
        "fused moe op(topk + dispatch + ffn + combine) in XPU");

  // 添加XPU错误信息的异常处理类
  py::register_exception<XPUError>(m, "XPUError");
}
