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
/*
 * copyright (C) 2022 KUNLUNXIN, Inc
 */

#pragma once
#include "xpu/xdnn.h"

namespace fd_xpu3 {
typedef xpu3::int64_t int64_t;
}

namespace fastdeploy {
namespace plugin {

namespace api = baidu::xpu::api;

template <typename T>
DLL_EXPORT int set_stop_value_multi_ends(api::Context* ctx,
                                         bool* stop_flags,
                                         T* topk_ids,
                                         T* next_tokens,
                                         const T* end_ids,
                                         const int* seq_lens,
                                         const int bs,
                                         const int end_length,
                                         const bool beam_search);

DLL_EXPORT int set_value_by_flags_and_idx(api::Context* ctx,
                                          const bool* stop_flags,
                                          int64_t* pre_ids_all,
                                          const int64_t* input_ids,
                                          const int* seq_lens_encoder,
                                          const int* seq_lens_decoder,
                                          const int64_t* step_idx,
                                          int bs,
                                          int length,
                                          int length_input_ids);

template <typename T>
DLL_EXPORT int token_penalty_multi_scores(api::Context* ctx,
                                          const int64_t* pre_ids,
                                          T* logits,
                                          const T* penalty_scores,
                                          const T* frequency_scores,
                                          const T* presence_scores,
                                          const float* temperatures,
                                          const int64_t* cur_len,
                                          const int64_t* min_len,
                                          const int64_t* eos_token_id,
                                          const int64_t* bad_words,
                                          const int64_t bs,
                                          const int64_t length,
                                          const int64_t length_id,
                                          const int64_t end_length,
                                          const int64_t length_bad_words);

DLL_EXPORT int get_padding_offset(api::Context* ctx,
                                  int* batch_id_per_token,
                                  int* cum_offsets_out,
                                  int* cu_seqlens_q,
                                  int* cu_seqlens_k,
                                  int64_t* x_remove_padding,
                                  const int64_t* input_ids,
                                  const int* seq_lens,
                                  const int max_seq_len,
                                  const int bs,
                                  const int64_t token_num);

DLL_EXPORT int speculate_get_padding_offset(api::Context* ctx,
                                            int* batch_id_per_token,
                                            int* cum_offsets_out,
                                            int* cu_seqlens_q,
                                            int* cu_seqlens_k,
                                            const int* cum_offsets,
                                            const int* seq_lens,
                                            const int max_seq_len,
                                            int bsz);

DLL_EXPORT int draft_model_preprocess(api::Context* ctx,
                                      int64_t* draft_tokens,
                                      int64_t* input_ids,
                                      bool* stop_flags,
                                      int* seq_lens_this_time,
                                      int* seq_lens_encoder,
                                      int* seq_lens_decoder,
                                      int64_t* step_idx,
                                      bool* not_need_stop,
                                      bool* is_block_step,
                                      bool* batch_drop,
                                      int64_t* pre_ids,
                                      const int64_t* accept_tokens,
                                      const int* accept_num,
                                      const int* base_model_seq_lens_this_time,
                                      const int* base_model_seq_lens_encoder,
                                      const int* base_model_seq_lens_decoder,
                                      const int64_t* base_model_step_idx,
                                      const bool* base_model_stop_flags,
                                      const bool* base_model_is_block_step,
                                      int64_t* base_model_draft_tokens,
                                      const int bsz,
                                      const int num_model_step,
                                      const int accept_tokens_len,
                                      const int draft_tokens_len,
                                      const int input_ids_len,
                                      const int base_model_draft_tokens_len,
                                      const int pre_ids_len,
                                      const bool truncate_first_token,
                                      const bool splitwise_prefill,
                                      const bool kvcache_scheduler_v1);

DLL_EXPORT int update_inputs(api::Context* ctx,
                             bool* not_need_stop,
                             int* seq_lens_this_time,
                             int* seq_lens_encoder,
                             int* seq_lens_decoder,
                             int64_t* input_ids,
                             const bool* stop_flags,
                             const bool* is_block_step,
                             const int64_t* next_tokens,
                             const int bsz,
                             const int max_bsz,
                             const int input_ids_stride);

DLL_EXPORT int free_and_dispatch_block(api::Context* ctx,
                                       bool* stop_flags,
                                       int* seq_lens_this_time,
                                       int* seq_lens_decoder,
                                       int* block_tables,
                                       int* encoder_block_lens,
                                       bool* is_block_step,
                                       int* step_block_list,  // [bsz]
                                       int* step_len,
                                       int* recover_block_list,
                                       int* recover_len,
                                       int* need_block_list,
                                       int* need_block_len,
                                       int* used_list_len,
                                       int* free_list,
                                       int* free_list_len,
                                       int64_t* first_token_ids,
                                       const int bsz,
                                       const int block_size,
                                       const int block_num_per_seq,
                                       const int max_decoder_block_num);

DLL_EXPORT int speculate_free_and_dispatch_block(
    api::Context* ctx,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_decoder,
    int* block_tables,
    int* encoder_block_lens,
    bool* is_block_step,
    int* step_block_list,  // [bsz]
    int* step_len,
    int* recover_block_list,
    int* recover_len,
    int* need_block_list,
    int* need_block_len,
    int* used_list_len,
    int* free_list,
    int* free_list_len,
    int64_t* first_token_ids,
    int* accept_num,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_decoder_block_num,
    const int max_draft_tokens);

DLL_EXPORT int recover_block(api::Context* ctx,
                             int* recover_block_list,  // [bsz]
                             int* recover_len,
                             bool* stop_flags,
                             int* seq_lens_this_time,
                             const int* ori_seq_lens_encoder,
                             int* seq_lens_encoder,
                             const int* seq_lens_decoder,
                             int* block_tables,
                             int* free_list,
                             int* free_list_len,
                             int64_t* input_ids,
                             const int64_t* pre_ids,
                             const int64_t* step_idx,
                             const int* encoder_block_lens,
                             const int* used_list_len,
                             const int64_t* next_tokens,
                             const int64_t* first_token_ids,
                             const int bsz,
                             const int block_num_per_seq,
                             const int length,
                             const int pre_id_length);

DLL_EXPORT int speculate_recover_block(api::Context* ctx,
                                       int* recover_block_list,  // [bsz]
                                       int* recover_len,
                                       bool* stop_flags,
                                       int* seq_lens_this_time,
                                       const int* ori_seq_lens_encoder,
                                       const int* ori_seq_lens_decoder,
                                       int* seq_lens_encoder,
                                       int* seq_lens_decoder,
                                       int* block_tables,
                                       int* free_list,
                                       int* free_list_len,
                                       int64_t* input_ids,
                                       const int64_t* pre_ids,
                                       const int64_t* step_idx,
                                       const int* encoder_block_lens,
                                       const int* used_list_len,
                                       const int64_t* next_tokens,
                                       const int64_t* first_token_ids,
                                       const int bsz,
                                       const int block_num_per_seq,
                                       const int length,
                                       const int pre_id_length);

DLL_EXPORT int recover_decode_task(api::Context* ctx,
                                   bool* stop_flags,
                                   int* seq_lens_this_time,
                                   int* seq_lens_encoder,
                                   int* seq_lens_decoder,
                                   int* step_seq_lens_decoder,
                                   int* block_tables,
                                   bool* is_block_step,
                                   const int bsz,
                                   const int block_num_per_seq,
                                   const int block_size);

DLL_EXPORT int recover_spec_decode_task(api::Context* ctx,
                                        bool* stop_flags,
                                        int* seq_lens_this_time,
                                        int* seq_lens_encoder,
                                        int* seq_lens_decoder,
                                        int* step_seq_lens_decoder,
                                        int* block_tables,
                                        bool* is_block_step,
                                        int64_t* draft_tokens,
                                        const int64_t* step_draft_tokens,
                                        const int* step_seq_lens_this_time,
                                        const int bsz,
                                        const int block_num_per_seq,
                                        const int block_size,
                                        const int draft_tokens_len,
                                        const int num_extra_tokens);

DLL_EXPORT int update_inputs_v1(api::Context* ctx,
                                bool* not_need_stop,
                                int* seq_lens_this_time,
                                int* seq_lens_encoder,
                                int* seq_lens_decoder,
                                int* step_seq_lens_decoder,
                                int64_t* prompt_lens,
                                int64_t* topk_ids,
                                int64_t* input_ids,
                                int* block_tables,
                                bool* stop_flags,
                                bool* is_block_step,
                                const int64_t* next_tokens,
                                const int bsz,
                                const int max_bsz,
                                const int input_ids_stride,
                                const int block_num_per_seq,
                                const int block_size);

template <typename TX, typename TY>
DLL_EXPORT int eb_adjust_batch(
    api::Context* ctx,
    const TX* x,
    TY* y,
    api::VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& decoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    api::VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TY>
DLL_EXPORT int eb_gather_next_token(
    api::Context* ctx,
    const TX* x,
    TY* y,
    api::VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    api::VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TY>
DLL_EXPORT int eb_mtp_gather_next_token(
    api::Context* ctx,
    const TX* x,
    TY* y,
    api::VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& decoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    api::VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TY>
DLL_EXPORT int eb_recover_batch_sequence(
    api::Context* ctx,
    const TX* x,
    TY* y,
    api::VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& decoder_seqs_lods,  // NOLINT
    api::VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    api::VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TSCALE = float, typename TY = int8_t>
DLL_EXPORT int quant2d_per_channel(api::Context* ctx,
                                   const TX* x,
                                   const TSCALE* scale_in,
                                   TY* y,
                                   TSCALE* scale_out,
                                   int64_t m,
                                   int64_t n);

DLL_EXPORT int text_image_index_out(api::Context* ctx,
                                    const int* token_type_ids,  // x
                                    int* text_index,            // y1
                                    int* image_index,           // y2
                                    const int64_t token_num);

template <typename T>
DLL_EXPORT int text_image_gather_scatter(api::Context* ctx,
                                         T* input,
                                         T* text_input,
                                         T* image_input,
                                         int* token_type_ids,
                                         int* text_index,
                                         int* image_index,
                                         int64_t token_num,
                                         int64_t text_token_num,
                                         int64_t image_token_num,
                                         int64_t hidden_size,
                                         bool is_scatter);

DLL_EXPORT int limit_thinking_content_length_kernel_v1(
    api::Context* ctx,
    int64_t* next_tokens,
    const int* max_think_lens,
    const int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_think_status,
    bool* stop_flags,
    const int64_t think_end_id,
    const int bs,
    const int eos_token_id_len);

DLL_EXPORT int limit_thinking_content_length_kernel_v2(
    api::Context* ctx,
    int64_t* next_tokens,
    const int* max_think_lens,
    const int64_t* step_idx,
    int* limit_think_status,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t line_break_id,
    const int bs);

/*--------------------------------------- MTP being
 * --------------------------------------------*/

template <typename T>
DLL_EXPORT int speculate_token_penalty_multi_scores(
    api::Context* ctx,
    const int64_t* pre_ids,
    T* logits,
    const T* penalty_scores,
    const T* frequency_scores,
    const T* presence_scores,
    const float* temperatures,
    const int64_t* cur_len,
    const int64_t* min_len,
    const int64_t* eos_token_id,
    const int64_t* bad_words,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t end_length,
    const int64_t length_bad_words,
    const int64_t token_num,
    const int64_t max_seq_len);
DLL_EXPORT int mtp_free_and_dispatch_block(api::Context* ctx,
                                           bool* base_model_stop_flags,
                                           bool* stop_flags,
                                           bool* batch_drop,
                                           int* seq_lens_this_time,
                                           int* seq_lens_decoder,
                                           int* block_tables,
                                           int* encoder_block_lens,
                                           int* used_list_len,
                                           int* free_list,
                                           int* free_list_len,
                                           const int bsz,
                                           const int block_size,
                                           const int block_num_per_seq,
                                           const int max_draft_tokens);

template <bool ENABLE_TOPP, bool USE_TOPK>
DLL_EXPORT int speculate_verify(api::Context* ctx,
                                const int64_t* sampled_token_ids,
                                int64_t* accept_tokens,
                                int* accept_num,
                                int64_t* step_idx,
                                bool* stop_flags,
                                const int* seq_lens_encoder,
                                const int* seq_lens_decoder,
                                const int64_t* draft_tokens,
                                const int* actual_draft_token_nums,
                                const float* dev_curand_states,
                                const float* topp,
                                const int* seq_lens_this_time,
                                const int64_t* verify_tokens,
                                const float* verify_scores,
                                const int64_t* max_dec_len,
                                const int64_t* end_tokens,
                                const bool* is_block_step,
                                const int* cu_seqlens_q_output,
                                const int* actual_candidate_len,
                                const int real_bsz,
                                const int max_draft_tokens,
                                const int end_length,
                                const int max_seq_len,
                                const int max_candidate_len,
                                const int verify_window,
                                const bool prefill_one_step_stop,
                                const bool benchmark_mode,
                                const bool accept_all_drafts,
                                const bool use_target_sampling);

DLL_EXPORT int speculate_clear_accept_nums(api::Context* ctx,
                                           int* accept_num,
                                           const int* seq_lens_decoder,
                                           const int max_bsz);

DLL_EXPORT int speculate_get_seq_lens_output(api::Context* ctx,
                                             int* seq_lens_output,
                                             const int* seq_lens_this_time,
                                             const int* seq_lens_encoder,
                                             const int* seq_lens_decoder,
                                             const int real_bsz);

DLL_EXPORT int draft_model_update(api::Context* ctx,
                                  const int64_t* inter_next_tokens,
                                  int64_t* draft_tokens,
                                  int64_t* pre_ids,
                                  int* seq_lens_this_time,
                                  int* seq_lens_encoder,
                                  int* seq_lens_decoder,
                                  int64_t* step_idx,
                                  const int* cu_seqlens_q_output,
                                  bool* stop_flags,
                                  bool* not_need_stop,
                                  const int64_t* max_dec_len,
                                  const int64_t* end_ids,
                                  int64_t* base_model_draft_tokens,
                                  const int bsz,
                                  const int max_draft_token,
                                  const int pre_id_length,
                                  const int max_base_model_draft_token,
                                  const int end_ids_len,
                                  const int max_seq_len,
                                  const int substep,
                                  const bool prefill_one_step_stop);

DLL_EXPORT int speculate_set_stop_value_multi_seqs(api::Context* ctx,
                                                   bool* stop_flags,
                                                   int64_t* accept_tokens,
                                                   int* accept_nums,
                                                   const int64_t* pre_ids,
                                                   const int64_t* step_idx,
                                                   const int64_t* stop_seqs,
                                                   const int* stop_seqs_len,
                                                   const int* seq_lens,
                                                   const int64_t* end_ids,
                                                   const int64_t* min_tokens,
                                                   const int bs_now,
                                                   const int accept_tokens_len,
                                                   const int stop_seqs_bs,
                                                   const int stop_seqs_max_len,
                                                   const int pre_ids_len);
template <typename T>
DLL_EXPORT int speculate_rebuild_append_padding(api::Context* ctx,
                                                T* full_hidden_states,
                                                int* cum_offsets,
                                                int* seq_len_encoder,
                                                int* seq_len_decoder,
                                                int* output_padding_offset,
                                                int max_seq_len,
                                                int dim_embed,
                                                int elem_nums,
                                                T* out);

template <typename T>
DLL_EXPORT int speculate_remove_padding(api::Context* ctx,
                                        T* x_remove_padding,
                                        const T* input_ids,
                                        const T* draft_tokens,
                                        const int* seq_lens,
                                        const int* seq_lens_encoder,
                                        const int* cum_offsets_out,
                                        int seq_length,
                                        int max_draft_tokens,
                                        int bsz,
                                        int token_num_data);

DLL_EXPORT int compute_self_order(api::Context* ctx,
                                  const int* last_seq_lens_this_time,
                                  const int* seq_lens_this_time,
                                  const int64_t* step_idx,
                                  int* src_map,
                                  int* output_token_num,
                                  int bsz);

DLL_EXPORT int compute_order(api::Context* ctx,
                             const int* seq_lens_this_time,
                             const int* seq_lens_encoder,
                             const int* base_model_seq_lens_this_time,
                             const int* base_model_seq_lens_encoder,
                             const int* accept_nums,
                             int* position_map,
                             int* output_token_num,
                             const int bsz,
                             const int actual_draft_token_num,
                             const int input_token_num);

DLL_EXPORT int draft_model_postprocess(api::Context* ctx,
                                       const int64_t* base_model_draft_tokens,
                                       int* base_model_seq_lens_this_time,
                                       const int* base_model_seq_lens_encoder,
                                       const bool* base_model_stop_flags,
                                       int bsz,
                                       int base_model_draft_token_len);

DLL_EXPORT int speculate_set_value_by_flag_and_id(api::Context* ctx,
                                                  int64_t* pre_ids_all,
                                                  const int64_t* accept_tokens,
                                                  int* accept_num,
                                                  const bool* stop_flags,
                                                  const int* seq_lens_encoder,
                                                  int* seq_lens_decoder,
                                                  const int64_t* step_idx,
                                                  int bs,
                                                  int length,
                                                  int max_draft_tokens);

DLL_EXPORT int speculate_get_output_padding_offset(
    api::Context* ctx,
    int* output_padding_offset,
    int* output_cum_offsets,
    const int* output_cum_offsets_tmp,
    const int* seq_lens_output,
    const int bsz,
    const int max_seq_len);

template <typename T, int MaxLength, int TopPBeamTopK>
DLL_EXPORT int top_p_candidates(api::Context* ctx,
                                const T* src,
                                const T* top_ps,
                                const int* batch_id_per_token_output,
                                int64_t* out_id,
                                T* out_val,
                                int* actual_candidates_lens,
                                int vocab_size,
                                int token_num,
                                int max_cadidate_len,
                                int max_seq_len);

DLL_EXPORT int speculate_free_and_reschedule(api::Context* ctx,
                                             bool* stop_flags,
                                             int* seq_lens_this_time,
                                             int* seq_lens_decoder,
                                             int* block_tables,
                                             int* encoder_block_lens,
                                             bool* is_block_step,
                                             int* step_block_list,  // [bsz]
                                             int* step_len,
                                             int* recover_block_list,
                                             int* recover_len,
                                             int* need_block_list,
                                             int* need_block_len,
                                             int* used_list_len,
                                             int* free_list,
                                             int* free_list_len,
                                             int64_t* first_token_ids,
                                             const int bsz,
                                             const int block_size,
                                             const int block_num_per_seq,
                                             const int max_decoder_block_num,
                                             const int max_draft_tokens);

DLL_EXPORT int speculate_schedule_cache(api::Context* ctx,
                                        const int64_t* draft_tokens,
                                        int* block_tables,
                                        bool* stop_flags,
                                        const int64_t* prompt_lens,
                                        int* seq_lens_this_time,
                                        int* seq_lens_encoder,
                                        int* seq_lens_decoder,
                                        int* step_seq_lens_decoder,
                                        int64_t* step_draft_tokens,
                                        int* step_seq_lens_this_time,
                                        int* accept_num,
                                        int64_t* accept_tokens,
                                        bool* is_block_step,
                                        bool* not_need_stop,
                                        const int real_bsz,
                                        const int max_bsz,
                                        const int max_next_step_tokens,
                                        const int draft_tokens_len,
                                        const int accept_tokens_len,
                                        const int block_size,
                                        const int block_num_per_seq,
                                        const bool prefill_one_step_stop);

DLL_EXPORT int speculate_preprocess(api::Context* ctx,
                                    int64_t* ids_remove_padding,
                                    int* batch_id_per_token,
                                    int* cu_seqlens_q,
                                    int* cu_seqlens_k,
                                    int* seq_lens_output,
                                    int* cu_seq_lens_q_output,
                                    int* batch_id_per_token_output,
                                    int* real_output_token_num,
                                    const int64_t* input_data,
                                    const int* seq_lens,
                                    const int64_t* draft_tokens,
                                    const int* seq_lens_encoder,
                                    const int max_seq_len,
                                    const int max_draft_tokens_per_batch,
                                    const int token_num_data,
                                    const int real_bs);

DLL_EXPORT int speculate_update_v3(api::Context* ctx,
                                   int* seq_lens_encoder,
                                   int* seq_lens_decoder,
                                   bool* not_need_stop,
                                   int64_t* draft_tokens,
                                   int* actual_draft_token_nums,
                                   const int64_t* accept_tokens,
                                   const int* accept_num,
                                   const bool* stop_flags,
                                   const int* seq_lens_this_time,
                                   const bool* is_block_step,
                                   const int64_t* stop_nums,
                                   const int real_bsz,
                                   const int max_bsz,
                                   const int max_draft_tokens);

DLL_EXPORT int speculate_update(api::Context* ctx,
                                int* seq_lens_encoder,
                                int* seq_lens_decoder,
                                bool* not_need_stop,
                                int64_t* draft_tokens,
                                int* actual_draft_token_nums,
                                const int64_t* accept_tokens,
                                const int* accept_num,
                                const bool* stop_flags,
                                const int* seq_lens_this_time,
                                const bool* is_block_step,
                                int* mask_rollback,
                                const int real_bsz,
                                const int max_bsz,
                                const int max_draft_tokens);

DLL_EXPORT int unified_update_model_status(api::Context* ctx,
                                           int* seq_lens_encoder,
                                           int* seq_lens_decoder,
                                           bool* has_running_seqs,
                                           int* mask_rollback,
                                           int64_t* step_input_ids,
                                           int* adaptive_step_input_len,
                                           int64_t* step_output_ids,
                                           int* step_output_len,
                                           bool* stop_flags,
                                           int* seq_lens_this_time,
                                           const bool* is_paused,
                                           int64_t* token_ids_all,
                                           const int64_t* prompt_lens,
                                           int64_t* step_idx,
                                           const int64_t* end_tokens,
                                           const int64_t* max_dec_len,
                                           int real_bsz,
                                           int max_bsz,
                                           int max_step_tokens,
                                           int max_model_len,
                                           int num_end_tokens,
                                           bool is_naive_mode,
                                           bool prefill_one_step_stop);

template <typename T>
DLL_EXPORT int rebuild_hidden_states(api::Context* ctx,
                                     const T* input,
                                     const int* position_map,
                                     T* out,
                                     int dim_embed,
                                     int elem_cnt,
                                     int output_token_num);
template <typename T>
DLL_EXPORT int rebuild_self_hidden_states(api::Context* ctx,
                                          const T* input,
                                          int* src_map,
                                          T* output,
                                          int input_token_num,
                                          int dim_embed,
                                          int elem_cnt);

DLL_EXPORT int speculate_get_logits(api::Context* ctx,
                                    float* draft_logits,
                                    int* next_token_num,
                                    int* batch_token_num,
                                    int* cu_next_token_offset,
                                    int* cu_batch_token_offset,
                                    const float* logits,
                                    const float* first_token_logits,
                                    const int* seq_lens_this_time,
                                    const int* seq_lens_encoder,
                                    const int real_bsz,
                                    const int vocab_size);

DLL_EXPORT int update_attn_mask_offsets(api::Context* ctx,
                                        int* attn_mask_offsets,
                                        const int* seq_lens_this_time,
                                        const int* seq_lens_encoder,
                                        const int* seq_lens_decoder,
                                        const int* cu_seqlens_q,
                                        const int* attn_mask_offsets_full,
                                        int* attn_mask_offsets_decoder,
                                        const bool* is_block_step,
                                        int* decode_states,
                                        int* mask_rollback,
                                        int real_bsz,
                                        int max_model_len,
                                        int decode_states_len);

DLL_EXPORT int speculate_limit_thinking_content_length_kernel(
    api::Context* ctx,
    int64_t* next_tokens,
    const int* max_think_lens,
    int* max_reply_lens,
    int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_status,
    int* accept_num,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t* inject_token_ids,
    const int tokens_per_step,
    const int bs,
    const int eos_token_id_len,
    const int inject_len,
    const bool splitwise_role_is_decode);
/*--------------------------------------- MTP end
 * --------------------------------------------*/

}  // namespace plugin
}  // namespace fastdeploy
