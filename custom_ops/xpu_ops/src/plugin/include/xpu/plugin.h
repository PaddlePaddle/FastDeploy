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

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
DLL_EXPORT int set_stop_value_multi_ends(Context* ctx,
                                         bool* stop_flags,
                                         T* topk_ids,
                                         T* next_tokens,
                                         const T* end_ids,
                                         const int* seq_lens,
                                         const int bs,
                                         const int end_length,
                                         const bool beam_search);

DLL_EXPORT int set_value_by_flags_and_idx(Context* ctx,
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
DLL_EXPORT int token_penalty_multi_scores(Context* ctx,
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

DLL_EXPORT int get_padding_offset(Context* ctx,
                                  int* padding_offset,
                                  int* cum_offsets_out,
                                  int* cu_seqlens_q,
                                  int* cu_seqlens_k,
                                  int64_t* x_remove_padding,
                                  const int64_t* input_ids,
                                  const int* cum_offsets,
                                  const int* seq_lens,
                                  const int max_seq_len,
                                  const int bs);

DLL_EXPORT int speculate_get_padding_offset(Context* ctx,
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

DLL_EXPORT int update_inputs(Context* ctx,
                             bool* not_need_stop,
                             int* seq_lens_this_time,
                             int* seq_lens_encoder,
                             int* seq_lens_decoder,
                             int64_t* input_ids,
                             const int64_t* stop_nums,
                             const bool* stop_flags,
                             const bool* is_block_step,
                             const int64_t* next_tokens,
                             const int bsz,
                             const int max_bsz,
                             const int input_ids_stride);

DLL_EXPORT int free_and_dispatch_block(Context* ctx,
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
    Context* ctx,
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

DLL_EXPORT int recover_block(Context* ctx,
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

DLL_EXPORT int speculate_recover_block(Context* ctx,
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

DLL_EXPORT int recover_decode_task(Context* ctx,
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

DLL_EXPORT int update_inputs_v1(Context* ctx,
                                bool* not_need_stop,
                                int* seq_lens_this_time,
                                int* seq_lens_encoder,
                                int* seq_lens_decoder,
                                int* step_seq_lens_decoder,
                                int64_t* prompt_lens,
                                int64_t* topk_ids,
                                int64_t* input_ids,
                                int* block_tables,
                                const int64_t* stop_nums,
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
    Context* ctx,
    const TX* x,
    TY* y,
    VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    VectorParam<int32_t>& decoder_seqs_lods,  // NOLINT
    VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TY>
DLL_EXPORT int eb_gather_next_token(
    Context* ctx,
    const TX* x,
    TY* y,
    VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TY>
DLL_EXPORT int eb_mtp_gather_next_token(
    Context* ctx,
    const TX* x,
    TY* y,
    VectorParam<int32_t>& encoder_seqs_lods,  // NOLINT
    VectorParam<int32_t>& decoder_seqs_lods,  // NOLINT
    VectorParam<int32_t>& encoder_batch_map,  // NOLINT
    VectorParam<int32_t>& decoder_batch_map,  // NOLINT
    int64_t hidden_dim);

template <typename TX, typename TSCALE = float, typename TY = int8_t>
DLL_EXPORT int quant2d_per_channel(api::Context* ctx,
                                   const TX* x,
                                   const TSCALE* scale_in,
                                   TY* y,
                                   TSCALE* scale_out,
                                   int64_t m,
                                   int64_t n);

DLL_EXPORT int text_image_index_out(Context* ctx,
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
    Context* ctx,
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
    const int* output_padding_offset,
    const int* output_cum_offsets,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t end_length,
    const int64_t length_bad_words,
    const int64_t token_num,
    const int64_t max_seq_len);
DLL_EXPORT int mtp_free_and_dispatch_block(Context* ctx,
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
DLL_EXPORT int speculate_verify(Context* ctx,
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
                                const int* output_cum_offsets,
                                const int* actual_candidate_len,
                                const int real_bsz,
                                const int max_draft_tokens,
                                const int end_length,
                                const int max_seq_len,
                                const int max_candidate_len,
                                const int verify_window,
                                const bool prefill_one_step_stop,
                                const bool benchmark_mode);

DLL_EXPORT int speculate_clear_accept_nums(Context* ctx,
                                           int* accept_num,
                                           const int* seq_lens_decoder,
                                           const int max_bsz);

DLL_EXPORT int speculate_get_seq_lens_output(Context* ctx,
                                             int* seq_lens_output,
                                             const int* seq_lens_this_time,
                                             const int* seq_lens_encoder,
                                             const int* seq_lens_decoder,
                                             const int real_bsz);

DLL_EXPORT int draft_model_update(Context* ctx,
                                  const int64_t* inter_next_tokens,
                                  int64_t* draft_tokens,
                                  int64_t* pre_ids,
                                  int* seq_lens_this_time,
                                  int* seq_lens_encoder,
                                  int* seq_lens_decoder,
                                  int64_t* step_idx,
                                  const int* output_cum_offsets,
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

DLL_EXPORT int speculate_set_stop_value_multi_seqs(Context* ctx,
                                                   bool* stop_flags,
                                                   int64_t* accept_tokens,
                                                   int* accept_nums,
                                                   const int64_t* pre_ids,
                                                   const int64_t* step_idx,
                                                   const int64_t* stop_seqs,
                                                   const int* stop_seqs_len,
                                                   const int* seq_lens,
                                                   const int64_t* end_ids,
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
DLL_EXPORT int speculate_remove_padding(Context* ctx,
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

DLL_EXPORT int draft_model_postprocess(Context* ctx,
                                       const int64_t* base_model_draft_tokens,
                                       int* base_model_seq_lens_this_time,
                                       const int* base_model_seq_lens_encoder,
                                       const bool* base_model_stop_flags,
                                       int bsz,
                                       int base_model_draft_token_len);

DLL_EXPORT int speculate_set_value_by_flag_and_id(Context* ctx,
                                                  int64_t* pre_ids_all,
                                                  const int64_t* accept_tokens,
                                                  const int* accept_num,
                                                  const bool* stop_flags,
                                                  const int* seq_lens_encoder,
                                                  const int* seq_lens_decoder,
                                                  const int64_t* step_idx,
                                                  int bs,
                                                  int length,
                                                  int max_draft_tokens);

DLL_EXPORT int speculate_get_output_padding_offset(
    Context* ctx,
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
                                const int* output_padding_offset,
                                int64_t* out_id,
                                T* out_val,
                                int* actual_candidates_lens,
                                int vocab_size,
                                int token_num,
                                int max_cadidate_len,
                                int max_seq_len);

DLL_EXPORT int speculate_free_and_reschedule(Context* ctx,
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

DLL_EXPORT int speculate_update_v3(Context* ctx,
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
template <typename T>
DLL_EXPORT int rebuild_hidden_states(api::Context* ctx,
                                     const T* input,
                                     const int* position_map,
                                     T* out,
                                     int dim_embed,
                                     int elem_cnt);
template <typename T>
DLL_EXPORT int rebuild_self_hidden_states(api::Context* ctx,
                                          const T* input,
                                          int* src_map,
                                          T* output,
                                          int dim_embed,
                                          int elem_cnt);
/*--------------------------------------- MTP end
 * --------------------------------------------*/

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
