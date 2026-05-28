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

#include "xpu/plugin.h"
#include "xpu/refactor/impl/launch_strategy.h"
#include "xpu/refactor/impl_public/wrapper_check.h"
#include "xpu/xdnn.h"

namespace fd_xpu3 {
__attribute__((global)) void draft_model_preprocess(
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    bool* not_need_stop,
    int64_t* pre_ids,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* target_model_seq_lens_encoder,
    const int* target_model_seq_lens_decoder,
    const int64_t* target_model_step_idx,
    const bool* target_model_stop_flags,
    const int64_t* max_dec_len,
    int64_t* target_model_draft_tokens,
    const int bsz,
    const int num_model_step,
    const int accept_tokens_len,
    const int draft_tokens_len,
    const int input_ids_len,
    const int target_model_draft_tokens_len,
    const int pre_ids_len,
    const bool is_splitwise_prefill);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int64_t* draft_tokens,
                       int64_t* input_ids,
                       bool* stop_flags,
                       int* seq_lens_this_time,
                       int* seq_lens_encoder,
                       int* seq_lens_decoder,
                       int64_t* step_idx,
                       bool* not_need_stop,
                       int64_t* pre_ids,
                       const int64_t* accept_tokens,
                       const int* accept_num,
                       const int* target_model_seq_lens_encoder,
                       const int* target_model_seq_lens_decoder,
                       const int64_t* target_model_step_idx,
                       const bool* target_model_stop_flags,
                       const int64_t* max_dec_len,
                       int64_t* target_model_draft_tokens,
                       const int bsz,
                       const int num_model_step,
                       const int accept_tokens_len,
                       const int draft_tokens_len,
                       const int input_ids_len,
                       const int target_model_draft_tokens_len,
                       const int pre_ids_len,
                       const bool is_splitwise_prefill) {
  int64_t not_stop_flag_sum = 0;
  for (int tid = 0; tid < bsz; tid++) {
    auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
    auto* draft_tokens_now = draft_tokens + tid * draft_tokens_len;
    const int32_t accept_num_now = accept_num[tid];
    auto* input_ids_now = input_ids + tid * input_ids_len;
    auto* target_model_draft_tokens_now =
        target_model_draft_tokens + tid * target_model_draft_tokens_len;
    auto* pre_ids_now = pre_ids + tid * pre_ids_len;
    const auto target_step = target_model_step_idx[tid];
    auto seq_len_encoder = seq_lens_encoder[tid];

    for (int i = 1; i < target_model_draft_tokens_len; i++) {
      target_model_draft_tokens_now[i] = -1;
    }

    bool should_skip = false;
    if (target_model_stop_flags[tid]) {
      should_skip = true;
    }
    if (!is_splitwise_prefill &&
        target_step + num_model_step >= max_dec_len[tid]) {
      should_skip = true;
    }

    if (should_skip) {
      stop_flags[tid] = true;
      seq_lens_this_time[tid] = 0;
      seq_lens_decoder[tid] = 0;
      seq_lens_encoder[tid] = 0;
      step_idx[tid] = 0;
    } else {
      not_stop_flag_sum += 1;
      stop_flags[tid] = false;

      if (seq_len_encoder > 0) {
        int64_t target_model_first_token = accept_tokens_now[0];
        pre_ids_now[0] = target_model_first_token;
        input_ids_now[seq_len_encoder - 1] = target_model_first_token;
        seq_lens_this_time[tid] = seq_len_encoder;
        step_idx[tid] = target_step - 1;
      } else {
        int32_t need_compute_token = accept_num_now;
        seq_lens_decoder[tid] =
            target_model_seq_lens_decoder[tid] - need_compute_token;
        step_idx[tid] = target_model_step_idx[tid] - need_compute_token;

        for (int i = 0; i < accept_num_now; i++) {
          draft_tokens_now[i] = accept_tokens_now[i];
          const int pre_id_pos =
              target_model_step_idx[tid] - (accept_num_now - i);
          pre_ids_now[pre_id_pos] = accept_tokens_now[i];
        }
        seq_lens_this_time[tid] = accept_num_now;
      }
    }
  }
  not_need_stop[0] = not_stop_flag_sum > 0;
  return api::SUCCESS;
}

static int xpu3_wrapper(api::Context* ctx,
                        int64_t* draft_tokens,
                        int64_t* input_ids,
                        bool* stop_flags,
                        int* seq_lens_this_time,
                        int* seq_lens_encoder,
                        int* seq_lens_decoder,
                        int64_t* step_idx,
                        bool* not_need_stop,
                        int64_t* pre_ids,
                        const int64_t* accept_tokens,
                        const int* accept_num,
                        const int* target_model_seq_lens_encoder,
                        const int* target_model_seq_lens_decoder,
                        const int64_t* target_model_step_idx,
                        const bool* target_model_stop_flags,
                        const int64_t* max_dec_len,
                        int64_t* target_model_draft_tokens,
                        const int bsz,
                        const int num_model_step,
                        const int accept_tokens_len,
                        const int draft_tokens_len,
                        const int input_ids_len,
                        const int target_model_draft_tokens_len,
                        const int pre_ids_len,
                        const bool is_splitwise_prefill) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre = fd_xpu3::draft_model_preprocess<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64*>(draft_tokens),
      reinterpret_cast<XPU_INT64*>(input_ids),
      stop_flags,
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      reinterpret_cast<XPU_INT64*>(step_idx),
      not_need_stop,
      reinterpret_cast<XPU_INT64*>(pre_ids),
      reinterpret_cast<const XPU_INT64*>(accept_tokens),
      accept_num,
      target_model_seq_lens_encoder,
      target_model_seq_lens_decoder,
      reinterpret_cast<const XPU_INT64*>(target_model_step_idx),
      target_model_stop_flags,
      reinterpret_cast<const XPU_INT64*>(max_dec_len),
      reinterpret_cast<XPU_INT64*>(target_model_draft_tokens),
      bsz,
      num_model_step,
      accept_tokens_len,
      draft_tokens_len,
      input_ids_len,
      target_model_draft_tokens_len,
      pre_ids_len,
      is_splitwise_prefill);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int draft_model_preprocess(api::Context* ctx,
                           int64_t* draft_tokens,
                           int64_t* input_ids,
                           bool* stop_flags,
                           int* seq_lens_this_time,
                           int* seq_lens_encoder,
                           int* seq_lens_decoder,
                           int64_t* step_idx,
                           bool* not_need_stop,
                           int64_t* pre_ids,
                           const int64_t* accept_tokens,
                           const int* accept_num,
                           const int* target_model_seq_lens_encoder,
                           const int* target_model_seq_lens_decoder,
                           const int64_t* target_model_step_idx,
                           const bool* target_model_stop_flags,
                           const int64_t* max_dec_len,
                           int64_t* target_model_draft_tokens,
                           const int bsz,
                           const int num_model_step,
                           const int accept_tokens_len,
                           const int draft_tokens_len,
                           const int input_ids_len,
                           const int target_model_draft_tokens_len,
                           const int pre_ids_len,
                           const bool is_splitwise_prefill) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "draft_model_preprocess", int64_t);
  WRAPPER_DUMP_PARAM6(ctx,
                      draft_tokens,
                      input_ids,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder);
  WRAPPER_DUMP_PARAM4(ctx, step_idx, not_need_stop, pre_ids, accept_tokens);
  WRAPPER_DUMP_PARAM4(ctx,
                      accept_num,
                      target_model_seq_lens_encoder,
                      target_model_seq_lens_decoder,
                      target_model_step_idx);
  WRAPPER_DUMP_PARAM4(ctx,
                      target_model_stop_flags,
                      max_dec_len,
                      target_model_draft_tokens,
                      bsz);
  WRAPPER_DUMP_PARAM6(ctx,
                      num_model_step,
                      accept_tokens_len,
                      draft_tokens_len,
                      input_ids_len,
                      target_model_draft_tokens_len,
                      pre_ids_len);
  WRAPPER_DUMP_PARAM1(ctx, is_splitwise_prefill);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * accept_tokens_len, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * input_ids_len, input_ids);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * draft_tokens_len, draft_tokens);
  WRAPPER_CHECK_PTR(ctx,
                    int64_t,
                    bsz * target_model_draft_tokens_len,
                    target_model_draft_tokens);

  WRAPPER_ASSERT_GT(ctx, bsz, 0);
  WRAPPER_ASSERT_LT(ctx, accept_tokens_len, 128);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       draft_tokens,
                       input_ids,
                       stop_flags,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       step_idx,
                       not_need_stop,
                       pre_ids,
                       accept_tokens,
                       accept_num,
                       target_model_seq_lens_encoder,
                       target_model_seq_lens_decoder,
                       target_model_step_idx,
                       target_model_stop_flags,
                       max_dec_len,
                       target_model_draft_tokens,
                       bsz,
                       num_model_step,
                       accept_tokens_len,
                       draft_tokens_len,
                       input_ids_len,
                       target_model_draft_tokens_len,
                       pre_ids_len,
                       is_splitwise_prefill);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        draft_tokens,
                        input_ids,
                        stop_flags,
                        seq_lens_this_time,
                        seq_lens_encoder,
                        seq_lens_decoder,
                        step_idx,
                        not_need_stop,
                        pre_ids,
                        accept_tokens,
                        accept_num,
                        target_model_seq_lens_encoder,
                        target_model_seq_lens_decoder,
                        target_model_step_idx,
                        target_model_stop_flags,
                        max_dec_len,
                        target_model_draft_tokens,
                        bsz,
                        num_model_step,
                        accept_tokens_len,
                        draft_tokens_len,
                        input_ids_len,
                        target_model_draft_tokens_len,
                        pre_ids_len,
                        is_splitwise_prefill);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
