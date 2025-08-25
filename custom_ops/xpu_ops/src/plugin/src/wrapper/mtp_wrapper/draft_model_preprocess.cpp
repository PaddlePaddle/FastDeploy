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

namespace xpu3 {
namespace plugin {
__attribute__((global)) void draft_model_preprocess(
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    int* seq_lens_encoder_record,
    int* seq_lens_decoder_record,
    bool* not_need_stop,
    bool* batch_drop,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* base_model_seq_lens_encoder,
    const int* base_model_seq_lens_decoder,
    const int64_t* base_model_step_idx,
    const bool* base_model_stop_flags,
    const bool* base_model_is_block_step,
    int64_t* base_model_draft_tokens,
    int real_bsz,
    int max_draft_token,
    int accept_tokens_len,
    int draft_tokens_len,
    int input_ids_len,
    int base_model_draft_tokens_len,
    bool truncate_first_token,
    bool splitwise_prefill);
}  // namespace plugin
}  // namespace xpu3

namespace xpu2 {
namespace plugin {}  // namespace plugin
}  // namespace xpu2

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(api::Context* ctx,
                       int64_t* draft_tokens,
                       int64_t* input_ids,
                       bool* stop_flags,
                       int* seq_lens_this_time,
                       int* seq_lens_encoder,
                       int* seq_lens_decoder,
                       int64_t* step_idx,
                       int* seq_lens_encoder_record,
                       int* seq_lens_decoder_record,
                       bool* not_need_stop,
                       bool* batch_drop,
                       const int64_t* accept_tokens,
                       const int* accept_num,
                       const int* base_model_seq_lens_encoder,
                       const int* base_model_seq_lens_decoder,
                       const int64_t* base_model_step_idx,
                       const bool* base_model_stop_flags,
                       const bool* base_model_is_block_step,
                       int64_t* base_model_draft_tokens,
                       int real_bsz,
                       int max_draft_token,
                       int accept_tokens_len,
                       int draft_tokens_len,
                       int input_ids_len,
                       int base_model_draft_tokens_len,
                       bool truncate_first_token,
                       bool splitwise_prefill) {
  int64_t not_stop_flag_sum = 0;
  int64_t not_stop_flag = 0;
  for (int tid = 0; tid < real_bsz; tid++) {
    if (splitwise_prefill) {
      int base_model_step_idx_now = base_model_step_idx[tid];
      auto* input_ids_now = input_ids + tid * input_ids_len;
      auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
      // printf("bid: %d, base_model_step_idx_now: %d seq_lens_encoder_record:
      // %d\n", tid, base_model_step_idx_now, seq_lens_encoder_record[tid]);
      if (base_model_step_idx_now == 1 && seq_lens_encoder_record[tid] > 0) {
        not_stop_flag = 1;
        int seq_len_encoder_record = seq_lens_encoder_record[tid];
        seq_lens_encoder[tid] = seq_len_encoder_record;
        seq_lens_encoder_record[tid] = -1;
        stop_flags[tid] = false;
        int64_t base_model_first_token = accept_tokens_now[0];
        int position = seq_len_encoder_record;
        if (truncate_first_token) {
          input_ids_now[position - 1] = base_model_first_token;
          seq_lens_this_time[tid] = seq_len_encoder_record;
        } else {
          input_ids_now[position] = base_model_first_token;
          seq_lens_this_time[tid] = seq_len_encoder_record + 1;
        }
      } else {
        stop_flags[tid] = true;
        seq_lens_this_time[tid] = 0;
        seq_lens_decoder[tid] = 0;
        seq_lens_encoder[tid] = 0;
        not_stop_flag = 0;
      }
      not_stop_flag_sum += not_stop_flag;
    } else {
      auto base_model_step_idx_now = base_model_step_idx[tid];
      auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
      auto* draft_tokens_now = draft_tokens + tid * draft_tokens_len;
      auto accept_num_now = accept_num[tid];
      auto* input_ids_now = input_ids + tid * input_ids_len;
      auto* base_model_draft_tokens_now =
          base_model_draft_tokens + tid * base_model_draft_tokens_len;
      for (int i = 1; i < base_model_draft_tokens_len; i++) {
        base_model_draft_tokens_now[i] = -1;
      }
      if (base_model_stop_flags[tid] && base_model_is_block_step[tid]) {
        batch_drop[tid] = true;
        stop_flags[tid] = true;
      }

      if (!(base_model_stop_flags[tid] || batch_drop[tid])) {
        not_stop_flag = 1;
        // 1. first token

        if (base_model_step_idx_now == 0) {
          seq_lens_this_time[tid] = 0;
          not_stop_flag = 0;
        } else if (base_model_step_idx_now == 1 &&
                   seq_lens_encoder_record[tid] > 0) {
          // Can be extended to first few tokens
          int seq_len_encoder_record = seq_lens_encoder_record[tid];
          seq_lens_encoder[tid] = seq_len_encoder_record;
          seq_lens_encoder_record[tid] = -1;
          seq_lens_decoder[tid] = seq_lens_decoder_record[tid];
          seq_lens_decoder_record[tid] = 0;
          stop_flags[tid] = false;
          int64_t base_model_first_token = accept_tokens_now[0];
          int position = seq_len_encoder_record;
          if (truncate_first_token) {
            input_ids_now[position - 1] = base_model_first_token;
            seq_lens_this_time[tid] = seq_len_encoder_record;
          } else {
            input_ids_now[position] = base_model_first_token;
            seq_lens_this_time[tid] = seq_len_encoder_record + 1;
          }
        } else if (accept_num_now <=
                   max_draft_token) /*Accept partial draft tokens*/ {
          // Base Model reject stop
          if (stop_flags[tid]) {
            stop_flags[tid] = false;
            seq_lens_decoder[tid] = base_model_seq_lens_decoder[tid];
            step_idx[tid] = base_model_step_idx[tid];
          } else {
            seq_lens_decoder[tid] -= max_draft_token - accept_num_now;
            step_idx[tid] -= max_draft_token - accept_num_now;
          }
          int64_t modified_token = accept_tokens_now[accept_num_now - 1];
          draft_tokens_now[0] = modified_token;
          seq_lens_this_time[tid] = 1;
        } else /*Accept all draft tokens*/ {
          draft_tokens_now[1] = accept_tokens_now[max_draft_token];
          seq_lens_this_time[tid] = 2;
        }
      } else {
        stop_flags[tid] = true;
        seq_lens_this_time[tid] = 0;
        seq_lens_decoder[tid] = 0;
        seq_lens_encoder[tid] = 0;
      }
      not_stop_flag_sum += not_stop_flag;
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
                        int* seq_lens_encoder_record,
                        int* seq_lens_decoder_record,
                        bool* not_need_stop,
                        bool* batch_drop,
                        const int64_t* accept_tokens,
                        const int* accept_num,
                        const int* base_model_seq_lens_encoder,
                        const int* base_model_seq_lens_decoder,
                        const int64_t* base_model_step_idx,
                        const bool* base_model_stop_flags,
                        const bool* base_model_is_block_step,
                        int64_t* base_model_draft_tokens,
                        int real_bsz,
                        int max_draft_token,
                        int accept_tokens_len,
                        int draft_tokens_len,
                        int input_ids_len,
                        int base_model_draft_tokens_len,
                        bool truncate_first_token,
                        bool splitwise_prefill) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;

  // NOTE: Don't change 16 to 64, because kernel use gsm
  xpu3::plugin::draft_model_preprocess<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<XPU_INT64*>(draft_tokens),
      reinterpret_cast<XPU_INT64*>(input_ids),
      stop_flags,
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      reinterpret_cast<XPU_INT64*>(step_idx),
      seq_lens_encoder_record,
      seq_lens_decoder_record,
      not_need_stop,
      batch_drop,
      reinterpret_cast<const XPU_INT64*>(accept_tokens),
      accept_num,
      base_model_seq_lens_encoder,
      base_model_seq_lens_decoder,
      reinterpret_cast<const XPU_INT64*>(base_model_step_idx),
      base_model_stop_flags,
      base_model_is_block_step,
      reinterpret_cast<XPU_INT64*>(base_model_draft_tokens),
      real_bsz,
      max_draft_token,
      accept_tokens_len,
      draft_tokens_len,
      input_ids_len,
      base_model_draft_tokens_len,
      truncate_first_token,
      splitwise_prefill);
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
                           int* seq_lens_encoder_record,
                           int* seq_lens_decoder_record,
                           bool* not_need_stop,
                           bool* batch_drop,
                           const int64_t* accept_tokens,
                           const int* accept_num,
                           const int* base_model_seq_lens_encoder,
                           const int* base_model_seq_lens_decoder,
                           const int64_t* base_model_step_idx,
                           const bool* base_model_stop_flags,
                           const bool* base_model_is_block_step,
                           int64_t* base_model_draft_tokens,
                           int real_bsz,
                           int max_draft_token,
                           int accept_tokens_len,
                           int draft_tokens_len,
                           int input_ids_len,
                           int base_model_draft_tokens_len,
                           bool truncate_first_token,
                           bool splitwise_prefill) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "draft_model_preprocess", int64_t);
  WRAPPER_DUMP_PARAM6(ctx,
                      draft_tokens,
                      input_ids,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder);
  WRAPPER_DUMP_PARAM5(ctx,
                      step_idx,
                      seq_lens_encoder_record,
                      seq_lens_decoder_record,
                      not_need_stop,
                      batch_drop);
  WRAPPER_DUMP_PARAM3(
      ctx, accept_tokens, accept_num, base_model_seq_lens_encoder);
  WRAPPER_DUMP_PARAM3(ctx,
                      base_model_seq_lens_decoder,
                      base_model_step_idx,
                      base_model_stop_flags);
  WRAPPER_DUMP_PARAM3(
      ctx, base_model_is_block_step, base_model_draft_tokens, real_bsz);
  WRAPPER_DUMP_PARAM3(
      ctx, max_draft_token, accept_tokens_len, draft_tokens_len);
  WRAPPER_DUMP_PARAM3(
      ctx, input_ids_len, base_model_draft_tokens_len, truncate_first_token);
  WRAPPER_DUMP_PARAM1(ctx, splitwise_prefill);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * accept_tokens_len, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * input_ids_len, input_ids);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * draft_tokens_len, draft_tokens);
  WRAPPER_CHECK_PTR(ctx,
                    int64_t,
                    real_bsz * base_model_draft_tokens_len,
                    base_model_draft_tokens);

  WRAPPER_ASSERT_GT(ctx, real_bsz, 0);
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
                       seq_lens_encoder_record,
                       seq_lens_decoder_record,
                       not_need_stop,
                       batch_drop,
                       accept_tokens,
                       accept_num,
                       base_model_seq_lens_encoder,
                       base_model_seq_lens_decoder,
                       base_model_step_idx,
                       base_model_stop_flags,
                       base_model_is_block_step,
                       base_model_draft_tokens,
                       real_bsz,
                       max_draft_token,
                       accept_tokens_len,
                       draft_tokens_len,
                       input_ids_len,
                       base_model_draft_tokens_len,
                       truncate_first_token,
                       splitwise_prefill);
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
                        seq_lens_encoder_record,
                        seq_lens_decoder_record,
                        not_need_stop,
                        batch_drop,
                        accept_tokens,
                        accept_num,
                        base_model_seq_lens_encoder,
                        base_model_seq_lens_decoder,
                        base_model_step_idx,
                        base_model_stop_flags,
                        base_model_is_block_step,
                        base_model_draft_tokens,
                        real_bsz,
                        max_draft_token,
                        accept_tokens_len,
                        draft_tokens_len,
                        input_ids_len,
                        base_model_draft_tokens_len,
                        truncate_first_token,
                        splitwise_prefill);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
