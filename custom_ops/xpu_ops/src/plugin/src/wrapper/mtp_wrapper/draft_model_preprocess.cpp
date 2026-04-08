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
                       const bool kvcache_scheduler_v1) {
  int64_t not_stop_flag_sum = 0;
  int64_t not_stop_flag = 0;
  for (int tid = 0; tid < bsz; tid++) {
    if (splitwise_prefill) {
      auto* input_ids_now = input_ids + tid * input_ids_len;
      auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
      if (seq_lens_encoder[tid] > 0) {
        not_stop_flag = 1;
        int seq_len_encoder = seq_lens_encoder[tid];
        stop_flags[tid] = false;
        int64_t base_model_first_token = accept_tokens_now[0];
        int position = seq_len_encoder;
        if (truncate_first_token) {
          input_ids_now[position - 1] = base_model_first_token;
          seq_lens_this_time[tid] = seq_len_encoder;
        } else {
          input_ids_now[position] = base_model_first_token;
          seq_lens_this_time[tid] = seq_len_encoder + 1;
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
      auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
      auto* draft_tokens_now = draft_tokens + tid * draft_tokens_len;
      auto accept_num_now = accept_num[tid];
      auto* input_ids_now = input_ids + tid * input_ids_len;
      auto* base_model_draft_tokens_now =
          base_model_draft_tokens + tid * base_model_draft_tokens_len;
      auto base_model_seq_len_decoder = base_model_seq_lens_decoder[tid];
      const int32_t base_model_seq_len_this_time =
          base_model_seq_lens_this_time[tid];
      auto* pre_ids_now = pre_ids + tid * pre_ids_len;
      for (int i = 1; i < base_model_draft_tokens_len; i++) {
        base_model_draft_tokens_now[i] = -1;
      }
      if (kvcache_scheduler_v1) {
        if (base_model_stop_flags[tid] && base_model_is_block_step[tid]) {
          stop_flags[tid] = true;
          is_block_step[tid] = true;
          // Need to continue infer
        }
      } else {
        if (base_model_stop_flags[tid] && base_model_is_block_step[tid]) {
          batch_drop[tid] = true;
          stop_flags[tid] = true;
        }
      }

      if (!(base_model_stop_flags[tid] || batch_drop[tid])) {
        not_stop_flag = 1;
        // prefill generation
        if (seq_lens_encoder[tid] > 0) {
          // Can be extended to first few tokens
          int seq_len_encoder = seq_lens_encoder[tid];
          stop_flags[tid] = false;
          int64_t base_model_first_token = accept_tokens_now[0];
          pre_ids_now[0] = base_model_first_token;
          int position = seq_len_encoder;
          if (truncate_first_token) {
            input_ids_now[position - 1] = base_model_first_token;
            seq_lens_this_time[tid] = seq_len_encoder;
          } else {
            input_ids_now[position] = base_model_first_token;
            seq_lens_this_time[tid] = seq_len_encoder + 1;
          }
        } else {  // decode generation
          if (kvcache_scheduler_v1) {
            // 3. try to recover mtp infer in V1 mode
            if (!base_model_is_block_step[tid] && is_block_step[tid]) {
              is_block_step[tid] = false;
            }
          }
          if (stop_flags[tid]) {
            stop_flags[tid] = false;
            // TODO: check
            seq_lens_decoder[tid] =
                base_model_seq_len_decoder - base_model_seq_len_this_time;
            step_idx[tid] =
                base_model_step_idx[tid] - base_model_seq_len_this_time;
          } else {
            // 2: Last base model generated token and first MTP
            // token
            seq_lens_decoder[tid] -= num_model_step - 1;
            step_idx[tid] -= num_model_step - 1;
          }
          for (int i = 0; i < accept_num_now; i++) {
            draft_tokens_now[i] = accept_tokens_now[i];
            const int pre_id_pos =
                base_model_step_idx[tid] - (accept_num_now - i);
            const int64_t accept_token = accept_tokens_now[i];
            pre_ids_now[pre_id_pos] = accept_token;
          }
          seq_lens_this_time[tid] = accept_num_now;
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
                        const bool kvcache_scheduler_v1) {
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
      not_need_stop,
      is_block_step,
      batch_drop,
      reinterpret_cast<XPU_INT64*>(pre_ids),
      reinterpret_cast<const XPU_INT64*>(accept_tokens),
      accept_num,
      base_model_seq_lens_this_time,
      base_model_seq_lens_encoder,
      base_model_seq_lens_decoder,
      reinterpret_cast<const XPU_INT64*>(base_model_step_idx),
      base_model_stop_flags,
      base_model_is_block_step,
      reinterpret_cast<XPU_INT64*>(base_model_draft_tokens),
      bsz,
      num_model_step,
      accept_tokens_len,
      draft_tokens_len,
      input_ids_len,
      base_model_draft_tokens_len,
      pre_ids_len,
      truncate_first_token,
      splitwise_prefill,
      kvcache_scheduler_v1);
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
                           const bool kvcache_scheduler_v1) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "draft_model_preprocess", int64_t);
  WRAPPER_DUMP_PARAM6(ctx,
                      draft_tokens,
                      input_ids,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder);
  WRAPPER_DUMP_PARAM5(
      ctx, step_idx, not_need_stop, is_block_step, batch_drop, pre_ids);
  WRAPPER_DUMP_PARAM3(
      ctx, accept_tokens, accept_num, base_model_seq_lens_encoder);
  WRAPPER_DUMP_PARAM4(ctx,
                      base_model_seq_lens_encoder,
                      base_model_seq_lens_decoder,
                      base_model_step_idx,
                      base_model_stop_flags);
  WRAPPER_DUMP_PARAM3(
      ctx, base_model_is_block_step, base_model_draft_tokens, bsz);
  WRAPPER_DUMP_PARAM3(ctx, num_model_step, accept_tokens_len, draft_tokens_len);
  WRAPPER_DUMP_PARAM4(ctx,
                      input_ids_len,
                      base_model_draft_tokens_len,
                      pre_ids_len,
                      truncate_first_token);
  WRAPPER_DUMP_PARAM2(ctx, splitwise_prefill, kvcache_scheduler_v1);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * accept_tokens_len, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * input_ids_len, input_ids);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * draft_tokens_len, draft_tokens);
  WRAPPER_CHECK_PTR(
      ctx, int64_t, bsz * base_model_draft_tokens_len, base_model_draft_tokens);

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
                       is_block_step,
                       batch_drop,
                       pre_ids,
                       accept_tokens,
                       accept_num,
                       base_model_seq_lens_this_time,
                       base_model_seq_lens_encoder,
                       base_model_seq_lens_decoder,
                       base_model_step_idx,
                       base_model_stop_flags,
                       base_model_is_block_step,
                       base_model_draft_tokens,
                       bsz,
                       num_model_step,
                       accept_tokens_len,
                       draft_tokens_len,
                       input_ids_len,
                       base_model_draft_tokens_len,
                       pre_ids_len,
                       truncate_first_token,
                       splitwise_prefill,
                       kvcache_scheduler_v1);
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
                        is_block_step,
                        batch_drop,
                        pre_ids,
                        accept_tokens,
                        accept_num,
                        base_model_seq_lens_this_time,
                        base_model_seq_lens_encoder,
                        base_model_seq_lens_decoder,
                        base_model_step_idx,
                        base_model_stop_flags,
                        base_model_is_block_step,
                        base_model_draft_tokens,
                        bsz,
                        num_model_step,
                        accept_tokens_len,
                        draft_tokens_len,
                        input_ids_len,
                        base_model_draft_tokens_len,
                        pre_ids_len,
                        truncate_first_token,
                        splitwise_prefill,
                        kvcache_scheduler_v1);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
