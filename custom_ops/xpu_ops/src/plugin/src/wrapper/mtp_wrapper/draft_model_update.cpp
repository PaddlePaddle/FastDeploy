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
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace xpu3 {
namespace plugin {
__attribute__((global)) void draft_model_update(
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
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

bool is_in_end(int64_t token, const int64_t* end_ids, int end_ids_len) {
  for (int i = 0; i < end_ids_len; ++i) {
    if (end_ids[i] == token) {
      return true;
    }
  }
  return false;
}

static int cpu_wrapper(Context* ctx,
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
                       const bool prefill_one_step_stop) {
  int64_t stop_sum = 0;

  // 遍历所有batch
  for (int tid = 0; tid < bsz; ++tid) {
    auto* draft_token_now = draft_tokens + tid * max_draft_token;
    auto* pre_ids_now = pre_ids + tid * pre_id_length;
    auto* base_model_draft_tokens_now =
        base_model_draft_tokens + tid * max_base_model_draft_token;
    const int next_tokens_start_id =
        tid * max_seq_len - output_cum_offsets[tid];
    auto* next_tokens_start = inter_next_tokens + next_tokens_start_id;
    auto seq_len_this_time = seq_lens_this_time[tid];
    auto seq_len_encoder = seq_lens_encoder[tid];
    auto seq_len_decoder = seq_lens_decoder[tid];

    int64_t stop_flag_now_int = 0;

    // 1. update step_idx && seq_lens_dec
    if (!stop_flags[tid]) {
      int64_t token_this_time = -1;
      // decoder step
      if (seq_len_decoder > 0 && seq_len_encoder <= 0) {
        seq_lens_decoder[tid] += seq_len_this_time;
        token_this_time = next_tokens_start[seq_len_this_time - 1];
        draft_token_now[0] = next_tokens_start[seq_len_this_time - 1];
        base_model_draft_tokens_now[substep + 1] = token_this_time;
        for (int i = 0; i < seq_len_this_time; ++i) {
          pre_ids_now[step_idx[tid] + 1 + i] = next_tokens_start[i];
        }
        step_idx[tid] += seq_len_this_time;

      } else {
        token_this_time = next_tokens_start[0];
        seq_lens_decoder[tid] = seq_len_encoder + seq_len_decoder;
        seq_lens_encoder[tid] = 0;
        pre_ids_now[1] = token_this_time;
        step_idx[tid] += 1;
        draft_token_now[0] = token_this_time;
        base_model_draft_tokens_now[substep + 1] = token_this_time;
      }

      // multi_end
      if (is_in_end(token_this_time, end_ids, end_ids_len) ||
          prefill_one_step_stop) {
        stop_flags[tid] = true;
        stop_flag_now_int = 1;
        // max_dec_len
      } else if (step_idx[tid] >= max_dec_len[tid]) {
        stop_flags[tid] = true;
        draft_token_now[seq_len_this_time - 1] = end_ids[0];
        base_model_draft_tokens_now[substep + 1] = end_ids[0];
        stop_flag_now_int = 1;
      }

    } else {
      draft_token_now[0] = -1;
      base_model_draft_tokens_now[substep + 1] = -1;
      stop_flag_now_int = 1;
    }

    // 2. set end
    if (!stop_flags[tid]) {
      seq_lens_this_time[tid] = 1;
    } else {
      seq_lens_this_time[tid] = 0;
      seq_lens_encoder[tid] = 0;
    }

    stop_sum += stop_flag_now_int;
  }

  // 等价于CUDA中的BlockReduce求和
  not_need_stop[0] = stop_sum < bsz;
  return SUCCESS;
}

static int xpu2or3_wrapper(Context* ctx,
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
                           const bool prefill_one_step_stop) {
  ctx_guard RAII_GUARD(ctx);
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::draft_model_update<<<1, 64, ctx->xpu_stream>>>(
      reinterpret_cast<const XPU_INT64*>(inter_next_tokens),
      reinterpret_cast<XPU_INT64*>(draft_tokens),
      reinterpret_cast<XPU_INT64*>(pre_ids),
      seq_lens_this_time,
      seq_lens_encoder,
      seq_lens_decoder,
      reinterpret_cast<XPU_INT64*>(step_idx),
      output_cum_offsets,
      stop_flags,
      not_need_stop,
      reinterpret_cast<const XPU_INT64*>(max_dec_len),
      reinterpret_cast<const XPU_INT64*>(end_ids),
      reinterpret_cast<XPU_INT64*>(base_model_draft_tokens),
      bsz,
      max_draft_token,
      pre_id_length,
      max_base_model_draft_token,
      end_ids_len,
      max_seq_len,
      substep,
      prefill_one_step_stop);

  return api::SUCCESS;
}

int draft_model_update(Context* ctx,
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
                       const bool prefill_one_step_stop) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "draft_model_update", int);
  WRAPPER_DUMP_PARAM6(ctx,
                      inter_next_tokens,
                      draft_tokens,
                      pre_ids,
                      seq_lens_this_time,
                      seq_lens_encoder,
                      seq_lens_decoder);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_idx,
                      output_cum_offsets,
                      stop_flags,
                      not_need_stop,
                      max_dec_len,
                      end_ids);
  WRAPPER_DUMP_PARAM6(ctx,
                      base_model_draft_tokens,
                      bsz,
                      max_draft_token,
                      pre_id_length,
                      max_base_model_draft_token,
                      end_ids_len);
  WRAPPER_DUMP_PARAM3(ctx, max_seq_len, substep, prefill_one_step_stop);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * max_seq_len, inter_next_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * max_draft_token, draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz * pre_id_length, pre_ids);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz, step_idx);
  WRAPPER_CHECK_PTR(ctx, int, bsz, output_cum_offsets);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, bool, 1, not_need_stop);
  WRAPPER_CHECK_PTR(ctx, int64_t, bsz, max_dec_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, end_ids_len, end_ids);
  WRAPPER_CHECK_PTR(
      ctx, int64_t, bsz * max_base_model_draft_token, base_model_draft_tokens);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       inter_next_tokens,
                       draft_tokens,
                       pre_ids,
                       seq_lens_this_time,
                       seq_lens_encoder,
                       seq_lens_decoder,
                       step_idx,
                       output_cum_offsets,
                       stop_flags,
                       not_need_stop,
                       max_dec_len,
                       end_ids,
                       base_model_draft_tokens,
                       bsz,
                       max_draft_token,
                       pre_id_length,
                       max_base_model_draft_token,
                       end_ids_len,
                       max_seq_len,
                       substep,
                       prefill_one_step_stop);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                           inter_next_tokens,
                           draft_tokens,
                           pre_ids,
                           seq_lens_this_time,
                           seq_lens_encoder,
                           seq_lens_decoder,
                           step_idx,
                           output_cum_offsets,
                           stop_flags,
                           not_need_stop,
                           max_dec_len,
                           end_ids,
                           base_model_draft_tokens,
                           bsz,
                           max_draft_token,
                           pre_id_length,
                           max_base_model_draft_token,
                           end_ids_len,
                           max_seq_len,
                           substep,
                           prefill_one_step_stop);
  }

  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
