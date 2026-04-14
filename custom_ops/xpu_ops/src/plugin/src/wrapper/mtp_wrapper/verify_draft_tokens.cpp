// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <numeric>

#include "xpu/plugin.h"
#include "xpu/refactor/impl_public/wrapper_check.h"

namespace fd_xpu3 {
__attribute__((global)) void verify_draft_tokens(
    // Core I/O
    int64_t *step_output_ids,
    int *step_output_len,
    const int64_t *step_input_ids,  // draft tokens
    // Target model outputs (strategy-dependent interpretation)
    const int64_t
        *target_tokens,  // GREEDY:argmax, TARGET_MATCH:sampled, TOPP:unused
    // Candidate set for TOPP/GREEDY (TARGET_MATCH: unused)
    const int64_t *candidate_ids,
    const float *candidate_scores,
    const int *candidate_lens,
    // Sampling params
    const float *curand_states,  // nullptr for GREEDY/TARGET_MATCH
    const float *topp,
    // Metadata
    const bool *stop_flags,
    const int *seq_lens_encoder,
    const int *seq_lens_this_time,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *reasoning_status,
    // max_dec_len / step_idx for EOS/max-len detection (read-only)
    const int64_t *max_dec_len,
    const int64_t *step_idx,
    // Dimensions and config
    const int max_bsz,
    const int real_bsz,
    const int max_step_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    const int verify_strategy,  // 0=TOPP, 1=GREEDY, 2=TARGET_MATCH
    const bool reject_all,
    const bool accept_all);
}  // namespace fd_xpu3

namespace fastdeploy {
namespace plugin {

// ============================================================
// Phase 1 helpers — single-step draft token verification
// ============================================================

// Check if draft_token appears in the candidate set
static inline bool is_in(const int64_t *candidates,
                         const int64_t draft,
                         const int candidate_len) {
  for (int i = 0; i < candidate_len; i++) {
    if (draft == candidates[i]) {
      return true;
    }
  }
  return false;
}
// TOPP: draft in top-p filtered candidate set
static inline bool verify_one_topp(const int64_t *verify_tokens_row,
                                   int64_t draft_token,
                                   int actual_cand_len) {
  return is_in(verify_tokens_row, draft_token, actual_cand_len);
}

// GREEDY / TARGET_MATCH: exact single-token match
static inline bool verify_one_match(int64_t target_token, int64_t draft_token) {
  return target_token == draft_token;
}

static inline bool is_in_end(const int64_t id,
                             const int64_t *end_ids,
                             int length) {
  bool flag = false;
  for (int i = 0; i < length; i++) {
    if (id == end_ids[i]) {
      return true;
    }
  }
  return flag;
}

// ============================================================
// VerifyContext — per-batch mutable state + accept helpers.
// Eliminates repeated EOS/max_dec_len check and output write
// patterns across Phase 1 and Phase 2.
// ============================================================
struct VerifyContext {
  // Immutable per-batch (set once at kernel entry)
  int bid;
  int max_step_tokens;
  int end_length;
  const int64_t *end_tokens;
  const int64_t *max_dec_len;
  const int64_t *step_input_ids_now;
  int64_t *step_output_ids;

  // Mutable per-batch state
  int64_t cur_step_idx;
  int output_len_now;
  bool stopped;

  // Emit a token at position `pos` to output in Phase 1.
  // Performs: step_idx check, EOS detection, token replacement, output write.
  // Returns true if this sequence should stop (EOS or max_dec_len hit).
  bool emit_token(int pos, int64_t token) {
    cur_step_idx++;
    bool is_eos = is_in_end(token, end_tokens, end_length);
    bool max_len_hit = (cur_step_idx >= max_dec_len[bid]);
    if ((is_eos || max_len_hit) && !is_eos) {
      token = end_tokens[0];
    }
    step_output_ids[bid * max_step_tokens + pos] = token;
    output_len_now++;
    if (is_eos || max_len_hit) {
      stopped = true;
      return true;
    }
    return false;
  }

  // Emit the final token at position `pos` in Phase 2.
  // Same EOS/max_dec_len logic. Increments output_len_now since
  // Phase 2 produces one additional token.
  void emit_final_token(int pos, int64_t token) {
    cur_step_idx++;
    bool is_eos = is_in_end(token, end_tokens, end_length);
    bool max_len_hit = (cur_step_idx >= max_dec_len[bid]);
    if ((is_eos || max_len_hit) && !is_eos) {
      token = end_tokens[0];
    }
    step_output_ids[bid * max_step_tokens + pos] = token;
    output_len_now++;
  }

  // TOPP-only: verify_window bulk-accept fallback.
  //
  // When draft token is NOT in top-p set but IS the top-2 token,
  // check verify_window consecutive positions for top-1 match.
  // If all match, bulk-accept from position i through ii.
  //
  // Returns the new loop position (i) after handling.
  // Sets *rejected=true if fallback was not triggered (caller should break).
  int try_verify_window_fallback(int i,
                                 bool *rejected,
                                 const int64_t *verify_tokens_now,
                                 int seq_len_this_time,
                                 int max_candidate_len,
                                 int verify_window) {
    int ii = i;
    if (max_candidate_len >= 2 &&
        verify_tokens_now[ii * max_candidate_len + 1] ==
            step_input_ids_now[ii + 1]) {
      // top-2 matches — scan verify_window consecutive top-1 matches
      int j = 0;
      ii += 1;
      for (; j < verify_window && ii < seq_len_this_time - 1; j++, ii++) {
        if (verify_tokens_now[ii * max_candidate_len] !=
            step_input_ids_now[ii + 1]) {
          break;
        }
      }
      if (j >= verify_window) {
        // Bulk accept all tokens from i to ii
        for (; i < ii; i++) {
          if (emit_token(i, step_input_ids_now[i + 1])) return i;
        }
        return i;  // continue outer loop from position ii
      }
    }
    // Fallback not triggered or insufficient window — reject
    *rejected = true;
    return i;
  }
};

static int64_t topp_sampling_kernel(const int64_t *candidate_ids,
                                    const float *candidate_scores,
                                    const float *dev_curand_states,
                                    const int candidate_len,
                                    const float topp,
                                    int tid) {
  // const int tid = core_id();
  float sum_scores = 0.0f;
  float rand_top_p = *dev_curand_states * topp;
  for (int i = 0; i < candidate_len; i++) {
    // printf("debug cpu sample i:%d scores:%f,ids:%ld
    // rand_top_p:%f,candidate_len:%d\n",
    // i,candidate_scores[i],candidate_ids[i],rand_top_p,candidate_len);
    sum_scores += candidate_scores[i];
    if (rand_top_p <= sum_scores) {
      return candidate_ids[i];
    }
  }
  return candidate_ids[0];
}

static int cpu_wrapper(
    api::Context *ctx,
    // Core I/O
    int64_t *step_output_ids,
    int *step_output_len,
    const int64_t *step_input_ids,  // draft tokens
    // Target model outputs (strategy-dependent interpretation)
    const int64_t
        *target_tokens,  // GREEDY:argmax, TARGET_MATCH:sampled, TOPP:unused
    // Candidate set for TOPP/GREEDY (TARGET_MATCH: unused)
    const int64_t *candidate_ids,
    const float *candidate_scores,
    const int *candidate_lens,
    // Sampling params
    const float *curand_states,  // nullptr for GREEDY/TARGET_MATCH
    const float *topp,
    // Metadata
    const bool *stop_flags,
    const int *seq_lens_encoder,
    const int *seq_lens_this_time,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *reasoning_status,
    // max_dec_len / step_idx for EOS/max-len detection (read-only)
    const int64_t *max_dec_len,
    const int64_t *step_idx,
    // Dimensions and config
    const int max_bsz,
    const int real_bsz,
    const int max_step_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    const int verify_strategy,  // 0=TOPP, 1=GREEDY, 2=TARGET_MATCH
    const bool reject_all,
    const bool accept_all) {
  for (int bid = 0; bid < max_bsz; bid++) {
    step_output_len[bid] = 0;

    if (bid >= real_bsz || is_block_step[bid] || stop_flags[bid]) continue;

    const int start_token_id = cu_seqlens_q_output[bid];
    // Pointers are strategy-dependent (may be nullptr for unused params)
    auto *candidate_ids_now =
        candidate_ids ? candidate_ids + start_token_id * max_candidate_len
                      : nullptr;
    auto *candidate_scores_now =
        candidate_scores ? candidate_scores + start_token_id * max_candidate_len
                         : nullptr;
    auto *candidate_lens_now =
        candidate_lens ? candidate_lens + start_token_id : nullptr;
    auto *target_tokens_now =
        target_tokens ? target_tokens + start_token_id : nullptr;

    // Initialize per-batch verification context
    VerifyContext v_ctx;
    v_ctx.bid = bid;
    v_ctx.max_step_tokens = max_step_tokens;
    v_ctx.end_length = end_length;
    v_ctx.end_tokens = end_tokens;
    v_ctx.max_dec_len = max_dec_len;
    v_ctx.step_input_ids_now = step_input_ids + bid * max_step_tokens;
    v_ctx.step_output_ids = step_output_ids;
    v_ctx.cur_step_idx = step_idx[bid];
    v_ctx.output_len_now = 0;
    v_ctx.stopped = false;

    // ======== Phase 1: Verify draft tokens ========
    int i = 0;
    for (; i < seq_lens_this_time[bid] - 1; i++) {
      // Early exit conditions: reject-all, prefill, reasoning
      if (reject_all || seq_lens_encoder[bid] != 0 ||
          reasoning_status[bid] == 1) {
        break;
      }

      // Accept-all override (debug/warmup)
      if (accept_all) {
        if (v_ctx.emit_token(i, v_ctx.step_input_ids_now[i + 1])) break;
        continue;
      }

      // Strategy dispatch
      bool accepted = false;
      switch (verify_strategy) {
        case 0: {  // TOPP
          auto actual_cand_len = candidate_lens_now[i] > max_candidate_len
                                     ? max_candidate_len
                                     : candidate_lens_now[i];
          accepted = verify_one_topp(candidate_ids_now + i * max_candidate_len,
                                     v_ctx.step_input_ids_now[i + 1],
                                     actual_cand_len);
          if (!accepted) {
            bool rejected = false;
            i = v_ctx.try_verify_window_fallback(i,
                                                 &rejected,
                                                 candidate_ids_now,
                                                 seq_lens_this_time[bid],
                                                 max_candidate_len,
                                                 verify_window);
            if (v_ctx.stopped || rejected) goto phase1_done;
            continue;  // bulk accept succeeded, continue from new i
          }
          break;
        }
        case 1:  // GREEDY
        case 2:  // TARGET_MATCH
          accepted = verify_one_match(target_tokens_now[i],
                                      v_ctx.step_input_ids_now[i + 1]);
          break;
      }

      if (accepted) {
        if (v_ctx.emit_token(i, v_ctx.step_input_ids_now[i + 1])) break;
      } else {
        break;  // reject
      }
    }
  phase1_done:

    // ======== Phase 2: Output token for rejected/last position ========
    if (!v_ctx.stopped) {
      int64_t output_token = 0;
      switch (verify_strategy) {
        case 0: {  // TOPP — stochastic sampling from candidate set
          auto actual_cand_len = candidate_lens_now[i] > max_candidate_len
                                     ? max_candidate_len
                                     : candidate_lens_now[i];
          output_token =
              topp_sampling_kernel(candidate_ids_now + i * max_candidate_len,
                                   candidate_scores_now + i * max_candidate_len,
                                   curand_states + bid,
                                   actual_cand_len,
                                   topp[bid],
                                   bid);
          break;
        }
        case 1:  // GREEDY — deterministic argmax from target_tokens
        case 2:  // TARGET_MATCH — target model's sampled token
          output_token = target_tokens_now[i];
          break;
      }
      v_ctx.emit_final_token(i, output_token);
    }
    step_output_len[bid] = v_ctx.output_len_now;
  }

  return api::SUCCESS;
}

static int xpu3_wrapper(
    api::Context *ctx,
    // Core I/O
    int64_t *step_output_ids,
    int *step_output_len,
    const int64_t *step_input_ids,  // draft tokens
    // Target model outputs (strategy-dependent interpretation)
    const int64_t
        *target_tokens,  // GREEDY:argmax, TARGET_MATCH:sampled, TOPP:unused
    // Candidate set for TOPP/GREEDY (TARGET_MATCH: unused)
    const int64_t *candidate_ids,
    const float *candidate_scores,
    const int *candidate_lens,
    // Sampling params
    const float *curand_states,  // nullptr for GREEDY/TARGET_MATCH
    const float *topp,
    // Metadata
    const bool *stop_flags,
    const int *seq_lens_encoder,
    const int *seq_lens_this_time,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *reasoning_status,
    // max_dec_len / step_idx for EOS/max-len detection (read-only)
    const int64_t *max_dec_len,
    const int64_t *step_idx,
    // Dimensions and config
    const int max_bsz,
    const int real_bsz,
    const int max_step_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    const int verify_strategy,  // 0=TOPP, 1=GREEDY, 2=TARGET_MATCH
    const bool reject_all,
    const bool accept_all) {
  using XPU_INT64 = typename api::XPUIndexType<int64_t>::type;
  int32_t ret_xre =
      fd_xpu3::verify_draft_tokens<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64 *>(step_output_ids),
          step_output_len,
          reinterpret_cast<const XPU_INT64 *>(step_input_ids),
          reinterpret_cast<const XPU_INT64 *>(target_tokens),
          reinterpret_cast<const XPU_INT64 *>(candidate_ids),
          candidate_scores,
          candidate_lens,
          curand_states,
          topp,
          stop_flags,
          seq_lens_encoder,
          seq_lens_this_time,
          reinterpret_cast<const XPU_INT64 *>(end_tokens),
          is_block_step,
          cu_seqlens_q_output,
          reasoning_status,
          reinterpret_cast<const XPU_INT64 *>(max_dec_len),
          reinterpret_cast<const XPU_INT64 *>(step_idx),
          max_bsz,
          real_bsz,
          max_step_tokens,
          end_length,
          max_seq_len,
          max_candidate_len,
          verify_window,
          verify_strategy,
          reject_all,
          accept_all);
  KERNEL_ASSERT_SUCCESS(ctx, ret_xre);
  return api::SUCCESS;
}

int verify_draft_tokens(
    api::Context *ctx,
    // Core I/O
    int64_t *step_output_ids,
    int *step_output_len,
    const int64_t *step_input_ids,  // draft tokens
    // Target model outputs (strategy-dependent interpretation)
    const int64_t
        *target_tokens,  // GREEDY:argmax, TARGET_MATCH:sampled, TOPP:unused
    // Candidate set for TOPP/GREEDY (TARGET_MATCH: unused)
    const int64_t *candidate_ids,
    const float *candidate_scores,
    const int *candidate_lens,
    // Sampling params
    const float *curand_states,  // nullptr for GREEDY/TARGET_MATCH
    const float *topp,
    // Metadata
    const bool *stop_flags,
    const int *seq_lens_encoder,
    const int *seq_lens_this_time,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *reasoning_status,
    // max_dec_len / step_idx for EOS/max-len detection (read-only)
    const int64_t *max_dec_len,
    const int64_t *step_idx,
    // Dimensions and config
    const int max_bsz,
    const int real_bsz,
    const int max_step_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    const int verify_strategy,  // 0=TOPP, 1=GREEDY, 2=TARGET_MATCH
    const bool reject_all,
    const bool accept_all) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "verify_draft_tokens", int64_t);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_output_ids,
                      step_output_len,
                      step_input_ids,
                      target_tokens,
                      candidate_ids,
                      candidate_scores);
  WRAPPER_DUMP_PARAM6(ctx,
                      candidate_lens,
                      curand_states,
                      topp,
                      stop_flags,
                      seq_lens_encoder,
                      seq_lens_this_time);

  WRAPPER_DUMP_PARAM6(ctx,
                      end_tokens,
                      is_block_step,
                      cu_seqlens_q_output,
                      reasoning_status,
                      max_dec_len,
                      step_idx);

  WRAPPER_DUMP_PARAM6(ctx,
                      max_bsz,
                      real_bsz,
                      max_step_tokens,
                      end_length,
                      max_seq_len,
                      max_candidate_len);

  WRAPPER_DUMP_PARAM4(
      ctx, verify_window, verify_strategy, reject_all, accept_all);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * max_step_tokens, step_output_ids);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * max_step_tokens, step_input_ids);
  // len(target_tokens) = cu_seqlens_q_output[-1]
  WRAPPER_CHECK_PTR_OR_NULL(ctx, int64_t, real_bsz, target_tokens);
  WRAPPER_CHECK_PTR_OR_NULL(ctx, int64_t, real_bsz, candidate_lens);
  WRAPPER_CHECK_PTR_OR_NULL(
      ctx, int64_t, real_bsz * max_candidate_len, candidate_ids);
  WRAPPER_CHECK_PTR_OR_NULL(
      ctx, float, real_bsz *max_candidate_len, candidate_scores);

  WRAPPER_CHECK_PTR(ctx, float, real_bsz, curand_states);
  WRAPPER_CHECK_PTR(ctx, float, real_bsz, topp);
  WRAPPER_CHECK_PTR(ctx, bool, real_bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int64_t, end_length, end_tokens);

  WRAPPER_CHECK_PTR(ctx, bool, real_bsz, is_block_step);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, cu_seqlens_q_output);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, reasoning_status);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz, max_dec_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz, step_idx);
  // param check sm size limit
  WRAPPER_ASSERT_GT(ctx, real_bsz, 0);
  WRAPPER_ASSERT_LE(ctx, real_bsz, 1024);
  WRAPPER_ASSERT_LE(ctx, real_bsz * max_candidate_len, 2048);
  WRAPPER_ASSERT_LE(ctx, verify_window * max_candidate_len, 128);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, step_output_len);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       step_output_ids,
                       step_output_len,
                       step_input_ids,
                       target_tokens,
                       candidate_ids,
                       candidate_scores,
                       candidate_lens,

                       curand_states,
                       topp,
                       stop_flags,
                       seq_lens_encoder,
                       seq_lens_this_time,
                       end_tokens,
                       is_block_step,
                       cu_seqlens_q_output,
                       reasoning_status,
                       max_dec_len,
                       step_idx,
                       max_bsz,
                       real_bsz,
                       max_step_tokens,
                       end_length,
                       max_seq_len,
                       max_candidate_len,
                       verify_window,
                       verify_strategy,
                       reject_all,
                       accept_all);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        step_output_ids,
                        step_output_len,
                        step_input_ids,
                        target_tokens,
                        candidate_ids,
                        candidate_scores,
                        candidate_lens,
                        curand_states,
                        topp,
                        stop_flags,
                        seq_lens_encoder,
                        seq_lens_this_time,
                        end_tokens,
                        is_block_step,
                        cu_seqlens_q_output,
                        reasoning_status,
                        max_dec_len,
                        step_idx,
                        max_bsz,
                        real_bsz,
                        max_step_tokens,
                        end_length,
                        max_seq_len,
                        max_candidate_len,
                        verify_window,
                        verify_strategy,
                        reject_all,
                        accept_all);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace fastdeploy
