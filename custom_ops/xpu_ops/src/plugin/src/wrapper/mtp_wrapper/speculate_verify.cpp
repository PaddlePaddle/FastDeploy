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

namespace xpu3 {
namespace plugin {
typedef uint32_t curandStatePhilox4_32_10_t;

template <bool ENABLE_TOPP, bool USE_TOPK>
__attribute__((global)) void speculate_verify(
    int64_t *accept_tokens,
    int *accept_num,
    int64_t *step_idx,
    bool *stop_flags,
    const int *seq_lens_encoder,
    const int *seq_lens_decoder,
    const int64_t *draft_tokens,
    const int *actual_draft_token_nums,
    const float *dev_curand_states,
    const float *topp,
    const int *seq_lens_this_time,
    const int64_t *verify_tokens,
    const float *verify_scores,
    const int64_t *max_dec_len,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *output_cum_offsets,
    const int *actual_candidate_len,
    const int real_bsz,
    const int max_draft_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    const bool prefill_one_step_stop,
    const bool benchmark_mode);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

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

static inline unsigned int xorwow(unsigned int &state) {  // NOLINT
  state ^= state >> 7;
  state ^= state << 9;
  state ^= state >> 13;
  return state;
}

typedef uint32_t curandStatePhilox4_32_10_t;

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

template <bool ENABLE_TOPP, bool USE_TOPK>
static int cpu_wrapper(Context *ctx,
                       int64_t *accept_tokens,
                       int *accept_num,
                       int64_t *step_idx,
                       bool *stop_flags,
                       const int *seq_lens_encoder,
                       const int *seq_lens_decoder,
                       const int64_t *draft_tokens,
                       const int *actual_draft_token_nums,
                       const float *dev_curand_states,
                       const float *topp,
                       const int *seq_lens_this_time,
                       const int64_t *verify_tokens,
                       const float *verify_scores,
                       const int64_t *max_dec_len,
                       const int64_t *end_tokens,
                       const bool *is_block_step,
                       const int *output_cum_offsets,
                       const int *actual_candidate_len,
                       const int real_bsz,
                       const int max_draft_tokens,
                       const int end_length,
                       const int max_seq_len,
                       const int max_candidate_len,
                       const int verify_window,
                       const bool prefill_one_step_stop,
                       const bool benchmark_mode) {
  for (int bid = 0; bid < real_bsz; ++bid) {
    // verify and set stop flags
    int accept_num_now = 1;
    int stop_flag_now_int = 0;

    if (!(is_block_step[bid] || bid >= real_bsz)) {
      const int start_token_id = bid * max_seq_len - output_cum_offsets[bid];
      // printf("debug cpu bid:%d,start_token_id:%d\n",bid, start_token_id);
      // printf("bid %d\n", bid);

      if (stop_flags[bid]) {
        stop_flag_now_int = 1;
      } else {  // 这里prefill阶段也会进入，但是因为draft
                // tokens会置零，因此会直接到最后的采样阶段
        auto *verify_tokens_now =
            verify_tokens + start_token_id * max_candidate_len;
        auto *draft_tokens_now = draft_tokens + bid * max_draft_tokens;
        auto *actual_candidate_len_now = actual_candidate_len + start_token_id;

        int i = 0;
        // printf("seq_lens_this_time[%d]-1: %d \n",bid,
        // seq_lens_this_time[bid]-1);
        for (; i < seq_lens_this_time[bid] - 1; i++) {
          if (benchmark_mode) {
            break;
          }
          if (seq_lens_encoder[bid] != 0) {
            break;
          }
          if (USE_TOPK) {
            if (verify_tokens_now[i * max_candidate_len] ==
                draft_tokens_now[i + 1]) {
              // accept_num_now++;
              step_idx[bid]++;
              auto accept_token = draft_tokens_now[i + 1];
              // printf("[USE_TOPK] bid %d Top 1 verify write accept
              // %d is %lld\n", bid, i, accept_token);
              accept_tokens[bid * max_draft_tokens + i] = accept_token;
              if (is_in_end(accept_token, end_tokens, end_length) ||
                  step_idx[bid] >= max_dec_len[bid]) {
                stop_flags[bid] = true;
                stop_flag_now_int = 1;
                if (step_idx[bid] >= max_dec_len[bid])
                  accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
                // printf("[USE_TOPK] bid %d Top 1 verify write
                // accept %d is %lld\n", bid, i, accept_token);
                break;
              } else {
                accept_num_now++;
              }
            } else {
              break;
            }
          } else {
            auto actual_candidate_len_value =
                actual_candidate_len_now[i] > max_candidate_len
                    ? max_candidate_len
                    : actual_candidate_len_now[i];
            if (is_in(verify_tokens_now + i * max_candidate_len,
                      draft_tokens_now[i + 1],
                      actual_candidate_len_value)) {
              // Top P verify
              // accept_num_now++;
              step_idx[bid]++;
              auto accept_token = draft_tokens_now[i + 1];
              accept_tokens[bid * max_draft_tokens + i] = accept_token;

              if (is_in_end(accept_token, end_tokens, end_length) ||
                  step_idx[bid] >= max_dec_len[bid]) {
                stop_flags[bid] = true;
                stop_flag_now_int = 1;
                if (step_idx[bid] >= max_dec_len[bid])
                  accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
                // printf("bid %d Top P verify write accept %d is
                // %lld\n", bid, i, accept_token);
                break;
              } else {
                accept_num_now++;
              }
            } else {
              // TopK verify
              int ii = i;
              if (max_candidate_len >= 2 &&
                  verify_tokens_now[ii * max_candidate_len + 1] ==
                      draft_tokens_now[ii + 1]) {  // top-2
                int j = 0;
                ii += 1;
                for (; j < verify_window && ii < seq_lens_this_time[bid] - 1;
                     j++, ii++) {
                  if (verify_tokens_now[ii * max_candidate_len] !=
                      draft_tokens_now[ii + 1]) {
                    break;
                  }
                }
                if (j >= verify_window) {  // accept all
                  accept_num_now += verify_window + 1;
                  step_idx[bid] += verify_window + 1;
                  for (; i < ii; i++) {
                    auto accept_token = draft_tokens_now[i + 1];
                    accept_tokens[bid * max_draft_tokens + i] = accept_token;
                    // printf("bid %d TopK verify write accept %dis "
                    // "%lld\n",bid,i,accept_token);
                    if (is_in_end(accept_token, end_tokens, end_length) ||
                        step_idx[bid] >= max_dec_len[bid]) {
                      stop_flags[bid] = true;
                      stop_flag_now_int = 1;
                      if (step_idx[bid] >= max_dec_len[bid])
                        accept_tokens[bid * max_draft_tokens + i] =
                            end_tokens[0];
                      // printf("bid %d TopK verify write accept %d is %lld\n",
                      // bid, i,end_tokens[0]);
                      accept_num_now--;
                      step_idx[bid]--;
                      break;
                    }
                  }
                }
              }
              break;
            }
          }
        }
        // sampling阶段
        // 第一种，draft_token[i+1]被拒绝，需要从verify_tokens_now[i]中选一个
        // 第二种，i == seq_lens_this_time[bid]-1,
        // 也是从verify_tokens_now[i]中选一个 但是停止的情况不算
        if (!stop_flag_now_int) {
          int64_t accept_token;
          const float *verify_scores_now =
              verify_scores + start_token_id * max_candidate_len;
          step_idx[bid]++;
          if (ENABLE_TOPP) {
            auto actual_candidate_len_value =
                actual_candidate_len_now[i] > max_candidate_len
                    ? max_candidate_len
                    : actual_candidate_len_now[i];

            accept_token =
                topp_sampling_kernel(verify_tokens_now + i * max_candidate_len,
                                     verify_scores_now + i * max_candidate_len,
                                     dev_curand_states + i,
                                     actual_candidate_len_value,
                                     topp[bid],
                                     bid);
          } else {
            accept_token = verify_tokens_now[i * max_candidate_len];
          }
          accept_tokens[bid * max_draft_tokens + i] = accept_token;
          if (prefill_one_step_stop) {
            stop_flags[bid] = true;
          }
          if (is_in_end(accept_token, end_tokens, end_length) ||
              step_idx[bid] >= max_dec_len[bid]) {
            stop_flags[bid] = true;
            stop_flag_now_int = 1;
            if (step_idx[bid] >= max_dec_len[bid])
              accept_tokens[bid * max_draft_tokens + i] = end_tokens[0];
          }
        }
        accept_num[bid] = accept_num_now;
      }
    }
  }
  return api::SUCCESS;
}

template <bool ENABLE_TOPP, bool USE_TOPK>
static int xpu3_wrapper(Context *ctx,
                        int64_t *accept_tokens,
                        int *accept_num,
                        int64_t *step_idx,
                        bool *stop_flags,
                        const int *seq_lens_encoder,
                        const int *seq_lens_decoder,
                        const int64_t *draft_tokens,
                        const int *actual_draft_token_nums,
                        const float *dev_curand_states,
                        const float *topp,
                        const int *seq_lens_this_time,
                        const int64_t *verify_tokens,
                        const float *verify_scores,
                        const int64_t *max_dec_len,
                        const int64_t *end_tokens,
                        const bool *is_block_step,
                        const int *output_cum_offsets,
                        const int *actual_candidate_len,
                        const int real_bsz,
                        const int max_draft_tokens,
                        const int end_length,
                        const int max_seq_len,
                        const int max_candidate_len,
                        const int verify_window,
                        const bool prefill_one_step_stop,
                        const bool benchmark_mode) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::speculate_verify<ENABLE_TOPP, USE_TOPK>
      <<<1, 64, ctx->xpu_stream>>>(
          reinterpret_cast<XPU_INT64 *>(accept_tokens),
          accept_num,
          reinterpret_cast<XPU_INT64 *>(step_idx),
          stop_flags,
          seq_lens_encoder,
          seq_lens_decoder,
          reinterpret_cast<const XPU_INT64 *>(draft_tokens),
          actual_draft_token_nums,
          dev_curand_states,
          topp,
          seq_lens_this_time,
          reinterpret_cast<const XPU_INT64 *>(verify_tokens),
          verify_scores,
          reinterpret_cast<const XPU_INT64 *>(max_dec_len),
          reinterpret_cast<const XPU_INT64 *>(end_tokens),
          is_block_step,
          output_cum_offsets,
          actual_candidate_len,
          real_bsz,
          max_draft_tokens,
          end_length,
          max_seq_len,
          max_candidate_len,
          verify_window,
          prefill_one_step_stop,
          benchmark_mode);
  return api::SUCCESS;
}
template <bool ENABLE_TOPP, bool USE_TOPK>
int speculate_verify(Context *ctx,
                     int64_t *accept_tokens,
                     int *accept_num,
                     int64_t *step_idx,
                     bool *stop_flags,
                     const int *seq_lens_encoder,
                     const int *seq_lens_decoder,
                     const int64_t *draft_tokens,
                     const int *actual_draft_token_nums,
                     const float *dev_curand_states,
                     const float *topp,
                     const int *seq_lens_this_time,
                     const int64_t *verify_tokens,
                     const float *verify_scores,
                     const int64_t *max_dec_len,
                     const int64_t *end_tokens,
                     const bool *is_block_step,
                     const int *output_cum_offsets,
                     const int *actual_candidate_len,
                     const int real_bsz,
                     const int max_draft_tokens,
                     const int end_length,
                     const int max_seq_len,
                     const int max_candidate_len,
                     const int verify_window,
                     const bool prefill_one_step_stop,
                     const bool benchmark_mode) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_verify", int64_t);
  WRAPPER_DUMP_PARAM3(ctx, accept_tokens, accept_num, step_idx);
  WRAPPER_DUMP_PARAM6(ctx,
                      stop_flags,
                      seq_lens_encoder,
                      seq_lens_decoder,
                      draft_tokens,
                      actual_draft_token_nums,
                      topp);
  WRAPPER_DUMP_PARAM5(ctx,
                      seq_lens_this_time,
                      verify_tokens,
                      verify_scores,
                      max_dec_len,
                      end_tokens);
  WRAPPER_DUMP_PARAM5(ctx,
                      is_block_step,
                      output_cum_offsets,
                      actual_candidate_len,
                      real_bsz,
                      max_draft_tokens);
  WRAPPER_DUMP_PARAM6(ctx,
                      end_length,
                      max_seq_len,
                      max_candidate_len,
                      verify_window,
                      prefill_one_step_stop,
                      benchmark_mode);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * max_draft_tokens, accept_tokens);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, accept_num);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz, step_idx);
  WRAPPER_CHECK_PTR(ctx, bool, real_bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_encoder);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz * max_draft_tokens, draft_tokens);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, actual_draft_token_nums);
  WRAPPER_CHECK_PTR(ctx, float, real_bsz, dev_curand_states);
  WRAPPER_CHECK_PTR(ctx, float, real_bsz, topp);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, seq_lens_this_time);
  // WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz, verify_tokens);
  // WRAPPER_CHECK_PTR(ctx, float, real_bsz, verify_scores);
  WRAPPER_CHECK_PTR(ctx, int64_t, real_bsz, max_dec_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, end_length, end_tokens);
  WRAPPER_CHECK_PTR(ctx, bool, real_bsz, is_block_step);
  WRAPPER_CHECK_PTR(ctx, int, real_bsz, output_cum_offsets);
  // WRAPPER_CHECK_PTR(ctx, int, real_bsz, actual_candidate_len);

  // param check sm size limit
  WRAPPER_ASSERT_GT(ctx, real_bsz, 0);
  WRAPPER_ASSERT_LE(ctx, real_bsz, 1024);
  WRAPPER_ASSERT_LE(ctx, real_bsz * max_candidate_len, 2048);
  WRAPPER_ASSERT_LE(ctx, verify_window * max_candidate_len, 128);
  // int sum = 0;
  // for (int i=0;i < real_bsz; i++){
  //     sum+= seq_lens_this_time[i];
  // }
  // WRAPPER_ASSERT_LE(ctx, sum * max_draft_tokens, 2048);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<ENABLE_TOPP, USE_TOPK>(ctx,
                                              accept_tokens,
                                              accept_num,
                                              step_idx,
                                              stop_flags,
                                              seq_lens_encoder,
                                              seq_lens_decoder,
                                              draft_tokens,
                                              actual_draft_token_nums,
                                              dev_curand_states,
                                              topp,
                                              seq_lens_this_time,
                                              verify_tokens,
                                              verify_scores,
                                              max_dec_len,
                                              end_tokens,
                                              is_block_step,
                                              output_cum_offsets,
                                              actual_candidate_len,
                                              real_bsz,
                                              max_draft_tokens,
                                              end_length,
                                              max_seq_len,
                                              max_candidate_len,
                                              verify_window,
                                              prefill_one_step_stop,
                                              benchmark_mode);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<ENABLE_TOPP, USE_TOPK>(ctx,
                                               accept_tokens,
                                               accept_num,
                                               step_idx,
                                               stop_flags,
                                               seq_lens_encoder,
                                               seq_lens_decoder,
                                               draft_tokens,
                                               actual_draft_token_nums,
                                               dev_curand_states,
                                               topp,
                                               seq_lens_this_time,
                                               verify_tokens,
                                               verify_scores,
                                               max_dec_len,
                                               end_tokens,
                                               is_block_step,
                                               output_cum_offsets,
                                               actual_candidate_len,
                                               real_bsz,
                                               max_draft_tokens,
                                               end_length,
                                               max_seq_len,
                                               max_candidate_len,
                                               verify_window,
                                               prefill_one_step_stop,
                                               benchmark_mode);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

#define INSTANTIATE_SPECULATE_VERIFY(ENABLE_TOPP, USE_TOPK)         \
  template int                                                      \
  baidu::xpu::api::plugin::speculate_verify<ENABLE_TOPP, USE_TOPK>( \
      baidu::xpu::api::Context *, /* xpu_ctx */                     \
      int64_t *,                  /* accept_tokens */               \
      int *,                      /* accept_num */                  \
      int64_t *,                  /* step_idx */                    \
      bool *,                     /* stop_flags */                  \
      const int *,                /* seq_lens_encoder */            \
      const int *,                /* seq_lens_decoder */            \
      const int64_t *,            /* draft_tokens */                \
      const int *,                /* actual_draft_token_nums */     \
      const float *,              /* dev_curand_states or topp */   \
      const float *,              /* topp or nullptr */             \
      const int *,                /* seq_lens_this_time */          \
      const int64_t *,            /* verify_tokens */               \
      const float *,              /* verify_scores */               \
      const int64_t *,            /* max_dec_len */                 \
      const int64_t *,            /* end_tokens */                  \
      const bool *,               /* is_block_step */               \
      const int *,                /* output_cum_offsets */          \
      const int *,                /* actual_candidate_len */        \
      int,                        /* real_bsz */                    \
      int,                        /* max_draft_tokens */            \
      int,                        /* end_length */                  \
      int,                        /* max_seq_len */                 \
      int,                        /* max_candidate_len */           \
      int,                        /* verify_window */               \
      bool,                                                         \
      bool); /* prefill_one_step_stop */

INSTANTIATE_SPECULATE_VERIFY(false, false)
INSTANTIATE_SPECULATE_VERIFY(false, true)
INSTANTIATE_SPECULATE_VERIFY(true, false)
INSTANTIATE_SPECULATE_VERIFY(true, true)

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
