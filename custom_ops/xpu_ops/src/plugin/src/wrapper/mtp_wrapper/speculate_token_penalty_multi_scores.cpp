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

template <typename T>
__attribute__((global)) void speculate_min_length_logits_process(
    T* logits,
    const int64_t* cur_len,
    const int64_t* min_len,
    const int64_t* eos_token_id,
    const int* output_padding_offset,
    const int* output_cum_offsets,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t end_length,
    const int64_t token_num,
    const int64_t max_seq_len);
__attribute__((global)) void speculate_update_repeat_times(
    const int64_t* pre_ids,
    const int64_t* cur_len,
    int* repeat_times,
    const int* output_padding_offset,
    const int64_t bs,
    const int64_t length,
    const int64_t length_id,
    const int64_t token_num,
    const int64_t max_seq_len);
template <typename T>
__attribute__((global)) void speculate_update_value_by_repeat_times(
    const int* repeat_times,
    const T* penalty_scores,
    const T* frequency_score,
    const T* presence_score,
    const float* temperatures,
    T* logits,
    const int* output_padding_offset,
    const int64_t bs,
    const int64_t length,
    const int64_t token_num,
    const int64_t max_seq_len);
template <typename T>
__attribute__((global)) void speculate_update_value_by_repeat_times_simd(
    const int* repeat_times,
    const T* penalty_scores,
    const T* frequency_score,
    const T* presence_score,
    const float* temperatures,
    T* logits,
    const int* output_padding_offset,
    const int64_t bs,
    const int64_t length,
    const int64_t token_num,
    const int64_t max_seq_len);
template <typename T>
__attribute__((global)) void speculate_ban_bad_words(
    T* logits,
    const int64_t* bad_words_list,
    const int* output_padding_offset,
    const int64_t bs,
    const int64_t length,
    const int64_t bad_words_length,
    const int64_t token_num,
    const int64_t max_seq_len);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

void update_repeat_times_cpu(const int64_t* pre_ids,
                             const int64_t* cur_len,
                             int* repeat_times,
                             const int* output_padding_offset,
                             const int64_t bs,
                             const int64_t length,
                             const int64_t length_id,
                             const int64_t token_num,
                             const int64_t max_seq_len) {
  for (int64_t i = 0; i < token_num; i++) {
    int64_t bi = (i + output_padding_offset[i]) / max_seq_len;
    if (bi < bs && cur_len[bi] >= 0) {
      for (int64_t j = 0; j < length_id; j++) {
        int64_t id = pre_ids[bi * length_id + j];
        if (id < 0) {
          break;
        } else if (id >= length) {
          continue;
        } else {
          repeat_times[i * length + id] += 1;
        }
      }
    }
  }
}

void ban_bad_words_cpu(float* logits,
                       const int64_t* bad_words_list,
                       const int* output_padding_offset,
                       const int64_t bs,
                       const int64_t length,
                       const int64_t bad_words_length,
                       const int64_t token_num,
                       const int64_t max_seq_len) {
  for (int64_t i = 0; i < token_num; i++) {
    int64_t bi = (i + output_padding_offset[i]) / max_seq_len;
    if (bi >= bs) {
      continue;
    }
    float* logits_now = logits + i * length;
    for (int64_t j = 0; j < bad_words_length; j++) {
      int64_t bad_words_token_id = bad_words_list[j];
      if (bad_words_token_id >= length || bad_words_token_id < 0) continue;
      logits_now[bad_words_token_id] = -1e10;
    }
  }
}

template <typename T>
static int cpu_wrapper(Context* ctx,
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
                       const int64_t max_seq_len) {
  std::vector<float> logitsfp32(token_num * length);
  std::vector<float> penalty_scoresfp32(bs);
  std::vector<float> frequency_scoresfp32(bs);
  std::vector<float> presence_scoresfp32(bs);
  std::vector<int> repeat_times_buffer(token_num * length, 0);
  int ret =
      api::cast<T, float>(ctx, logits, logitsfp32.data(), token_num * length);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::cast<T, float>(ctx, penalty_scores, penalty_scoresfp32.data(), bs);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret = api::cast<T, float>(
      ctx, frequency_scores, frequency_scoresfp32.data(), bs);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  ret =
      api::cast<T, float>(ctx, presence_scores, presence_scoresfp32.data(), bs);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);
  for (int64_t i = 0; i < token_num; i++) {
    int64_t bi = (i + output_padding_offset[i]) / max_seq_len;
    int64_t query_start_token_idx = bi * max_seq_len - output_cum_offsets[bi];
    if (bi < bs && cur_len[bi] >= 0 &&
        (cur_len[bi] + (i - query_start_token_idx) < min_len[bi])) {
      for (int64_t j = 0; j < end_length; j++) {
        logitsfp32[i * length + eos_token_id[j]] =
            std::is_same<T, float16>::value ? -1e4 : -1e10;
      }
    }
  }
  int* repeat_times = &(repeat_times_buffer[0]);
  update_repeat_times_cpu(pre_ids,
                          cur_len,
                          repeat_times,
                          output_padding_offset,
                          bs,
                          length,
                          length_id,
                          token_num,
                          max_seq_len);
  for (int64_t i = 0; i < token_num; i++) {
    int64_t bi = (i + output_padding_offset[i]) / max_seq_len;
    if (bi >= bs) {
      continue;
    }
    float alpha = penalty_scoresfp32[bi];
    float beta = frequency_scoresfp32[bi];
    float gamma = presence_scoresfp32[bi];
    float temperature = temperatures[bi];
    for (int64_t j = 0; j < length; j++) {
      int times = repeat_times[i * length + j];
      float logit_now = logitsfp32[i * length + j];
      if (times != 0) {
        logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
        logit_now = logit_now - times * beta - gamma;
      }
      logitsfp32[i * length + j] = logit_now / temperature;
    }
  }
  if (bad_words && length_bad_words > 0) {
    ban_bad_words_cpu(logitsfp32.data(),
                      bad_words,
                      output_padding_offset,
                      bs,
                      length,
                      length_bad_words,
                      token_num,
                      max_seq_len);
  }
  ret = api::cast<float, T>(ctx, logitsfp32.data(), logits, token_num * length);
  return ret;
}

template <typename T>
static int xpu3_wrapper(Context* ctx,
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
                        const int64_t max_seq_len) {
  api::ctx_guard RAII_GUARD(ctx);
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto min_length_logits_process_kernel =
      xpu3::plugin::speculate_min_length_logits_process<T>;
  auto update_repeat_times_kernel = xpu3::plugin::speculate_update_repeat_times;
  auto update_value_by_repeat_times_kernel =
      xpu3::plugin::speculate_update_value_by_repeat_times<T>;
  if (length % 16 == 0) {
    update_value_by_repeat_times_kernel =
        xpu3::plugin::speculate_update_value_by_repeat_times_simd<T>;
  }
  auto ban_bad_words_kernel = xpu3::plugin::speculate_ban_bad_words<T>;

  int* repeat_times = RAII_GUARD.alloc_l3_or_gm<int>(token_num * length);
  WRAPPER_ASSERT_WORKSPACE(ctx, repeat_times);
  int ret = api::constant<int>(ctx, repeat_times, token_num * length, 0);
  WRAPPER_ASSERT_SUCCESS(ctx, ret);

  update_repeat_times_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      reinterpret_cast<const XPU_INT64*>(pre_ids),
      reinterpret_cast<const XPU_INT64*>(cur_len),
      repeat_times,
      output_padding_offset,
      bs,
      length,
      length_id,
      token_num,
      max_seq_len);
  min_length_logits_process_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      logits,
      reinterpret_cast<const XPU_INT64*>(cur_len),
      reinterpret_cast<const XPU_INT64*>(min_len),
      reinterpret_cast<const XPU_INT64*>(eos_token_id),
      output_padding_offset,
      output_cum_offsets,
      bs,
      length,
      length_id,
      end_length,
      token_num,
      max_seq_len);
  update_value_by_repeat_times_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      repeat_times,
      penalty_scores,
      frequency_scores,
      presence_scores,
      temperatures,
      logits,
      output_padding_offset,
      bs,
      length,
      token_num,
      max_seq_len);

  if (bad_words && length_bad_words > 0) {
    ban_bad_words_kernel<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
        logits,
        reinterpret_cast<const XPU_INT64*>(bad_words),
        output_padding_offset,
        bs,
        length,
        length_bad_words,
        token_num,
        max_seq_len);
  }
  return api::SUCCESS;
}

template <typename T>
int speculate_token_penalty_multi_scores(Context* ctx,
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
                                         const int64_t max_seq_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_token_penalty_multi_scores", T);
  WRAPPER_DUMP_PARAM6(ctx,
                      pre_ids,
                      logits,
                      penalty_scores,
                      frequency_scores,
                      presence_scores,
                      temperatures);
  WRAPPER_DUMP_PARAM6(ctx,
                      cur_len,
                      min_len,
                      eos_token_id,
                      bad_words,
                      output_padding_offset,
                      output_cum_offsets);
  WRAPPER_DUMP_PARAM4(ctx, bs, length, length_id, end_length);
  WRAPPER_DUMP_PARAM3(ctx, length_bad_words, token_num, max_seq_len);
  WRAPPER_DUMP(ctx);
  // TODO(mayang02) shape check
  int64_t pre_ids_len = -1;
  int64_t logits_len = -1;
  int64_t penalty_scores_len = -1;
  int64_t frequency_scores_len = -1;
  int64_t presence_scores_len = -1;
  int64_t temperatures_len = -1;
  int64_t cur_len_len = -1;
  int64_t min_len_len = -1;
  int64_t eos_token_id_len = -1;
  int64_t bad_words_len = -1;
  int64_t output_padding_offset_len = -1;
  int64_t output_cum_offsets_len = -1;
  WRAPPER_ASSERT_LE(ctx, bs, 640);
  WRAPPER_CHECK_SHAPE(ctx, &pre_ids_len, {bs, length_id});
  WRAPPER_CHECK_SHAPE(ctx, &logits_len, {token_num, length});
  WRAPPER_CHECK_SHAPE(ctx, &penalty_scores_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &frequency_scores_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &presence_scores_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &temperatures_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &cur_len_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &min_len_len, {bs});
  WRAPPER_CHECK_SHAPE(ctx, &eos_token_id_len, {end_length});
  WRAPPER_CHECK_SHAPE(ctx, &bad_words_len, {length_bad_words});
  WRAPPER_CHECK_SHAPE(ctx, &output_padding_offset_len, {token_num});
  WRAPPER_CHECK_SHAPE(ctx, &output_cum_offsets_len, {bs});
  WRAPPER_CHECK_PTR(ctx, int64_t, pre_ids_len, pre_ids);
  WRAPPER_CHECK_PTR(ctx, T, logits_len, logits);
  WRAPPER_CHECK_PTR(ctx, T, penalty_scores_len, penalty_scores);
  WRAPPER_CHECK_PTR(ctx, T, frequency_scores_len, frequency_scores);
  WRAPPER_CHECK_PTR(ctx, T, presence_scores_len, presence_scores);
  WRAPPER_CHECK_PTR(ctx, float, temperatures_len, temperatures);
  WRAPPER_CHECK_PTR(ctx, int64_t, cur_len_len, cur_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, min_len_len, min_len);
  WRAPPER_CHECK_PTR(ctx, int64_t, eos_token_id_len, eos_token_id);
  WRAPPER_CHECK_PTR(ctx, int64_t, bad_words_len, bad_words);
  WRAPPER_CHECK_PTR(ctx, int, output_padding_offset_len, output_padding_offset);
  WRAPPER_CHECK_PTR(ctx, int, output_cum_offsets_len, output_cum_offsets);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(ctx,
                          pre_ids,
                          logits,
                          penalty_scores,
                          frequency_scores,
                          presence_scores,
                          temperatures,
                          cur_len,
                          min_len,
                          eos_token_id,
                          bad_words,
                          output_padding_offset,
                          output_cum_offsets,
                          bs,
                          length,
                          length_id,
                          end_length,
                          length_bad_words,
                          token_num,
                          max_seq_len);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(ctx,
                           pre_ids,
                           logits,
                           penalty_scores,
                           frequency_scores,
                           presence_scores,
                           temperatures,
                           cur_len,
                           min_len,
                           eos_token_id,
                           bad_words,
                           output_padding_offset,
                           output_cum_offsets,
                           bs,
                           length,
                           length_id,
                           end_length,
                           length_bad_words,
                           token_num,
                           max_seq_len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int speculate_token_penalty_multi_scores<float>(
    Context* ctx,
    const int64_t* pre_ids,
    float* logits,
    const float* penalty_scores,
    const float* frequency_scores,
    const float* presence_scores,
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
template int speculate_token_penalty_multi_scores<float16>(
    Context* ctx,
    const int64_t* pre_ids,
    float16* logits,
    const float16* penalty_scores,
    const float16* frequency_scores,
    const float16* presence_scores,
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
template int speculate_token_penalty_multi_scores<bfloat16>(
    Context* ctx,
    const int64_t* pre_ids,
    bfloat16* logits,
    const bfloat16* penalty_scores,
    const bfloat16* frequency_scores,
    const bfloat16* presence_scores,
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

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
