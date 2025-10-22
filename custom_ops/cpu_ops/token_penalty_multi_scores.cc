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

#include <paddle/extension.h>
#include <vector>
#include "paddle/phi/core/kernel_registry.h"

void min_length_logits_process(float *logits,
                               const int64_t *cur_len,
                               const int64_t *min_len,
                               const int64_t *eos_token_id,
                               const int64_t bs,
                               const int64_t length,
                               const int64_t end_length) {
  for (int bi = 0; bi < bs; ++bi) {
    if (cur_len[bi] < 0) {
      continue;
    }
    if (cur_len[bi] < min_len[bi]) {
      for (int i = 0; i < end_length; ++i) {
        logits[bi * length + eos_token_id[i]] = -1e10;
      }
    }
  }
}

void update_repeat_times(const int64_t *pre_ids,
                         const int64_t *cur_len,
                         int *repeat_times,
                         const int64_t bs,
                         const int64_t length,
                         const int64_t length_id) {
  for (int bi = 0; bi < bs; ++bi) {
    if (cur_len[bi] < 0) {
      continue;
    }
    const int64_t *pre_ids_now = pre_ids + bi * length_id;
    int *repeat_times_now = repeat_times + bi * length;
    for (int i = 0; i < length_id; i++) {
      int64_t id = pre_ids_now[i];
      if (id < 0) {
        break;
      }
      repeat_times_now[id] += 1;
    }
  }
}

void update_value_by_repeat_times(const int *repeat_times,
                                  const float *penalty_scores,
                                  const float *frequency_score,
                                  const float *presence_score,
                                  const float *temperatures,
                                  float *logits,
                                  const int64_t bs,
                                  const int64_t length) {
  for (int bi = 0; bi < bs; ++bi) {
    float *logits_now = logits + bi * length;
    const int *repeat_times_now = repeat_times + bi * length;
    float alpha = static_cast<float>(penalty_scores[bi]);
    float beta = static_cast<float>(frequency_score[bi]);
    float gamma = static_cast<float>(presence_score[bi]);
    for (int i = 0; i < length; ++i) {
      int times = repeat_times_now[i];
      float logit_now = static_cast<float>(logits_now[i]);
      if (times == 0) {
        logits_now[i] = static_cast<float>(logit_now / temperatures[bi]);
      }
      logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
      logits_now[i] = static_cast<float>(logit_now - times * beta - gamma);
    }
  }
}

void ban_bad_words(float *logits,
                   const int64_t *bad_words_list,
                   const int64_t bs,
                   const int64_t length,
                   const int64_t bad_words_length) {
  for (int bi = 0; bi < bs; ++bi) {
    float *logits_now = logits + bi * length;
    for (int bwid = 0; bwid < bad_words_length; ++bwid) {
      const int64_t bad_words_token_id = bad_words_list[bwid];
      if (bad_words_token_id >= length || bad_words_token_id < 0) continue;
      logits_now[bad_words_token_id] = -1e10;
    }
  }
}

template <paddle::DataType D>
void token_penalty_multi_scores_kernel(const paddle::Tensor &pre_ids,
                                       const paddle::Tensor &logits,
                                       const paddle::Tensor &penalty_scores,
                                       const paddle::Tensor &frequency_score,
                                       const paddle::Tensor &presence_score,
                                       const paddle::Tensor &temperatures,
                                       const paddle::Tensor &bad_tokens,
                                       const paddle::Tensor &cur_len,
                                       const paddle::Tensor &min_len,
                                       const paddle::Tensor &eos_token_id) {
  std::vector<int64_t> shape = logits.shape();
  auto repeat_times =
      paddle::full(shape, 0, paddle::DataType::INT32, pre_ids.place());
  int64_t bs = shape[0];
  int64_t length = shape[1];
  int64_t length_id = pre_ids.shape()[1];
  int64_t end_length = eos_token_id.shape()[0];
  int64_t length_bad_words = bad_tokens.shape()[0];

  min_length_logits_process(const_cast<float *>(logits.data<float>()),
                            cur_len.data<int64_t>(),
                            min_len.data<int64_t>(),
                            eos_token_id.data<int64_t>(),
                            bs,
                            length,
                            end_length);

  update_repeat_times(pre_ids.data<int64_t>(),
                      cur_len.data<int64_t>(),
                      repeat_times.data<int>(),
                      bs,
                      length,
                      length_id);

  update_value_by_repeat_times(repeat_times.data<int>(),
                               penalty_scores.data<float>(),
                               frequency_score.data<float>(),
                               presence_score.data<float>(),
                               temperatures.data<float>(),
                               const_cast<float *>(logits.data<float>()),
                               bs,
                               length);

  ban_bad_words(const_cast<float *>(logits.data<float>()),
                bad_tokens.data<int64_t>(),
                bs,
                length,
                length_bad_words);
}

void TokenPenaltyMultiScores(const paddle::Tensor &pre_ids,
                             const paddle::Tensor &logits,
                             const paddle::Tensor &penalty_scores,
                             const paddle::Tensor &frequency_scores,
                             const paddle::Tensor &presence_scores,
                             const paddle::Tensor &temperatures,
                             const paddle::Tensor &bad_tokens,
                             const paddle::Tensor &cur_len,
                             const paddle::Tensor &min_len,
                             const paddle::Tensor &eos_token_id) {
  return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT32>(
      pre_ids,
      logits,
      penalty_scores,
      frequency_scores,
      presence_scores,
      temperatures,
      bad_tokens,
      cur_len,
      min_len,
      eos_token_id);
}

PD_BUILD_STATIC_OP(get_token_penalty_multi_scores_cpu)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "temperatures",
             "bad_tokens",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores));
