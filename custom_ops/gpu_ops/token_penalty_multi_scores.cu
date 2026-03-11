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

#include "helper.h"

template <typename T>
__global__ inline void min_dec_length_logits_process(
    T *logits,
    const int64_t *cur_dec_lens,
    const int64_t *min_dec_lens,
    const int64_t *eos_token_id,
    const int64_t bs,
    const int64_t vocab_size,
    const int64_t eos_len) {
  int bi = threadIdx.x;
  if (bi >= bs || cur_dec_lens[bi] < 0) {
    return;
  }
  if (cur_dec_lens[bi] < min_dec_lens[bi]) {
    for (int i = 0; i < eos_len; i++) {
      logits[bi * vocab_size + eos_token_id[i]] = -1e10;
    }
  }
}

template <>
__global__ inline void min_dec_length_logits_process<half>(
    half *logits,
    const int64_t *cur_dec_len,
    const int64_t *min_dec_len,
    const int64_t *eos_token_id,
    const int64_t bs,
    const int64_t vocab_size,
    const int64_t eos_len) {
  int bi = threadIdx.x;
  if (bi >= bs) return;
  if (cur_dec_len[bi] < 0) {
    return;
  }
  if (cur_dec_len[bi] < min_dec_len[bi]) {
    for (int i = 0; i < eos_len; i++) {
      logits[bi * vocab_size + eos_token_id[i]] = -1e4;
    }
  }
}

__global__ void update_repeat_times(const int64_t *token_ids_all,
                                    const int64_t *prompt_len,
                                    const int64_t *cur_dec_len,
                                    const float *penalty_scores,
                                    const float *frequency_score,
                                    const float *presence_score,
                                    const int64_t bs,
                                    const int64_t vocab_size,
                                    const int64_t max_model_len,
                                    int *repeat_times) {
  int64_t bi = blockIdx.x;
  float alpha = penalty_scores[bi];
  float beta = frequency_score[bi];
  float gamma = presence_score[bi];
  if (alpha == 1.f && beta == 0.f && gamma == 0.f) {
    return;
  }

  if (bi >= bs || cur_dec_len[bi] < 0) {
    return;
  }

  int64_t tid = threadIdx.x;
  const int64_t prompt_len_now = prompt_len[bi];
  const int64_t *token_ids_now = token_ids_all + bi * max_model_len;
  int *repeat_times_now = repeat_times + bi * vocab_size;

  // Pass 1: mark prompt tokens (set slot to -1 if absent from generated)
  for (int64_t i = tid; i < prompt_len_now; i += blockDim.x) {
    int64_t id = token_ids_now[i];
    if (id >= 0) {
      atomicCAS(&repeat_times_now[id], 0, -1);
    }
  }
  // Ensure all prompt marks complete before counting generated tokens,
  // otherwise atomicCAS can race with atomicMax+atomicAdd for the same slot.
  __syncthreads();

  // Pass 2: count generated tokens
  for (int64_t i = prompt_len_now + tid; i < max_model_len; i += blockDim.x) {
    int64_t id = token_ids_now[i];
    if (id < 0) continue;
    atomicMax(&repeat_times_now[id], 0);
    atomicAdd(&repeat_times_now[id], 1);
  }
}

template <typename T>
__global__ void update_value_by_repeat_times(const int *repeat_times,
                                             const T *penalty_scores,
                                             const T *frequency_score,
                                             const T *presence_score,
                                             const float *temperatures,
                                             T *logits,
                                             const int64_t bs,
                                             const int64_t vocab_size) {
  int bi = blockIdx.x;
  int tid = threadIdx.x;
  T *logits_now = logits + bi * vocab_size;
  const int *repeat_times_now = repeat_times + bi * vocab_size;
  float alpha = static_cast<float>(penalty_scores[bi]);
  float beta = static_cast<float>(frequency_score[bi]);
  float gamma = static_cast<float>(presence_score[bi]);
  float temperature = temperatures[bi];
  if (alpha == 1.f && beta == 0.f && gamma == 0.f && temperature == 1.f) {
    return;
  }

  for (int i = tid; i < vocab_size; i += blockDim.x) {
    int times = repeat_times_now[i];
    float logit_now = static_cast<float>(logits_now[i]);
    if (times != 0) {
      logit_now = logit_now < 0 ? logit_now * alpha : logit_now / alpha;
    }
    if (times > 0) {
      logit_now = logit_now - times * beta - gamma;
    }
    logits_now[i] = static_cast<T>(logit_now / temperature);
  }
}

template <typename T>
__global__ void ban_bad_words(T *logits,
                              const int64_t *bad_tokens,
                              const int64_t *bad_tokens_lens,
                              const int64_t bs,
                              const int64_t vocab_size,
                              const int64_t bad_words_len) {
  const int bi = blockIdx.x;
  int tid = threadIdx.x;
  T *logits_now = logits + bi * vocab_size;

  const int64_t *bad_tokens_now = bad_tokens + bi * bad_words_len;
  const int32_t bad_tokens_len =
      static_cast<int32_t>(min(bad_tokens_lens[bi], bad_words_len));
  for (int i = tid; i < bad_tokens_len; i += blockDim.x) {
    const int64_t bad_words_token_id = bad_tokens_now[i];
    if (bad_words_token_id >= vocab_size || bad_words_token_id < 0) continue;
    logits_now[bad_words_token_id] = -1e10;
  }
}

template <paddle::DataType D>
void token_penalty_multi_scores_kernel(const paddle::Tensor &token_ids_all,
                                       const paddle::Tensor &logits,
                                       const paddle::Tensor &penalty_scores,
                                       const paddle::Tensor &frequency_score,
                                       const paddle::Tensor &presence_score,
                                       const paddle::Tensor &temperatures,
                                       const paddle::Tensor &bad_tokens,
                                       const paddle::Tensor &bad_tokens_lens,
                                       const paddle::Tensor &prompt_lens,
                                       const paddle::Tensor &cur_dec_lens,
                                       const paddle::Tensor &min_dec_lens,
                                       const paddle::Tensor &eos_token_id) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(logits.place()));
  auto cu_stream = dev_ctx->stream();
#else
  auto cu_stream = logits.stream();
#endif
  std::vector<int64_t> shape = logits.shape();

  // repeat_times:
  // 0: Absent in both prompt and generated tokens.
  // >= 1: Frequency of occurrence in generated tokens.
  // -1: Only present in the prompt.
  auto repeat_times =
      paddle::full(shape, 0, paddle::DataType::INT32, logits.place());

  int64_t bs = shape[0];
  int64_t vocab_size = shape[1];
  int64_t max_model_len = token_ids_all.shape()[1];
  int64_t bad_words_len = bad_tokens.shape()[1];
  int64_t eos_len = eos_token_id.shape()[0];

  int block_size = (bs + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
  min_dec_length_logits_process<<<1, block_size, 0, cu_stream>>>(
      reinterpret_cast<DataType_ *>(
          const_cast<data_t *>(logits.data<data_t>())),
      cur_dec_lens.data<int64_t>(),
      min_dec_lens.data<int64_t>(),
      eos_token_id.data<int64_t>(),
      bs,
      vocab_size,
      eos_len);

  block_size = (max_model_len + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
#ifdef PADDLE_WITH_COREX
  block_size = std::min(block_size, 512);
#else
  block_size = min(block_size, 512);
#endif
  update_repeat_times<<<bs, block_size, 0, cu_stream>>>(
      token_ids_all.data<int64_t>(),
      prompt_lens.data<int64_t>(),
      cur_dec_lens.data<int64_t>(),
      penalty_scores.data<float>(),
      frequency_score.data<float>(),
      presence_score.data<float>(),
      bs,
      vocab_size,
      max_model_len,
      repeat_times.data<int>());

  block_size = (vocab_size + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
#ifdef PADDLE_WITH_COREX
  block_size = std::min(block_size, 512);
#else
  block_size = min(block_size, 512);
#endif
  update_value_by_repeat_times<DataType_><<<bs, block_size, 0, cu_stream>>>(
      repeat_times.data<int>(),
      reinterpret_cast<DataType_ *>(
          const_cast<data_t *>(penalty_scores.data<data_t>())),
      reinterpret_cast<DataType_ *>(
          const_cast<data_t *>(frequency_score.data<data_t>())),
      reinterpret_cast<DataType_ *>(
          const_cast<data_t *>(presence_score.data<data_t>())),
      temperatures.data<float>(),
      reinterpret_cast<DataType_ *>(
          const_cast<data_t *>(logits.data<data_t>())),
      bs,
      vocab_size);

  block_size = (bad_words_len + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE;
#ifdef PADDLE_WITH_COREX
  block_size = std::min(block_size, 512);
#else
  block_size = min(block_size, 512);
#endif
  ban_bad_words<DataType_><<<bs, block_size, 0, cu_stream>>>(
      reinterpret_cast<DataType_ *>(
          const_cast<data_t *>(logits.data<data_t>())),
      bad_tokens.data<int64_t>(),
      bad_tokens_lens.data<int64_t>(),
      bs,
      vocab_size,
      bad_words_len);
}

void TokenPenaltyMultiScores(const paddle::Tensor &token_ids_all,
                             const paddle::Tensor &logits,
                             const paddle::Tensor &penalty_scores,
                             const paddle::Tensor &frequency_scores,
                             const paddle::Tensor &presence_scores,
                             const paddle::Tensor &temperatures,
                             const paddle::Tensor &bad_tokens,
                             const paddle::Tensor &bad_tokens_lens,
                             const paddle::Tensor &prompt_lens,
                             const paddle::Tensor &cur_dec_lens,
                             const paddle::Tensor &min_dec_lens,
                             const paddle::Tensor &eos_token_id) {
  switch (logits.type()) {
    case paddle::DataType::BFLOAT16: {
      return token_penalty_multi_scores_kernel<paddle::DataType::BFLOAT16>(
          token_ids_all,
          logits,
          penalty_scores,
          frequency_scores,
          presence_scores,
          temperatures,
          bad_tokens,
          bad_tokens_lens,
          prompt_lens,
          cur_dec_lens,
          min_dec_lens,
          eos_token_id);
    }
    case paddle::DataType::FLOAT16: {
      return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT16>(
          token_ids_all,
          logits,
          penalty_scores,
          frequency_scores,
          presence_scores,
          temperatures,
          bad_tokens,
          bad_tokens_lens,
          prompt_lens,
          cur_dec_lens,
          min_dec_lens,
          eos_token_id);
    }
    case paddle::DataType::FLOAT32: {
      return token_penalty_multi_scores_kernel<paddle::DataType::FLOAT32>(
          token_ids_all,
          logits,
          penalty_scores,
          frequency_scores,
          presence_scores,
          temperatures,
          bad_tokens,
          bad_tokens_lens,
          prompt_lens,
          cur_dec_lens,
          min_dec_lens,
          eos_token_id);
    }
    default: {
      PD_THROW(
          "NOT supported data type. "
          "Only float16, bfloat16 and float32 are supported. ");
      break;
    }
  }
}

PD_BUILD_STATIC_OP(get_token_penalty_multi_scores)
    .Inputs({"token_ids_all",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "temperatures",
             "bad_tokens",
             "bad_tokens_lens",
             "prompt_lens",
             "cur_dec_lens",
             "min_dec_lens",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetInplaceMap({{"logits", "logits_out"}})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores));
