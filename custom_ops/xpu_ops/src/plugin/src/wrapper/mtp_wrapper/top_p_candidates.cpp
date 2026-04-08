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
template <typename T, int MaxLength, int TopPBeamTopK>
__attribute__((global)) void top_p_candidates(const T* src,
                                              const T* top_ps,
                                              const int* output_padding_offset,
                                              int64_t* out_id,
                                              T* out_val,
                                              int* actual_candidates_lens,
                                              int vocab_size,
                                              int token_num,
                                              int max_candidate_len,
                                              int max_seq_len);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T, int MaxLength, int TopPBeamTopK>
static int cpu_wrapper(Context* ctx,
                       const T* src,
                       const T* top_ps,
                       const int* output_padding_offset,
                       int64_t* out_id,
                       T* out_val,
                       int* actual_candidates_lens,
                       int vocab_size,
                       int token_num,
                       int candidate_len,
                       int max_seq_len) {
  int64_t local_out_id[TopPBeamTopK];
  T local_out_val[TopPBeamTopK];

  for (int64_t i = 0; i < token_num; i++) {
    float sum_prob = 0.0f;
    for (int j = 0; j < TopPBeamTopK; j++) {
      local_out_id[j] = -1;
      local_out_val[j] = std::numeric_limits<T>::min();
    }
    const T* cur_row_src = src + i * vocab_size;
    for (int id = 0; id < vocab_size; id++) {
      if (cur_row_src[id] > local_out_val[TopPBeamTopK - 1] ||
          (cur_row_src[id] == local_out_val[TopPBeamTopK - 1] &&
           id < local_out_id[TopPBeamTopK - 1])) {
        local_out_id[TopPBeamTopK - 1] = id;
        local_out_val[TopPBeamTopK - 1] = cur_row_src[id];
        for (int k = TopPBeamTopK - 2; k >= 0; k--) {
          if (local_out_val[k + 1] > local_out_val[k] ||
              (local_out_val[k + 1] == local_out_val[k] &&
               local_out_id[k + 1] < local_out_id[k])) {
            std::swap(local_out_id[k + 1], local_out_id[k]);
            std::swap(local_out_val[k + 1], local_out_val[k]);
          }
        }
      }
    }
    int ori_token_id = i + output_padding_offset[i];
    int bid = ori_token_id / max_seq_len;
    float top_p_value = static_cast<float>(top_ps[bid]);
    bool set_to_default_val = false;
    for (int j = 0; j < TopPBeamTopK; j++) {
      if (set_to_default_val) {
        out_id[i * candidate_len + j] = 0;
        out_val[i * candidate_len + j] = 0;
      } else {
        out_id[i * candidate_len + j] = local_out_id[j];
        out_val[i * candidate_len + j] = local_out_val[j];
        float val = static_cast<float>(local_out_val[j]);
        sum_prob += val;
        if (sum_prob >= top_p_value) {
          actual_candidates_lens[i] = j + 1;
          set_to_default_val = true;
        }
      }
    }
  }
  return api::SUCCESS;
}

template <typename T, int MaxLength, int TopPBeamTopK>
static int xpu3_wrapper(Context* ctx,
                        const T* src,
                        const T* top_ps,
                        const int* output_padding_offset,
                        int64_t* out_id,
                        T* out_val,
                        int* actual_candidates_lens,
                        int vocab_size,
                        int token_num,
                        int candidate_len,
                        int max_seq_len) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  xpu3::plugin::top_p_candidates<T, MaxLength, TopPBeamTopK>
      <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
          src,
          top_ps,
          output_padding_offset,
          reinterpret_cast<XPU_INT64*>(out_id),
          out_val,
          actual_candidates_lens,
          vocab_size,
          token_num,
          candidate_len,
          max_seq_len);
  return api::SUCCESS;
}

template <typename T, int MaxLength, int TopPBeamTopK>
int top_p_candidates(Context* ctx,
                     const T* src,
                     const T* top_ps,
                     const int* output_padding_offset,
                     int64_t* out_id,
                     T* out_val,
                     int* actual_candidates_lens,
                     int vocab_size,
                     int token_num,
                     int candidate_len,
                     int max_seq_len) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "top_p_candidates", T);
  WRAPPER_DUMP_PARAM5(ctx, src, top_ps, output_padding_offset, out_id, out_val);
  WRAPPER_DUMP_PARAM5(ctx,
                      actual_candidates_lens,
                      vocab_size,
                      token_num,
                      candidate_len,
                      max_seq_len);
  WRAPPER_DUMP(ctx);

  WRAPPER_CHECK_PTR(ctx, T, token_num * vocab_size, src);
  WRAPPER_CHECK_PTR(ctx, T, token_num, output_padding_offset);
  WRAPPER_CHECK_PTR(ctx, T, token_num * candidate_len, out_id);
  WRAPPER_CHECK_PTR(ctx, T, token_num * candidate_len, out_val);

  WRAPPER_ASSERT_GT(ctx, vocab_size, 0);
  WRAPPER_ASSERT_GT(ctx, token_num, 0);
  WRAPPER_ASSERT_GT(ctx, candidate_len, 0);
  WRAPPER_ASSERT_GT(ctx, max_seq_len, 0);
  WRAPPER_ASSERT_GT(ctx, TopPBeamTopK, 0);
  WRAPPER_ASSERT_LE(ctx, TopPBeamTopK, 10);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T, MaxLength, TopPBeamTopK>(ctx,
                                                   src,
                                                   top_ps,
                                                   output_padding_offset,
                                                   out_id,
                                                   out_val,
                                                   actual_candidates_lens,
                                                   vocab_size,
                                                   token_num,
                                                   candidate_len,
                                                   max_seq_len);
  } else if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T, MaxLength, TopPBeamTopK>(ctx,
                                                    src,
                                                    top_ps,
                                                    output_padding_offset,
                                                    out_id,
                                                    out_val,
                                                    actual_candidates_lens,
                                                    vocab_size,
                                                    token_num,
                                                    candidate_len,
                                                    max_seq_len);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

#define _XPU_DEF_TOP_P_CANDIDATES_WRAPPER(T, MaxLength)       \
  template int top_p_candidates<T, MaxLength, 2>(Context*,    \
                                                 const T*,    \
                                                 const T*,    \
                                                 const int*,  \
                                                 int64_t*,    \
                                                 T*,          \
                                                 int*,        \
                                                 int,         \
                                                 int,         \
                                                 int,         \
                                                 int);        \
  template int top_p_candidates<T, MaxLength, 3>(Context*,    \
                                                 const T*,    \
                                                 const T*,    \
                                                 const int*,  \
                                                 int64_t*,    \
                                                 T*,          \
                                                 int*,        \
                                                 int,         \
                                                 int,         \
                                                 int,         \
                                                 int);        \
  template int top_p_candidates<T, MaxLength, 4>(Context*,    \
                                                 const T*,    \
                                                 const T*,    \
                                                 const int*,  \
                                                 int64_t*,    \
                                                 T*,          \
                                                 int*,        \
                                                 int,         \
                                                 int,         \
                                                 int,         \
                                                 int);        \
  template int top_p_candidates<T, MaxLength, 5>(Context*,    \
                                                 const T*,    \
                                                 const T*,    \
                                                 const int*,  \
                                                 int64_t*,    \
                                                 T*,          \
                                                 int*,        \
                                                 int,         \
                                                 int,         \
                                                 int,         \
                                                 int);        \
  template int top_p_candidates<T, MaxLength, 8>(Context*,    \
                                                 const T*,    \
                                                 const T*,    \
                                                 const int*,  \
                                                 int64_t*,    \
                                                 T*,          \
                                                 int*,        \
                                                 int,         \
                                                 int,         \
                                                 int,         \
                                                 int);        \
  template int top_p_candidates<T, MaxLength, 10>(Context*,   \
                                                  const T*,   \
                                                  const T*,   \
                                                  const int*, \
                                                  int64_t*,   \
                                                  T*,         \
                                                  int*,       \
                                                  int,        \
                                                  int,        \
                                                  int,        \
                                                  int);

_XPU_DEF_TOP_P_CANDIDATES_WRAPPER(bfloat16, 2);
_XPU_DEF_TOP_P_CANDIDATES_WRAPPER(float, 2);
_XPU_DEF_TOP_P_CANDIDATES_WRAPPER(float16, 2);

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
