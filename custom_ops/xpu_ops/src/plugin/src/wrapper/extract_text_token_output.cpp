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
template <typename T>
__attribute__((global)) void extract_text_token_output(int *max_seq_len,
                                          int *max_seq_len_index,
                                          int *mm_token_num_len,
                                          int *seq_lens_this_time,
                                          int *cu_seqlens_q,
                                          T *score_text,
                                          T *output,
                                          const int bsz,
                                          const int hidden_size);
}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

template <typename T>
static int cpu_wrapper( Context* ctx,
                        int *max_seq_len,
                        int *max_seq_len_index,
                        int *mm_token_num_len,
                        int *seq_lens_this_time,
                        int *cu_seqlens_q,
                        T *score_text,
                        T *output,
                        const int bsz,
                        const int hidden_size)
{
  int max_seq_len_val = max_seq_len[0];
  int max_seq_len_index_val = max_seq_len_index[0];
  int mm_token_num_len_val = mm_token_num_len[0];
  
  // 主循环处理
  for (int bsz_index = 0; bsz_index < bsz; ++bsz_index) {
      int true_bsz = cu_seqlens_q[bsz_index + 1] - 1;
      if (bsz_index >= max_seq_len_index_val) {
          true_bsz = true_bsz - mm_token_num_len_val;
      }
      
      if (max_seq_len_val == mm_token_num_len_val && bsz_index == max_seq_len_index_val) {
          // 将整行置为0
          for (int j = 0; j < hidden_size; ++j) {
              output[bsz_index * hidden_size + j] = 0.0f;
          }
      } else {
          if (seq_lens_this_time[bsz_index] != 0) {
              // 复制整行数据
              for (int j = 0; j < hidden_size; ++j) {
                  output[bsz_index * hidden_size + j] = score_text[true_bsz * hidden_size + j];
              }
          }
      }
  }

  return api::SUCCESS;
}

template <typename T>
static int xpu3_wrapper(
  Context* ctx,
  int *max_seq_len,
  int *max_seq_len_index,
  int *mm_token_num_len,
  int *seq_lens_this_time,
  int *cu_seqlens_q,
  T *score_text,
  T *output,
  const int bsz,
  const int hidden_size) {
  xpu3::plugin::extract_text_token_output<T> <<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
    max_seq_len, max_seq_len_index, mm_token_num_len, seq_lens_this_time, 
    cu_seqlens_q, score_text, output, bsz, hidden_size
  );
  return api::SUCCESS;
}

template <typename T>
int extract_text_token_output(Context* ctx,
                              int *max_seq_len,
                              int *max_seq_len_index,
                              int *mm_token_num_len,
                              int *seq_lens_this_time,
                              int *cu_seqlens_q,
                              T *score_text,
                              T *output,
                              const int bsz,
                              const int hidden_size)
{
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "extract_text_token_output", T);
  WRAPPER_DUMP_PARAM5(ctx, max_seq_len, max_seq_len_index, mm_token_num_len, seq_lens_this_time, cu_seqlens_q);
  WRAPPER_DUMP_PARAM4(ctx, score_text, output, bsz, hidden_size);
  WRAPPER_DUMP(ctx);
  WRAPPER_CHECK_PTR(ctx, int, 1, max_seq_len);
  WRAPPER_CHECK_PTR(ctx, int, 1, max_seq_len_index);
  WRAPPER_CHECK_PTR(ctx, int, 1, mm_token_num_len);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz+1, cu_seqlens_q);

  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper<T>(
      ctx, max_seq_len, max_seq_len_index, mm_token_num_len, seq_lens_this_time, 
      cu_seqlens_q, score_text, output, bsz, hidden_size
    );
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper<T>(
      ctx, max_seq_len, max_seq_len_index, mm_token_num_len, seq_lens_this_time, 
      cu_seqlens_q, score_text, output, bsz, hidden_size
    );
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

template int extract_text_token_output(Context*, int*, int*, int*, int*, int*, float*, float*, const int, const int);
} // namespace plugin
} // namespace api
} // namespace xpu
} // namespace baidu

