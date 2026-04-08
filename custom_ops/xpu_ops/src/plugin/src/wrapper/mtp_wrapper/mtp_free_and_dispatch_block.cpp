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

namespace xpu2 {
namespace plugin {}  // namespace plugin
}  // namespace xpu2

namespace xpu3 {
namespace plugin {

__attribute__((global)) void mtp_free_and_dispatch_block(
    bool *base_model_stop_flags,
    bool *stop_flags,
    bool *batch_drop,
    int *seq_lens_this_time,
    int *seq_lens_decoder,
    int *block_tables,
    int *encoder_block_lens,
    int *used_list_len,
    int *free_list,
    int *free_list_len,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_draft_tokens);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context *ctx,
                       bool *base_model_stop_flags,
                       bool *stop_flags,
                       bool *batch_drop,
                       int *seq_lens_this_time,
                       int *seq_lens_decoder,
                       int *block_tables,
                       int *encoder_block_lens,
                       int *used_list_len,
                       int *free_list,
                       int *free_list_len,
                       const int bsz,
                       const int block_size,
                       const int block_num_per_seq,
                       const int max_draft_tokens) {
  int need_block_len = 0;
  int need_block_list[640];
  for (int tid = 0; tid < bsz; tid++) {
    need_block_list[tid] = 0;
    int *block_table_now = block_tables + tid * block_num_per_seq;
    if (base_model_stop_flags[tid] || batch_drop[tid]) {
      // 回收block块
      const int encoder_block_len = encoder_block_lens[tid];
      const int decoder_used_len = used_list_len[tid];
      if (decoder_used_len > 0) {
        for (int i = 0; i < decoder_used_len; i++) {
          free_list[free_list_len[0] + i] =
              block_table_now[encoder_block_len + i];
          block_table_now[encoder_block_len + i] = -1;
        }
        free_list_len[0] += decoder_used_len;
        encoder_block_lens[tid] = 0;
        used_list_len[tid] = 0;
      }
    }
  }
  for (int tid = 0; tid < bsz; tid++) {
    int *block_table_now = block_tables + tid * block_num_per_seq;
    int max_possible_block_idx =
        (seq_lens_decoder[tid] + max_draft_tokens + 1) / block_size;
    if (!base_model_stop_flags[tid] && !batch_drop[tid] &&
        max_possible_block_idx < block_num_per_seq &&
        block_table_now[max_possible_block_idx] == -1) {
      need_block_list[need_block_len] = tid;
      need_block_len++;
    }
  }
  // 这里直接从 bid 0 开始遍历
  while (need_block_len > free_list_len[0]) {
    int max_used_list_len_id = 0;
    int max_used_list_len = 0;
    for (int i = 0; i < bsz; i++) {
      if (!base_model_stop_flags[i] && used_list_len[i] > max_used_list_len) {
        max_used_list_len = used_list_len[i];
        max_used_list_len_id = i;
      }
    }
    const int encoder_block_len = encoder_block_lens[max_used_list_len_id];
    int *block_table_now =
        block_tables + max_used_list_len_id * block_num_per_seq;
    for (int i = 0; i < max_used_list_len; i++) {
      free_list[free_list_len[0] + i] = block_table_now[encoder_block_len + i];
      block_table_now[encoder_block_len + i] = -1;
    }
    stop_flags[max_used_list_len_id] = true;
    batch_drop[max_used_list_len_id] = true;
    seq_lens_this_time[max_used_list_len_id] = 0;
    seq_lens_decoder[max_used_list_len_id] = 0;
    used_list_len[max_used_list_len_id] = 0;
    free_list_len[0] += max_used_list_len;
  }
  for (int tid = 0; tid < need_block_len; tid++) {
    const int need_block_id = need_block_list[tid];
    // 这里必须用 batch_drop, 不能用 stop_flags
    if (!batch_drop[need_block_id]) {
      used_list_len[need_block_id] += 1;
      int *block_table_now = block_tables + need_block_id * block_num_per_seq;
      block_table_now[(seq_lens_decoder[need_block_id] + max_draft_tokens + 1) /
                      block_size] = free_list[free_list_len[0] - 1];
      free_list_len[0] -= 1;
    }
  }
  return api::SUCCESS;
}

static int xpu2or3_wrapper(Context *ctx,
                           bool *base_model_stop_flags,
                           bool *stop_flags,
                           bool *batch_drop,
                           int *seq_lens_this_time,
                           int *seq_lens_decoder,
                           int *block_tables,
                           int *encoder_block_lens,
                           int *used_list_len,
                           int *free_list,
                           int *free_list_len,
                           const int bsz,
                           const int block_size,
                           const int block_num_per_seq,
                           const int max_draft_tokens) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  bool is_xpu3 = ctx->dev().type() == api::kXPU3;
  if (!is_xpu3) {
    WRAPPER_UNIMPLEMENTED(ctx);
  }
  auto mtp_free_and_dispatch_block = xpu3::plugin::mtp_free_and_dispatch_block;
  mtp_free_and_dispatch_block<<<12, 64, ctx->xpu_stream>>>(
      base_model_stop_flags,
      stop_flags,
      batch_drop,
      seq_lens_this_time,
      seq_lens_decoder,
      block_tables,
      encoder_block_lens,
      used_list_len,
      free_list,
      free_list_len,
      bsz,
      block_size,
      block_num_per_seq,
      max_draft_tokens);
  return api::SUCCESS;
}

int mtp_free_and_dispatch_block(Context *ctx,
                                bool *base_model_stop_flags,
                                bool *stop_flags,
                                bool *batch_drop,
                                int *seq_lens_this_time,
                                int *seq_lens_decoder,
                                int *block_tables,
                                int *encoder_block_lens,
                                int *used_list_len,
                                int *free_list,
                                int *free_list_len,
                                const int bsz,
                                const int block_size,
                                const int block_num_per_seq,
                                const int max_draft_tokens) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "mtp_free_and_dispatch_block", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      base_model_stop_flags,
                      stop_flags,
                      batch_drop,
                      seq_lens_this_time,
                      seq_lens_decoder,
                      block_tables);
  WRAPPER_DUMP_PARAM4(
      ctx, encoder_block_lens, used_list_len, free_list, free_list_len);
  WRAPPER_DUMP_PARAM4(
      ctx, bsz, block_size, block_num_per_seq, max_draft_tokens);
  WRAPPER_ASSERT_LE(ctx, bsz, 640);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, base_model_stop_flags);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, stop_flags);
  WRAPPER_CHECK_PTR(ctx, bool, bsz, batch_drop);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_this_time);
  WRAPPER_CHECK_PTR(ctx, int, bsz, seq_lens_decoder);
  WRAPPER_CHECK_PTR(ctx, int, bsz *block_num_per_seq, block_tables);
  WRAPPER_CHECK_PTR(ctx, int, bsz, encoder_block_lens);
  WRAPPER_CHECK_PTR(ctx, int, bsz, used_list_len);
  WRAPPER_CHECK_PTR(ctx, int, 1, free_list_len);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       base_model_stop_flags,
                       stop_flags,
                       batch_drop,
                       seq_lens_this_time,
                       seq_lens_decoder,
                       block_tables,
                       encoder_block_lens,
                       used_list_len,
                       free_list,
                       free_list_len,
                       bsz,
                       block_size,
                       block_num_per_seq,
                       max_draft_tokens);
  }
  if (ctx->dev().type() == api::kXPU2 || ctx->dev().type() == api::kXPU3) {
    return xpu2or3_wrapper(ctx,
                           base_model_stop_flags,
                           stop_flags,
                           batch_drop,
                           seq_lens_this_time,
                           seq_lens_decoder,
                           block_tables,
                           encoder_block_lens,
                           used_list_len,
                           free_list,
                           free_list_len,
                           bsz,
                           block_size,
                           block_num_per_seq,
                           max_draft_tokens);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
