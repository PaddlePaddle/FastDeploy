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

__attribute__((global)) void speculate_free_and_reschedule(
    bool *stop_flags,
    int *seq_lens_this_time,
    int *seq_lens_decoder,
    int *block_tables,
    int *encoder_block_lens,
    bool *is_block_step,
    int *step_block_list,  // [bsz]
    int *step_len,
    int *recover_block_list,
    int *recover_len,
    int *need_block_list,
    int *need_block_len,
    int *used_list_len,
    int *free_list,
    int *free_list_len,
    int64_t *first_token_ids,
    const int bsz,
    const int block_size,
    const int block_num_per_seq,
    const int max_decoder_block_num,
    const int max_draft_tokens);

}  // namespace plugin
}  // namespace xpu3

namespace baidu {
namespace xpu {
namespace api {
namespace plugin {

static int cpu_wrapper(Context *ctx,
                       bool *stop_flags,
                       int *seq_lens_this_time,
                       int *seq_lens_decoder,
                       int *block_tables,
                       int *encoder_block_lens,
                       bool *is_block_step,
                       int *step_block_list,  // [bsz]
                       int *step_len,
                       int *recover_block_list,
                       int *recover_len,
                       int *need_block_list,
                       int *need_block_len,
                       int *used_list_len,
                       int *free_list,
                       int *free_list_len,
                       int64_t *first_token_ids,
                       const int bsz,
                       const int block_size,
                       const int block_num_per_seq,
                       const int max_decoder_block_num,
                       const int max_draft_tokens) {
  return -1;
}

static int xpu3_wrapper(Context *ctx,
                        bool *stop_flags,
                        int *seq_lens_this_time,
                        int *seq_lens_decoder,
                        int *block_tables,
                        int *encoder_block_lens,
                        bool *is_block_step,
                        int *step_block_list,  // [bsz]
                        int *step_len,
                        int *recover_block_list,
                        int *recover_len,
                        int *need_block_list,
                        int *need_block_len,
                        int *used_list_len,
                        int *free_list,
                        int *free_list_len,
                        int64_t *first_token_ids,
                        const int bsz,
                        const int block_size,
                        const int block_num_per_seq,
                        const int max_decoder_block_num,
                        const int max_draft_tokens) {
  using XPU_INT64 = typename XPUIndexType<int64_t>::type;
  auto speculate_free_and_reschedule =
      xpu3::plugin::speculate_free_and_reschedule;
  speculate_free_and_reschedule<<<ctx->ncluster(), 64, ctx->xpu_stream>>>(
      stop_flags,
      seq_lens_this_time,
      seq_lens_decoder,
      block_tables,
      encoder_block_lens,
      is_block_step,
      step_block_list,
      step_len,
      recover_block_list,
      recover_len,
      need_block_list,
      need_block_len,
      used_list_len,
      free_list,
      free_list_len,
      reinterpret_cast<XPU_INT64 *>(first_token_ids),
      bsz,
      block_size,
      block_num_per_seq,
      max_decoder_block_num,
      max_draft_tokens);
  return api::SUCCESS;
}

int speculate_free_and_reschedule(Context *ctx,
                                  bool *stop_flags,
                                  int *seq_lens_this_time,
                                  int *seq_lens_decoder,
                                  int *block_tables,
                                  int *encoder_block_lens,
                                  bool *is_block_step,
                                  int *step_block_list,  // [bsz]
                                  int *step_len,
                                  int *recover_block_list,
                                  int *recover_len,
                                  int *need_block_list,
                                  int *need_block_len,
                                  int *used_list_len,
                                  int *free_list,
                                  int *free_list_len,
                                  int64_t *first_token_ids,
                                  const int bsz,
                                  const int block_size,
                                  const int block_num_per_seq,
                                  const int max_decoder_block_num,
                                  const int max_draft_tokens) {
  WRAPPER_CHECK_CTX(ctx);
  WRAPPER_DUMP_FUNCTION_T1(ctx, "speculate_free_and_reschedule", float);
  WRAPPER_DUMP_PARAM6(ctx,
                      stop_flags,
                      seq_lens_this_time,
                      seq_lens_decoder,
                      block_tables,
                      encoder_block_lens,
                      is_block_step);
  WRAPPER_DUMP_PARAM6(ctx,
                      step_block_list,
                      step_len,
                      recover_block_list,
                      recover_len,
                      need_block_list,
                      need_block_len);
  WRAPPER_DUMP_PARAM4(
      ctx, used_list_len, free_list, free_list_len, first_token_ids);
  WRAPPER_DUMP_PARAM5(ctx,
                      bsz,
                      block_size,
                      block_num_per_seq,
                      max_decoder_block_num,
                      max_draft_tokens);
  WRAPPER_DUMP(ctx);
  if (ctx->dev().type() == api::kCPU) {
    return cpu_wrapper(ctx,
                       stop_flags,
                       seq_lens_this_time,
                       seq_lens_decoder,
                       block_tables,
                       encoder_block_lens,
                       is_block_step,
                       step_block_list,
                       step_len,
                       recover_block_list,
                       recover_len,
                       need_block_list,
                       need_block_len,
                       used_list_len,
                       free_list,
                       free_list_len,
                       first_token_ids,
                       bsz,
                       block_size,
                       block_num_per_seq,
                       max_decoder_block_num,
                       max_draft_tokens);
  }
  if (ctx->dev().type() == api::kXPU3) {
    return xpu3_wrapper(ctx,
                        stop_flags,
                        seq_lens_this_time,
                        seq_lens_decoder,
                        block_tables,
                        encoder_block_lens,
                        is_block_step,
                        step_block_list,
                        step_len,
                        recover_block_list,
                        recover_len,
                        need_block_list,
                        need_block_len,
                        used_list_len,
                        free_list,
                        free_list_len,
                        first_token_ids,
                        bsz,
                        block_size,
                        block_num_per_seq,
                        max_decoder_block_num,
                        max_draft_tokens);
  }
  WRAPPER_UNIMPLEMENTED(ctx);
}

}  // namespace plugin
}  // namespace api
}  // namespace xpu
}  // namespace baidu
