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

#include "helper.h"

template <int THREADBLOCK_SIZE>
__global__ void speculate_schedula_cache(const int64_t *draft_tokens,
                                         int *block_tables,
                                         bool *stop_flags,
                                         const int64_t *prompt_lens,
                                         int *seq_lens_this_time,
                                         int *seq_lens_encoder,
                                         int *seq_lens_decoder,
                                         int *step_seq_lens_decoder,
                                         int64_t *step_draft_tokens,
                                         int *step_seq_lens_this_time,
                                         int *accept_num,
                                         int64_t *accept_tokens,
                                         bool *is_block_step,
                                         bool *not_need_stop,
                                         const int64_t *stop_nums,
                                         const int real_bsz,
                                         const int max_bsz,
                                         const int max_next_step_tokens,
                                         const int draft_tokens_len,
                                         const int accept_tokens_len,
                                         const int block_size,
                                         const int block_num_per_seq,
                                         const bool prefill_one_step_stop) {
  const int bid = threadIdx.x;
  int stop_flag_now_int = 0;
  if (bid < real_bsz) {
    if (!stop_flags[bid]) {
      const int64_t *draft_tokens_now = draft_tokens + bid * draft_tokens_len;
      int64_t *step_draft_tokens_now =
          step_draft_tokens + bid * draft_tokens_len;
      int *block_table_now = block_tables + bid * block_num_per_seq;
      int64_t *accept_tokens_now = accept_tokens + bid * accept_tokens_len;

      if (seq_lens_decoder[bid] >= prompt_lens[bid]) {
        const int max_possible_block_idx =
            (seq_lens_decoder[bid] + max_next_step_tokens) / block_size;

        if (prefill_one_step_stop) {
          stop_flags[bid] = true;
          seq_lens_this_time[bid] = 0;
          seq_lens_decoder[bid] = 0;
          seq_lens_encoder[bid] = 0;
          accept_num[bid] = 0;
          stop_flag_now_int = 1;
        } else if (max_possible_block_idx < block_num_per_seq &&
                   block_table_now[max_possible_block_idx] == -1) {
          is_block_step[bid] = true;
          step_seq_lens_this_time[bid] = seq_lens_this_time[bid];
          seq_lens_this_time[bid] = 0;
          stop_flags[bid] = true;
          stop_flag_now_int = 1;
          step_seq_lens_decoder[bid] = seq_lens_decoder[bid];
          seq_lens_decoder[bid] = 0;
          accept_num[bid] = 0;
          for (int i = 0; i < accept_tokens_len; i++) {
            accept_tokens_now[i] = -1;
          }
          for (int i = 0; i < draft_tokens_len; i++) {
            step_draft_tokens_now[i] = draft_tokens_now[i];
          }
        }
      } else {
        // prefill
        stop_flags[bid] = true;
        seq_lens_this_time[bid] = 0;
        seq_lens_decoder[bid] = 0;
        seq_lens_encoder[bid] = 0;
        accept_num[bid] = 0;
        stop_flag_now_int = 1;
      }

    } else {
      stop_flag_now_int = 1;
    }
  } else if (bid >= real_bsz && bid < max_bsz) {
    stop_flag_now_int = 1;
  }
  __syncthreads();
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // printf("stop_flag_now_int %d \n", stop_flag_now_int);
  int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);

  if (threadIdx.x == 0) {
    // printf("stop_sum %d \n", stop_sum);
    not_need_stop[0] = stop_sum < stop_nums[0];
  }
}

void SpeculateScheduleCache(const paddle::Tensor &draft_tokens,
                            const paddle::Tensor &block_tables,
                            const paddle::Tensor &stop_flags,
                            const paddle::Tensor &prompt_lens,
                            const paddle::Tensor &seq_lens_this_time,
                            const paddle::Tensor &seq_lens_encoder,
                            const paddle::Tensor &seq_lens_decoder,
                            const paddle::Tensor &step_seq_lens_decoder,
                            const paddle::Tensor &step_draft_tokens,
                            const paddle::Tensor &step_seq_lens_this_time,
                            const paddle::Tensor &accept_num,
                            const paddle::Tensor &accept_tokens,
                            const paddle::Tensor &is_block_step,
                            const paddle::Tensor &not_need_stop,
                            const paddle::Tensor &stop_nums,
                            const int block_size,
                            const int max_draft_tokens) {
  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  const int accept_tokens_len = accept_tokens.shape()[1];
  const int draft_token_len = draft_tokens.shape()[1];
  const int block_num_per_seq = block_tables.shape()[1];

  constexpr int BlockSize = 512;
  const int max_next_step_tokens = 2 * max_draft_tokens + 2;
  bool prefill_one_step_stop = false;
  if (const char *env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP_V1")) {
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }
  auto not_need_stop_gpu = not_need_stop.copy_to(stop_flags.place(), false);
  speculate_schedula_cache<BlockSize>
      <<<1, BlockSize, 0, seq_lens_this_time.stream()>>>(
          draft_tokens.data<int64_t>(),
          const_cast<int *>(block_tables.data<int>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          prompt_lens.data<int64_t>(),
          const_cast<int *>(seq_lens_this_time.data<int>()),
          const_cast<int *>(seq_lens_encoder.data<int>()),
          const_cast<int *>(seq_lens_decoder.data<int>()),
          const_cast<int *>(step_seq_lens_decoder.data<int>()),
          const_cast<int64_t *>(step_draft_tokens.data<int64_t>()),
          const_cast<int *>(step_seq_lens_this_time.data<int>()),
          const_cast<int *>(accept_num.data<int>()),
          const_cast<int64_t *>(accept_tokens.data<int64_t>()),
          const_cast<bool *>(is_block_step.data<bool>()),
          const_cast<bool *>(not_need_stop_gpu.data<bool>()),
          stop_nums.data<int64_t>(),
          real_bsz,
          max_bsz,
          max_next_step_tokens,
          draft_token_len,
          accept_tokens_len,
          block_size,
          block_num_per_seq,
          prefill_one_step_stop);

  auto not_need_stop_cpu =
      not_need_stop_gpu.copy_to(not_need_stop.place(), true);
  bool *not_need_stop_data = const_cast<bool *>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}

PD_BUILD_STATIC_OP(speculate_schedule_cache)
    .Inputs({"draft_tokens",
             "block_tables",
             "stop_flags",
             "prompt_lens",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_seq_lens_decoder",
             "step_draft_tokens",
             "step_seq_lens_this_time",
             "accept_num",
             "accept_tokens",
             "is_block_step",
             "not_need_stop",
             "stop_nums"})
    .Attrs({"block_size: int", "max_draft_tokens: int"})
    .Outputs({"draft_tokens_out",
              "block_tables_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_seq_lens_decoder_out",
              "step_draft_tokens_out",
              "step_seq_lens_this_time_out",
              "accept_num_out",
              "accept_tokens_out",
              "is_block_step_out",
              "not_need_stop_out"})
    .SetInplaceMap({
        {"draft_tokens", "draft_tokens_out"},
        {"block_tables", "block_tables_out"},
        {"stop_flags", "stop_flags_out"},
        {"seq_lens_this_time", "seq_lens_this_time_out"},
        {"seq_lens_encoder", "seq_lens_encoder_out"},
        {"seq_lens_decoder", "seq_lens_decoder_out"},
        {"step_seq_lens_decoder", "step_seq_lens_decoder_out"},
        {"step_draft_tokens", "step_draft_tokens_out"},
        {"step_seq_lens_this_time", "step_seq_lens_this_time_out"},
        {"accept_num", "accept_num_out"},
        {"accept_tokens", "accept_tokens_out"},
        {"is_block_step", "is_block_step_out"},
        {"not_need_stop", "not_need_stop_out"},
    })
    .SetKernelFn(PD_KERNEL(SpeculateScheduleCache));
