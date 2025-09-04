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
#include "paddle/extension.h"


#define DISPATCH_BLOCKSIZE(BLOCK_SIZE, ...) \
  do {                                     \
    constexpr int BlockSize = BLOCK_SIZE;  \
    __VA_ARGS__;                           \
  } while (0)

#define DISPATCH_TRUNCATE_FIRST_TOKEN(truncate_first_token, TRUNCATE_FIRST_TOKEN, ...) \
  do {                                                                                 \
    if (truncate_first_token) {                                                        \
      constexpr bool TRUNCATE_FIRST_TOKEN = true;                                     \
      __VA_ARGS__;                                                                     \
    } else {                                                                           \
      constexpr bool TRUNCATE_FIRST_TOKEN = false;                                    \
      __VA_ARGS__;                                                                     \
    }                                                                                  \
  } while (0)

#define DISPATCH_KVCACHE_SCHEDULER(kvcache_scheduler_v1, KVCACHE_SCHEDULER_V1, ...) \
  do {                                                                              \
    if (kvcache_scheduler_v1) {                                                     \
      constexpr bool KVCACHE_SCHEDULER_V1 = true;                                   \
      __VA_ARGS__;                                                                  \
    } else {                                                                        \
      constexpr bool KVCACHE_SCHEDULER_V1 = false;                                  \
      __VA_ARGS__;                                                                  \
    }                                                                               \
  } while (0)

#define DISPATCH_SPLITWISE_PREFILL(splitwise_prefill, SPLITWISE_PREFILL, ...) \
  do {                                                                        \
    if (splitwise_prefill) {                                                  \
      constexpr bool SPLITWISE_PREFILL = true;                                \
      __VA_ARGS__;                                                            \
    } else {                                                                  \
      constexpr bool SPLITWISE_PREFILL = false;                               \
      __VA_ARGS__;                                                            \
    }                                                                         \
  } while (0)


template <int THREADBLOCK_SIZE, bool TRUNCATE_FIRST_TOKEN, bool KVCACHE_SCHEDULER_V1>
__global__ void process_splitwise_prefill(
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    bool* not_need_stop,
    bool* is_block_step,
    bool* batch_drop,
    int64_t* pre_ids,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const int* base_model_seq_lens_decoder,
    const int64_t* base_model_step_idx,
    const bool* base_model_stop_flags,
    const bool* base_model_is_block_step,
    int64_t* base_model_draft_tokens,
    const int bsz,
    const int num_model_step,
    const int accept_tokens_len,
    const int draft_tokens_len,
    const int input_ids_len,
    const int base_model_draft_tokens_len,
    const int pre_ids_len) {
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t not_stop_flag = 0;

  int tid = threadIdx.x;
  if (tid < bsz) {
    int base_model_step_idx_now = base_model_step_idx[tid];
    auto* input_ids_now = input_ids + tid * input_ids_len;
    auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
    if (seq_lens_encoder[tid] > 0) {
      not_stop_flag = 1;
      int seq_len_encoder = seq_lens_encoder[tid];
      stop_flags[tid] = false;
      int64_t base_model_first_token = accept_tokens_now[0];
      int position = seq_len_encoder;
      if (TRUNCATE_FIRST_TOKEN) {
        input_ids_now[position - 1] = base_model_first_token;
        seq_lens_this_time[tid] = seq_len_encoder;
      } else {
        input_ids_now[position] = base_model_first_token;
        seq_lens_this_time[tid] = seq_len_encoder + 1;
      }
    } else {
      stop_flags[tid] = true;
      seq_lens_this_time[tid] = 0;
      seq_lens_decoder[tid] = 0;
      seq_lens_encoder[tid] = 0;
      not_stop_flag = 0;
    }
  }
  __syncthreads();

  int64_t not_stop_flag_sum = BlockReduce(temp_storage).Sum(not_stop_flag);
  if (tid == 0) {
    not_need_stop[0] = not_stop_flag_sum > 0;
  }
}




template <int THREADBLOCK_SIZE, bool TRUNCATE_FIRST_TOKEN, bool KVCACHE_SCHEDULER_V1>
__global__ void draft_model_preprocess_kernel(
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    bool* not_need_stop,
    bool* is_block_step,
    bool* batch_drop,
    int64_t* pre_ids,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const int* base_model_seq_lens_decoder,
    const int64_t* base_model_step_idx,
    const bool* base_model_stop_flags,
    const bool* base_model_is_block_step,
    int64_t* base_model_draft_tokens,
    const int bsz,
    const int num_model_step,
    const int accept_tokens_len,
    const int draft_tokens_len,
    const int input_ids_len,
    const int base_model_draft_tokens_len,
    const int pre_ids_len) {
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t not_stop_flag = 0;

  int tid = threadIdx.x;

  if (tid < bsz) {
    const int32_t base_model_step_idx_now = base_model_step_idx[tid];
    auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
    auto* draft_tokens_now = draft_tokens + tid * draft_tokens_len;
    const int32_t accept_num_now = accept_num[tid];
    auto* input_ids_now = input_ids + tid * input_ids_len;
    auto* base_model_draft_tokens_now =
        base_model_draft_tokens + tid * base_model_draft_tokens_len;
    auto base_model_seq_len_decoder = base_model_seq_lens_decoder[tid];
    const int32_t base_model_seq_len_this_time = base_model_seq_lens_this_time[tid];
    auto* pre_ids_now = pre_ids + tid * pre_ids_len;
#pragma unroll
    for (int i = 1; i < base_model_draft_tokens_len; i++) {
      base_model_draft_tokens_now[i] = -1;
    }

    // 1. process block_step situation
    //    -- In v0 mode, block_step will drop mtp query.
    //    -- In v1 mode, block_step will continue to infer.
    if constexpr(KVCACHE_SCHEDULER_V1) {
      if (base_model_stop_flags[tid] && base_model_is_block_step[tid]) {
        stop_flags[tid] = true;
        is_block_step[tid] = true;
        // Need to continue infer
      }
    } else {
      if (base_model_stop_flags[tid] && base_model_is_block_step[tid]) {
        batch_drop[tid] = true;
        stop_flags[tid] = true;
      }
    }

    // 2. process normal query, not in any special case.
    if (!(base_model_stop_flags[tid] || batch_drop[tid])) {
      not_stop_flag = 1;
      // prefill generation
      if (seq_lens_encoder[tid] > 0) {
        // Can be extended to first few tokens
        int seq_len_encoder = seq_lens_encoder[tid];
        stop_flags[tid] = false;
        int64_t base_model_first_token = accept_tokens_now[0];
        pre_ids_now[0] = base_model_first_token;
        int position = seq_len_encoder;
        if (TRUNCATE_FIRST_TOKEN) {
          input_ids_now[position - 1] = base_model_first_token;
          seq_lens_this_time[tid] = seq_len_encoder;
        } else {
          input_ids_now[position] = base_model_first_token;
          seq_lens_this_time[tid] = seq_len_encoder + 1;
        }
      } else {  // decode generation
        if constexpr (KVCACHE_SCHEDULER_V1) {
        // 3. try to recover mtp infer in V1 mode
          if (!base_model_is_block_step[tid] && is_block_step[tid]) {
            is_block_step[tid] = false;
          }
        }
        if (stop_flags[tid]) {
          stop_flags[tid] = false;
          // TODO: check
          seq_lens_decoder[tid] = base_model_seq_len_decoder - base_model_seq_len_this_time;
          step_idx[tid] = base_model_step_idx[tid] - base_model_seq_len_this_time;
        } else {
          // 2: Last base model generated token and first MTP token
          seq_lens_decoder[tid] -= num_model_step - 1;
          step_idx[tid] -= num_model_step - 1;
        }
        for (int i = 0; i < accept_num_now; i++) {
          draft_tokens_now[i] = accept_tokens_now[i];
          const int pre_id_pos = base_model_step_idx[tid] - (accept_num_now - i);
          const int64_t accept_token = accept_tokens_now[i];
          pre_ids_now[pre_id_pos] = accept_token;
        }
        seq_lens_this_time[tid] = accept_num_now;
      }
    } else {
      stop_flags[tid] = true;
      seq_lens_this_time[tid] = 0;
      seq_lens_decoder[tid] = 0;
      seq_lens_encoder[tid] = 0;
    }
  }
  __syncthreads();
  int64_t not_stop_flag_sum = BlockReduce(temp_storage).Sum(not_stop_flag);
  if (tid == 0) {
    not_need_stop[0] = not_stop_flag_sum > 0;
  }
}


void DispatchRunner(
    const cudaStream_t &stream,
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    bool* not_need_stop,
    bool* is_block_step,
    bool* batch_drop,
    int64_t* pre_ids,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* base_model_seq_lens_this_time,
    const int* base_model_seq_lens_encoder,
    const int* base_model_seq_lens_decoder,
    const int64_t* base_model_step_idx,
    const bool* base_model_stop_flags,
    const bool* base_model_is_block_step,
    int64_t* base_model_draft_tokens,
    const int bsz,
    const int num_model_step,
    const int accept_tokens_len,
    const int draft_tokens_len,
    const int input_ids_len,
    const int base_model_draft_tokens_len,
    const int pre_ids_len,
    const bool truncate_first_token,
    const bool splitwise_prefill,
    const bool kvcache_scheduler_v1) {
  DISPATCH_BLOCKSIZE(512, {
    DISPATCH_TRUNCATE_FIRST_TOKEN(truncate_first_token, TRUNCATE_FIRST_TOKEN, {
      DISPATCH_KVCACHE_SCHEDULER(kvcache_scheduler_v1, KVCACHE_SCHEDULER_V1, {
        DISPATCH_SPLITWISE_PREFILL(splitwise_prefill, SPLITWISE_PREFILL, {
          if constexpr (SPLITWISE_PREFILL) {
            process_splitwise_prefill<BlockSize, TRUNCATE_FIRST_TOKEN, KVCACHE_SCHEDULER_V1>
                <<<1, BlockSize, 0, stream>>>(
                    draft_tokens,
                    input_ids,
                    stop_flags,
                    seq_lens_this_time,
                    seq_lens_encoder,
                    seq_lens_decoder,
                    step_idx,
                    not_need_stop,
                    is_block_step,
                    batch_drop,
                    pre_ids,
                    accept_tokens,
                    accept_num,
                    base_model_seq_lens_this_time,
                    base_model_seq_lens_encoder,
                    base_model_seq_lens_decoder,
                    base_model_step_idx,
                    base_model_stop_flags,
                    base_model_is_block_step,
                    base_model_draft_tokens,
                    bsz,
                    num_model_step,
                    accept_tokens_len,
                    draft_tokens_len,
                    input_ids_len,
                    base_model_draft_tokens_len,
                    pre_ids_len);
          } else {
            draft_model_preprocess_kernel<BlockSize, TRUNCATE_FIRST_TOKEN, KVCACHE_SCHEDULER_V1>
                <<<1, BlockSize, 0, stream>>>(
                    draft_tokens,
                    input_ids,
                    stop_flags,
                    seq_lens_this_time,
                    seq_lens_encoder,
                    seq_lens_decoder,
                    step_idx,
                    not_need_stop,
                    is_block_step,
                    batch_drop,
                    pre_ids,
                    accept_tokens,
                    accept_num,
                    base_model_seq_lens_this_time,
                    base_model_seq_lens_encoder,
                    base_model_seq_lens_decoder,
                    base_model_step_idx,
                    base_model_stop_flags,
                    base_model_is_block_step,
                    base_model_draft_tokens,
                    bsz,
                    num_model_step,
                    accept_tokens_len,
                    draft_tokens_len,
                    input_ids_len,
                    base_model_draft_tokens_len,
                    pre_ids_len);
          }
        });
      });
    });
  });
}

void DraftModelPreprocess(const paddle::Tensor& draft_tokens,
                          const paddle::Tensor& input_ids,
                          const paddle::Tensor& stop_flags,
                          const paddle::Tensor& seq_lens_this_time,
                          const paddle::Tensor& seq_lens_encoder,
                          const paddle::Tensor& seq_lens_decoder,
                          const paddle::Tensor& step_idx,
                          const paddle::Tensor& not_need_stop,
                          const paddle::Tensor& is_block_step,
                          const paddle::Tensor& batch_drop,
                          const paddle::Tensor& pre_ids,
                          const paddle::Tensor& accept_tokens,
                          const paddle::Tensor& accept_num,
                          const paddle::Tensor& base_model_seq_lens_this_time,
                          const paddle::Tensor& base_model_seq_lens_encoder,
                          const paddle::Tensor& base_model_seq_lens_decoder,
                          const paddle::Tensor& base_model_step_idx,
                          const paddle::Tensor& base_model_stop_flags,
                          const paddle::Tensor& base_model_is_block_step,
                          const paddle::Tensor& base_model_draft_tokens,
                          const int num_model_step,
                          const bool truncate_first_token,
                          const bool splitwise_prefill,
                          const bool kvcache_scheduler_v1) {
  int real_bsz = seq_lens_this_time.shape()[0];
  int accept_tokens_len = accept_tokens.shape()[1];
  int input_ids_len = input_ids.shape()[1];
  int draft_tokens_len = draft_tokens.shape()[1];
  int pre_ids_len = pre_ids.shape()[1];
  auto cu_stream = seq_lens_this_time.stream();
  constexpr int BlockSize = 512;
  int base_model_draft_tokens_len = base_model_draft_tokens.shape()[1];
  auto not_need_stop_gpu =
      not_need_stop.copy_to(seq_lens_this_time.place(), false);

  DispatchRunner(
      cu_stream,
      const_cast<int64_t*>(draft_tokens.data<int64_t>()),
      const_cast<int64_t*>(input_ids.data<int64_t>()),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      const_cast<bool*>(not_need_stop_gpu.data<bool>()),
      const_cast<bool*>(is_block_step.data<bool>()),
      const_cast<bool*>(batch_drop.data<bool>()),
      const_cast<int64_t*>(pre_ids.data<int64_t>()),
      accept_tokens.data<int64_t>(),
      accept_num.data<int>(),
      base_model_seq_lens_this_time.data<int>(),
      base_model_seq_lens_encoder.data<int>(),
      base_model_seq_lens_decoder.data<int>(),
      base_model_step_idx.data<int64_t>(),
      base_model_stop_flags.data<bool>(),
      base_model_is_block_step.data<bool>(),
      const_cast<int64_t*>(base_model_draft_tokens.data<int64_t>()),
      real_bsz,
      num_model_step,
      accept_tokens_len,
      draft_tokens_len,
      input_ids_len,
      base_model_draft_tokens_len,
      pre_ids_len,
      truncate_first_token,
      splitwise_prefill,
      kvcache_scheduler_v1);

  auto not_need_stop_cpu =
      not_need_stop_gpu.copy_to(not_need_stop.place(), false);
  bool* not_need_stop_data = const_cast<bool*>(not_need_stop.data<bool>());
  not_need_stop_data[0] = not_need_stop_cpu.data<bool>()[0];
}


PD_BUILD_STATIC_OP(draft_model_preprocess)
    .Inputs({"draft_tokens",
             "input_ids",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "not_need_stop",
             "is_block_step",
             "batch_drop",
             "pre_ids",
             "accept_tokens",
             "accept_num",
             "base_model_seq_lens_this_time",
             "base_model_seq_lens_encoder",
             "base_model_seq_lens_decoder",
             "base_model_step_idx",
             "base_model_stop_flags",
             "base_model_is_block_step",
             "base_model_draft_tokens"})
    .Outputs({"draft_tokens_out",
              "input_ids_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_idx_out",
              "not_need_stop_out",
              "batch_drop_out",
              "pre_ids_out"})
    .Attrs({"num_model_step: int", "truncate_first_token: bool", "splitwise_prefill: bool", "kvcache_scheduler_v1: bool"})
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"input_ids", "input_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"step_idx", "step_idx_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"batch_drop", "batch_drop_out"},
                    {"pre_ids", "pre_ids_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelPreprocess));
