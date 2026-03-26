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

template <int THREADBLOCK_SIZE>
__global__ void draft_model_update_kernel(const int64_t* inter_next_tokens,
                                          int64_t* draft_tokens,
                                          int64_t* pre_ids,
                                          int* seq_lens_this_time,
                                          int* seq_lens_encoder,
                                          int* seq_lens_decoder,
                                          int64_t* step_idx,
                                          const int* cu_seqlens_q_output,
                                          bool* stop_flags,
                                          bool* not_need_stop,
                                          const int64_t* max_dec_len,
                                          const int64_t* end_ids,
                                          int64_t* base_model_draft_tokens,
                                          const int bsz,
                                          const int max_draft_token,
                                          const int pre_id_length,
                                          const int max_base_model_draft_token,
                                          const int end_ids_len,
                                          const int max_seq_len,
                                          const int substep,
                                          const bool prefill_one_step_stop) {
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t stop_flag_now_int = 0;

  int tid = threadIdx.x;
  if (tid < bsz && seq_lens_this_time[tid] > 0) {
    int seq_len_this_time = seq_lens_this_time[tid];
    int seq_len_encoder = seq_lens_encoder[tid];
    int seq_len_decoder = seq_lens_decoder[tid];

    int next_tokens_start_id = 0;
    for (int i = 0; i < tid; i++) {
      next_tokens_start_id += seq_lens_this_time[i] > 0 ? 1 : 0;
    }

    int64_t* draft_token_now = draft_tokens + tid * max_draft_token;
    int64_t* pre_ids_now = pre_ids + tid * pre_id_length;
    int64_t* base_model_draft_tokens_now =
        base_model_draft_tokens + tid * max_base_model_draft_token;
    const int64_t* next_tokens_start = inter_next_tokens + next_tokens_start_id;

    // 1. update step_idx && seq_lens_dec
    if (!stop_flags[tid]) {
      int64_t token_this_time = -1;
      if (seq_len_encoder > 0) {
        token_this_time = next_tokens_start[0];
        seq_lens_decoder[tid] = seq_len_encoder + seq_len_decoder;
        seq_lens_encoder[tid] = 0;
        pre_ids_now[1] = token_this_time;
        step_idx[tid] += 1;
        draft_token_now[0] = token_this_time;
        if (step_idx[tid] < max_dec_len[tid]) {
          base_model_draft_tokens_now[substep + 1] = token_this_time;
        }
      } else if (seq_len_decoder > 0) {
        if (step_idx[tid] >= max_dec_len[tid] - 1) {
          // If up to max_dec_len -1. Recompute but not update.
          base_model_draft_tokens_now[substep + 1] = -1;
        } else {
          seq_lens_decoder[tid] += seq_len_this_time;
          token_this_time = next_tokens_start[0];
          draft_token_now[0] = token_this_time;
          base_model_draft_tokens_now[substep + 1] = token_this_time;
          step_idx[tid] += seq_len_this_time;
          pre_ids_now[step_idx[tid]] = token_this_time;
        }
      }
    } else {
      draft_token_now[0] = -1;
      base_model_draft_tokens_now[substep + 1] = -1;
      stop_flag_now_int = 1;
    }

    // 2. set end
    if (!stop_flags[tid]) {
      seq_lens_this_time[tid] = 1;
    } else {
      seq_lens_this_time[tid] = 0;
      seq_lens_encoder[tid] = 0;
    }
  }
  __syncthreads();
  int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_now_int);
  if (tid == 0) {
    not_need_stop[0] = stop_sum < bsz;
  }
}

void DraftModelUpdate(const paddle::Tensor& inter_next_tokens,
                      const paddle::Tensor& draft_tokens,
                      const paddle::Tensor& pre_ids,
                      const paddle::Tensor& seq_lens_this_time,
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& seq_lens_decoder,
                      const paddle::Tensor& step_idx,
                      const paddle::Tensor& cu_seqlens_q_output,
                      const paddle::Tensor& stop_flags,
                      const paddle::Tensor& not_need_stop,
                      const paddle::Tensor& max_dec_len,
                      const paddle::Tensor& end_ids,
                      const paddle::Tensor& base_model_draft_tokens,
                      const int max_seq_len,
                      const int substep) {
  auto seq_lens_this_time_shape = seq_lens_this_time.shape();
  auto cu_stream = seq_lens_this_time.stream();
  const int real_bsz = seq_lens_this_time_shape[0];
  const int end_ids_len = end_ids.shape()[0];
  const int max_draft_token = draft_tokens.shape()[1];
  const int pre_id_length = pre_ids.shape()[1];
  const int max_base_model_draft_token = base_model_draft_tokens.shape()[1];
  constexpr int BlockSize = 512;

  bool prefill_one_step_stop = false;
  if (const char* env_p = std::getenv("PREFILL_NODE_ONE_STEP_STOP")) {
    // std::cout << "Your PATH is: " << env_p << '\n';
    if (env_p[0] == '1') {
      prefill_one_step_stop = true;
    }
  }

  draft_model_update_kernel<BlockSize><<<1, BlockSize, 0, cu_stream>>>(
      inter_next_tokens.data<int64_t>(),
      const_cast<int64_t*>(draft_tokens.data<int64_t>()),
      const_cast<int64_t*>(pre_ids.data<int64_t>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      cu_seqlens_q_output.data<int>(),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<bool*>(not_need_stop.data<bool>()),
      max_dec_len.data<int64_t>(),
      end_ids.data<int64_t>(),
      const_cast<int64_t*>(base_model_draft_tokens.data<int64_t>()),
      real_bsz,
      max_draft_token,
      pre_id_length,
      max_base_model_draft_token,
      end_ids_len,
      max_seq_len,
      substep,
      prefill_one_step_stop);
}

PD_BUILD_STATIC_OP(draft_model_update)
    .Inputs({"inter_next_tokens",
             "draft_tokens",
             "pre_ids",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "step_idx",
             "cu_seqlens_q_output",
             "stop_flags",
             "not_need_stop",
             "max_dec_len",
             "end_ids",
             "base_model_draft_tokens"})
    .Attrs({"max_seq_len: int", "substep: int"})
    .Outputs({"draft_tokens_out",
              "pre_ids_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_idx_out",
              "stop_flags_out",
              "not_need_stop_out",
              "base_model_draft_tokens_out"})
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"pre_ids", "pre_ids_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"step_idx", "step_idx_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"base_model_draft_tokens", "base_model_draft_tokens_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelUpdate));
