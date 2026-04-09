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

// Main draft preprocess kernel.
// MTP state (seq_lens_decoder, step_idx) is "shadow state":
//   - Initialized from target model state each round
//   - Used for MTP forward, but not committed until verify
//   - No rollback needed since it's always re-initialized
//
// is_splitwise_prefill: set true on P-D disaggregated prefill node.
//   In that mode, only prefill requests (seq_lens_encoder > 0) run MTP;
//   decode requests are marked stopped and skipped.
template <int THREADBLOCK_SIZE>
__global__ void draft_model_preprocess_kernel(
    int64_t* draft_tokens,
    int64_t* input_ids,
    bool* stop_flags,
    int* seq_lens_this_time,
    int* seq_lens_encoder,
    int* seq_lens_decoder,
    int64_t* step_idx,
    bool* not_need_stop,
    int64_t* pre_ids,
    const int64_t* accept_tokens,
    const int* accept_num,
    const int* target_model_seq_lens_encoder,
    const int* target_model_seq_lens_decoder,
    const int64_t* target_model_step_idx,
    const bool* target_model_stop_flags,
    const int64_t* max_dec_len,
    int64_t* target_model_draft_tokens,
    const int bsz,
    const int num_model_step,
    const int accept_tokens_len,
    const int draft_tokens_len,
    const int input_ids_len,
    const int target_model_draft_tokens_len,
    const int pre_ids_len,
    const bool is_splitwise_prefill) {
  typedef cub::BlockReduce<int64_t, THREADBLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int64_t not_stop_flag = 0;

  int tid = threadIdx.x;

  if (tid < bsz) {
    auto* accept_tokens_now = accept_tokens + tid * accept_tokens_len;
    auto* draft_tokens_now = draft_tokens + tid * draft_tokens_len;
    const int32_t accept_num_now = accept_num[tid];
    auto* input_ids_now = input_ids + tid * input_ids_len;
    auto* target_model_draft_tokens_now =
        target_model_draft_tokens + tid * target_model_draft_tokens_len;
    auto* pre_ids_now = pre_ids + tid * pre_ids_len;
    const auto target_step = target_model_step_idx[tid];
    auto seq_len_encoder = seq_lens_encoder[tid];

    // Clear target_model_draft_tokens (keep first token)
#pragma unroll
    for (int i = 1; i < target_model_draft_tokens_len; i++) {
      target_model_draft_tokens_now[i] = -1;
    }

    // ============================================================
    // Decision: Should MTP/Draft model run?
    // ============================================================
    bool should_skip = false;

    // Target model stopped
    if (target_model_stop_flags[tid]) {
      should_skip = true;
    }

    // Near end of max_dec_len in no prefill node
    if ((not is_splitwise_prefill &&
         target_step + num_model_step >= max_dec_len[tid])) {
      should_skip = true;
    }

    // ============================================================
    // Execute based on decision
    // ============================================================
    if (should_skip) {
      stop_flags[tid] = true;
      seq_lens_this_time[tid] = 0;
      seq_lens_decoder[tid] = 0;
      seq_lens_encoder[tid] = 0;
      step_idx[tid] = 0;
      not_stop_flag = 0;
    } else {
      not_stop_flag = 1;
      stop_flags[tid] = false;

      if (seq_len_encoder > 0) {
        // prefill | chunk_prefill | prompt_cache | recover after preempted
        int64_t target_model_first_token = accept_tokens_now[0];
        pre_ids_now[0] = target_model_first_token;

        input_ids_now[seq_len_encoder - 1] = target_model_first_token;
        seq_lens_this_time[tid] = seq_len_encoder;

        // Shadow state: initialize from target model (prefill just finished)
        step_idx[tid] = target_step - 1;
      } else {
        // Decode: MTP shadow state from target model
        // Shadow state is initialized from target model state each round
        // This is the key simplification: no rollback needed
        int32_t need_compute_token = accept_num_now;
        seq_lens_decoder[tid] =
            target_model_seq_lens_decoder[tid] - need_compute_token;
        step_idx[tid] = target_model_step_idx[tid] - need_compute_token;

        // Prepare draft input tokens from accepted tokens
        for (int i = 0; i < accept_num_now; i++) {
          draft_tokens_now[i] = accept_tokens_now[i];
          const int pre_id_pos =
              target_model_step_idx[tid] - (accept_num_now - i);
          pre_ids_now[pre_id_pos] = accept_tokens_now[i];
        }
        seq_lens_this_time[tid] = accept_num_now;
      }
    }
  }

  __syncthreads();
  int64_t not_stop_flag_sum = BlockReduce(temp_storage).Sum(not_stop_flag);
  if (tid == 0) {
    not_need_stop[0] = not_stop_flag_sum > 0;
  }
}

void DraftModelPreprocess(const paddle::Tensor& draft_tokens,
                          const paddle::Tensor& input_ids,
                          const paddle::Tensor& stop_flags,
                          const paddle::Tensor& seq_lens_this_time,
                          const paddle::Tensor& seq_lens_encoder,
                          const paddle::Tensor& seq_lens_decoder,
                          const paddle::Tensor& step_idx,
                          const paddle::Tensor& not_need_stop,
                          const paddle::Tensor& pre_ids,
                          const paddle::Tensor& accept_tokens,
                          const paddle::Tensor& accept_num,
                          const paddle::Tensor& target_model_seq_lens_encoder,
                          const paddle::Tensor& target_model_seq_lens_decoder,
                          const paddle::Tensor& target_model_step_idx,
                          const paddle::Tensor& target_model_stop_flags,
                          const paddle::Tensor& max_dec_len,
                          const paddle::Tensor& target_model_draft_tokens,
                          const int num_model_step,
                          const bool is_splitwise_prefill) {
  constexpr int kBlockSize = 1024;
  int real_bsz = seq_lens_this_time.shape()[0];
  PADDLE_ENFORCE_LE(
      real_bsz,
      kBlockSize,
      phi::errors::InvalidArgument(
          "draft_model_preprocess: real_bsz (%d) exceeds kBlockSize (%d).",
          real_bsz,
          kBlockSize));
  int accept_tokens_len = accept_tokens.shape()[1];
  int input_ids_len = input_ids.shape()[1];
  int draft_tokens_len = draft_tokens.shape()[1];
  int pre_ids_len = pre_ids.shape()[1];
  auto cu_stream = seq_lens_this_time.stream();
  int target_model_draft_tokens_len = target_model_draft_tokens.shape()[1];

  draft_model_preprocess_kernel<kBlockSize><<<1, kBlockSize, 0, cu_stream>>>(
      const_cast<int64_t*>(draft_tokens.data<int64_t>()),
      const_cast<int64_t*>(input_ids.data<int64_t>()),
      const_cast<bool*>(stop_flags.data<bool>()),
      const_cast<int*>(seq_lens_this_time.data<int>()),
      const_cast<int*>(seq_lens_encoder.data<int>()),
      const_cast<int*>(seq_lens_decoder.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      const_cast<bool*>(not_need_stop.data<bool>()),
      const_cast<int64_t*>(pre_ids.data<int64_t>()),
      accept_tokens.data<int64_t>(),
      accept_num.data<int>(),
      target_model_seq_lens_encoder.data<int>(),
      target_model_seq_lens_decoder.data<int>(),
      target_model_step_idx.data<int64_t>(),
      target_model_stop_flags.data<bool>(),
      max_dec_len.data<int64_t>(),
      const_cast<int64_t*>(target_model_draft_tokens.data<int64_t>()),
      real_bsz,
      num_model_step,
      accept_tokens_len,
      draft_tokens_len,
      input_ids_len,
      target_model_draft_tokens_len,
      pre_ids_len,
      is_splitwise_prefill);
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
             "pre_ids",
             "accept_tokens",
             "accept_num",
             "target_model_seq_lens_encoder",
             "target_model_seq_lens_decoder",
             "target_model_step_idx",
             "target_model_stop_flags",
             "max_dec_len",
             "target_model_draft_tokens"})
    .Outputs({"draft_tokens_out",
              "input_ids_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "step_idx_out",
              "not_need_stop_out",
              "pre_ids_out"})
    .Attrs({"num_model_step: int", "is_splitwise_prefill: bool"})
    .SetInplaceMap({{"draft_tokens", "draft_tokens_out"},
                    {"input_ids", "input_ids_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"step_idx", "step_idx_out"},
                    {"not_need_stop", "not_need_stop_out"},
                    {"pre_ids", "pre_ids_out"}})
    .SetKernelFn(PD_KERNEL(DraftModelPreprocess));
