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

/**
 * @file unified_update_model_status.cu
 * @brief Unified kernel for updating model status after token generation.
 *
 * Launched as a single block of 1024 threads (max_bsz <= 1024).
 */

/**
 * @brief Check if token is an end token.
 */
__device__ __forceinline__ bool is_end_token(int64_t token,
                                             const int64_t *end_tokens,
                                             int num_end_tokens) {
#pragma unroll 4
  for (int i = 0; i < num_end_tokens; i++) {
    if (token == end_tokens[i]) return true;
  }
  return false;
}

/**
 * @brief Main unified update kernel.
 */
template <int BLOCK_SIZE>
__global__ void unified_update_model_status_kernel(int *seq_lens_encoder,
                                                   int *seq_lens_decoder,
                                                   bool *has_running_seqs,
                                                   int64_t *step_input_ids,
                                                   int64_t *step_output_ids,
                                                   int *step_output_len,
                                                   bool *stop_flags,
                                                   int *seq_lens_this_time,
                                                   const bool *is_paused,
                                                   int64_t *token_ids_all,
                                                   const int64_t *prompt_lens,
                                                   int64_t *step_idx,
                                                   const int64_t *end_tokens,
                                                   const int64_t *max_dec_len,
                                                   int real_bsz,
                                                   int max_bsz,
                                                   int max_step_tokens,
                                                   int max_model_len,
                                                   int num_end_tokens) {
  const int batch_id = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  const bool is_valid_slot = batch_id < max_bsz;
  int stop_flag_int = 0;

  if (is_valid_slot) {
    // 1. Read state
    int cur_seq_len_encoder = seq_lens_encoder[batch_id];
    int cur_seq_len_decoder = seq_lens_decoder[batch_id];
    bool cur_stop_flag = stop_flags[batch_id];
    int output_len = step_output_len[batch_id];
    int64_t cur_step_idx = step_idx[batch_id];
    bool cur_is_paused = is_paused[batch_id];
    int64_t prompt_len = prompt_lens[batch_id];

    bool is_running = !cur_stop_flag && !cur_is_paused;

    // 2. EOS detection
    if (is_running && output_len > 0) {
      int64_t *output_ids = &step_output_ids[batch_id * max_step_tokens];

      for (int i = 0; i < output_len; i++) {
        cur_step_idx++;
        int64_t token = output_ids[i];
        bool is_eos = is_end_token(token, end_tokens, num_end_tokens);
        bool max_len_hit = (cur_step_idx >= max_dec_len[batch_id]);

        if (is_eos || max_len_hit) {
          if (!is_eos) output_ids[i] = end_tokens[0];
          output_len = i + 1;
          cur_stop_flag = true;
          break;
        }
      }
    }

    if (is_running) {
      // 3. Update state and write back
      if (cur_seq_len_encoder > 0) {
        cur_seq_len_decoder += cur_seq_len_encoder;
        cur_seq_len_encoder = 0;
      } else if (cur_seq_len_decoder > 0) {
        cur_seq_len_decoder += output_len;
      }

      if (cur_stop_flag) {
        // It should clear seq_lens_decoder in next step for save_output
        stop_flag_int = 1;
        stop_flags[batch_id] = true;
      }

      // 4. Update model status
      seq_lens_encoder[batch_id] = cur_seq_len_encoder;
      seq_lens_decoder[batch_id] = cur_seq_len_decoder;
      step_output_len[batch_id] = output_len;
      step_idx[batch_id] = cur_step_idx;

      // 5. Write history to token_ids_all (forward loop: position base+k =
      // output_ids[k])
      if (output_len > 0) {
        // Bounds check: highest write index is prompt_len + cur_step_idx
        if (prompt_len + cur_step_idx < max_model_len) {
          int64_t *token_ids_all_now =
              &token_ids_all[batch_id * max_model_len + prompt_len];
          int64_t *output_ids = &step_output_ids[batch_id * max_step_tokens];
          int64_t base = cur_step_idx - output_len;
          for (int i = 0; i < output_len; i++) {
            token_ids_all_now[base + i] = output_ids[i];
          }
        }
      }

      // 6. Prepare next step input[0]
      if (output_len > 0) {
        step_input_ids[batch_id * max_step_tokens] =
            step_output_ids[batch_id * max_step_tokens + output_len - 1];
      }
    } else if (batch_id >= real_bsz) {
      // Padding slot: just count as stopped, don't modify state
      stop_flag_int = 1;
    } else {
      // Stopped or paused slot (batch_id < real_bsz)
      stop_flag_int = 1;
      stop_flags[batch_id] = true;
      seq_lens_encoder[batch_id] = 0;
      seq_lens_decoder[batch_id] = 0;
      seq_lens_this_time[batch_id] = 0;
      step_output_len[batch_id] = 0;
    }
  }

  // Simple block-level reduction using shared memory
  __syncthreads();
  typedef cub::BlockReduce<int64_t, BLOCK_SIZE> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int64_t stop_sum = BlockReduce(temp_storage).Sum(stop_flag_int);

  if (threadIdx.x == 0) {
    has_running_seqs[0] = stop_sum < max_bsz;
  }
}

// Host interface
void UnifiedUpdateModelStatus(const paddle::Tensor &seq_lens_encoder,
                              const paddle::Tensor &seq_lens_decoder,
                              const paddle::Tensor &has_running_seqs,
                              const paddle::Tensor &step_input_ids,
                              const paddle::Tensor &step_output_ids,
                              const paddle::Tensor &step_output_len,
                              const paddle::Tensor &stop_flags,
                              const paddle::Tensor &seq_lens_this_time,
                              const paddle::Tensor &is_paused,
                              const paddle::Tensor &token_ids_all,
                              const paddle::Tensor &prompt_lens,
                              const paddle::Tensor &step_idx,
                              const paddle::Tensor &end_tokens,
                              const paddle::Tensor &max_dec_len) {
  const int real_bsz = seq_lens_this_time.shape()[0];
  const int max_bsz = stop_flags.shape()[0];
  PADDLE_ENFORCE_LE(
      max_bsz,
      1024,
      phi::errors::InvalidArgument(
          "unified_update_model_status: max_bsz (%d) must be <= 1024 "
          "(single-block launch limit).",
          max_bsz));
  const int max_step_tokens = step_input_ids.shape()[1];
  const int max_model_len = token_ids_all.shape()[1];
  const int num_end_tokens = end_tokens.shape()[0];

  constexpr int BlockSize = 1024;

  // has_running_seqs is CPU tensor, need to copy to GPU first
  unified_update_model_status_kernel<BlockSize>
      <<<1, BlockSize, 0, seq_lens_this_time.stream()>>>(
          const_cast<int *>(seq_lens_encoder.data<int>()),
          const_cast<int *>(seq_lens_decoder.data<int>()),
          const_cast<bool *>(has_running_seqs.data<bool>()),
          const_cast<int64_t *>(step_input_ids.data<int64_t>()),
          const_cast<int64_t *>(step_output_ids.data<int64_t>()),
          const_cast<int *>(step_output_len.data<int>()),
          const_cast<bool *>(stop_flags.data<bool>()),
          const_cast<int *>(seq_lens_this_time.data<int>()),
          const_cast<bool *>(is_paused.data<bool>()),
          const_cast<int64_t *>(token_ids_all.data<int64_t>()),
          prompt_lens.data<int64_t>(),
          const_cast<int64_t *>(step_idx.data<int64_t>()),
          end_tokens.data<int64_t>(),
          max_dec_len.data<int64_t>(),
          real_bsz,
          max_bsz,
          max_step_tokens,
          max_model_len,
          num_end_tokens);
}

PD_BUILD_STATIC_OP(unified_update_model_status)
    .Inputs({"seq_lens_encoder",
             "seq_lens_decoder",
             "has_running_seqs",
             "step_input_ids",
             "step_output_ids",
             "step_output_len",
             "stop_flags",
             "seq_lens_this_time",
             "is_paused",
             "token_ids_all",
             "prompt_lens",
             "step_idx",
             "end_tokens",
             "max_dec_len"})
    .Outputs({"seq_lens_encoder_out",
              "seq_lens_decoder_out",
              "has_running_seqs_out",
              "step_input_ids_out",
              "step_output_ids_out",
              "step_output_len_out",
              "stop_flags_out",
              "seq_lens_this_time_out",
              "token_ids_all_out",
              "step_idx_out"})
    .SetInplaceMap({{"seq_lens_encoder", "seq_lens_encoder_out"},
                    {"seq_lens_decoder", "seq_lens_decoder_out"},
                    {"has_running_seqs", "has_running_seqs_out"},
                    {"step_input_ids", "step_input_ids_out"},
                    {"step_output_ids", "step_output_ids_out"},
                    {"step_output_len", "step_output_len_out"},
                    {"stop_flags", "stop_flags_out"},
                    {"seq_lens_this_time", "seq_lens_this_time_out"},
                    {"token_ids_all", "token_ids_all_out"},
                    {"step_idx", "step_idx_out"}})
    .SetKernelFn(PD_KERNEL(UnifiedUpdateModelStatus));
