// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>
#include "helper.h"

// ================================================================
// Reasoning Phase State Machine
//
// reasoning_status meanings:
//
//   x = 0 : Thinking phase
//         - Model is generating hidden reasoning content
//         - No token constraint is applied
//
//         Transition condition (x = 0 -> x = 1):
//         - Check whether <think_end> token appears
//           in the last 4 generated tokens
//
// ------------------------------------------------
//
//   x = 1 : Generating "\n</think>\n\n"
//         - Model is emitting the explicit boundary pattern
//         - In non-MTP mode, accept_num is implicitly 1
//           and does not need to be manually set
//         - In MTP mode, accept_num must be 1 in verify kernel
//
//         Transition condition (x = 1 -> x = 2):
//         - step_idx >= 3
//         - pre_ids[-4:] exactly match:
//               "\n</think>\n\n"
//
// ------------------------------------------------
//
//   x = 2 : Generating <response> / <tool_call> phase
//         - Model starts generating visible response or tool calls
//         - Token constraint is enforced at the first token of this phase
//         - Logits are masked to allow only a predefined token set
//
//         Kernel applied:
//         - apply_token_enforce_generation_scores_kernel
//
//         Transition condition (x = 2 -> x = 3):
//         - Automatically advance after one step
//
// ------------------------------------------------
//
//   x = 3 : End state
//         - Reasoning boundary handling is complete
//         - No further state transitions
//
// ================================================================
__global__ void update_reasoning_status_kernel(
    const bool* stop_flags,       // [bs]
    const int* seq_lens_encoder,  // [bs]
    const int64_t* step_idx,      // [bs]
    const int64_t* pre_ids,       // [bs, max_seq_len]
    const bool* enable_thinking,  // [bs]
    int32_t* reasoning_status,    // [bs]
    int32_t bs,
    int32_t max_seq_len,
    int64_t think_end_id,
    int64_t line_break_id) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= bs) return;
  bool enable_thinking_flag = enable_thinking[tid];
  int32_t status = reasoning_status[tid];
  if (stop_flags[tid] || status == 3) return;

  int64_t cur_step = step_idx[tid];
  const int64_t* pre_ids_now = pre_ids + tid * max_seq_len;
  int64_t t0 = (cur_step >= 0) ? pre_ids_now[cur_step] : -1;
  int64_t t1 = (cur_step >= 1) ? pre_ids_now[cur_step - 1] : -1;
  int64_t t2 = (cur_step >= 2) ? pre_ids_now[cur_step - 2] : -1;
  int64_t t3 = (cur_step >= 3) ? pre_ids_now[cur_step - 3] : -1;

  int32_t new_status = status;

  // x = 0 -> x = 1
  if (status == 0) {
    if (!enable_thinking_flag && seq_lens_encoder[tid] > 0 && cur_step == 0) {
      // x = 0 -> x = 2 (only for first token when thinking is disabled)
      new_status = 2;
    } else if (t0 == think_end_id || t1 == think_end_id || t2 == think_end_id ||
               t3 == think_end_id) {
      new_status = 1;
    }
  }

  // x = 1 -> x = 2 (include think_end_id)
  // or x = 1 -> x = 3 (not include think_end_id)
  // Here must be serial judge
  if (new_status == 1 && cur_step >= 3) {
    if (t3 == line_break_id && t2 == think_end_id && t1 == line_break_id &&
        t0 == line_break_id) {
      new_status = 2;
    } else if (t3 != think_end_id && t2 != think_end_id && t1 != think_end_id &&
               t0 != think_end_id) {
      new_status = 3;
    }
  } else if (status == 2) {
    // x = 2 -> x = 3
    new_status = 3;
  }
  reasoning_status[tid] = new_status;
}
// ================================================================
// Kernel 2: apply enforce generation scores
// ================================================================
template <typename T>
__global__ void apply_token_enforce_generation_scores_kernel(
    const T* __restrict__ logits_src,            // logits_tmp (backup)
    T* __restrict__ logits_dst,                  // logits (output)
    const int64_t* __restrict__ allowed_tokens,  // [allowed_len]
    const int32_t* __restrict__ reasoning_status,
    const int* batch_id_per_token_output,
    const int* cu_seqlens_q_output,
    const int max_bsz,
    const int max_seq_len,
    const int vocab_size,
    const int allowed_tokens_len) {
  int token_idx = blockIdx.x;
  int tid = threadIdx.x;

  const int bs_idx = batch_id_per_token_output[token_idx];
  const int query_start_token_idx = cu_seqlens_q_output[bs_idx];
  bool is_batch_first_token = (token_idx == query_start_token_idx);

  if (allowed_tokens_len == 0 || !is_batch_first_token) {
    return;
  }

  if (bs_idx < max_bsz && reasoning_status[bs_idx] == 2) {
    const T* src = logits_src + token_idx * vocab_size;
    T* dst = logits_dst + token_idx * vocab_size;

    // 1. clear all logits
    for (int i = tid; i < vocab_size; i += blockDim.x) {
      dst[i] = static_cast<T>(-1e10f);
    }
    __syncthreads();

    // 2. restore allowed tokens
    for (int i = tid; i < allowed_tokens_len; i += blockDim.x) {
      int64_t token_id = allowed_tokens[i];
      if ((unsigned)token_id < (unsigned)vocab_size) {
        dst[token_id] = src[token_id];
      }
    }
  }
}

// ================================================================
// C++ Launcher
// ================================================================
template <paddle::DataType D>
void reasoning_phase_token_constraint(
    const paddle::Tensor& logits,  // inplace output
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& allowed_tokens,
    const paddle::Tensor& reasoning_status,
    const paddle::Tensor& batch_id_per_token_output,
    const paddle::Tensor& cu_seqlens_q_output,
    const paddle::Tensor& enable_thinking,
    int64_t think_end_id,
    int64_t line_break_id) {
  typedef PDTraits<D> traits_;
  typedef typename traits_::DataType DataType_;
  typedef typename traits_::data_t data_t;

  auto stream = logits.stream();

  int bs = seq_lens_this_time.shape()[0];
  int token_num = logits.shape()[0];
  int vocab_size = logits.shape()[1];
  int max_seq_len = pre_ids.shape()[1];
  int allowed_tokens_len = allowed_tokens.shape()[0];

  // ------------------------------------------------
  // Kernel 1: update reasoning status
  // ------------------------------------------------
  // int block1 = (bs + 31) / 32 * 32;

  const int block_size = 512;
  const int gird_size = (bs + block_size - 1) / block_size;

  update_reasoning_status_kernel<<<gird_size, block_size, 0, stream>>>(
      stop_flags.data<bool>(),
      seq_lens_encoder.data<int>(),
      step_idx.data<int64_t>(),
      pre_ids.data<int64_t>(),
      enable_thinking.data<bool>(),
      const_cast<int32_t*>(reasoning_status.data<int32_t>()),
      bs,
      max_seq_len,
      think_end_id,
      line_break_id);

  // ------------------------------------------------
  // backup logits
  // ------------------------------------------------
  auto logits_tmp = logits.copy_to(logits.place(), false);

  // ------------------------------------------------
  // Kernel 2: enforce generation
  // ------------------------------------------------
  int block_size_2 = (vocab_size + 31) / 32 * 32;
  block_size_2 = std::min(block_size_2, 512);

  apply_token_enforce_generation_scores_kernel<<<token_num,
                                                 block_size_2,
                                                 0,
                                                 stream>>>(
      reinterpret_cast<DataType_*>(logits_tmp.data<data_t>()),
      reinterpret_cast<DataType_*>(const_cast<data_t*>(logits.data<data_t>())),
      allowed_tokens.data<int64_t>(),
      reasoning_status.data<int32_t>(),
      batch_id_per_token_output.data<int32_t>(),
      cu_seqlens_q_output.data<int32_t>(),
      bs,
      max_seq_len,
      vocab_size,
      allowed_tokens_len);
}

void ReasoningPhaseTokenConstraint(
    const paddle::Tensor& logits,
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& seq_lens_this_time,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& allowed_tokens,
    const paddle::Tensor& reasoning_status,
    const paddle::Tensor& batch_id_per_token_output,
    const paddle::Tensor& cu_seqlens_q_output,
    const paddle::Tensor& enable_thinking,
    int64_t think_end_id,
    int64_t line_break_id) {
  switch (logits.type()) {
    case paddle::DataType::FLOAT16:
      return reasoning_phase_token_constraint<paddle::DataType::FLOAT16>(
          logits,
          pre_ids,
          stop_flags,
          seq_lens_this_time,
          seq_lens_encoder,
          step_idx,
          allowed_tokens,
          reasoning_status,
          batch_id_per_token_output,
          cu_seqlens_q_output,
          enable_thinking,
          think_end_id,
          line_break_id);
    case paddle::DataType::BFLOAT16:
      return reasoning_phase_token_constraint<paddle::DataType::BFLOAT16>(
          logits,
          pre_ids,
          stop_flags,
          seq_lens_this_time,
          seq_lens_encoder,
          step_idx,
          allowed_tokens,
          reasoning_status,
          batch_id_per_token_output,
          cu_seqlens_q_output,
          enable_thinking,
          think_end_id,
          line_break_id);
    case paddle::DataType::FLOAT32:
      return reasoning_phase_token_constraint<paddle::DataType::FLOAT32>(
          logits,
          pre_ids,
          stop_flags,
          seq_lens_this_time,
          seq_lens_encoder,
          step_idx,
          allowed_tokens,
          reasoning_status,
          batch_id_per_token_output,
          cu_seqlens_q_output,
          enable_thinking,
          think_end_id,
          line_break_id);
    default:
      PD_THROW("Unsupported data type.");
  }
}

// ================================================================
// PD_BUILD_STATIC_OP
// ================================================================
PD_BUILD_STATIC_OP(reasoning_phase_token_constraint)
    .Inputs({"logits",
             "pre_ids",
             "stop_flags",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "step_idx",
             "allowed_tokens",
             "reasoning_status",
             "batch_id_per_token_output",
             "cu_seqlens_q_output",
             "enable_thinking"})
    .Outputs({"logits_out", "reasoning_status_out"})
    .Attrs({"think_end_id: int64_t", "line_break_id: int64_t"})
    .SetInplaceMap({{"logits", "logits_out"},
                    {"reasoning_status", "reasoning_status_out"}})
    .SetKernelFn(PD_KERNEL(ReasoningPhaseTokenConstraint));
