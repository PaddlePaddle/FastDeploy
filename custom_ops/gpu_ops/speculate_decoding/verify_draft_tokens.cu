// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

// Pure verification kernel — outputs step_output_ids + step_output_len only.
// All state management (step_idx, stop_flags, EOS/max_dec_len detection)
// is handled by unified_update_model_status.
//
// Verification strategies:
//   0 = TOPP         : draft token in top-p candidate set (+ verify_window
//   fallback) 1 = GREEDY       : draft token == top-1 token (strict argmax
//   match) 2 = TARGET_MATCH : draft token == target model's sampled token

#include <curand_kernel.h>
#include "helper.h"  // NOLINT

// ============================================================
// Persistent curand state — allocated once, reused across calls.
// Only needed for TOPP strategy (Phase 2 stochastic sampling).
// ============================================================
static curandState_t *dev_curand_states = nullptr;
static int allocated_bsz = 0;
static uint64_t seed = 0;
static uint64_t offset = 0;

__global__ void setup_seed_kernel(curandState_t *state,
                                  const uint64_t seed,
                                  const uint64_t offset,
                                  const int bs,
                                  const bool need_batch_random) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < bs; i += gridDim.x * blockDim.x) {
    if (need_batch_random) {
      curand_init(seed, i, offset, &state[i]);
    } else {
      curand_init(seed, 0, offset, &state[i]);
    }
  }
}

// ============================================================
// Phase 1 helpers — single-step draft token verification
// ============================================================

// Check if draft_token appears in the candidate set
__device__ inline bool is_in(const int64_t *candidates,
                             const int64_t draft,
                             const int candidate_len) {
  for (int i = 0; i < candidate_len; i++) {
    if (draft == candidates[i]) return true;
  }
  return false;
}

// TOPP: draft in top-p filtered candidate set
__device__ inline bool verify_one_topp(const int64_t *verify_tokens_row,
                                       int64_t draft_token,
                                       int actual_cand_len) {
  return is_in(verify_tokens_row, draft_token, actual_cand_len);
}

// GREEDY / TARGET_MATCH: exact single-token match
__device__ inline bool verify_one_match(int64_t target_token,
                                        int64_t draft_token) {
  return target_token == draft_token;
}

// ============================================================
// TOPP-only: verify_window bulk-accept fallback
//
// When draft token is NOT in top-p set but IS the top-2 token,
// check verify_window consecutive positions for top-1 match.
// If all match, bulk-accept from position i through ii.
//
// Returns the new loop position (i) after handling.
// Sets *stopped=true and adjusts *accept_num_now if EOS hit.
// Sets *rejected=true if fallback was not triggered (caller should break).
// ============================================================
__device__ inline int try_verify_window_fallback(
    int bid,
    int i,
    int *output_len_now,
    bool *stopped,
    bool *rejected,
    const int64_t *verify_tokens_now,
    const int64_t *step_input_ids_now,
    int64_t *step_output_ids,
    const int64_t *end_tokens,
    int seq_len_this_time,
    int max_candidate_len,
    int max_step_tokens,
    int end_length,
    int verify_window) {
  int ii = i;
  if (max_candidate_len >= 2 && verify_tokens_now[ii * max_candidate_len + 1] ==
                                    step_input_ids_now[ii + 1]) {
    // top-2 matches — scan verify_window consecutive top-1 matches
    int j = 0;
    ii += 1;
    for (; j < verify_window && ii < seq_len_this_time - 1; j++, ii++) {
      if (verify_tokens_now[ii * max_candidate_len] !=
          step_input_ids_now[ii + 1]) {
        break;
      }
    }
    if (j >= verify_window) {
      // Bulk accept all tokens from i to ii
      for (; i < ii; i++) {
        auto token = step_input_ids_now[i + 1];
        step_output_ids[bid * max_step_tokens + i] = token;
        (*output_len_now)++;
        if (is_in_end(token, end_tokens, end_length)) {
          *stopped = true;
          return i;
        }
      }
      return i;  // continue outer loop from position ii
    }
  }
  // Fallback not triggered or insufficient window — reject
  *rejected = true;
  return i;
}

// ============================================================
// Phase 2 helpers — sample token for rejected/last position
// ============================================================

__device__ inline int64_t topp_sampling_kernel(const int64_t *candidate_ids,
                                               const float *candidate_scores,
                                               curandState_t *curand_states,
                                               const int candidate_len,
                                               const float topp) {
  // Use bid (blockIdx.x-based) index, not threadIdx.x — curand_states is
  // allocated with size bsz, and each batch element uses one thread.
  const int bid = blockIdx.x * blockDim.x + threadIdx.x;
  float sum_scores = 0.0f;
  float rand_top_p = curand_uniform(curand_states + bid) * topp;
  for (int i = 0; i < candidate_len; i++) {
    sum_scores += candidate_scores[i];
    if (rand_top_p <= sum_scores) {
      return candidate_ids[i];
    }
  }
  return candidate_ids[0];
}

// ============================================================
// Main verification kernel
// ============================================================
//
// Input parameter groups by strategy:
//   - target_tokens:        GREEDY=argmax, TARGET_MATCH=sampled, TOPP=unused
//   (None)
//   - candidate_ids/scores: TOPP=full candidate set, GREEDY/TARGET_MATCH=unused
//   (None)
//   - candidate_lens:       TOPP=actual length per position,
//   GREEDY/TARGET_MATCH=unused (None)
//
// All parameters may be empty tensors for strategies that don't use them.
//
__global__ void verify_draft_tokens(
    // Core I/O
    int64_t *step_output_ids,
    int *step_output_len,
    const int64_t *step_input_ids,  // draft tokens
    // Target model outputs (strategy-dependent interpretation)
    const int64_t
        *target_tokens,  // GREEDY:argmax, TARGET_MATCH:sampled, TOPP:unused
    // Candidate set for TOPP/GREEDY (TARGET_MATCH: unused)
    const int64_t *candidate_ids,
    const float *candidate_scores,
    const int *candidate_lens,
    // Sampling params
    curandState_t *curand_states,  // nullptr for GREEDY/TARGET_MATCH
    const float *topp,
    // Metadata
    const bool *stop_flags,
    const int *seq_lens_encoder,
    const int *seq_lens_this_time,
    const int64_t *end_tokens,
    const bool *is_block_step,
    const int *cu_seqlens_q_output,
    const int *reasoning_status,
    // Dimensions and config
    const int max_bsz,
    const int max_step_tokens,
    const int end_length,
    const int max_seq_len,
    const int max_candidate_len,
    const int verify_window,
    const int verify_strategy,  // 0=TOPP, 1=GREEDY, 2=TARGET_MATCH
    const bool reject_all,
    const bool accept_all) {
  const int bid = threadIdx.x;
  int output_len_now = 1;
  bool stopped = false;

  // Initialize step_output_len to 0 for ALL slots
  if (bid < max_bsz) {
    step_output_len[bid] = 0;
  }

  if (bid >= real_bsz || is_block_step[bid] || stop_flags[bid]) return;

  const int start_token_id = cu_seqlens_q_output[bid];
  // Pointers are strategy-dependent (may be nullptr for unused params)
  auto *candidate_ids_now =
      candidate_ids ? candidate_ids + start_token_id * max_candidate_len
                    : nullptr;
  auto *candidate_scores_now =
      candidate_scores ? candidate_scores + start_token_id * max_candidate_len
                       : nullptr;
  auto *candidate_lens_now =
      candidate_lens ? candidate_lens + start_token_id : nullptr;
  auto *target_tokens_now =
      target_tokens ? target_tokens + start_token_id : nullptr;
  auto *step_input_ids_now = step_input_ids + bid * max_step_tokens;

  // ======== Phase 1: Verify draft tokens ========
  int i = 0;
  for (; i < seq_lens_this_time[bid] - 1; i++) {
    // Early exit conditions: reject-all, prefill, reasoning
    if (reject_all || seq_lens_encoder[bid] != 0 ||
        reasoning_status[bid] == 1) {
      break;
    }

    // Accept-all override (debug/warmup)
    if (accept_all) {
      auto draft_token = step_input_ids_now[i + 1];
      step_output_ids[bid * max_step_tokens + i] = draft_token;
      output_len_now++;
      if (is_in_end(draft_token, end_tokens, end_length)) {
        stopped = true;
        break;
      }
      continue;
    }

    // Strategy dispatch
    bool accepted = false;
    switch (verify_strategy) {
      case 0: {  // TOPP
        auto actual_cand_len = candidate_lens_now[i] > max_candidate_len
                                   ? max_candidate_len
                                   : candidate_lens_now[i];
        accepted = verify_one_topp(candidate_ids_now + i * max_candidate_len,
                                   step_input_ids_now[i + 1],
                                   actual_cand_len);
        if (!accepted) {
          bool rejected = false;
          i = try_verify_window_fallback(bid,
                                         i,
                                         &output_len_now,
                                         &stopped,
                                         &rejected,
                                         candidate_ids_now,
                                         step_input_ids_now,
                                         step_output_ids,
                                         end_tokens,
                                         seq_lens_this_time[bid],
                                         max_candidate_len,
                                         max_step_tokens,
                                         end_length,
                                         verify_window);
          if (stopped || rejected) goto phase1_done;
          continue;  // bulk accept succeeded, continue from new i
        }
        break;
      }
      case 1:  // GREEDY
        accepted =
            verify_one_match(target_tokens_now[i], step_input_ids_now[i + 1]);
        break;
      case 2:  // TARGET_MATCH
        accepted =
            verify_one_match(target_tokens_now[i], step_input_ids_now[i + 1]);
        break;
    }

    if (accepted) {
      step_output_ids[bid * max_step_tokens + i] = step_input_ids_now[i + 1];
      output_len_now++;
      if (is_in_end(step_input_ids_now[i + 1], end_tokens, end_length)) {
        stopped = true;
        break;
      }
    } else {
      break;  // reject
    }
  }
phase1_done:

  // ======== Phase 2: Output token for rejected/last position ========
  if (!stopped) {
    int64_t output_token;
    switch (verify_strategy) {
      case 0: {  // TOPP — stochastic sampling from candidate set
        auto actual_cand_len = candidate_lens_now[i] > max_candidate_len
                                   ? max_candidate_len
                                   : candidate_lens_now[i];
        output_token =
            topp_sampling_kernel(candidate_ids_now + i * max_candidate_len,
                                 candidate_scores_now + i * max_candidate_len,
                                 curand_states,
                                 actual_cand_len,
                                 topp[bid]);
        break;
      }
      case 1:  // GREEDY — deterministic argmax from target_tokens
        output_token = target_tokens_now[i];
        break;
      case 2:  // TARGET_MATCH — target model's sampled token
        output_token = target_tokens_now[i];
        break;
    }
    step_output_ids[bid * max_step_tokens + i] = output_token;
  }
  step_output_len[bid] = output_len_now;
}

// ============================================================
// Host function
// ============================================================
void VerifyDraftTokens(
    // Core I/O
    const paddle::Tensor &step_output_ids,
    const paddle::Tensor &step_output_len,
    const paddle::Tensor &step_input_ids,
    // Target model outputs (optional, required for TARGET_MATCH)
    const paddle::optional<paddle::Tensor> &target_tokens,
    // Candidate set (optional, required for TOPP/GREEDY)
    const paddle::optional<paddle::Tensor> &candidate_ids,
    const paddle::optional<paddle::Tensor> &candidate_scores,
    const paddle::optional<paddle::Tensor> &candidate_lens,
    // Sampling params
    const paddle::Tensor &topp,
    // Metadata
    const paddle::Tensor &stop_flags,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_this_time,
    const paddle::Tensor &end_tokens,
    const paddle::Tensor &is_block_step,
    const paddle::Tensor &cu_seqlens_q_output,
    const paddle::Tensor &reasoning_status,
    int max_seq_len,
    int verify_window,
    int verify_strategy,
    bool reject_all,
    bool accept_all) {
  auto bsz = step_output_ids.shape()[0];
  auto max_step_tokens = step_input_ids.shape()[1];
  auto end_length = end_tokens.shape()[0];
  // max_candidate_len: 1 if candidate_ids not provided, else from shape
  int max_candidate_len = candidate_ids ? candidate_ids->shape()[1] : 1;

  constexpr int BlockSize = 512;
  auto stream = step_output_ids.stream();

  // curand state: only needed for TOPP(0) strategy (stochastic sampling)
  curandState_t *curand_ptr = nullptr;
  if (verify_strategy ==
      0 /* TOPP only - GREEDY and TARGET_MATCH use deterministic output */) {
    if (dev_curand_states == nullptr || bsz > allocated_bsz) {
      if (dev_curand_states) cudaFree(dev_curand_states);
      cudaMalloc(&dev_curand_states, sizeof(curandState_t) * bsz);
      allocated_bsz = bsz;
    }
    setup_seed_kernel<<<1, BlockSize, 0, stream>>>(
        dev_curand_states, seed, offset, bsz, true);
    seed++;
    offset++;
    curand_ptr = dev_curand_states;
  }

  // Get data pointers (nullptr if optional not provided)
  const int64_t *target_tokens_ptr =
      target_tokens ? target_tokens->data<int64_t>() : nullptr;
  const int64_t *candidate_ids_ptr =
      candidate_ids ? candidate_ids->data<int64_t>() : nullptr;
  const float *candidate_scores_ptr =
      candidate_scores ? candidate_scores->data<float>() : nullptr;
  const int *candidate_lens_ptr =
      candidate_lens ? candidate_lens->data<int>() : nullptr;

  // Validate parameters based on verify_strategy.
  // Note: empty_input_forward may lead to empty optional tensors — only
  // validate when bsz > 0 (i.e. there are active sequences).
  if (bsz > 0) {
    if (verify_strategy == 0 /* TOPP */) {
      if (!candidate_ids_ptr || !candidate_scores_ptr || !candidate_lens_ptr) {
        PD_THROW(
            "verify_strategy=TOPP (0) requires candidate_ids, "
            "candidate_scores, candidate_lens");
      }
    } else if (verify_strategy == 1 /* GREEDY */) {
      if (!target_tokens_ptr) {
        PD_THROW("verify_strategy=GREEDY (1) requires target_tokens (argmax)");
      }
    } else if (verify_strategy == 2 /* TARGET_MATCH */) {
      if (!target_tokens_ptr) {
        PD_THROW(
            "verify_strategy=TARGET_MATCH (2) requires target_tokens "
            "(sampled)");
      }
    }
  }

  verify_draft_tokens<<<1, BlockSize, 0, stream>>>(
      // Core I/O
      const_cast<int64_t *>(step_output_ids.data<int64_t>()),
      const_cast<int *>(step_output_len.data<int>()),
      step_input_ids.data<int64_t>(),
      // Target model outputs
      target_tokens_ptr,
      // Candidate set
      candidate_ids_ptr,
      candidate_scores_ptr,
      candidate_lens_ptr,
      // Sampling params
      curand_ptr,
      topp.data<float>(),
      // Metadata
      stop_flags.data<bool>(),
      seq_lens_encoder.data<int>(),
      seq_lens_this_time.data<int>(),
      end_tokens.data<int64_t>(),
      is_block_step.data<bool>(),
      cu_seqlens_q_output.data<int>(),
      reasoning_status.data<int>(),
      // Dimensions and config
      bsz,  // max_bsz
      max_step_tokens,
      end_length,
      max_seq_len,
      max_candidate_len,
      verify_window,
      verify_strategy,
      reject_all,
      accept_all);
}

PD_BUILD_STATIC_OP(verify_draft_tokens)
    .Inputs({"step_output_ids",
             "step_output_len",
             "step_input_ids",
             paddle::Optional("target_tokens"),
             paddle::Optional("candidate_ids"),
             paddle::Optional("candidate_scores"),
             paddle::Optional("candidate_lens"),
             "topp",
             "stop_flags",
             "seq_lens_encoder",
             "seq_lens_this_time",
             "end_tokens",
             "is_block_step",
             "cu_seqlens_q_output",
             "reasoning_status"})
    .Outputs({"step_output_ids_out", "step_output_len_out"})
    .Attrs({"max_seq_len: int",
            "verify_window: int",
            "verify_strategy: int",
            "reject_all: bool",
            "accept_all: bool"})
    .SetInplaceMap({{"step_output_ids", "step_output_ids_out"},
                    {"step_output_len", "step_output_len_out"}})
    .SetKernelFn(PD_KERNEL(VerifyDraftTokens));
