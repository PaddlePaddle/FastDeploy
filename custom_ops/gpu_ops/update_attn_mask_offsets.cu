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

__global__ void update_attn_mask_offsets_kernel(
    int* attn_mask_offsets,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* seq_lens_decoder,
    const int* cu_seqlens_q,
    const int* attn_mask_offsets_full,
    int* attn_mask_offsets_decoder,
    const bool* is_block_step,
    int* decode_states,
    int* mask_rollback,
    const int real_bsz,
    const int max_model_len,
    const int decode_states_len) {
  int tid = threadIdx.x;

  for (int bid = tid; bid < real_bsz; bid += blockDim.x) {
    int seq_len_this_time = seq_lens_this_time[bid];
    int seq_len_encoder = seq_lens_encoder[bid];
    int seq_len_decoder = seq_lens_decoder[bid];
    int query_start_id = cu_seqlens_q[bid];

    const int* attn_mask_offsets_full_now =
        attn_mask_offsets_full + bid * max_model_len;
    int* decode_states_now = decode_states + bid * decode_states_len;
    if (!is_block_step[bid]) {
      if (seq_len_encoder == 0 && seq_len_decoder == 0) {
        // Status: stop
      } else if (seq_len_encoder > 0) {
        for (int i = 0; i < seq_len_this_time; i++) {
          if (*decode_states_now == 2 && seq_len_decoder > 0) {
            // Status: vision generate phase
            attn_mask_offsets[(query_start_id + i) * 2 + 1] =
                seq_len_decoder + seq_len_this_time;
          } else {
            // Status: prefill -- normal or chunk_prefill
            attn_mask_offsets[(query_start_id + i) * 2 + 1] =
                attn_mask_offsets_full_now[i] + 1;
          }
        }
      } else if (seq_len_decoder > 0) {
        // Status: decoder -- normal or chunk_prefill
        // TODO: support speculative decoding.
        attn_mask_offsets_decoder[bid] -= mask_rollback[bid];
        mask_rollback[bid] = 0;
        for (int i = 0; i < seq_len_this_time; i++) {
          attn_mask_offsets[(query_start_id + i) * 2 + 1] =
              attn_mask_offsets_decoder[bid] + 1 + i;
        }
        attn_mask_offsets_decoder[bid] += seq_len_this_time;

        // Speculative decoding in text_generation
        if (seq_len_this_time > 1) {
          for (int i = 0; i < decode_states_len; i++) {
            if (i < seq_len_this_time) {
              decode_states_now[i] = 0;
            } else {
              decode_states_now[i] = -1;
            }
          }
        }
      }
    }
  }
}

std::vector<paddle::Tensor> UpdateAttnMaskOffsets(
    const paddle::Tensor& ids_remove_padding,
    const paddle::Tensor& seq_lens_this_time,  // only on cpu
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& cu_seqlens_q,
    const paddle::Tensor& attn_mask_offsets_full,
    const paddle::Tensor& attn_mask_offsets_decoder,
    const paddle::Tensor& is_block_step,
    const paddle::Tensor& decode_states,
    const paddle::Tensor& mask_rollback) {
  int max_model_len = attn_mask_offsets_full.shape()[1];
  int real_bsz = seq_lens_this_time.shape()[0];
  int batch_seq_lens = ids_remove_padding.shape()[0];
  int decode_states_len = decode_states.shape()[1];

  auto attn_mask_offsets = paddle::full({batch_seq_lens * 2},
                                        0,
                                        paddle::DataType::INT32,
                                        ids_remove_padding.place());

  // launch config
  int blockSize = 512;

  update_attn_mask_offsets_kernel<<<1,
                                    blockSize,
                                    0,
                                    ids_remove_padding.stream()>>>(
      attn_mask_offsets.data<int>(),
      seq_lens_this_time.data<int>(),
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      cu_seqlens_q.data<int>(),
      attn_mask_offsets_full.data<int>(),
      const_cast<int*>(attn_mask_offsets_decoder.data<int>()),
      is_block_step.data<bool>(),
      const_cast<int*>(decode_states.data<int>()),
      const_cast<int*>(mask_rollback.data<int>()),
      real_bsz,
      max_model_len,
      decode_states_len);

  return {attn_mask_offsets};
}

PD_BUILD_STATIC_OP(update_attn_mask_offsets)
    .Inputs({"ids_remove_padding",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "cu_seqlens_q",
             "attn_mask_offsets_full",
             "attn_mask_offsets_decoder",
             "is_block_step",
             "decode_states",
             "mask_rollback"})
    .Outputs({"attn_mask_offsets", "decode_states_out", "mask_rollback_out"})
    .SetInplaceMap({{"decode_states", "decode_states_out"},
                    {"mask_rollback", "mask_rollback_out"}})
    .SetKernelFn(PD_KERNEL(UpdateAttnMaskOffsets));
