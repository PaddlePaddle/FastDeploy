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

__global__ void GetPositionIdsAndMaskEncoderBatchKernel(
    const int* seq_lens_encoder,  // [bsz] 每个批次的 encoder 长度
    const int* seq_lens_decoder,  // [bsz] 每个批次的 decoder 长度 (对于 MLA
                                  // prefix cache，这是 cached KV 长度)
    const int* seq_lens_this_time,
    int* position_ids,  // 输出的一维 position_ids
    const int bsz) {    // 批次大小
  // 当前线程索引（每个线程对应一个批次）
  int tid = threadIdx.x;
  if (tid >= bsz) return;

  // 动态计算当前批次的偏移量。
  // 每个 batch 只会贡献 encoder_len 或 seq_lens_this_time 中的一个，
  // 而非两者之和（chunked prefill 时 encoder_len > 0 与 decoder_len > 0
  // 同时成立，
  //  但该 batch 只有 encoder_len 个真实 token）。
  int offset = 0;
  for (int i = 0; i < tid; i++) {
    if (seq_lens_encoder[i] > 0) {
      offset += seq_lens_encoder[i];
    } else if (seq_lens_decoder[i] > 0) {
      offset += seq_lens_this_time[i];
    }
  }

  // 当前批次的 encoder 和 decoder 长度
  int encoder_len = seq_lens_encoder[tid];
  int decoder_len = seq_lens_decoder[tid];
  int seq_len_this_time = seq_lens_this_time[tid];

  // For MLA with prefix cache support:
  // - Chunked prefill (encoder_len > 0 && decoder_len > 0):
  //   only writes encoder positions starting at decoder_len (cached length).
  // - First-chunk prefill (encoder_len > 0 && decoder_len == 0):
  //   writes encoder positions starting at 0.
  // - Pure decode (encoder_len == 0 && decoder_len > 0):
  //   writes seq_lens_this_time decode positions starting at decoder_len.
  if (encoder_len > 0) {
    int start_pos = (decoder_len > 0) ? decoder_len : 0;
    for (int i = 0; i < encoder_len; i++) {
      position_ids[offset + i] = start_pos + i;
    }
  } else if (decoder_len > 0) {
    for (int i = 0; i < seq_len_this_time; i++) {
      position_ids[offset + i] = decoder_len + i;
    }
  }
}

void GetPositionIdsAndMaskEncoderBatch(const paddle::Tensor& seq_lens_encoder,
                                       const paddle::Tensor& seq_lens_decoder,
                                       const paddle::Tensor& seq_lens_this_time,
                                       const paddle::Tensor& position_ids) {
  const int bsz = seq_lens_this_time.shape()[0];

  GetPositionIdsAndMaskEncoderBatchKernel<<<1, bsz, 0, position_ids.stream()>>>(
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      seq_lens_this_time.data<int>(),
      const_cast<int*>(position_ids.data<int>()),
      bsz);
}

PD_BUILD_STATIC_OP(get_position_ids_and_mask_encoder_batch)
    .Inputs({
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "position_ids",
    })
    .Outputs({"position_ids_out"})
    .SetInplaceMap({{"position_ids", "position_ids_out"}})
    .SetKernelFn(PD_KERNEL(GetPositionIdsAndMaskEncoderBatch));
