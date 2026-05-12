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

__global__ void GetPositionIdsKernel(const int* __restrict__ seq_lens_encoder,
                                     const int* __restrict__ seq_lens_decoder,
                                     const int* __restrict__ seq_lens_this_time,
                                     int* __restrict__ position_ids,
                                     const int bsz) {
  int current_bid = threadIdx.x;
  if (current_bid >= bsz) return;

  // Caculate the offset of current batch in the position_ids buffer
  int buffer_offset = 0;
  for (int i = 0; i < current_bid; i++) {
    buffer_offset += seq_lens_this_time[i];
  }

  // Caculate the token offset in the current batch
  int token_offset = seq_lens_decoder[current_bid];
  int token_num_this_batch = seq_lens_this_time[current_bid];
  if (token_num_this_batch == 0) return;

// Write position ids for current batch
#pragma unroll
  for (int i = 0; i < token_num_this_batch; i++) {
    position_ids[buffer_offset + i] = token_offset + i;
  }
}

void GetPositionIds(const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& seq_lens_this_time,
                    const paddle::Tensor& position_ids) {
  const int bsz = seq_lens_this_time.shape()[0];

  GetPositionIdsKernel<<<1, bsz, 0, position_ids.stream()>>>(
      seq_lens_encoder.data<int>(),
      seq_lens_decoder.data<int>(),
      seq_lens_this_time.data<int>(),
      const_cast<int*>(position_ids.data<int>()),
      bsz);
}

PD_BUILD_STATIC_OP(get_position_ids)
    .Inputs({
        "seq_lens_encoder",
        "seq_lens_decoder",
        "seq_lens_this_time",
        "position_ids",
    })
    .Outputs({"position_ids_out"})
    .SetInplaceMap({{"position_ids", "position_ids_out"}})
    .SetKernelFn(PD_KERNEL(GetPositionIds));
