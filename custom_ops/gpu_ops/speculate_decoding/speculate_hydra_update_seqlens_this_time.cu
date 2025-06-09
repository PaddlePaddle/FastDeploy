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

__global__ void hydra_update_this_time(int* seq_lens_this_time,
                                       const int* seq_lens_encoder,
                                       const int* seq_lens_decoder,
                                       const float* topk_scores,
                                       const float* score_threshold,
                                       int real_bsz,
                                       int idx) {
    int linear_idx = threadIdx.x;
    // verify and set stop flags
    for (; linear_idx < real_bsz; linear_idx += blockDim.x) {
        if (seq_lens_encoder[linear_idx] == 0 &&
            seq_lens_decoder[linear_idx] != 0) {
            if (topk_scores[linear_idx] > score_threshold[linear_idx] &&
                seq_lens_this_time[linear_idx] == idx + 1) {
                seq_lens_this_time[linear_idx]++;
            }
        } else if (seq_lens_encoder[linear_idx] == 0 &&
                   seq_lens_decoder[linear_idx] == 0) {
            seq_lens_this_time[linear_idx] = 0;
        }
    }
}

void HydraUpdateThisTime(const paddle::Tensor& seq_lens_this_time,
                         const paddle::Tensor& seq_lens_encoder,
                         const paddle::Tensor& seq_lens_decoder,
                         const paddle::Tensor& topk_scores,
                         const paddle::Tensor& score_threshold,
                         const int real_bsz,
                         const int idx) {
    constexpr int BlockSize = 512;

    hydra_update_this_time<<<1, BlockSize, 0, seq_lens_this_time.stream()>>>(
        const_cast<int*>(seq_lens_this_time.data<int>()),
        seq_lens_encoder.data<int>(),
        seq_lens_decoder.data<int>(),
        topk_scores.data<float>(),
        score_threshold.data<float>(),
        real_bsz,
        idx);
}

PD_BUILD_STATIC_OP(speculate_hydra_update_seqlens_this_time)
    .Inputs({"seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "topk_scores",
             "score_threshold"})
    .Outputs({"seq_lens_this_time_out"})
    .Attrs({"real_bsz: int", "idx: int"})
    .SetInplaceMap({{"seq_lens_this_time", "seq_lens_this_time_out"}})
    .SetKernelFn(PD_KERNEL(HydraUpdateThisTime));
