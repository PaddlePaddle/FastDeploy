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

#include "paddle/extension.h"

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

__global__ void SpeculateHydraSetScoreThresholdKernel(
    float* threshold,
    const int* seq_lens_this_time,
    const int* seq_lens_encoder,
    const int* accept_num,
    const int real_bsz,
    const float default_threshold = 0.3,
    const float upper_threshold = 0.8,
    const float lower_threshold = 0.0,
    const float threshold_step = 0.1,
    const float threshold_step_fac = 0.5) {
    for (int bid = threadIdx.x; bid < real_bsz; bid += blockDim.x) {
        if (seq_lens_encoder[bid] > 0) {
            threshold[bid] = default_threshold;
        } else if (seq_lens_this_time[bid] <= 1) {
            continue;
        } else if (accept_num[bid] >= seq_lens_this_time[bid] &&
                   threshold[bid] >
                       lower_threshold + threshold_step * threshold_step_fac) {
            threshold[bid] -= threshold_step * threshold_step_fac;
        } else if (accept_num[bid] < seq_lens_this_time[bid] &&
                   threshold[bid] < upper_threshold - threshold_step) {
            threshold[bid] += threshold_step;
        }
    }
}

void SpeculateHydraSetScoreThreshold(const paddle::Tensor& seq_lens_this_time,
                                     const paddle::Tensor& seq_lens_encoder,
                                     const paddle::Tensor& accept_num,
                                     const paddle::Tensor& threshold) {
    auto cu_stream = seq_lens_this_time.stream();
    std::vector<int64_t> seq_lens_this_time_shape = seq_lens_this_time.shape();
    const int bsz = seq_lens_this_time_shape[0];

    SpeculateHydraSetScoreThresholdKernel<<<1, 256, 0, cu_stream>>>(
        const_cast<float*>(threshold.data<float>()),
        seq_lens_this_time.data<int>(),
        seq_lens_encoder.data<int>(),
        accept_num.data<int>(),
        bsz);
}

PD_BUILD_STATIC_OP(speculate_hydra_set_score_threshold)
    .Inputs(
        {"seq_lens_this_time", "seq_lens_encoder", "accept_num", "threshold"})
    .Outputs({"threshold_out"})
    .SetInplaceMap({{"threshold", "threshold_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateHydraSetScoreThreshold));
