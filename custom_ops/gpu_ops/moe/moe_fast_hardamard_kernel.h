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

#pragma once

#include "helper.h"

template <typename T, typename OutT>
void MoeFastHardamardWrapper(const T *x_data,
                             const int64_t *expert_idx_per_token,
                             const int64_t *recv_expert_count,
                             const T *shift,
                             const T *smooth,
                             const float *quant_scales,
                             const int quant_round_type,
                             const float quant_max_bound,
                             const float quant_min_bound,
                             const int64_t token_num,
                             const int64_t dim,
                             const int num_max_tokens_per_expert,
                             bool used_in_ep_low_latency,
                             const int hadamard_block_size,
                             OutT *out,
                             cudaStream_t &stream);
