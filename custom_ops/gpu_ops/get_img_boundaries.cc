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
#include <map>

std::vector<paddle::Tensor> GetImgBoundaries(
                            const paddle::Tensor& task_input_ids,
                            const paddle::Tensor& grid_thw,
                            int64_t image_token_id) {
    // All tensor in cpu
    auto input_ids_cpu = task_input_ids.data<int64_t>();
    int64_t seq_lens_origin = task_input_ids.numel();
    auto grid_thw_cpu = grid_thw.data<int64_t>();
    std::vector<int> img_boundaries;
    img_boundaries.emplace_back(0);

    int st_idx = 0;
    int last_st_ib = 0;
    while (st_idx < seq_lens_origin) {
        if (input_ids_cpu[st_idx] != image_token_id) { // 1. 当前st_idx为文本，找到文本末尾
            do {
                st_idx ++;
            } while (st_idx < seq_lens_origin && input_ids_cpu[st_idx] != image_token_id);
            img_boundaries.emplace_back(st_idx); // 记录划分chunk的末尾位置，此处为文本的末位+1
        } else { // 2. 当前st_idx为多模，根据多模token的长度找到末尾
            int ib = last_st_ib;
            int cur_st_len = 0;
            int token_times = 4;
            cur_st_len = (grid_thw_cpu[ib * 3 + 1] * grid_thw_cpu[ib * 3 + 2]) / token_times;
            img_boundaries.emplace_back(st_idx + cur_st_len);
            last_st_ib = ++ib;
            st_idx += cur_st_len;
        }
    }

    auto out = paddle::full({static_cast<int64_t>(img_boundaries.size())}, 0, paddle::DataType::INT64, paddle::CPUPlace());

    for (int i = 0; i < img_boundaries.size(); i++) {
        out.data<int64_t>()[i] = img_boundaries[i];
    }

    return {out};
}

PD_BUILD_OP(get_img_boundaries)
    .Inputs({"task_input_ids", "grid_thw"})
    .Attrs({"image_token_id: int64_t"})
    .Outputs({"img_boundaries"})
    .SetKernelFn(PD_KERNEL(GetImgBoundaries));
