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

template <int VecSize>
__global__ void text_image_index_out_kernel(
    const int32_t* token_type_ids,
    int32_t* text_index,
    int32_t* image_index,
    const int64_t token_num
) {
    int global_thread_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (global_thread_idx >= 1) return;
    int text_count = 0;
    int images_count = 0;

    for (int i = 0; i < token_num; ++i) {
        // printf(" %d %d  %d %d \n", text_index[i], text_count, images_count, i);
        if (token_type_ids[i] == 0) {
            text_index[i] = text_count;
            text_count += 1;
        } else if (token_type_ids[i] == 1) {
            image_index[i] = images_count;
            images_count += 1;
        } else {
            // skip cuda graph padding value
            continue;
        }
    }
}

void TextImageIndexOut(
            const paddle::Tensor& token_type_ids,
             paddle::Tensor& text_index,
             paddle::Tensor& image_index) {

    const int64_t token_num = token_type_ids.shape()[0];
    auto stream = token_type_ids.stream();
    text_image_index_out_kernel<1><<<1, 1, 0, stream>>>(
        token_type_ids.data<int32_t>(),
        text_index.data<int32_t>(),
        image_index.data<int32_t>(),
        token_num
    );
}


PD_BUILD_STATIC_OP(text_image_index_out)
    .Inputs({"token_type_ids",
             "text_index",
             "image_index"})
    .Outputs({"text_index_out",
              "image_index_out"})
    .SetInplaceMap({{"text_index", "text_index_out"},
                    {"image_index", "image_index_out"}})
    .SetKernelFn(PD_KERNEL(TextImageIndexOut));
