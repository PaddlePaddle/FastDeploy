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

std::vector<paddle::Tensor> GetImgBoundaries(
    const paddle::Tensor& task_input_ids,
    const paddle::Tensor& grid_thw,
    const int64_t image_patch_id) {
  // All tensor in cpu
  auto input_ids_ptr = task_input_ids.data<int64_t>();
  int64_t seq_lens_origin = task_input_ids.numel();
  auto grid_thw_ptr = grid_thw.data<int64_t>();

  int token_times = 4;
  int token_idx = 0;
  int image_idx = 0;
  std::vector<int> img_boundaries, img_nums;
  img_boundaries.emplace_back(0);
  img_nums.emplace_back(0);
  while (token_idx < seq_lens_origin) {
    if (input_ids_ptr[token_idx] != image_patch_id) {
      do {
        token_idx++;
      } while (token_idx < seq_lens_origin &&
               input_ids_ptr[token_idx] != image_patch_id);
    } else {
      int cur_image_token_len =
          (grid_thw_ptr[image_idx * 3 + 1] * grid_thw_ptr[image_idx * 3 + 2]) /
          token_times;
      image_idx++;
      token_idx += cur_image_token_len;
    }
    img_boundaries.emplace_back(token_idx);
    img_nums.emplace_back(image_idx);
  }

  int64_t num_img_boundaries = static_cast<int64_t>(img_boundaries.size());
  auto out = paddle::full(
      {2, num_img_boundaries}, 0, paddle::DataType::INT64, paddle::CPUPlace());

  for (int i = 0; i < num_img_boundaries; i++) {
    out.data<int64_t>()[i] = img_boundaries[i];
    out.data<int64_t>()[num_img_boundaries + i] = img_nums[i];
  }

  return {out};
}

PD_BUILD_OP(get_img_boundaries)
    .Inputs({"task_input_ids", "grid_thw"})
    .Attrs({"image_patch_id: int64_t"})
    .Outputs({"img_boundaries"})
    .SetKernelFn(PD_KERNEL(GetImgBoundaries));
