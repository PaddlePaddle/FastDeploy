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


std::vector<paddle::Tensor> GetMmSplitFuseV2(const paddle::Tensor& task_input_ids,
                            const paddle::Tensor& task_image_type_ids,
                            const paddle::Tensor& task_input_ids_image_token_count,
                            const paddle::Tensor& grid_thw,
                            int64_t image_token_id,
                            int64_t img_total,
                            int seq_lens_origin,
                            int split_fuse_text_size) {
    // All tensor in cpu
    auto input_ids_cpu = task_input_ids.data<int64_t>();
    auto task_input_ids_image_token_count_cpu = task_input_ids_image_token_count.data<int>();
    auto grid_thw_cpu = grid_thw.data<int64_t>();
    std::vector<int> image_chunk_selections_vector;  // 当前chunk 图片数目
    std::vector<int> split_fuse_cur_seq_lens_vector; // 当前chunk 长度
    std::vector<int> split_fuse_cur_mm_lens_vector;  // 当前chunk mm_token数目
    // [预处理] 记录可划分chunk的位置
    std::map<int, int> mp;
    mp[0] = 1; // init
    int st_idx = 0, last_st_ib = 0;
    int idx = 0;
    while (st_idx < seq_lens_origin) {
        // 1. 当前st_idx为文本，找到文本末尾
        if (input_ids_cpu[st_idx] != image_token_id) {
            do {
                st_idx ++;
            } while (st_idx < seq_lens_origin && input_ids_cpu[st_idx] != image_token_id);
            mp[st_idx] = 1; // 记录划分chunk的末尾位置，此处为文本的末位+1
        } else { // 2. 当前 st_idx 为多模，根据多模token的长度找到末尾
            int ib = last_st_ib;
            int cur_st_len = 0;
            int token_times = 4;
            cur_st_len = (grid_thw_cpu[ib * 3 + 1] * grid_thw_cpu[ib * 3 + 2]) / token_times;
            mp[st_idx + cur_st_len] = 1;
            last_st_ib = ++ib;
            st_idx += cur_st_len;
        }
    }
    int chunk_image_number = 0;
    int last_id = 0;
    for (idx = 0; idx < seq_lens_origin; idx++) {
        if (mp[idx] == 1 && input_ids_cpu[idx] == image_token_id) {
            chunk_image_number ++;
        }
        if (idx > 0 && (idx + 1) % split_fuse_text_size == 0 || idx == seq_lens_origin - 1) {
            int chunk_start = last_id * split_fuse_text_size;
            int chunk_end = idx;
            int chunk_image_token_number = task_input_ids_image_token_count_cpu[chunk_end + 1] - task_input_ids_image_token_count_cpu[chunk_start];
            image_chunk_selections_vector.emplace_back(chunk_image_number);
            split_fuse_cur_seq_lens_vector.emplace_back(chunk_end - chunk_start + 1);
            split_fuse_cur_mm_lens_vector.emplace_back(chunk_image_token_number);
            chunk_image_number = 0;
            last_id = (idx + 1) / split_fuse_text_size;
        }
    }
    // vector to cpu tensor
    auto image_chunk_selections_out_cpu = paddle::from_blob(image_chunk_selections_vector.data(), {image_chunk_selections_vector.size()}, task_image_type_ids.dtype());
    auto split_fuse_cur_seq_lens_out_cpu = paddle::from_blob(split_fuse_cur_seq_lens_vector.data(), {split_fuse_cur_seq_lens_vector.size()}, task_image_type_ids.dtype());
    auto split_fuse_cur_mm_lens_out_cpu = paddle::from_blob(split_fuse_cur_mm_lens_vector.data(), {split_fuse_cur_mm_lens_vector.size()}, task_image_type_ids.dtype());
    // cpu tensor to gpu tensor
    auto image_chunk_selections_out = paddle::experimental::copy_to(image_chunk_selections_out_cpu, task_image_type_ids.place(), false);
    auto split_fuse_cur_seq_lens_out = paddle::experimental::copy_to(split_fuse_cur_seq_lens_out_cpu, task_image_type_ids.place(), false);
    auto split_fuse_cur_mm_lens_out = paddle::experimental::copy_to(split_fuse_cur_mm_lens_out_cpu, task_image_type_ids.place(), false);
    return {image_chunk_selections_out, split_fuse_cur_seq_lens_out, split_fuse_cur_mm_lens_out};
}

PD_BUILD_OP(get_mm_split_fuse_v2)
    .Inputs({"task_input_ids", "task_image_type_ids", "task_input_ids_image_token_count", "grid_thw"})
    .Attrs({"image_token_id: int64_t", "img_total: int64_t", "seq_lens_origin: int", "split_fuse_text_size: int"})
    .Outputs({"image_chunk_selections", "split_fuse_cur_seq_lens", "split_fuse_cur_mm_lens_out"})
    .SetKernelFn(PD_KERNEL(GetMmSplitFuseV2));
