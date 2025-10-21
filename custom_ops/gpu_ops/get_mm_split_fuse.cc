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

#ifndef PD_BUILD_STATIC_OP
#define PD_BUILD_STATIC_OP(name) PD_BUILD_OP(static_op_##name)
#endif

std::vector<paddle::Tensor> GetMmSplitFuse(const paddle::Tensor& task_input_ids,
                            const paddle::Tensor& task_image_type_ids,
                            const paddle::Tensor& task_input_ids_image_token_count,
                            const paddle::Tensor& grid_thw,
                            int64_t image_token_id,
                            int64_t img_total,
                            int batch_idx,
                            int seq_lens_origin,
                            int split_fuse_img_size,
                            int split_fuse_text_size,
                            int max_chunk_token_size) {
    // All tensor in cpu
    auto input_ids_cpu = task_input_ids.data<int64_t>();
    auto image_type_ids_cpu = task_image_type_ids.data<int>();
    auto task_input_ids_image_token_count_cpu = task_input_ids_image_token_count.data<int>();
    auto grid_thw_cpu = grid_thw.data<int64_t>();
    int chunk_token_count = 0;
    int chunk_image_cout = 0;
    int idx = 0;
    std::vector<int> image_chunk_selections_vector;
    std::vector<int> split_fuse_cur_seq_lens_vector;
    std::vector<int> split_fuse_cur_idx_vector;
    split_fuse_cur_idx_vector.emplace_back(0);
    int image_idx = 0;
    int last_ib = 0;
    // 打表参数, mp记录可划分chunk的位置
    std::map<int, int> mp;
    int st_idx = 0;
    int last_st_ib = 0;
    while (st_idx < seq_lens_origin) {
        // 1. 当前st_idx为文本，找到文本末尾
        if (input_ids_cpu[st_idx] != image_token_id) {
            do {
                st_idx ++;
            } while (st_idx < seq_lens_origin && input_ids_cpu[st_idx] != image_token_id);
            mp[st_idx] = 1; // 记录划分chunk的末尾位置，此处为文本的末位+1
        } else { // 2. 当前st_idx为多模，根据多模token的长度找到末尾
            int ib = last_st_ib;
            int cur_st_len = 0;
            int token_times = 4;
            cur_st_len = (grid_thw_cpu[ib * 3 + 1] * grid_thw_cpu[ib * 3 + 2]) / token_times;
            mp[st_idx + cur_st_len] = 1;
            last_st_ib = ++ib;
            st_idx += cur_st_len;
        }
    }

    while (idx < seq_lens_origin) {
        idx = idx + split_fuse_text_size;
        if (idx >= seq_lens_origin) {
            // idx 超过最大seq_len，应该包含n个图片和文本
            idx = seq_lens_origin;
            int last_idx = split_fuse_cur_idx_vector.back();
            int chunk_image_token_number = task_input_ids_image_token_count_cpu[idx] - task_input_ids_image_token_count_cpu[last_idx];
            int chunk_image_number = 0;
            int cur_img_len = 0;
            int ib = last_ib;
            while (ib < img_total && cur_img_len < chunk_image_token_number){
                int token_times = 4;
                cur_img_len += (grid_thw_cpu[ib * 3 + 1] * grid_thw_cpu[ib * 3 + 2]) / token_times;
                ib ++;
                chunk_image_number ++;
            }
            image_chunk_selections_vector.emplace_back(chunk_image_number);
            split_fuse_cur_seq_lens_vector.emplace_back(idx - last_idx);
            split_fuse_cur_idx_vector.emplace_back(idx);
            continue;
        }
        // text
        if (input_ids_cpu[idx-1] != image_token_id) {
            // case1. 如果切到text, 直接分chunk
            int last_idx = split_fuse_cur_idx_vector.back();
            int chunk_image_token_number = task_input_ids_image_token_count_cpu[idx] - task_input_ids_image_token_count_cpu[last_idx];
            int chunk_image_number = 0;
            int cur_img_len = 0;
            int ib = last_ib;
            while (ib < img_total && cur_img_len < chunk_image_token_number) {
                int token_times = 4;
                cur_img_len += (grid_thw_cpu[ib * 3 + 1] * grid_thw_cpu[ib * 3 + 2]) / token_times;
                ib ++;
                chunk_image_number ++;
            }
            image_chunk_selections_vector.emplace_back(chunk_image_number);
            split_fuse_cur_seq_lens_vector.emplace_back(idx - last_idx); // split_fuse_text_size
            last_ib = ib; // last_ib记录遍历到第几张图
            split_fuse_cur_idx_vector.emplace_back(idx);
            continue;
        } else {
            // case2. 如果切到图片，从当前图片往后找。往前找会出现边界问题
            // case2.1 如果split_size = img_token_num, mp[idx]==1, 直接按当前idx切分chunk
            while (idx < seq_lens_origin && mp[idx] != 1) {
                idx++;
            } // idx指向切分chunk的位置
            int last_idx = split_fuse_cur_idx_vector.back();
            int chunk_image_token_number = task_input_ids_image_token_count_cpu[idx] - task_input_ids_image_token_count_cpu[last_idx];
            int chunk_image_number = 0;
            int cur_img_len = 0;
            int ib = last_ib;
            while (ib < img_total && cur_img_len < chunk_image_token_number) {
                int token_times = 4;
                cur_img_len += (grid_thw_cpu[ib * 3 + 1] * grid_thw_cpu[ib * 3 + 2]) / token_times;
                ib ++;
                chunk_image_number ++;
            }
            image_chunk_selections_vector.emplace_back(chunk_image_number);
            split_fuse_cur_seq_lens_vector.emplace_back(idx - last_idx);
            split_fuse_cur_idx_vector.emplace_back(idx);
            last_ib = ib;
            continue;
        }
    }
    auto image_chunk_selections_out_cpu = paddle::from_blob(image_chunk_selections_vector.data(), {image_chunk_selections_vector.size()}, task_image_type_ids.dtype());
    auto split_fuse_cur_seq_lens_out_cpu = paddle::from_blob(split_fuse_cur_seq_lens_vector.data(), {split_fuse_cur_seq_lens_vector.size()}, task_image_type_ids.dtype());
    auto image_chunk_selections_out = paddle::experimental::copy_to(image_chunk_selections_out_cpu, task_image_type_ids.place(), false);
    auto split_fuse_cur_seq_lens_out = paddle::experimental::copy_to(split_fuse_cur_seq_lens_out_cpu, task_image_type_ids.place(), false);
    return {image_chunk_selections_out, split_fuse_cur_seq_lens_out};
}

PD_BUILD_STATIC_OP(get_mm_split_fuse)
    .Inputs({"task_input_ids", "task_image_type_ids", "task_input_ids_image_token_count", "grid_thw"})
    .Attrs({"image_token_id: int64_t", "img_total: int64_t", "batch_idx: int", "seq_lens_origin: int", "split_fuse_img_size: int", "split_fuse_text_size: int", "max_chunk_token_size: int"})
    .Outputs({"image_chunk_selections", "split_fuse_cur_seq_lens"})
    .SetKernelFn(PD_KERNEL(GetMmSplitFuse));
