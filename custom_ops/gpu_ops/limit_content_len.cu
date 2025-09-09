// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

__global__ void limit_content_len(
    int64_t* next_tokens,
    const int64_t* end_thinking_tokens,
    int* max_content_lens,
    const int* max_think_lens,
    int64_t* step_idx,
    const int64_t* eos_token_ids,
    int64_t* max_dec_lens,
    int* limit_content_status,
    const bool* enable_thinking,
    int* accept_num,
    int* seq_lens_decoder,
    bool* stop_flags,
    const int tokens_per_step,
    const int bs,
    const int end_thinking_token_num,
    const int eos_token_id_len) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= bs) return;

    if (!enable_thinking[idx]) return;

    const int original_accept_num = accept_num[idx];
    if (original_accept_num <= 0) return;

    int current_limit_content_status = limit_content_status[idx];
    // 如果在回复阶段, 且已经触发停止标志, 则直接返回, 无需多余执行.
    if (current_limit_content_status == 2 && stop_flags[idx]) {
        return;
    }

    const int max_think_len_reg = max_think_lens[idx];

    const int64_t end_thinking_token_reg = end_thinking_tokens[0];

    int64_t current_max_dec_len = max_dec_lens[idx];
    int new_accept_num = original_accept_num;

    const int64_t current_base_step = step_idx[idx] - original_accept_num + 1;

    for (int token_offset = 0; token_offset < original_accept_num; token_offset++) {
        const int token_idx = idx * tokens_per_step + token_offset;
        int64_t next_token_reg = next_tokens[token_idx];
        const int64_t current_step = current_base_step + token_offset;

        bool condition_triggered = false;
        bool is_eos = false;

        // ======================= 思考阶段控制 =======================
        // 阶段 1: 仍在思考 (status == 0), 检查是否需要强制结束
        if (current_limit_content_status < 1) {
            bool should_transform = false;

            // 当开启思考长度控制时，检查是否超时
            if (max_think_len_reg > 0 && current_step >= max_think_len_reg) {
                should_transform = true;
            } else {
                // 检查是否生成了EOS
                for (int j = 0; j < eos_token_id_len; j++) {
                    if (eos_token_ids[j] == next_token_reg) {
                        is_eos = true;
                        should_transform = true;
                        break;
                    }
                }
            }

            if (should_transform) {
                // 强制将当前token替换为结束思考的token
                next_token_reg = end_thinking_token_reg;
                // 将状态推进到 1, 表示 "正在结束思考"
                current_limit_content_status = 1;
                condition_triggered = true; // 因为修改了token，需要截断
                // 只在EOS触发时清除stop_flags
                if (is_eos && stop_flags[idx]) {
                    stop_flags[idx] = false;
                }
            }
        }

        // ======================= 思考结束处理 =======================
        // 阶段 2: 检查是否已满足结束思考的条件 (status < 2)
        // 这种情况会处理两种场景:
        // 1. status == 0: 模型自己生成了 end_thinking_token
        // 2. status == 1: 上一阶段强制注入了 end_thinking_token
        if (current_limit_content_status < 2) {
            if (next_token_reg == end_thinking_token_reg) {
                // 确认思考结束，将状态推进到 2 (响应阶段)
                current_limit_content_status = 2;
            }
        }

        next_tokens[token_idx] = next_token_reg;

        if (condition_triggered) {
            new_accept_num = token_offset + 1;
            break;
        }
    }

    // 更新全局状态
    int discarded_tokens = original_accept_num - new_accept_num;
    if (discarded_tokens > 0) {
        step_idx[idx] -= discarded_tokens;
        seq_lens_decoder[idx] -= discarded_tokens;
    }

    accept_num[idx] = new_accept_num;
    limit_content_status[idx] = current_limit_content_status;
    max_dec_lens[idx] = current_max_dec_len;
}

void LimitContentLen(const paddle::Tensor& next_tokens,
                     const paddle::Tensor& end_thinking_tokens,
                     const paddle::Tensor& max_content_len,
                     const paddle::Tensor& max_think_len,
                     const paddle::Tensor& step_idx,
                     const paddle::Tensor& eos_token_ids,
                     const paddle::Tensor& max_dec_len,
                     const paddle::Tensor& limit_content_status,
                     const paddle::Tensor& enable_thinking,
                     const paddle::Tensor& accept_num,
                     const paddle::Tensor& seq_lens_decoder,
                     const paddle::Tensor& stop_flags) {

    const int batch_size = next_tokens.shape()[0];
    const int tokens_per_step = next_tokens.shape()[1];
    const int end_thinking_token_num = end_thinking_tokens.shape()[0];
    const int end_length = eos_token_ids.shape()[0];
    PD_CHECK(end_thinking_token_num == 1, "limit_content_len only support end_thinking_token_num = 1 for now.");

    dim3 grid(1);
    dim3 block(1024);

    limit_content_len<<<grid, block>>>(
        const_cast<int64_t *>(next_tokens.data<int64_t>()),
        end_thinking_tokens.data<int64_t>(),
        const_cast<int *>(max_content_len.data<int>()),
        max_think_len.data<int>(),
        const_cast<int64_t *>(step_idx.data<int64_t>()),
        eos_token_ids.data<int64_t>(),
        const_cast<int64_t *>(max_dec_len.data<int64_t>()),
        const_cast<int *>(limit_content_status.data<int>()),
        enable_thinking.data<bool>(),
        const_cast<int *>(accept_num.data<int>()),
        const_cast<int *>(seq_lens_decoder.data<int>()),
        const_cast<bool *>(stop_flags.data<bool>()),
        tokens_per_step,
        batch_size,
        end_thinking_token_num,
        end_length);
}
PD_BUILD_STATIC_OP(limit_content_len)
    .Inputs({"next_tokens",
             "end_thinking_tokens",
             "max_content_len",
             "max_think_len",
             "step_idx",
             "eos_token_ids",
             "max_dec_len",
             "limit_content_status",
             "enable_thinking",
             "accept_num",
             "seq_lens_decoder",
             "stop_flags"})
    .Outputs({"next_tokens_out", "max_dec_len_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"},
                    {"max_dec_len", "max_dec_len_out"}})
    .SetKernelFn(PD_KERNEL(LimitContentLen));
