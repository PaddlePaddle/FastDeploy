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

__global__ void limit_thinking_content_length_kernel_v1(
    int64_t *next_tokens,
    const int *max_think_lens,
    const int64_t *step_idx,
    const int64_t *eos_token_ids,
    int *limit_think_status,
    bool *stop_flags,
    const int64_t think_end_id,
    const int bs,
    const int eos_token_id_len) {
  int bid = threadIdx.x;
  if (bid >= bs) return;

  // 如果该序列未启用思考功能，则直接返回，默认值为 -1，表示不限制思考长度
  const int max_think_len = max_think_lens[bid];
  if (max_think_len < 0) return;
  int current_limit_think_status = limit_think_status[bid];
  // 如果在回复阶段, 且已经触发停止标志, 则直接返回, 无需多余执行
  if (current_limit_think_status == 2 && stop_flags[bid]) {
    return;
  }

  int64_t next_token = next_tokens[bid];
  const int64_t step = step_idx[bid];

  // ======================= 思考阶段控制 =======================
  // 阶段 1: 仍在思考 (status == 0), 检查是否需要强制结束
  if (current_limit_think_status < 1) {
    // 当开启思考长度控制时，检查是否超时
    if (step >= max_think_len) {
      // 强制将当前token替换为结束思考的token
      next_token = think_end_id;
      // 将状态推进到 1, 表示 "正在结束思考"
      current_limit_think_status = 1;
    } else {
      // 检查是否生成了EOS
      for (int i = 0; i < eos_token_id_len; i++) {
        if (eos_token_ids[i] == next_token) {
          // 强制将当前token替换为结束思考的token
          next_token = think_end_id;
          // 将状态推进到 1, 表示 "正在结束思考"
          current_limit_think_status = 1;
          if (stop_flags[bid]) {
            stop_flags[bid] = false;
          }
          break;
        }
      }
    }
  }
  // ======================= 思考结束处理 =======================
  // 阶段 2: 检查是否已满足结束思考的条件 (status < 2)
  // 这种情况会处理两种场景:
  // 1. status == 0: 模型自己生成了 think_end_id
  // 2. status == 1: 上一阶段强制注入了 think_end_id
  if (current_limit_think_status < 2) {
    if (next_token == think_end_id) {
      // 确认思考结束，将状态推进到 2 (响应阶段)
      current_limit_think_status = 2;
    }
  }
  // 写回更新后的 token
  next_tokens[bid] = next_token;
  // 更新全局状态
  limit_think_status[bid] = current_limit_think_status;
}

void LimitThinkingContentLengthV1(const paddle::Tensor &next_tokens,
                                  const paddle::Tensor &max_think_lens,
                                  const paddle::Tensor &step_idx,
                                  const paddle::Tensor &limit_think_status,
                                  const paddle::Tensor &stop_flags,
                                  const paddle::Tensor &eos_token_ids,
                                  const int64_t think_end_id) {
  const int batch_size = next_tokens.shape()[0];
  const int eos_token_id_len = eos_token_ids.shape()[0];
  limit_thinking_content_length_kernel_v1<<<1, 1024, 0, next_tokens.stream()>>>(
      const_cast<int64_t *>(next_tokens.data<int64_t>()),
      max_think_lens.data<int>(),
      step_idx.data<int64_t>(),
      eos_token_ids.data<int64_t>(),
      const_cast<int *>(limit_think_status.data<int>()),
      const_cast<bool *>(stop_flags.data<bool>()),
      think_end_id,
      batch_size,
      eos_token_id_len);
}

PD_BUILD_STATIC_OP(limit_thinking_content_length_v1)
    .Inputs({"next_tokens",
             "max_think_lens",
             "step_idx",
             "limit_think_status",
             "stop_flags",
             "eos_token_ids"})
    .Attrs({"think_end_id: int64_t"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(LimitThinkingContentLengthV1));
