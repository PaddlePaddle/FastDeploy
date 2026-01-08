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

// limit_think_status:
// 0：思考生成阶段
// 1：注入第 1 个 token：\n
// 2：注入第 2 个 token：</think>
// 3：注入第 3 个 token：\n
// 4：注入第 4 个 token：\n
// 5：思考结束，进入回复阶段
__global__ void limit_thinking_content_length_kernel_v2(
    int64_t *next_tokens,
    const int *max_think_lens,
    const int64_t *step_idx,
    const int64_t *eos_token_ids,
    int *limit_think_status,
    const bool *stop_flags,
    const int64_t think_end_id,
    const int64_t line_break_id,
    const int bs,
    const int eos_token_id_len) {
  int bid = threadIdx.x;
  if (bid >= bs) return;
  // 如果该序列未启用思考功能，则直接返回，默认值为 -1，表示不限制思考长度
  const int max_think_len = max_think_lens[bid];
  if (max_think_len < 0) return;
  int current_limit_think_status = limit_think_status[bid];
  // 如果在回复阶段, 或者已经触发停止标志, 则直接返回, 无需多余执行
  if (current_limit_think_status == 5 || stop_flags[bid]) {
    return;
  }

  int64_t next_token = next_tokens[bid];
  const int64_t step = step_idx[bid];

  // ======================= 思考阶段控制 =======================
  // A) 超长触发：到达 max_think_len 时开始注入（从本 step 起输出 \n）
  if (current_limit_think_status == 0 && step == max_think_len) {
    current_limit_think_status = 1;
  }
  // B) 新增：思考阶段提前输出 eos，开始注入（从本 step 起覆盖 eos 为 \n）
  if (current_limit_think_status == 0) {
    for (int i = 0; i < eos_token_id_len; i++) {
      if (eos_token_ids[i] == next_token) {
        current_limit_think_status = 1;
        break;
      }
    }
  }

  if (current_limit_think_status == 1) {
    next_token = line_break_id;
    current_limit_think_status = 2;
  } else if (current_limit_think_status == 2) {
    // 强制将当前token替换为结束思考的token
    next_token = think_end_id;
    current_limit_think_status = 3;
  } else if (current_limit_think_status == 3) {
    // 强制将当前token替换为结束思考的token
    next_token = line_break_id;
    current_limit_think_status = 4;
  } else if (current_limit_think_status == 4) {
    // 强制将当前token替换为结束思考的token
    next_token = line_break_id;
    // 将状态推进到 1, 表示 "正在结束思考"
    current_limit_think_status = 5;
  } else {
    if (next_token == think_end_id) {
      // 模型可能自己生成了 </think>
      current_limit_think_status = 5;
    }
  }

  // 写回更新后的 token
  next_tokens[bid] = next_token;
  // 更新全局状态
  limit_think_status[bid] = current_limit_think_status;
}

void LimitThinkingContentLengthV2(const paddle::Tensor &next_tokens,
                                  const paddle::Tensor &max_think_lens,
                                  const paddle::Tensor &step_idx,
                                  const paddle::Tensor &limit_think_status,
                                  const paddle::Tensor &stop_flags,
                                  const paddle::Tensor &eos_token_ids,
                                  const int64_t think_end_id,
                                  const int64_t line_break_id) {
  const int batch_size = next_tokens.shape()[0];
  const int eos_token_id_len = eos_token_ids.shape()[0];
  limit_thinking_content_length_kernel_v2<<<1, 1024, 0, next_tokens.stream()>>>(
      const_cast<int64_t *>(next_tokens.data<int64_t>()),
      max_think_lens.data<int>(),
      step_idx.data<int64_t>(),
      eos_token_ids.data<int64_t>(),
      const_cast<int *>(limit_think_status.data<int>()),
      stop_flags.data<bool>(),
      think_end_id,
      line_break_id,
      batch_size,
      eos_token_id_len);
}

PD_BUILD_STATIC_OP(limit_thinking_content_length_v2)
    .Inputs({"next_tokens",
             "max_think_lens",
             "step_idx",
             "limit_think_status",
             "stop_flags",
             "eos_token_ids"})
    .Attrs({"think_end_id: int64_t", "line_break_id: int64_t"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(LimitThinkingContentLengthV2));
