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
__global__ void speculate_limit_thinking_content_length_kernel_v2(
    int64_t* next_tokens,
    const int* max_think_lens,
    int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_think_status,
    int* accept_num,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t line_break_id,
    const int tokens_per_step,
    const int bs,
    const int eos_token_id_len) {
  int bid = threadIdx.x;
  if (bid >= bs) return;

  const int original_accept_num = accept_num[bid];
  if (original_accept_num <= 0) return;

  // 如果该序列未启用思考功能，则直接返回，默认值为 -1，表示不限制思考长度
  const int max_think_len = max_think_lens[bid];
  if (max_think_len < 0) return;
  int current_limit_think_status = limit_think_status[bid];
  // 如果在回复阶段, 或者已经触发停止标志, 则直接返回, 无需多余执行.
  if (current_limit_think_status == 5 || stop_flags[bid]) {
    return;
  }

  int new_accept_num = original_accept_num;

  const int64_t current_base_step = step_idx[bid] - original_accept_num + 1;

  for (int token_offset = 0; token_offset < original_accept_num;
       token_offset++) {
    const int token_idx = bid * tokens_per_step + token_offset;
    int64_t next_token = next_tokens[token_idx];
    const int64_t current_step = current_base_step + token_offset;

    bool condition_triggered = false;

    // ======================= 思考阶段控制 =======================
    // ======================= 思考阶段控制 =======================
    // A) 超长触发：到达 max_think_len 时开始注入（从本 step 起输出 \n）
    if (current_limit_think_status == 0 && current_step == max_think_len) {
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
      // 强制将当前token替换为结束思考的token
      next_token = line_break_id;
      current_limit_think_status = 2;
      condition_triggered = true;  // 因为修改了token，需要截断
    } else if (current_limit_think_status == 2) {
      // 强制将当前token替换为结束思考的token
      next_token = think_end_id;
      current_limit_think_status = 3;
      condition_triggered = true;  // 因为修改了token，需要截断
    } else if (current_limit_think_status == 3) {
      // 强制将当前token替换为结束思考的token
      next_token = line_break_id;
      current_limit_think_status = 4;
      condition_triggered = true;  // 因为修改了token，需要截断
    } else if (current_limit_think_status == 4) {
      // 强制将当前token替换为结束思考的token
      next_token = line_break_id;
      // 将状态推进到 1, 表示 "正在结束思考"
      current_limit_think_status = 5;
      condition_triggered = true;  // 因为修改了token，需要截断
    } else {
      if (next_token == think_end_id) {
        // 模型可能自己生成了 </think>
        current_limit_think_status = 5;
      }
    }

    next_tokens[token_idx] = next_token;

    if (condition_triggered) {
      new_accept_num = token_offset + 1;
      break;
    }
  }

  // 更新全局状态
  int discarded_tokens = original_accept_num - new_accept_num;
  if (discarded_tokens > 0) {
    step_idx[bid] -= discarded_tokens;
  }

  accept_num[bid] = new_accept_num;
  limit_think_status[bid] = current_limit_think_status;
}

void SpeculateLimitThinkingContentLengthV2(
    const paddle::Tensor& next_tokens,
    const paddle::Tensor& max_think_lens,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& limit_think_status,
    const paddle::Tensor& accept_num,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& eos_token_ids,
    const int64_t think_end_id,
    const int64_t line_break_id) {
  const int batch_size = next_tokens.shape()[0];
  const int eos_token_id_len = eos_token_ids.shape()[0];
  const int tokens_per_step = next_tokens.shape()[1];

  speculate_limit_thinking_content_length_kernel_v2<<<1,
                                                      1024,
                                                      0,
                                                      next_tokens.stream()>>>(
      const_cast<int64_t*>(next_tokens.data<int64_t>()),
      max_think_lens.data<int>(),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      eos_token_ids.data<int64_t>(),
      const_cast<int*>(limit_think_status.data<int>()),
      const_cast<int*>(accept_num.data<int>()),
      stop_flags.data<bool>(),
      think_end_id,
      line_break_id,
      tokens_per_step,
      batch_size,
      eos_token_id_len);
}

PD_BUILD_STATIC_OP(speculate_limit_thinking_content_length_v2)
    .Inputs({"next_tokens",
             "max_think_lens",
             "step_idx",
             "limit_think_status",
             "accept_num",
             "stop_flags",
             "eos_token_ids"})
    .Attrs({"think_end_id: int64_t", "line_break_id: int64_t"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateLimitThinkingContentLengthV2));
