// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

__global__ void limit_thinking_content_length_kernel(
    int64_t* next_tokens,
    const int* max_think_lens,
    int* max_reply_lens,
    const int64_t* step_idx,
    const int64_t* eos_token_ids,
    int* limit_status,
    const bool* stop_flags,
    const int64_t think_end_id,
    const int64_t* inject_token_ids,
    const int bs,
    const int eos_token_id_len,
    const int inject_len,
    const bool splitwise_role_is_decode) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  if (bid >= bs) return;
  if (stop_flags[bid]) return;

  const int max_think_len =
      max_think_lens[bid];  // <0: 不强制截断思考，但仍有思考阶段
  int max_reply_len = max_reply_lens[bid];  // <0: 不限制回复

  // 两者都不限制，且你不需要维护状态机的话，可以直接 return
  // 但如果你希望一直维护状态（便于后续调试/统计），也可以不 return。
  if (max_think_len < 0 && max_reply_len < 0) return;

  const int done_status = (inject_len > 0) ? (inject_len + 1) : 1;
  const int reply_base = done_status + 1;

  int status = limit_status[bid];
  if (status < 0) status = 0;
  const int prev_status = status;

  int64_t next_token = next_tokens[bid];
  const int64_t step = step_idx[bid];

  // ======================= 1) 思考阶段：永远监听 think_end_id
  // ======================= 即使 max_think_len < 0（不强制截断），只要模型输出
  // </think>，也要把状态置为 done_status
  if (status == 0 && next_token == think_end_id) {
    status = done_status;
    if (max_reply_len >= 0) {
      max_reply_len += 2;
    }
  }

  // ======================= 2) 仅当启用"思考截断"时，才触发注入/覆盖 eos
  // =======================
  if (max_think_len >= 0 && status < reply_base) {
    // A) 超长触发：到达 max_think_len 时开始注入
    if (max_think_len > 0) {
      // A) 超长触发：到达 max_think_len 时开始注入（从本 token 起输出
      // inject_token_ids[0]）
      if (status == 0 && step == max_think_len) {
        status = (inject_len > 0) ? 1 : done_status;
      }
    } else if (max_think_len == 0) {
      // A) 超长触发：到达 max_think_len 时开始注入
      if (status == 0 && !splitwise_role_is_decode) {
        // 如果是集中式或 P 节点：从本 token 起输出 inject_token_ids[0]）
        status = (inject_len > 0) ? 1 : done_status;
      } else if (status == 0 && splitwise_role_is_decode) {
        // 如果是 D 节点下：从本 token 起输出 inject_token_ids[1]）
        status = (inject_len > 0) ? 2 : done_status + 1;
      }
    }

    // B) 思考阶段提前输出 eos：开始注入（覆盖 eos）
    if (status == 0 && inject_len > 0) {
      for (int i = 0; i < eos_token_id_len; i++) {
        if (eos_token_ids[i] == next_token) {
          status = 1;
          break;
        }
      }
    }

    // 注入序列
    if (inject_len > 0 && status >= 1 && status <= inject_len) {
      next_token = inject_token_ids[status - 1];
      status += 1;
      if (status > done_status) status = done_status;
    }
  }

  // 这一拍是否"刚刚进入 done_status"
  const bool became_done_this_step = (status == done_status) &&
                                     (prev_status != done_status) &&
                                     (prev_status < reply_base);

  // ======================= 3) 回复长度限制：必须在思考结束之后才生效
  // =======================
  if (max_reply_len >= 0) {
    // 关键：本 step 如果刚输出 </think> 或刚完成注入进入 done_status，不要在同
    // step 计回复
    if (!became_done_this_step) {
      // 只有在"前一拍已经是 done_status"，这一拍才允许切换到 reply_base
      // 开始计数
      if (status == done_status) {
        status = reply_base;  // reply_len = 0
      }

      if (status >= reply_base) {
        int reply_len = status - reply_base;

        if (reply_len >= max_reply_len) {
          // 强制 EOS；由后置 stop_flags 再判停
          if (eos_token_id_len > 0) next_token = eos_token_ids[0];
          status = reply_base + max_reply_len;
        } else {
          // 正常输出当前 token，并将回复计数 +1
          status = reply_base + (reply_len + 1);
        }
      }
    }
  }

  next_tokens[bid] = next_token;
  limit_status[bid] = status;
  max_reply_lens[bid] = max_reply_len;
}

void LimitThinkingContentLength(const paddle::Tensor& next_tokens,
                                const paddle::Tensor& max_think_lens,
                                const paddle::Tensor& max_reply_lens,
                                const paddle::Tensor& step_idx,
                                const paddle::Tensor& limit_status,
                                const paddle::Tensor& stop_flags,
                                const paddle::Tensor& eos_token_ids,
                                const paddle::Tensor& inject_token_ids,
                                const int64_t think_end_id,
                                const bool splitwise_role_is_decode) {
  const int batch_size = next_tokens.shape()[0];
  const int eos_token_id_len = eos_token_ids.shape()[0];
  const int inject_len = inject_token_ids.shape()[0];

  const int threads = 256;
  const int blocks = (batch_size + threads - 1) / threads;

  limit_thinking_content_length_kernel<<<blocks,
                                         threads,
                                         0,
                                         next_tokens.stream()>>>(
      const_cast<int64_t*>(next_tokens.data<int64_t>()),
      max_think_lens.data<int>(),
      const_cast<int*>(max_reply_lens.data<int>()),
      step_idx.data<int64_t>(),
      eos_token_ids.data<int64_t>(),
      const_cast<int*>(limit_status.data<int>()),
      stop_flags.data<bool>(),
      think_end_id,
      inject_token_ids.data<int64_t>(),
      batch_size,
      eos_token_id_len,
      inject_len,
      splitwise_role_is_decode);
}

PD_BUILD_STATIC_OP(limit_thinking_content_length)
    .Inputs({"next_tokens",
             "max_think_lens",
             "max_reply_lens",
             "step_idx",
             "limit_status",
             "stop_flags",
             "eos_token_ids",
             "inject_token_ids"})
    .Attrs({"think_end_id: int64_t", "splitwise_role_is_decode: bool"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(LimitThinkingContentLength));
