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

// 1) 支持 inject_token_ids（任意长度）
// 2) 支持限制回复长度 max_reply_lens
// 3) 语义对齐非MTP limit_thinking_content_length：
//    - max_think_len < 0：不强制截断思考，但仍会监听 think_end_id
//    来进入回复阶段
//    - max_reply_len >= 0：仅在"思考结束后的下一个 token"开始计回复长度
//
// 状态机（复用 limit_status）
// - 0：思考生成阶段
// - 1..inject_len：注入 inject_token_ids[status - 1]
// - done_status = inject_len + 1（inject_len==0 时
// done_status=1）：思考结束点（本 token 不计入回复）
// - reply_base = done_status + 1
// - status >= reply_base：回复阶段计数，reply_len = status -
// reply_base（已输出回复 token 数）
__global__ void speculate_limit_thinking_content_length_kernel(
    int64_t* next_tokens,          // [bs, tokens_per_step]
    const int* max_think_lens,     // [bs]
    int* max_reply_lens,           // [bs]
    int64_t* step_idx,             // [bs]
    const int64_t* eos_token_ids,  // [eos_len]
    int* limit_status,             // [bs]
    int* accept_num,               // [bs]
    const bool* stop_flags,        // [bs]
    const int64_t think_end_id,
    const int64_t* inject_token_ids,  // [inject_len]
    const int tokens_per_step,
    const int bs,
    const int eos_token_id_len,
    const int inject_len,
    const bool splitwise_role_is_decode) {
  const int bid = blockIdx.x * blockDim.x + threadIdx.x;
  if (bid >= bs) return;

  const int original_accept_num = accept_num[bid];
  if (original_accept_num <= 0) return;
  if (stop_flags[bid]) return;

  const int max_think_len =
      max_think_lens[bid];  // <0: 不强制截断思考（但仍监听 think_end_id）
  int max_reply_len = max_reply_lens[bid];  // <0: 不限制回复
  if (max_think_len < 0 && max_reply_len < 0) return;

  // 状态常量（允许 inject_len==0）
  const int done_status = (inject_len > 0) ? (inject_len + 1) : 1;
  const int reply_base = done_status + 1;

  int status = limit_status[bid];
  if (status < 0) status = 0;

  int new_accept_num = original_accept_num;

  // 本 step 的 token offset 对应的绝对 step
  const int64_t current_base_step = step_idx[bid] - original_accept_num + 1;

  for (int token_offset = 0; token_offset < original_accept_num;
       token_offset++) {
    const int token_idx = bid * tokens_per_step + token_offset;
    int64_t next_token = next_tokens[token_idx];
    const int64_t current_step = current_base_step + token_offset;

    const int prev_status = status;
    bool condition_triggered = false;

    // ======================= 1) 思考阶段监听 think_end_id（语义对齐非MTP）
    // =======================
    // 注意：这里必须放在"注入触发逻辑"之前，因为如果模型自然输出 </think>，
    // 这一 token 应该把 status 置为 done_status，但"本 token 不计入回复"。
    if (status == 0 && next_token == think_end_id) {
      status = done_status;
      // 不截断 accept_num：后续 token 可以继续（但回复计数从下一个 token
      // 才开始）
      if (max_reply_len >= 0) {
        max_reply_len += 2;
      }
    }

    // ======================= 2) 仅当启用"思考截断"(max_think_len>=0)
    // 时触发注入 =======================
    if (max_think_len >= 0 && status < reply_base) {
      if (max_think_len > 0) {
        // A) 超长触发：到达 max_think_len 时开始注入（从本 token 起输出
        // inject_token_ids[0]）
        if (status == 0 && current_step == max_think_len) {
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

      // B) 思考阶段提前输出 eos：开始注入（从本 token 起覆盖 eos 为
      // inject_token_ids[0]）
      if (status == 0 && inject_len > 0) {
        for (int i = 0; i < eos_token_id_len; i++) {
          if (eos_token_ids[i] == next_token) {
            status = 1;
            break;
          }
        }
      }

      // 注入序列：如果进入注入区间，则覆盖 next_token，并推进状态。
      // 由于覆盖了 token，必须截断 accept_num，避免一口气接受后续 token。
      if (inject_len > 0 && status >= 1 && status <= inject_len) {
        next_token = inject_token_ids[status - 1];
        status += 1;
        if (status > done_status) status = done_status;  // 防御
        condition_triggered = true;
      }
    }

    // 这一 token 是否"刚刚进入 done_status"
    // 典型场景：自然输出 </think>（0 -> done_status）
    //          或 注入最后一步（inject_len -> done_status）
    const bool became_done_this_token = (status == done_status) &&
                                        (prev_status != done_status) &&
                                        (prev_status < reply_base);

    // ======================= 3) 回复长度限制（对齐非MTP）
    // ======================= 关键：刚进入 done_status 的这一 token（</think>
    // 或注入 token）不计入回复，也不在这一 token 开始回复计数
    if (max_reply_len >= 0) {
      if (!became_done_this_token) {
        // 只有在"前一个 token 已经是 done_status"，当前 token 才允许进入
        // reply_base 开始计数
        if (status == done_status) {
          status = reply_base;  // reply_len = 0
        }

        if (status >= reply_base) {
          int reply_len =
              status - reply_base;  // 已输出回复 token 数（不含当前 token）

          if (reply_len >= max_reply_len) {
            // 达到上限：强制 EOS，并截断 accept_num
            if (eos_token_id_len > 0) next_token = eos_token_ids[0];
            status = reply_base + max_reply_len;
            condition_triggered = true;
          } else {
            // 正常输出当前 token，并回复计数 +1
            status = reply_base + (reply_len + 1);
          }
        }
      }
    }

    // 写回当前 token
    next_tokens[token_idx] = next_token;

    // 若本 token 触发了"覆盖 token"（注入 or 强制 EOS），则截断 accept_num
    if (condition_triggered) {
      new_accept_num = token_offset + 1;
      break;
    }
  }

  // 更新 step_idx / accept_num（被截断的 token 需要回退
  // step_idx）
  const int discarded_tokens = original_accept_num - new_accept_num;
  if (discarded_tokens > 0) {
    step_idx[bid] -= discarded_tokens;
  }

  accept_num[bid] = new_accept_num;
  limit_status[bid] = status;
  max_reply_lens[bid] = max_reply_len;
}

void SpeculateLimitThinkingContentLength(
    const paddle::Tensor& next_tokens,
    const paddle::Tensor& max_think_lens,
    const paddle::Tensor& max_reply_lens,  // 新增
    const paddle::Tensor& step_idx,
    const paddle::Tensor& limit_status,
    const paddle::Tensor& accept_num,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& eos_token_ids,
    const paddle::Tensor& inject_token_ids,  // 新增：支持任意长度注入序列
    const int64_t think_end_id,
    const bool splitwise_role_is_decode) {
  const int batch_size = next_tokens.shape()[0];
  const int tokens_per_step = next_tokens.shape()[1];
  const int eos_token_id_len = eos_token_ids.shape()[0];
  const int inject_len = inject_token_ids.shape()[0];

  const int threads = 256;
  int blocks = (batch_size + threads - 1) / threads;
  if (blocks > 1024) blocks = 1024;

  speculate_limit_thinking_content_length_kernel<<<blocks,
                                                   threads,
                                                   0,
                                                   next_tokens.stream()>>>(
      const_cast<int64_t*>(next_tokens.data<int64_t>()),
      max_think_lens.data<int>(),
      const_cast<int*>(max_reply_lens.data<int>()),
      const_cast<int64_t*>(step_idx.data<int64_t>()),
      eos_token_ids.data<int64_t>(),
      const_cast<int*>(limit_status.data<int>()),
      const_cast<int*>(accept_num.data<int>()),
      stop_flags.data<bool>(),
      think_end_id,
      (inject_len > 0) ? inject_token_ids.data<int64_t>() : nullptr,
      tokens_per_step,
      batch_size,
      eos_token_id_len,
      inject_len,
      splitwise_role_is_decode);
}

PD_BUILD_STATIC_OP(speculate_limit_thinking_content_length)
    .Inputs({"next_tokens",
             "max_think_lens",
             "max_reply_lens",
             "step_idx",
             "limit_status",
             "accept_num",
             "stop_flags",
             "eos_token_ids",
             "inject_token_ids"})
    .Attrs({"think_end_id: int64_t", "splitwise_role_is_decode: bool"})
    .Outputs({"next_tokens_out"})
    .SetInplaceMap({{"next_tokens", "next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(SpeculateLimitThinkingContentLength));
