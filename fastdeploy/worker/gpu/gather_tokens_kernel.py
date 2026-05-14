"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import paddle
import numpy as np
import triton
import triton.language as tl


@triton.jit
def count_tokens_per_request_kernel(
    idx_mapping_ptr,
    scheduler_updated_ptr,
    seq_lens_ptr,
    draft_valid_counts_ptr,  # 预计算的draft_tokens有效数量
    token_counts_ptr,
    num_requests,
):
    """
    计算每个请求需要收集的token数量

    规则：
    - 如果scheduler_updated为True: token数量 = seq_len
    - 否则: token数量 = 1 (last_sampled_token) + draft_valid_count
    """
    pid = tl.program_id(0)

    if pid >= num_requests:
        return

    # 获取原始索引
    original_idx = tl.load(idx_mapping_ptr + pid)

    # 判断是否是scheduler更新的请求
    is_updated = tl.load(scheduler_updated_ptr + original_idx).to(tl.int32)

    if is_updated == 1:
        # 从input_ids取，使用seq_len
        count = tl.load(seq_lens_ptr + original_idx)
    else:
        # 从last_sampled_tokens取1个，加上draft_tokens的有效数量
        draft_count = tl.load(draft_valid_counts_ptr + original_idx)
        count = 1 + draft_count

    tl.store(token_counts_ptr + pid, count)


@triton.jit
def gather_tokens_kernel(
    idx_mapping_ptr,
    scheduler_updated_ptr,
    input_ids_ptr,
    input_ids_stride_0,
    input_ids_stride_1,
    last_sampled_tokens_ptr,
    draft_tokens_ptr,
    draft_tokens_stride_0,
    draft_tokens_stride_1,
    output_offsets_ptr,  # 每个请求在输出中的起始偏移
    output_ptr,
    num_requests,
    max_seq_len,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    根据条件收集token到输出中

    每个program处理一个请求的所有token
    """
    pid = tl.program_id(0)

    if pid >= num_requests:
        return

    # 获取原始索引
    original_idx = tl.load(idx_mapping_ptr + pid)

    # 判断是否是scheduler更新的请求
    is_updated = tl.load(scheduler_updated_ptr + original_idx).to(tl.int32)

    # 获取输出偏移
    output_offset = tl.load(output_offsets_ptr + pid)

    if is_updated == 1:
        # 从input_ids取token
        # 需要根据seq_len来决定取多少个，但这里简化处理取max_seq_len个
        for i in range(0, max_seq_len, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < max_seq_len

            # 计算input_ids中的位置
            input_offsets = original_idx * input_ids_stride_0 + offsets * input_ids_stride_1
            token = tl.load(input_ids_ptr + input_offsets, mask=mask)

            # 写入输出
            output_offsets = output_offset + offsets
            tl.store(output_ptr + output_offsets, token, mask=mask)
    else:
        # 写入last_sampled_token
        token = tl.load(last_sampled_tokens_ptr + original_idx)
        tl.store(output_ptr + output_offset, token)

        # 如果有投机采样，拼上draft_tokens
        if num_speculative_steps > 0:
            for i in range(0, num_speculative_steps, BLOCK_SIZE):
                offsets = i + tl.arange(0, BLOCK_SIZE)
                mask = offsets < num_speculative_steps

                # 计算draft_tokens中的位置
                draft_offsets = original_idx * draft_tokens_stride_0 + offsets * draft_tokens_stride_1
                token = tl.load(draft_tokens_ptr + draft_offsets, mask=mask)

                # 写入输出（偏移1，因为第一个是last_sampled_token）
                output_positions = output_offset + 1 + offsets
                tl.store(output_ptr + output_positions, token, mask=mask)


def gather_tokens_with_triton(
    idx_mapping: paddle.Tensor,
    scheduler_updated: paddle.Tensor,
    input_ids: paddle.Tensor,
    last_sampled_tokens: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    seq_lens: paddle.Tensor,
    num_speculative_steps: int,
) -> paddle.Tensor:
    """
    根据idx_mapping和条件收集token

    功能说明：
    1. 根据idx_mapping找到原始idx
    2. 对于每个请求：
       - 如果scheduler_updated为True：从input_ids取该请求的所有token
       - 否则：取last_sampled_token，如果num_speculative_steps>0则拼上draft_tokens

    参数:
        idx_mapping: [num_requests] 映射后的索引到原始索引，如[2, 0, 3, 1]
        scheduler_updated: [max_num_seqs] bool数组，标记是否刚被scheduler更新
        input_ids: [max_num_seqs, max_seq_len] 输入token ids
        last_sampled_tokens: [max_num_seqs] 最后采样的token
        draft_tokens: [max_num_seqs, num_speculative_steps] 投机采样的draft tokens
        seq_lens: [max_num_seqs] 每个请求的实际序列长度
        num_speculative_steps: 投机采样步数

    返回:
        ids_remove_padding: [total_tokens] 连续的token序列（已去除padding）
    """
    num_requests = idx_mapping.shape[0]
    max_num_seqs = scheduler_updated.shape[0]
    max_seq_len = input_ids.shape[1]

    # 计算每个请求的draft_tokens有效数量（统计非-1的）
    draft_valid_counts = paddle.sum((draft_tokens != -1).astype(paddle.int32), axis=1)

    # 1. 计算每个请求的token数量
    token_counts = paddle.empty([num_requests], dtype=paddle.int32)
    grid = triton.cdiv(num_requests, 1)
    count_tokens_per_request_kernel[grid](
        idx_mapping,
        scheduler_updated,
        seq_lens,
        draft_valid_counts,
        token_counts,
        num_requests,
    )

    # 2. 计算每个请求的输出偏移（prefix sum）
    token_counts_np = token_counts.numpy()
    output_offsets_np = np.cumsum(token_counts_np) - token_counts_np[0]
    output_offsets = paddle.to_tensor(output_offsets_np)

    total_tokens = int(np.sum(token_counts_np))
    ids_remove_padding = paddle.full([total_tokens], -1, dtype=paddle.int32)

    # 3. 收集token
    BLOCK_SIZE = 128
    grid = triton.cdiv(num_requests, 1)
    gather_tokens_kernel[grid](
        idx_mapping,
        scheduler_updated,
        input_ids,
        input_ids.stride(0),
        input_ids.stride(1),
        last_sampled_tokens,
        draft_tokens,
        draft_tokens.stride(0),
        draft_tokens.stride(1),
        output_offsets,
        ids_remove_padding,
        num_requests,
        max_seq_len,
        num_speculative_steps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return ids_remove_padding


def gather_tokens_with_paddle(
    idx_mapping: paddle.Tensor,
    scheduler_updated: paddle.Tensor,
    input_ids: paddle.Tensor,
    last_sampled_tokens: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    seq_lens: paddle.Tensor,
    num_speculative_steps: int,
) -> paddle.Tensor:
    """
    使用Paddle原生API实现的版本（作为fallback或调试用）

    功能与gather_tokens_with_triton相同
    """
    idx_mapping_np = idx_mapping.numpy()
    scheduler_updated_np = scheduler_updated.numpy()
    input_ids_np = input_ids.numpy()
    last_sampled_tokens_np = last_sampled_tokens.numpy()
    draft_tokens_np = draft_tokens.numpy()
    seq_lens_np = seq_lens.numpy()

    all_tokens = []

    for mapped_idx, original_idx in enumerate(idx_mapping_np):
        if scheduler_updated_np[original_idx]:
            # 从input_ids取
            seq_len = seq_lens_np[original_idx]
            tokens = input_ids_np[original_idx, :seq_len]
            all_tokens.extend(tokens.tolist())
        else:
            # 从last_sampled_tokens取
            all_tokens.append(last_sampled_tokens_np[original_idx])

            # 如果有投机采样，拼上draft_tokens
            if num_speculative_steps > 0:
                draft = draft_tokens_np[original_idx]
                # 过滤掉-1
                valid_draft = draft[draft != -1]
                all_tokens.extend(valid_draft.tolist())

    return paddle.to_tensor(all_tokens, dtype=paddle.int32)


# 统一接口
def gather_tokens(
    idx_mapping: paddle.Tensor,
    scheduler_updated: paddle.Tensor,
    input_ids: paddle.Tensor,
    last_sampled_tokens: paddle.Tensor,
    draft_tokens: paddle.Tensor,
    seq_lens: paddle.Tensor,
    num_speculative_steps: int,
    use_triton: bool = True,
) -> paddle.Tensor:
    """
    统一的gather_tokens接口

    参数:
        use_triton: 是否使用Triton加速，默认True
    """
    if use_triton:
        return gather_tokens_with_triton(
            idx_mapping, scheduler_updated, input_ids,
            last_sampled_tokens, draft_tokens, seq_lens, num_speculative_steps
        )
    else:
        return gather_tokens_with_paddle(
            idx_mapping, scheduler_updated, input_ids,
            last_sampled_tokens, draft_tokens, seq_lens, num_speculative_steps
        )
