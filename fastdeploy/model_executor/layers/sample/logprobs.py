"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

from typing import Callable, List, Optional, Tuple

import paddle
import paddle.nn.functional as F
import triton
import triton.language as tl

from fastdeploy.model_executor.ops.triton_ops.triton_utils import (
    enable_compat_on_triton_kernel,
)
from fastdeploy.platforms import current_platform
from fastdeploy.worker.output import LogprobsTensors


@enable_compat_on_triton_kernel
@triton.jit
def count_greater_kernel(
    x_ptr,  # [num_tokens, n_elements]
    y_ptr,  # [num_tokens, 1]
    out_ptr,  # [num_tokens, 1]
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    b = tl.program_id(0)
    sum_val = 0.0
    y = tl.load(y_ptr + b * 1 + 0)
    for col_start_idx in range(0, tl.cdiv(n_elements, BLOCK_SIZE)):
        col_ids = col_start_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        col_mask = col_ids < n_elements
        x = tl.load(x_ptr + b * n_elements + col_ids, mask=col_mask, other=-float("inf"))
        compare_mask = x >= y
        cmp_mask = tl.where(compare_mask & col_mask, 1, 0)
        sum_val += tl.sum(cmp_mask, axis=0)
    tl.store(out_ptr + b, sum_val.to(tl.int64))


def batched_count_greater_than(x: paddle.Tensor, y: paddle.Tensor) -> paddle.Tensor:
    """
    Triton implementation: (x >= y).sum(-1)

    Args:
        x (paddle.Tensor): 2D tensor，shape [num_tokens, n_elements]，float32.
        y (paddle.Tensor): 2D tensor，shape [num_tokens, 1]，float32.

    Returns:
        paddle.Tensor: 1D tensor，shape [num_tokens].
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}D"
    assert y.dim() == 2 and y.shape[1] == 1, f"y must be 2D with shape [num_tokens, 1], got {y.shape}"
    assert x.shape[0] == y.shape[0], f"shape[0] mismatch: x has {x.shape[0]}, y has {y.shape[0]}"
    assert x.dtype == y.dtype, f"dtype mismatch: x is {x.dtype}, y is {y.dtype}"

    if current_platform.is_cuda():

        num_tokens, n_elements = x.shape
        dtype = paddle.int64

        out = paddle.empty([num_tokens], dtype=dtype, device=x.place)

        config = {"BLOCK_SIZE": 4096, "num_warps": 16}
        grid = (num_tokens,)

        count_greater_kernel[grid](
            x_ptr=x,
            y_ptr=y,
            out_ptr=out,
            n_elements=n_elements,
            BLOCK_SIZE=config["BLOCK_SIZE"],
            num_warps=config["num_warps"],
        )
    else:
        out = (x >= y).sum(-1)

    return out


def gather_logprobs(
    logprobs: paddle.Tensor,
    num_logprobs: int,
    token_ids: paddle.Tensor,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        num_logprobs: minimum number of logprobs to retain per token
        token_ids: prompt tokens (if prompt logprobs) or sampled tokens
                   (if sampled logprobs); 1D token ID tensor with (num tokens) elements.
                   Must be int64.

    Returns:
        LogprobsTensors with top-k indices, top-k logprobs, and token ranks.
    """
    assert token_ids.dtype == paddle.int64
    token_ids = token_ids.unsqueeze(1)
    logprobs.clip_(min=paddle.finfo(logprobs.dtype).min)
    token_logprobs = paddle.take_along_axis(logprobs, token_ids, axis=-1)

    token_ranks = batched_count_greater_than(logprobs, token_logprobs)

    if num_logprobs >= 1:
        topk_logprobs, topk_indices = paddle.topk(logprobs, num_logprobs, axis=-1)
        indices = paddle.concat([token_ids, topk_indices], axis=1)
        top_logprobs = paddle.concat([token_logprobs, topk_logprobs], axis=1)
    else:
        indices = token_ids
        top_logprobs = token_logprobs

    return LogprobsTensors(indices.cpu(), top_logprobs.cpu(), token_ranks.cpu())


def build_output_logprobs(
    logits: paddle.Tensor,
    sampling_metadata,
    share_inputs: List[paddle.Tensor],
    is_naive: bool = False,
    logprobs_mode: str = "default",
    compute_logprobs_fn: Optional[Callable] = None,
) -> Tuple[Optional[LogprobsTensors], Optional[paddle.Tensor], Optional[paddle.Tensor]]:
    """
    Build logprobs output for both NAIVE and speculative (MTP/Ngram) modes.

    This is a standalone function (not tied to any sampler) so that both
    naive and speculative decoding paths can share the same logprob logic.

    For NAIVE mode: logits are already per-token, no extraction needed.
    For speculative mode: extracts target logits for accepted token positions.

    Args:
        logits: Model output logits.
        sampling_metadata: Sampling parameters and metadata.
        share_inputs: Shared input tensors.
        is_naive: True for NAIVE mode (single token per request).
        logprobs_mode: One of "raw_logprobs", "raw_logits", or "default".
        compute_logprobs_fn: Callable for computing logprobs with temperature
            scaling and top_p normalization. Used when logprobs_mode == "raw_logprobs".

    Returns:
        tuple: (logprobs_tensors, cu_batch_token_offset, output_logits)
    """
    num_logprobs = sampling_metadata.max_num_logprobs
    logprobs_tensors = None
    cu_batch_token_offset = None

    real_bsz = share_inputs["seq_lens_this_time"].shape[0]

    if is_naive:
        # NAIVE mode: one token per request, logits are already correct
        output_logits = logits
        token_ids = share_inputs["accept_tokens"][:real_bsz, 0]
    else:
        # Speculative mode: extract target logits for accepted positions
        from fastdeploy.model_executor.layers.sample.ops import (
            speculate_get_target_logits,
        )

        batch_token_num = paddle.where(
            share_inputs["seq_lens_encoder"][:real_bsz] != 0,
            paddle.ones_like(share_inputs["seq_lens_encoder"][:real_bsz]),
            share_inputs["seq_lens_this_time"],
        ).flatten()

        share_inputs["batch_token_num"] = batch_token_num

        ori_cu_batch_token_offset = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(batch_token_num)]).astype(
            "int32"
        )
        cu_batch_token_offset = paddle.concat(
            [paddle.to_tensor([0]), paddle.cumsum(share_inputs["accept_num"][:real_bsz])]
        ).astype("int32")
        share_inputs["cu_batch_token_offset"] = cu_batch_token_offset

        output_logits = paddle.empty(
            [share_inputs["accept_num"][:real_bsz].sum(), logits.shape[1]],
            dtype=logits.dtype,
        )
        speculate_get_target_logits(
            output_logits,
            logits,
            cu_batch_token_offset,
            ori_cu_batch_token_offset,
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
            share_inputs["accept_num"],
        )

        idx = paddle.arange(share_inputs["accept_tokens"].shape[1], dtype="int32")
        mask = idx < share_inputs["accept_num"].unsqueeze(1)
        token_ids = paddle.masked_select(share_inputs["accept_tokens"], mask)

    # Adapt for sampling mask
    if num_logprobs is None:
        return None, None, output_logits

    # Compute logprobs with temperature scaling and top_p normalization
    if logprobs_mode == "raw_logprobs":
        raw_logprobs = compute_logprobs_fn(output_logits, sampling_metadata)
    elif logprobs_mode == "raw_logits":
        raw_logprobs = output_logits.clone()
    else:
        raw_logprobs = F.log_softmax(output_logits, axis=-1)

    logprobs_tensors = gather_logprobs(raw_logprobs, num_logprobs, token_ids=token_ids)
    # output_logits use to compute sampling_mask
    return logprobs_tensors, cu_batch_token_offset, output_logits


def logprobs_renormalize_with_logz(logprobs: paddle.Tensor, logz, logprobs_tensors: LogprobsTensors):
    """
    Renormalize logprobs to match truncated sampling distribution.
    Args:
        logprobs: tensor [B, max_num_logprobs + 1]
        logz: [B], log(sum(probs in candidate set K)) for each request.
              Can be np.ndarray or paddle.Tensor (CPU pinned memory).
        logprobs_tensors: LogprobsTensors
    """
    if isinstance(logz, paddle.Tensor):
        logz = logz.astype(logprobs.dtype)
    else:
        logz = paddle.to_tensor(logz, dtype=logprobs.dtype)
    # Renormalize: log π_masked = log π_full - log Z_K
    # Only normalize valid candidates; padding positions use -inf
    valid_mask = paddle.isfinite(logprobs)
    normalized_logprobs = paddle.where(
        valid_mask, logprobs - logz.unsqueeze(1), paddle.full_like(logprobs, float("-inf"))
    )
    # Update logprobs_tensors with normalized values
    return LogprobsTensors(
        logprob_token_ids=logprobs_tensors.logprob_token_ids,
        logprobs=normalized_logprobs,
        selected_token_ranks=logprobs_tensors.selected_token_ranks,
    )
