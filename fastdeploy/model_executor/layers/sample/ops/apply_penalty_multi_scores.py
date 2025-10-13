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

import paddle

from fastdeploy.platforms import current_platform


def apply_penalty_multi_scores(
    pre_token_ids: paddle.Tensor,
    prompt_ids: paddle.Tensor,
    prompt_lens: paddle.Tensor,
    logits: paddle.Tensor,
    repetition_penalties: paddle.Tensor,
    frequency_penalties: paddle.Tensor,
    presence_penalties: paddle.Tensor,
    temperature: paddle.Tensor,
    bad_words_token_ids: paddle.Tensor,
    step_idx: paddle.Tensor,
    min_dec_lens: paddle.Tensor,
    eos_token_ids: paddle.Tensor,
) -> paddle.Tensor:
    """
    apply_penalty_multi_scores
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import get_token_penalty_multi_scores

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            prompt_ids,
            prompt_lens,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    elif current_platform.is_dcu():
        from fastdeploy.model_executor.ops.gpu import get_token_penalty_multi_scores

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            prompt_ids,
            prompt_lens,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    elif current_platform.is_xpu():
        from fastdeploy.model_executor.ops.xpu import get_token_penalty_multi_scores

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    elif current_platform.is_iluvatar():
        from fastdeploy.model_executor.ops.iluvatar import (
            get_token_penalty_multi_scores,
        )

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            prompt_ids,
            prompt_lens,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    elif current_platform.is_gcu():
        from fastdeploy.model_executor.ops.gcu import get_token_penalty_multi_scores

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    elif current_platform.is_maca():
        from fastdeploy.model_executor.ops.gpu import get_token_penalty_multi_scores

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            prompt_ids,
            prompt_lens,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    elif current_platform.is_intel_hpu():
        from fastdeploy.model_executor.ops.intel_hpu import (
            get_token_penalty_multi_scores,
        )

        logits = get_token_penalty_multi_scores(
            pre_token_ids,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
        )
    else:
        raise NotImplementedError

    return logits


def apply_speculative_penalty_multi_scores(
    pre_token_ids: paddle.Tensor,
    logits: paddle.Tensor,
    repetition_penalties: paddle.Tensor,
    frequency_penalties: paddle.Tensor,
    presence_penalties: paddle.Tensor,
    temperature: paddle.Tensor,
    bad_words_token_ids: paddle.Tensor,
    step_idx: paddle.Tensor,
    min_dec_lens: paddle.Tensor,
    eos_token_ids: paddle.Tensor,
    seq_lens_this_time: paddle.Tensor,
    output_padding_offset: paddle.Tensor,
    output_cum_offsets: paddle.Tensor,
    max_len: int,
):
    """
    apply_speculative_penalty_multi_scores
    """
    if current_platform.is_cuda():
        from fastdeploy.model_executor.ops.gpu import (
            speculate_get_token_penalty_multi_scores,
        )

        speculate_get_token_penalty_multi_scores(
            pre_token_ids,
            logits,
            repetition_penalties,
            frequency_penalties,
            presence_penalties,
            temperature,
            bad_words_token_ids,
            step_idx,
            min_dec_lens,
            eos_token_ids,
            seq_lens_this_time,
            output_padding_offset,
            output_cum_offsets,
            max_len,
        )
    else:
        raise NotImplementedError
    # inplace
    return logits
