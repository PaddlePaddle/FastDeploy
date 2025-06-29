"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass
from typing import Optional

import paddle


@dataclass
class ModelOutputData:
    """
        OutputData by execute_model
    """

    """
        Tokens generated in the previous step
    """
    next_tokens: paddle.Tensor

    """
        Flags indicating whether decoding should stop
    """
    stop_flags: paddle.Tensor

    """
        Index of the current decoding step
    """
    step_idx: int

    """
        Maximum decoding length
    """
    max_dec_len: int

    """
        Previous ids used for decoding
    """
    pre_ids: paddle.Tensor

    """
        Sequence lengths for this step
    """
    seq_lens_this_time: paddle.Tensor

    """
        Eos token ID
    """
    eos_token_id: paddle.Tensor

    """
        Indicates if stopping conditions should be ignored
    """
    not_need_stop: bool

    """
        Sequence lengths of the encoder
    """
    seq_lens_encoder: paddle.Tensor

    """
        Sequence lengths of the decoder
    """
    seq_lens_decoder: paddle.Tensor

    """
        Indicates if this is a blocking step
    """
    is_block_step: bool

    """
        The ID of the message queue.
    """
    msg_queue_id: int

    """
        The model parallel rank
    """
    mp_rank: int

    """
        Use EP parallel
    """
    use_ep: bool

    """
        input ids
    """
    input_ids: paddle.Tensor

    """
        stop nums for every sequence
    """
    stop_nums: paddle.Tensor

    """
        for speculative decoding
        full hidden states before lm_head
    """
    full_hidden_states: paddle.Tensor

    """
         draft tokens for every sequence
    """
    draft_tokens: paddle.Tensor

    """
        draft token num for every sequence
    """
    actual_draft_token_num: paddle.Tensor

    """
        accepted tokens in current step
    """
    accept_tokens: paddle.Tensor

    """
        the number of accepted tokens in current step
    """
    accept_num: paddle.Tensor


@dataclass
class ModelRunnerOutput:
    """
        [WIP] ModelRunnerOutput is serialized and sent to the scheduler process.
    """

    """
        [num_reqs]
    """
    req_ids: list[str]

    """
        req_id -> index
    """
    req_id_to_index: dict[str, int]

    """
        [num_reqs, num_generated_tokens]
    """
    sampled_token_ids: list[list[int]]

    """
        [num_reqs, num_spec_tokens]
    """
    spec_token_ids: Optional[list[list[int]]]
