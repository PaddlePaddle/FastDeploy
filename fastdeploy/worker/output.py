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
from typing import NamedTuple, Optional

import paddle


class Logprob(NamedTuple):
    """
    A named tuple containing information about a token's log probability.
    """

    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


PromptLogprobs = list[dict[int, Logprob] | None]
# [{token_id, logprob}] for tokens sampled from the top-k
SampleLogprobs = list[dict[int, Logprob]]


class LogprobsLists(NamedTuple):
    """ """

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: list[list[int]]
    # [num_reqs, max_num_logprobs + 1]
    logprobs: list[list[float]]
    # [num_reqs]
    sampled_token_ranks: list[int]

    def slice_columns(self, start: int, end: int):
        """
        Slice columns (per-row top-k logprobs and token IDs).
        Keeps the number of requests unchanged.
        """
        return LogprobsLists(
            [row[start:end] for row in self.logprob_token_ids],
            [row[start:end] for row in self.logprobs],
            self.sampled_token_ranks,  # unchanged
        )

    def slice_rows(self, start: int, end: int):
        """
        Slice rows.
        Keeps the number of max_num_logprobs unchanged.
        """
        return LogprobsLists(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.sampled_token_ranks[start:end],
        )


class LogprobsTensors(NamedTuple):
    """ """

    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: paddle.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: paddle.Tensor
    # [num_reqs]
    selected_token_ranks: paddle.Tensor

    def tolists(self):
        """Convert to lists."""
        return LogprobsLists(
            self.logprob_token_ids.tolist(),
            self.logprobs.tolist(),
            self.selected_token_ranks.tolist(),
        )

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = paddle.empty([num_positions, num_tokens_per_position], device="cpu", dtype=paddle.int64)
        logprobs = paddle.empty_like(logprob_token_ids, device="cpu", dtype=paddle.float32)
        selected_token_ranks = paddle.empty([num_positions], device="cpu", dtype=paddle.int64)
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )

    @staticmethod
    def empty(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on default device."""

        logprob_token_ids = paddle.empty([num_positions, num_tokens_per_position], dtype=paddle.int64)
        logprobs = paddle.empty_like(logprob_token_ids, dtype=paddle.float32)
        selected_token_ranks = paddle.empty([num_positions], dtype=paddle.int64)
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )

    def slice_rows(self, start: int, end: int):
        """
        Slice rows.
        Keeps the number of max_num_logprobs unchanged.
        """
        with paddle.no_grad():
            return LogprobsTensors(
                paddle.to_tensor(self.logprob_token_ids[start:end], place=self.logprob_token_ids.place),
                paddle.to_tensor(self.logprobs[start:end], place=self.logprob_token_ids.place),
                paddle.to_tensor(self.selected_token_ranks[start:end], place=self.logprob_token_ids.place),
            )


@dataclass
class SamplerOutput:
    """ """

    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: paddle.Tensor
    logprobs_tensors: Optional[LogprobsTensors]
    token_num_per_batch: Optional[paddle.Tensor] = None
    cu_batch_token_offset: Optional[paddle.Tensor] = None


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

    """
        the token ids of stop sequence
    """
    stop_token_ids: paddle.Tensor = None

    """
        the length of stop sequence
    """
    stop_seqs_len: paddle.Tensor = None

    """
        the length of input prompt
    """
    prompt_lens: paddle.Tensor = None

    """
        step mask rollback in some cases
    """
    mask_rollback: paddle.Tensor = None

    """
        prompt_logprobs
    """
    prompt_logprobs_list: Optional[LogprobsTensors] = None


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

    """
    [num_reqs, hidden_size]
    """
    pooler_output: list[Optional[paddle.Tensor]]
