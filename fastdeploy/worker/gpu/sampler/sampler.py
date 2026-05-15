"""
# Copyright (c) 2026  PaddlePaddle Authors. All Rights Reserved.
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

from dataclasses import dataclass
from typing import Optional

import paddle

from fastdeploy.worker.gpu.sampler import post_process
from fastdeploy.worker.gpu.sampler.sampler_state import SamplingState


@dataclass
class SamplerInputs:
    """Per-step inputs assembled by the model runner."""

    # [num_seqs] int32 — batch_idx -> req_idx mapping.
    idx_mapping: paddle.Tensor
    # [max_num_seqs, vocab] int32 — per-request token frequency histogram,
    # maintained by post_update after each step.
    output_bin_counts: paddle.Tensor
    # [max_num_seqs] int32 — tokens emitted so far per request.
    step_idx: paddle.Tensor
    # [max_num_seqs] int32 — per-request min/max decode lengths.
    min_dec_len: paddle.Tensor
    max_dec_len: paddle.Tensor
    # [num_eos] int64 — global eos tokens (model-level).
    eos_token_ids: paddle.Tensor


class Sampler:
    """Minimal orchestration layer around the Triton post-process kernels."""

    def __init__(self, sampling_states: SamplingState):
        self.states = sampling_states

    # ------------------------------------------------------------------
    # logits post-processing
    # ------------------------------------------------------------------
    def preprocess_logits(
        self,
        logits: paddle.Tensor,
        inputs: SamplerInputs,
    ) -> paddle.Tensor:
        """Apply bad-words masking, repetition penalty and temperature
        scaling in place on ``logits`` ([num_seqs, vocab])."""
        s = self.states
        idx_map = inputs.idx_mapping

        post_process.apply_bad_words_mask(
            logits,
            bad_word_token_ids=s.bad_word_token_ids.gpu,
            bad_word_offsets=s.bad_word_offsets.gpu,
            num_bad_words=s.num_bad_words.gpu,
            idx_mapping=idx_map,
        )
        post_process.apply_repetition_penalty(
            logits,
            output_bin_counts=inputs.output_bin_counts,
            repetition_penalty=s.repetition_penalty.gpu,
            idx_mapping=idx_map,
        )
        post_process.apply_temperature(
            logits,
            temperature=s.temperature.gpu,
            idx_mapping=idx_map,
        )
        return logits

    # ------------------------------------------------------------------
    # stop detection
    # ------------------------------------------------------------------
    def check_stop(
        self,
        sampled_tokens: paddle.Tensor,
        num_sampled: paddle.Tensor,
        inputs: SamplerInputs,
        num_sampled_cu: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        """Return ``stop_flags[num_seqs]`` for the current batch only."""
        s = self.states
        if num_sampled_cu is None:
            # Build exclusive prefix-sum with a leading zero on GPU.
            zero = paddle.zeros([1], dtype=num_sampled.dtype)
            num_sampled_cu = paddle.concat([zero, paddle.cumsum(num_sampled)])

        return post_process.check_stop(
            sampled_tokens=sampled_tokens,
            num_sampled=num_sampled,
            num_sampled_cu=num_sampled_cu,
            idx_mapping=inputs.idx_mapping,
            eos_token_ids=inputs.eos_token_ids,
            stop_token_ids=s.stop_token_ids.gpu,
            num_stop_token_ids=s.num_stop_token_ids.gpu,
            min_dec_len=inputs.min_dec_len,
            max_dec_len=inputs.max_dec_len,
            step_idx=inputs.step_idx,
        )
