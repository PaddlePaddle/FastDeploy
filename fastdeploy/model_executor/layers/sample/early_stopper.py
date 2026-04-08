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

from abc import abstractmethod

import paddle

from fastdeploy.config import EarlyStopConfig


class EarlyStopper:
    @abstractmethod
    def initialize(self, batch_size: int, cfg: EarlyStopConfig):
        """
        Initialize the stopper and set hyper-parameters.
        args:
            - batch_size: int, the batch size of input
            - cfg: EarlyStopConfig
        """
        raise NotImplementedError

    @abstractmethod
    def process(self, probs: paddle.Tensor, next_tokens: paddle.Tensor, stop_flags: paddle.Tensor):
        """
        process the stopper and set the stop_flags corresponding to the batch that triggers early stop to True
        args:
            - probs: [batch_size, vocab_size], the probs of every sample
            - next_tokens: [batch_size, 1], the token index of every chosen sample
            - stop_flags: [batch_size, 1], determine which batch will be stopped
        """
        raise NotImplementedError


class RepetitionEarlyStopper(EarlyStopper):
    def initialize(self, batch_size: int, cfg: EarlyStopConfig):
        self.early_stop_cfg = cfg
        self.window_size = cfg.window_size
        self.threshold = cfg.threshold
        self.trunc_scores = paddle.zeros((batch_size, self.early_stop_cfg.window_size), dtype="float32")

    def process(self, probs: paddle.Tensor, next_tokens: paddle.Tensor, stop_flags: paddle.Tensor):
        """
        args:
            - probs: [batch_size, vocab_size], the probs of every sample
            - next_tokens: [batch_size, 1], the token index of every chosen sample
            - stop_flags: [batch_size, 1], determine which batch will be stopped
        """
        # It will use normal execute if there is no triton support, otherwise use triton
        try:
            self.process_triton(probs, next_tokens, stop_flags)
        except Exception:
            self.process_normal(probs, next_tokens, stop_flags)

    def process_normal(self, probs: paddle.Tensor, next_tokens: paddle.Tensor, stop_flags: paddle.Tensor):
        # Get the probability score corresponding to next_tokens in this step
        next_scores = paddle.index_sample(probs, next_tokens)
        real_bsz = probs.shape[0]

        # Sliding window: Move left one grid and insert new score
        self.trunc_scores[:real_bsz, :-1] = self.trunc_scores[:real_bsz, 1:]
        self.trunc_scores[:real_bsz, -1:] = next_scores

        # Determine which samples need to be terminated: all trunc_scores are greater than threshold
        need_trunc_all = paddle.all(self.trunc_scores > self.threshold, axis=-1).unsqueeze(-1)

        # Add the stop flags
        stop_flags[need_trunc_all[:real_bsz]] = True

        # Reset trunc_scores of truncated samples to 0 to avoid false triggering in the next step
        reset_mask = need_trunc_all.tile([1, self.window_size])
        self.trunc_scores = paddle.where(reset_mask, paddle.zeros_like(self.trunc_scores), self.trunc_scores)

    def process_triton(self, probs: paddle.Tensor, next_tokens: paddle.Tensor, stop_flags: paddle.Tensor):
        import triton

        from fastdeploy.model_executor.ops.triton_ops import (
            repetition_early_stopper_kernel,
        )

        B, W = self.trunc_scores.shape
        real_bsz, V = probs.shape
        BLOCK_W = triton.next_power_of_2(W)

        grid = (real_bsz,)
        repetition_early_stopper_kernel[grid](
            self.trunc_scores,
            probs,
            next_tokens,
            stop_flags,
            self.threshold,
            B,
            W,
            V,
            self.trunc_scores.shape[1],
            probs.shape[1],
            BLOCK_W=BLOCK_W,
        )
        return next_tokens


# mapping strategy name to class
EARLY_STOPPER_MAPPING = {
    "repetition": RepetitionEarlyStopper,
}


def get_early_stopper_cls_from_stragegy(strategy: str):
    """
    get early stopper class from strategy name
    args:
        - strategy: string, the strategy name
    """
    strategy = strategy.lower()
    assert (
        strategy in EARLY_STOPPER_MAPPING
    ), f"{strategy} is not supported yet, only support {EARLY_STOPPER_MAPPING.keys()}."
    return EARLY_STOPPER_MAPPING[strategy]
