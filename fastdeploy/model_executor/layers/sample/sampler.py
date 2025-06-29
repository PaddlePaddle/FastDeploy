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
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.guided_decoding.base_guided_decoding import \
    LogitsProcessorBase
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.ops import (
    apply_penalty_multi_scores, apply_speculative_penalty_multi_scores,
    top_p_sampling)
from fastdeploy.platforms import current_platform


class SamplerProcessor:
    """
    SamplingProcessor for guided decoding.
    """

    def __init__(self):
        self.async_step = None
        self.token_bitmask = None
        self.logits_processor: Dict[int, Optional[Any]] = dict()
        self.executor = ThreadPoolExecutor()
        self.logits_lock = threading.Lock()

    def add_logits_processor(self,
                             ids: int,
                             future: Optional[Any] = None,
                             prefill_tokens: List[int] = []):
        """ add logits processor to SamplerProcessor """
        with self.logits_lock:
            if future is None:
                if ids in self.logits_processor:
                    del self.logits_processor[ids]
                return

            if isinstance(future, LogitsProcessorBase):
                self.logits_processor[ids] = future
                for token in prefill_tokens:
                    self.logits_processor[ids].accept_token(token)
            elif future.done():
                self.logits_processor[ids] = future.result()
                for token in prefill_tokens:
                    self.logits_processor[ids].accept_token(token)
            else:
                self.logits_processor[ids] = [future, prefill_tokens]

    def update_vocab_mask(self, skip_idx_list: List[int] = []):
        """ update vocab mask. (cpu-heavy operation) """
        if len(self.logits_processor) == 0:
            return

        with self.logits_lock:
            for idx, processor in self.logits_processor.items():
                if processor is None:
                    del self.logits_processor[idx]
                    continue

                if not isinstance(processor, LogitsProcessorBase):
                    future, prefill_tokens = self.logits_processor[idx]
                    self.logits_processor[idx] = future.result()
                    for token in prefill_tokens:
                        self.logits_processor[idx].accept_token(token)

            available_processors = None
            for processor in self.logits_processor.values():
                if processor.is_terminated():
                    continue
                available_processors = processor
            if available_processors is None:
                return

        # allocate token bitmask
        self.token_bitmask = available_processors.allocate_token_bitmask()

        with self.logits_lock:
            # fill token bitmask
            for idx, processor in self.logits_processor.items():
                if processor.is_terminated() or idx in skip_idx_list:
                    continue

                processor.fill_token_bitmask(self.token_bitmask, idx)

    def apply_token_mask(self,
                         logits: paddle.Tensor,
                         skip_idx_list: List[int] = []):
        """ apply token mask to logits """
        if len(self.logits_processor) == 0 or self.token_bitmask is None:
            return logits

        # self.async_step.result()
        available_processors = None
        with self.logits_lock:
            for processor in self.logits_processor.values():
                if processor.is_terminated():
                    continue
                available_processors = processor
        if available_processors is None:
            return logits

        indices = list(self.logits_processor.keys())
        mask_idx = [i for i in indices if i not in skip_idx_list]
        return available_processors.apply_token_mask(logits,
                                                     self.token_bitmask,
                                                     indices=mask_idx)

    def _accept_token(self, idx: int, token: int):
        """ accept token """
        if idx not in self.logits_processor:
            raise ValueError(
                f"Invalid index, idx: {idx}, logit_processors.keys: {self.logits_processor.keys()}"
            )

        if self.logits_processor[idx].is_terminated():
            return

        self.logits_processor[idx].accept_token(token)

    def update_output_tokens(self,
                             next_tokens: paddle.Tensor,
                             skip_idx_list: List[int] = []):
        """ update output tokens """
        if len(self.logits_processor) == 0:
            return

        token_ids = next_tokens.numpy().tolist()
        with self.logits_lock:
            for idx in self.logits_processor.keys():
                token = token_ids[idx][0]
                if token < 0 or self.logits_processor[
                        idx] is None or idx in skip_idx_list:
                    continue

                self._accept_token(idx, token)

    def pre_process(self, skip_idx_list: List[int] = []):
        """ pre process before running """
        # create async operation for guided decoding
        # TODO: support async
        self.update_vocab_mask(skip_idx_list)
        # self.async_step = self.executor.submit(self.update_vocab_mask)


class Sampler(nn.Layer):
    """
    Sampler for normal generation.
    """

    def __init__(self):
        """
        """
        super().__init__()
        if current_platform.is_cuda() or current_platform.is_xpu():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError()

        self.processor = SamplerProcessor()

    def apply_logits_processor(self,
                               ids: int,
                               future: Optional[Any] = None,
                               prefill_tokens: List[int] = []):
        """ apply logits processor to sampler """
        self.processor.add_logits_processor(ids, future, prefill_tokens)

    def pre_process(self, skip_idx_list: List[int] = []):
        """ pre process before running """
        self.processor.pre_process(skip_idx_list)

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        skip_idx_list: List[int] = [],
    ) -> paddle.Tensor:
        """
        """
        logits = self.processor.apply_token_mask(logits, skip_idx_list)

        logits = apply_penalty_multi_scores(
            sampling_metadata.pre_token_ids,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
        )

        probs = F.softmax(logits)

        _, next_tokens = top_p_sampling(probs, sampling_metadata.top_p)

        self.processor.update_output_tokens(next_tokens, skip_idx_list)
        return next_tokens


class SpeculativeSampler(nn.Layer):
    """
    Sampler for speculative generation.
    """

    def __init__(self, fd_config: FDConfig):
        """
        """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError()
        self.speculative_verify_window = fd_config.speculative_config.verify_window
        self.speculative_max_candidate_len = fd_config.speculative_config.max_candidate_len

    def pre_process(self, skip_idx_list: List[int] = []):
        """ pre process before running """
        pass

    def apply_logits_processor(self,
                               ids: int,
                               future: Optional[Any] = None,
                               prefill_tokens: List[int] = []):
        """ apply logits processor to sampler """
        pass

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        max_model_len: int,
        share_inputs: List[paddle.Tensor],
    ) -> paddle.Tensor:
        """
        """

        from fastdeploy.model_executor.ops.gpu import (speculate_verify,
                                                       top_p_candidates)

        logits = apply_speculative_penalty_multi_scores(
            sampling_metadata.pre_token_ids,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
            share_inputs["seq_lens_this_time"],
            share_inputs["output_padding_offset"],
            share_inputs["output_cum_offsets"],
            max_model_len,
        )

        probs = F.softmax(logits)

        verify_scores, verify_tokens, actual_candidate_len = top_p_candidates(
            probs,
            sampling_metadata.top_p,
            share_inputs["output_padding_offset"],
            self.speculative_max_candidate_len,
            max_model_len,
        )

        speculate_verify(
            share_inputs["accept_tokens"],
            share_inputs["accept_num"],
            share_inputs["step_idx"],
            share_inputs["stop_flags"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            share_inputs[
                "draft_tokens"],  # Both input and output, need to write the last 1 token accepted to position 0.
            share_inputs["seq_lens_this_time"],
            verify_tokens,
            verify_scores,
            share_inputs["max_dec_len"],
            sampling_metadata.eos_token_ids,
            share_inputs["is_block_step"],
            share_inputs["output_cum_offsets"],
            actual_candidate_len,
            share_inputs["actual_draft_token_num"],
            sampling_metadata.top_p,
            max_model_len,
            self.speculative_verify_window,
            True,  # enable_topp
        )

        return None


class MTPSampler(nn.Layer):
    """
    """

    def __init__(self, fd_config: FDConfig):
        """
        """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError()

    def pre_process(self, skip_idx_list: List[int] = []):
        """ pre process before running """
        pass

    def apply_logits_processor(self,
                               ids: int,
                               future: Optional[Any] = None,
                               prefill_tokens: List[int] = []):
        """ apply logits processor to sampler """
        pass

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        max_model_len: int,
        share_inputs: List[paddle.Tensor],
    ) -> paddle.Tensor:
        """
        """
        logits = apply_speculative_penalty_multi_scores(
            sampling_metadata.pre_token_ids,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
            share_inputs["seq_lens_decoder"],
            max_model_len,
        )
        probs = F.softmax(logits)

        _, next_tokens = top_p_sampling(probs, sampling_metadata.top_p)
        return next_tokens
