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
import paddle.nn.functional as F
from paddle import nn

from fastdeploy.config import FDConfig
from fastdeploy.model_executor.guided_decoding import LogitsProcessorBase
from fastdeploy.model_executor.layers.sample.early_stopper import (
    get_early_stopper_cls_from_stragegy,
)
from fastdeploy.model_executor.layers.sample.meta_data import SamplingMetadata
from fastdeploy.model_executor.layers.sample.ops import (
    apply_penalty_multi_scores,
    apply_speculative_penalty_multi_scores,
    min_p_sampling,
    speculate_get_target_logits,
    speculate_insert_first_token,
    top_k_top_p_sampling,
)
from fastdeploy.platforms import current_platform
from fastdeploy.reasoning import ReasoningParser
from fastdeploy.worker.output import LogprobsTensors, SamplerOutput


def top_p_normalize_probs_paddle(
    probs: paddle.Tensor,
    top_ps: paddle.Tensor,
):
    probs_idx = probs.argsort(axis=-1, descending=True)
    probs_sort = paddle.take_along_axis(probs, probs_idx, axis=-1)
    probs_sum = paddle.cumsum(probs_sort, axis=-1)
    probs_sort = paddle.where((probs_sum - probs_sort) > top_ps, paddle.zeros_like(probs_sort), probs_sort)
    probs_sort.divide_(probs_sort.sum(axis=-1, keepdim=True))
    return paddle.zeros_like(probs_sort).put_along_axis_(indices=probs_idx, values=probs_sort, axis=-1)


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
        self.reasoning_parser = None

    def apply_reasoning_parser(self, reasoning_parser: Optional[ReasoningParser] = None):
        self.reasoning_parser = reasoning_parser

    def add_logits_processor(
        self,
        ids: int,
        future: Optional[Any] = None,
        prefill_tokens: List[int] = [],
    ):
        """add logits processor to SamplerProcessor"""
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
        """update vocab mask. (cpu-heavy operation)"""
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

    def apply_token_mask(self, logits: paddle.Tensor, skip_idx_list: List[int] = []):
        """apply token mask to logits"""
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

        indices = []
        for idx, processor in self.logits_processor.items():
            if processor is None or idx in skip_idx_list:
                continue
            if self.reasoning_parser is None or not processor.enable_reasoning or processor.reasoning_ended:
                indices.append(idx)

        return available_processors.apply_token_mask(logits, self.token_bitmask, indices=indices)

    def _accept_token(self, idx: int, token: int):
        """accept token"""
        if idx not in self.logits_processor:
            raise ValueError(f"Invalid index, idx: {idx}, logit_processors.keys: {self.logits_processor.keys()}")

        if self.logits_processor[idx].is_terminated():
            return

        if (
            self.reasoning_parser is not None
            and self.logits_processor[idx].enable_reasoning
            and not self.logits_processor[idx].reasoning_ended
        ):
            reasoning_ended = self.reasoning_parser.is_reasoning_end([token])
            self.logits_processor[idx].reasoning_ended = reasoning_ended
            return

        self.logits_processor[idx].accept_token(token)

    def update_output_tokens(self, next_tokens: paddle.Tensor, skip_idx_list: List[int] = []):
        """update output tokens"""
        if len(self.logits_processor) == 0:
            return

        token_ids = next_tokens.numpy().tolist()
        with self.logits_lock:
            for idx in self.logits_processor.keys():
                token = token_ids[idx][0]
                if token < 0 or self.logits_processor[idx] is None or idx in skip_idx_list:
                    continue

                self._accept_token(idx, token)

    def pre_process(self, skip_idx_list: List[int] = []):
        """pre process before running"""
        # create async operation for guided decoding
        # TODO: support async
        self.update_vocab_mask(skip_idx_list)
        # self.async_step = self.executor.submit(self.update_vocab_mask)


class Sampler(nn.Layer):
    """
    Sampler for normal generation.
    """

    def __init__(self, fd_config: FDConfig = None, logprobs_mode: str = "raw_logprobs"):
        """ """
        super().__init__()
        if (
            current_platform.is_cuda()
            or current_platform.is_xpu()
            or current_platform.is_iluvatar()
            or current_platform.is_gcu()
            or current_platform.is_dcu()
            or current_platform.is_maca()
        ):
            self.forward = self.forward_cuda
        elif current_platform.is_intel_hpu():
            self.forward = self.forward_intel_hpu
        else:
            raise NotImplementedError

        self.processor = SamplerProcessor()
        self.logprobs_mode = fd_config.model_config.logprobs_mode if fd_config is not None else logprobs_mode
        # Can only be created when fd_config.early_stopper_config.enable_early_stop = True
        if (
            fd_config is not None
            and fd_config.early_stop_config is not None
            and fd_config.early_stop_config.enable_early_stop
        ):
            early_stopper_cls = get_early_stopper_cls_from_stragegy(fd_config.early_stop_config.strategy)
            self.early_stopper = early_stopper_cls()
            self.early_stopper.initialize(fd_config.scheduler_config.max_num_seqs, fd_config.early_stop_config)

    def set_reasoning_parser(self, reasoning_parser: Optional[ReasoningParser] = None):
        """set reasoning parser"""
        self.processor.apply_reasoning_parser(reasoning_parser)

    def apply_logits_processor(self, ids: int, future: Optional[Any] = None, prefill_tokens: List[int] = []):
        """apply logits processor to sampler"""
        self.processor.add_logits_processor(ids, future, prefill_tokens)

    def pre_process(self, skip_idx_list: List[int] = []):
        """pre process before running"""
        self.processor.pre_process(skip_idx_list)

    def post_process(self, next_tokens: paddle.Tensor, skip_idx_list: List[int] = []):
        """post process after running"""
        self.processor.update_output_tokens(next_tokens, skip_idx_list)

    def compute_logprobs(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> paddle.Tensor:
        """ """
        last_logits = logits
        real_bsz = last_logits.shape[0]
        temp_scaled_logprobs = sampling_metadata.temp_scaled_logprobs
        top_p_normalized_logprobs = sampling_metadata.top_p_normalized_logprobs
        share_inputs = sampling_metadata.share_inputs
        if temp_scaled_logprobs is not None:
            real_bsz_temp_scaled = temp_scaled_logprobs[:real_bsz]
            temperature = sampling_metadata.temperature[:real_bsz]
            temp_temperature = paddle.where(real_bsz_temp_scaled, temperature, paddle.ones_like(temperature))
            last_logits = last_logits / temp_temperature

        last_logprobs = F.log_softmax(last_logits, axis=-1)
        top_p_logprob = None
        top_p_req_mask = None

        if top_p_normalized_logprobs is not None and share_inputs is not None:
            seq_lens_this_time = share_inputs["seq_lens_this_time"].reshape([-1, 1])[:real_bsz]
            seq_lens_encoder = share_inputs["seq_lens_encoder"].reshape([-1, 1])[:real_bsz]
            seq_lens_decoder = share_inputs["seq_lens_decoder"].reshape([-1, 1])[:real_bsz]
            seq_lens_time_sum = seq_lens_this_time + seq_lens_encoder + seq_lens_decoder
            real_req_mask = seq_lens_time_sum > 0
            top_p_req_mask = paddle.logical_and(top_p_normalized_logprobs[:real_bsz], real_req_mask)
            real_req_top_p = sampling_metadata.top_p[:real_bsz]
            # Normalize logprobs if top_p normalization is enabled
            # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
            top_p_req_mask = paddle.logical_and(top_p_req_mask, real_req_top_p != 1.0)
            if top_p_req_mask.any():
                probs = F.softmax(last_logits, axis=-1)
                probs = top_p_normalize_probs_paddle(probs, real_req_top_p)
                top_p_logprob = paddle.log(probs)
        if top_p_logprob is not None:
            last_logprobs = paddle.where(top_p_req_mask, top_p_logprob, last_logprobs)
        return last_logprobs

    def gather_logprobs(
        self,
        logprobs: paddle.Tensor,
        num_logprobs: int,
        token_ids: paddle.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.
        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.
        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == paddle.int64
        logprobs.clip_(min=paddle.finfo(logprobs.dtype).min)
        # Get with the logprob of the prompt or sampled token.
        token_logprobs = paddle.take_along_axis(logprobs, token_ids, axis=-1)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        if num_logprobs >= 1:
            # Find the topK values.
            topk_logprobs, topk_indices = paddle.topk(logprobs, num_logprobs, axis=-1)
            indices = paddle.concat([token_ids, topk_indices], axis=1)
            top_logprobs = paddle.concat([token_logprobs, topk_logprobs], axis=1)
        else:
            indices = token_ids
            top_logprobs = token_logprobs

        return LogprobsTensors(indices, top_logprobs, token_ranks)

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        skip_idx_list: List[int] = [],
    ) -> SamplerOutput:
        """ """
        logits = self.processor.apply_token_mask(logits, skip_idx_list)

        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits, sampling_metadata)
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        logits = apply_penalty_multi_scores(
            sampling_metadata.pre_token_ids,
            sampling_metadata.prompt_ids,
            sampling_metadata.prompt_lens,
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

        if num_logprobs is not None:
            if self.logprobs_mode == "processed_logprobs":
                raw_logprobs = self.compute_logprobs(logits, sampling_metadata)
            elif self.logprobs_mode == "processed_logits":
                raw_logprobs = logits.clone()

        probs = F.softmax(logits)

        probs = min_p_sampling(probs, sampling_metadata.min_p, sampling_metadata.min_p_list)
        _, next_tokens = top_k_top_p_sampling(
            probs,
            sampling_metadata.top_p,
            sampling_metadata.top_k,
            sampling_metadata.top_k_list,
            seed=sampling_metadata.seed[0, 0],
        )

        logprobs_tensors = (
            None if num_logprobs is None else self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=next_tokens)
        )
        if sampling_metadata.enable_early_stop:
            # will set the stop batch in stop_flags
            assert sampling_metadata.stop_flags is not None, "need stop_flags for early stop"
            self.early_stopper.process(probs, next_tokens, sampling_metadata.stop_flags)

        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=next_tokens,
            logprobs_tensors=logprobs_tensors,
        )

        return sampler_output

    def forward_intel_hpu(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        batch_ids: paddle.Tensor,
        max_batch: int,
        rank: int,
        local_rank: int,
    ) -> paddle.Tensor:
        if logits.dtype != paddle.float32:
            logits = paddle.cast(logits, paddle.float32)

        from fastdeploy.model_executor.ops.intel_hpu import fused_sampler

        _, next_tokens = fused_sampler(
            sampling_metadata.pre_token_ids,
            sampling_metadata.prompt_ids,
            sampling_metadata.seq_lens_encoder,
            sampling_metadata.seq_lens_decoder,
            sampling_metadata.step_idx,
            sampling_metadata.stop_flags,
            logits,
            sampling_metadata.repetition_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.presence_penalties,
            sampling_metadata.temperature,
            sampling_metadata.bad_words_token_ids,
            sampling_metadata.step_idx,
            sampling_metadata.min_dec_lens,
            sampling_metadata.eos_token_ids,
            sampling_metadata.top_p,
            rank,
            local_rank,
        )

        if next_tokens.shape[0] != max_batch:
            dim = next_tokens.shape[-1]
            tmp_tokens = paddle.full((max_batch, dim), -1 if local_rank == 0 else 0, dtype=next_tokens.dtype)
            tmp_tokens = paddle.scatter(tmp_tokens, batch_ids, next_tokens[: batch_ids.shape[0], :])
            return tmp_tokens

        return next_tokens


class SpeculativeSampler(nn.Layer):
    """
    Sampler for speculative generation.
    """

    def __init__(self, fd_config: FDConfig):
        """ """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError
        self.logprobs_mode = fd_config.model_config.logprobs_mode
        self.speculative_verify_window = fd_config.speculative_config.verify_window
        self.speculative_max_candidate_len = fd_config.speculative_config.max_candidate_len
        self.speculative_benchmark_mode = fd_config.speculative_config.benchmark_mode

    def pre_process(self, skip_idx_list: List[int] = []):
        """pre process before running"""
        pass

    def set_reasoning_parser(self, reasoning_parser: Optional[ReasoningParser] = None):
        """set reasoning parser"""
        pass

    def post_process(self, next_tokens: paddle.Tensor, skip_idx_list: List[int] = []):
        """post process after running"""
        pass

    def apply_logits_processor(self, ids: int, future: Optional[Any] = None, prefill_tokens: List[int] = []):
        """apply logits processor to sampler"""
        pass

    def compute_logprobs(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> paddle.Tensor:
        """compute logprobs"""
        share_inputs = sampling_metadata.share_inputs
        last_logits = logits
        real_bsz = share_inputs["seq_lens_this_time"].shape[0]
        batch_token_num = share_inputs["batch_token_num"][:real_bsz]

        temp_scaled_logprobs = sampling_metadata.temp_scaled_logprobs
        top_p_normalized_logprobs = sampling_metadata.top_p_normalized_logprobs
        if temp_scaled_logprobs is not None:
            real_bsz_temp_scaled = temp_scaled_logprobs[:real_bsz]
            temperature = sampling_metadata.temperature[:real_bsz]
            real_bsz_temp_scaled = (
                real_bsz_temp_scaled.astype("int32").squeeze(1).repeat_interleave(batch_token_num).astype("bool")
            )
            temperature = temperature.squeeze(1).repeat_interleave(batch_token_num)
            temp_temperature = paddle.where(
                real_bsz_temp_scaled, temperature, paddle.ones_like(temperature)
            ).unsqueeze(1)
            last_logits = last_logits / temp_temperature

        last_logprobs = F.log_softmax(last_logits, axis=-1)
        top_p_logprob = None
        top_p_token_mask = None

        if top_p_normalized_logprobs is not None and share_inputs is not None:
            real_token_top_p = (
                sampling_metadata.top_p[:real_bsz].squeeze(1).repeat_interleave(batch_token_num).unsqueeze(1)
            )
            top_p_normalized_logprobs = (
                top_p_normalized_logprobs[:real_bsz]
                .astype("int32")
                .squeeze(1)
                .repeat_interleave(batch_token_num)
                .astype("bool")
                .unsqueeze(1)
            )
            top_p_token_mask = paddle.logical_and(top_p_normalized_logprobs, real_token_top_p != 1.0)
            if top_p_token_mask.any():
                probs = F.softmax(last_logits, axis=-1)
                probs = top_p_normalize_probs_paddle(probs, real_token_top_p)
                top_p_logprob = paddle.log(probs)
        if top_p_logprob is not None:
            last_logprobs = paddle.where(top_p_token_mask, top_p_logprob, last_logprobs)
        return last_logprobs

    def gather_logprobs(
        self,
        logprobs: paddle.Tensor,
        num_logprobs: int,
        token_ids: paddle.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.
        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.
        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == paddle.int64
        token_ids = token_ids.unsqueeze(1)
        logprobs.clip_(min=paddle.finfo(logprobs.dtype).min)
        # Get with the logprob of the prompt or sampled token.
        token_logprobs = paddle.take_along_axis(logprobs, token_ids, axis=-1)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        if num_logprobs >= 1:
            # Find the topK values.
            topk_logprobs, topk_indices = paddle.topk(logprobs, num_logprobs, axis=-1)
            indices = paddle.concat([token_ids, topk_indices], axis=1)
            top_logprobs = paddle.concat([token_logprobs, topk_logprobs], axis=1)
        else:
            indices = token_ids
            top_logprobs = token_logprobs

        return LogprobsTensors(indices, top_logprobs, token_ranks)

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        max_model_len: int,
        share_inputs: List[paddle.Tensor],
        accept_all_drafts: bool = False,
    ) -> paddle.Tensor:
        """ """

        from fastdeploy.model_executor.ops.gpu import speculate_verify, top_p_candidates

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
                "draft_tokens"
            ],  # Both input and output, need to write the last 1 token accepted to position 0.
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
            self.speculative_benchmark_mode,
            accept_all_drafts,
        )

        num_logprobs = sampling_metadata.max_num_logprobs
        batch_token_num = None
        if num_logprobs is not None:
            real_bsz = share_inputs["seq_lens_this_time"].shape[0]
            batch_token_num = paddle.where(
                share_inputs["seq_lens_encoder"][:real_bsz] != 0,
                paddle.ones_like(share_inputs["seq_lens_encoder"][:real_bsz]),
                share_inputs["accept_num"][:real_bsz].unsqueeze(1),
            ).squeeze(1)
            share_inputs["batch_token_num"] = batch_token_num
            ori_cu_batch_token_offset = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(batch_token_num)]).astype(
                "int32"
            )
            cu_batch_token_offset = paddle.concat(
                [paddle.to_tensor([0]), paddle.cumsum(share_inputs["accept_num"][:real_bsz])]
            ).astype("int32")
            share_inputs["cu_batch_token_offset"] = cu_batch_token_offset
            target_logtis = paddle.empty(
                [share_inputs["accept_num"][:real_bsz].sum(), logits.shape[1]], dtype=logits.dtype
            )
            speculate_get_target_logits(
                target_logtis,
                logits,
                cu_batch_token_offset,
                ori_cu_batch_token_offset,
                share_inputs["seq_lens_this_time"],
                share_inputs["seq_lens_encoder"],
                share_inputs["accept_num"],
            )
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(target_logtis, sampling_metadata)
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = target_logtis.clone()

        logprobs_tensors = None
        token_ids = share_inputs["accept_tokens"]
        if num_logprobs is not None:
            token_ids = paddle.concat(
                [
                    share_inputs["accept_tokens"][i, : share_inputs["accept_num"][i]]
                    for i in range(share_inputs["accept_num"][:real_bsz].shape[0])
                ]
            )
            logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=token_ids)

        sampler_output = SamplerOutput(
            sampled_token_ids=token_ids,
            logprobs_tensors=logprobs_tensors,
            token_num_per_batch=batch_token_num,
            cu_batch_token_offset=share_inputs["cu_batch_token_offset"],
        )

        return sampler_output


class MTPSampler(nn.Layer):
    """ """

    def __init__(self, fd_config: FDConfig):
        """ """
        super().__init__()
        if current_platform.is_cuda():
            self.forward = self.forward_cuda
        else:
            raise NotImplementedError
        self.logprobs_mode = fd_config.model_config.logprobs_mode

    def pre_process(self, skip_idx_list: List[int] = []):
        """pre process before running"""
        pass

    def apply_logits_processor(
        self,
        ids: int,
        future: Optional[Any] = None,
        prefill_tokens: List[int] = [],
    ):
        """apply logits processor to sampler"""
        pass

    def set_reasoning_parser(self, reasoning_parser: Optional[ReasoningParser] = None):
        """set reasoning parser"""
        pass

    def post_process(self, next_tokens: paddle.Tensor, skip_idx_list: List[int] = []):
        """post process after running"""
        pass

    def compute_logprobs(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> paddle.Tensor:
        """compute logprobs"""
        share_inputs = sampling_metadata.share_inputs
        real_bsz = share_inputs["seq_lens_this_time"].shape[0]
        last_logits = logits
        temp_scaled_logprobs = sampling_metadata.temp_scaled_logprobs
        top_p_normalized_logprobs = sampling_metadata.top_p_normalized_logprobs
        if temp_scaled_logprobs is not None:
            real_bsz_temp_scaled = temp_scaled_logprobs[:real_bsz]
            temperature = sampling_metadata.temperature[:real_bsz]
            real_bsz_temp_scaled = (
                real_bsz_temp_scaled.astype("int32")
                .squeeze(1)
                .repeat_interleave(share_inputs["batch_token_num"][:real_bsz])
                .astype("bool")
            )
            temperature = temperature.squeeze(1).repeat_interleave(share_inputs["batch_token_num"][:real_bsz])
            temp_temperature = paddle.where(
                real_bsz_temp_scaled, temperature, paddle.ones_like(temperature)
            ).unsqueeze(1)
            last_logits = last_logits / temp_temperature

        last_logprobs = F.log_softmax(last_logits, axis=-1)
        top_p_logprob = None
        top_p_token_mask = None

        if top_p_normalized_logprobs is not None and share_inputs is not None:
            real_token_top_p = (
                sampling_metadata.top_p[:real_bsz]
                .squeeze(1)
                .repeat_interleave(share_inputs["batch_token_num"][:real_bsz])
                .unsqueeze(1)
            )
            top_p_normalized_logprobs = (
                top_p_normalized_logprobs[:real_bsz]
                .astype("int32")
                .squeeze(1)
                .repeat_interleave(share_inputs["batch_token_num"][:real_bsz])
                .astype("bool")
                .unsqueeze(1)
            )
            top_p_token_mask = paddle.logical_and(top_p_normalized_logprobs, real_token_top_p != 1.0)

            if top_p_token_mask.any():
                probs = F.softmax(last_logits, axis=-1)
                probs = top_p_normalize_probs_paddle(probs, real_token_top_p)
                top_p_logprob = paddle.log(probs)
        if top_p_logprob is not None:
            last_logprobs = paddle.where(top_p_token_mask, top_p_logprob, last_logprobs)
        return last_logprobs

    def gather_logprobs(
        self,
        logprobs: paddle.Tensor,
        num_logprobs: int,
        token_ids: paddle.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.
        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: minimum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.
        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == paddle.int64
        token_ids = token_ids.unsqueeze(1)
        logprobs.clip_(min=paddle.finfo(logprobs.dtype).min)
        # Get with the logprob of the prompt or sampled token.
        token_logprobs = paddle.take_along_axis(logprobs, token_ids, axis=-1)

        # Compute the ranks of the actual token.
        token_ranks = (logprobs >= token_logprobs).sum(-1)

        if num_logprobs >= 1:
            # Find the topK values.
            topk_logprobs, topk_indices = paddle.topk(logprobs, num_logprobs, axis=-1)
            indices = paddle.concat([token_ids, topk_indices], axis=1)
            top_logprobs = paddle.concat([token_logprobs, topk_logprobs], axis=1)
        else:
            indices = token_ids
            top_logprobs = token_logprobs

        return LogprobsTensors(indices, top_logprobs, token_ranks)

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        max_model_len: int,
        share_inputs: List[paddle.Tensor],
    ) -> paddle.Tensor:
        """ """
        num_logprobs = sampling_metadata.max_num_logprobs
        real_bsz = share_inputs["seq_lens_this_time"].shape[0]
        if num_logprobs is not None and share_inputs["substep"] == 0:
            real_token_num = share_inputs["batch_token_num"][:real_bsz].sum()
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(
                    share_inputs["draft_logits"][:real_token_num, :], sampling_metadata
                )
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = share_inputs["draft_logits"][:real_token_num, :].clone()

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

        _, next_tokens = top_k_top_p_sampling(
            probs, sampling_metadata.top_p, sampling_metadata.top_k, sampling_metadata.top_k_list
        )

        token_ids = None
        logprobs_tensors = None
        if num_logprobs is not None and share_inputs["substep"] == 0:
            token_ids = paddle.empty(real_token_num, dtype="int64")
            speculate_insert_first_token(
                token_ids,
                share_inputs["accept_tokens"],
                next_tokens,
                share_inputs["cu_next_token_offset"],
                share_inputs["cu_batch_token_offset"],
                share_inputs["seq_lens_this_time"],
                share_inputs["seq_lens_encoder"],
            )

            logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=token_ids)

        sampler_output = SamplerOutput(
            sampled_token_ids=token_ids,
            logprobs_tensors=logprobs_tensors,
            token_num_per_batch=share_inputs["batch_token_num"][:real_bsz],
            cu_batch_token_offset=share_inputs["cu_batch_token_offset"],
        )
        return next_tokens, sampler_output
