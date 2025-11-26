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

import multiprocessing
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Optional

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddleformers.utils.log import logger

from fastdeploy.config import FDConfig
from fastdeploy.envs import FD_FILL_BITMASK_BATCH
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


def padding_sampling_params(top_p, top_k, seq_lens_this_time, seq_lens_encoder):
    real_bsz = seq_lens_this_time.shape[0]
    repeats = paddle.where(seq_lens_encoder[:real_bsz] == 0, seq_lens_this_time, paddle.ones_like(seq_lens_this_time))
    top_p_padding = paddle.repeat_interleave(top_p[:real_bsz], repeats).unsqueeze(1)
    top_k_padding = paddle.repeat_interleave(top_k[:real_bsz], repeats).unsqueeze(1)
    return top_p_padding, top_k_padding


class GuidedDecoding:
    """
    processor for guided decoding.
    """

    def __init__(self, fd_config: FDConfig):
        self.token_bitmask = None
        self.max_num_seqs: int = int(
            fd_config.scheduler_config.max_num_seqs if fd_config.scheduler_config is not None else 1
        )
        self.logits_processors: List[Any] = [None] * self.max_num_seqs
        self.reasoning_parser = None
        self._prefill_done_idxs: List[bool] = [False] * self.max_num_seqs
        # for pd
        self._tokens_to_acc: List[None | List[int]] = [None] * self.max_num_seqs

        self.fill_bitmask_parallel_batch_size: int = FD_FILL_BITMASK_BATCH
        max_workers = max(
            1,
            min(multiprocessing.cpu_count() // 2, int(self.max_num_seqs) / int(self.fill_bitmask_parallel_batch_size)),
        )
        self.executor_for_fillmask = ThreadPoolExecutor(max_workers=int(max_workers))
        self._fillmask_futures: List[Future] = [None] * self.max_num_seqs
        self.is_cuda_platform = current_platform.is_cuda()
        logger.info(
            f"GuidedDecoding max_num_seqs={self.max_num_seqs} fill_bitmask_parallel_batch_size={self.fill_bitmask_parallel_batch_size} is_cuda_platform={self.is_cuda_platform} max_workers={max_workers}"
        )

    def apply_reasoning_parser(self, reasoning_parser: Optional[ReasoningParser] = None):
        self.reasoning_parser = reasoning_parser

    def add_logits_processor(
        self,
        idx: int,
        future: Optional[Any] = None,
        prefill_tokens: List[int] = [],
    ):
        """add logits processor to SamplerProcessor"""
        self._prefill_done_idxs[idx] = False

        if future is None:
            # normal request without guided_backend
            self.logits_processors[idx] = None
            return

        if len(prefill_tokens) != 0:
            # first_token from prefill node
            self._prefill_done_idxs[idx] = True

        if future.done():
            # cached xgrammar
            self.logits_processors[idx] = future.result()
            for token in prefill_tokens:
                self._accept_token(idx, token)
        else:
            # async
            self.logits_processors[idx] = future
            self._tokens_to_acc[idx] = prefill_tokens

    def should_fill_bitmask(self, idx: int) -> bool:
        """
        Determines whether to fill a bitmask for the logits processor at the given index.

        Args:
            idx (int): The index of the logits processor to check

        Returns:
            bool: True if the idx request bitmask should be filled

        """
        if self.reasoning_parser is not None:
            if self.logits_processors[idx].enable_reasoning:  # <think> guided
                return True
            if not self.logits_processors[idx].reasoning_ended:
                return False
        return True

    def reset_processor(self, idx: int):
        """reset idx"""
        self._prefill_done_idxs[idx] = False
        self.logits_processors[idx] = None

    def update_vocab_mask(self, prefill_done_idxs: List[int] = []):
        """update vocab mask. (cpu-heavy operation)"""
        for idx in prefill_done_idxs:
            if self.logits_processors[idx] is None:
                continue

            assert not self._prefill_done_idxs[idx]
            self._prefill_done_idxs[idx] = True
            if isinstance(self.logits_processors[idx], Future):
                continue

        idxs = []
        for idx, processor in enumerate(self.logits_processors):
            if processor is None or not self._prefill_done_idxs[idx]:
                continue
            # skip, join at apply_token_mask
            if isinstance(processor, Future):
                continue
            if processor.is_terminated:
                self.reset_processor(idx)
                continue

            self.accept_tokens_from_prefill_node(idx)

            if self.token_bitmask is None:
                self.token_bitmask = self.logits_processors[idx].allocate_token_bitmask()

            if self.should_fill_bitmask(idx):
                idxs.append(idx)
        self._async_batch_fill_token_bitmask(idxs)

    def batch_fill_token_bitmask(self, batch: List[int]):
        """
        Fills the token bitmask for a batch of logits processor indices.

        This method is typically called asynchronously via a thread pool executor
        to parallelize the bitmask filling operation. It is important that any
        shared data structures accessed within this method (such as
        `self.token_bitmask` and `self.logits_processors`) are thread-safe or
        properly synchronized to avoid race conditions.

        Args:
            batch (List[int]): List of indices for which to fill the token bitmask.
        """
        for idx in batch:
            self.logits_processors[idx].fill_token_bitmask(self.token_bitmask, idx)

    def _async_batch_fill_token_bitmask(self, idxs: List[int]):
        """launch async fill"""
        batch: List[int] = []
        for idx in idxs:
            batch.append(idx)
            if len(batch) == self.fill_bitmask_parallel_batch_size:
                promise = self.executor_for_fillmask.submit(self.batch_fill_token_bitmask, batch[:])
                self._fillmask_futures[idx] = promise
                batch = []
        if batch:
            promise = self.executor_for_fillmask.submit(self.batch_fill_token_bitmask, batch[:])
            self._fillmask_futures[batch[-1]] = promise

    def join_async_fillmask(self):
        """join all async fill futures"""
        for idx, furture in enumerate(self._fillmask_futures):
            if furture is not None:
                try:
                    furture.result()
                except Exception as e:
                    logger.error(f"Exception in async fillmask future at idx {idx}: {e}", exc_info=True)
                self._fillmask_futures[idx] = None

    def accept_tokens_from_prefill_node(self, idx: int):
        """accept prefill token, not future"""
        if self._tokens_to_acc[idx] is not None:
            # accept token from prefill node first
            for token in self._tokens_to_acc[idx]:
                self._accept_token(idx, token)
            self._tokens_to_acc[idx] = None

    def apply_token_mask(self, logits: paddle.Tensor, prefill_done_idxs: List[int] = []):
        """apply token mask to logits"""

        indices = []
        for idx, processor in enumerate(self.logits_processors):
            if processor is None or not self._prefill_done_idxs[idx]:
                continue

            # compiled done, check idx should fill,  fill_token_bitmask done in preprocess
            if not isinstance(processor, Future):
                if self.should_fill_bitmask(idx):
                    indices.append(idx)
                continue

            # is Future, processor async compiled not ready, need join and wait
            ts = time.time()
            wait = False
            if not processor.done():
                wait = True
            self.logits_processors[idx] = processor.result()
            if wait:
                logger.debug(f"[{idx} join async compile xgrammar, time_cost:{time.time() - ts}]")

            self.accept_tokens_from_prefill_node(idx)
            # Possible optimization: Extract 'think' content validation from logits_processors,
            # allowing join operations to complete immediately after 'think' terminates.
            # Furthermore, the current idx could be skipped, with compilation overhead
            # estimated at only a few milliseconds.

            # check idx for fill_token_mask
            if not self.should_fill_bitmask(idx):
                continue

            indices.append(idx)

            if self.token_bitmask is None:
                self.token_bitmask = self.logits_processors[idx].allocate_token_bitmask()

            # launch async fill
            self._async_batch_fill_token_bitmask([idx])

        if len(indices) == 0:
            return logits
        self.join_async_fillmask()
        from fastdeploy.model_executor.guided_decoding.xgrammar_backend import (
            apply_token_mask,
        )

        return apply_token_mask(logits, self.token_bitmask, indices=indices, is_cuda_platform=self.is_cuda_platform)

    def _accept_token(self, idx: int, token: int):
        """accept token"""

        if self.reasoning_parser is not None:
            if not self.logits_processors[idx].enable_reasoning:
                if not self.logits_processors[idx].reasoning_ended:
                    reasoning_ended = self.reasoning_parser.is_reasoning_end([token])
                    self.logits_processors[idx].reasoning_ended = reasoning_ended
                    return

        if not self.logits_processors[idx].accept_token(token) or self.logits_processors[idx].is_terminated:
            self.reset_processor(idx)

    def update_output_tokens(self, next_tokens: paddle.Tensor):
        """update output tokens"""
        if len(self.logits_processors) == 0:
            return

        token_ids = next_tokens.numpy().tolist()
        for idx, processor in enumerate(self.logits_processors):
            if not self._prefill_done_idxs[idx] or processor is None:
                continue
            if idx >= len(token_ids):
                continue
            token = token_ids[idx][0]
            if token < 0:
                self.reset_processor(idx)
                continue
            logger.debug(f"[{idx}]accept token{token}")
            self._accept_token(idx, token)

    def pre_process(self, prefill_done_idxs: List[int] = []):
        """pre process before running"""
        self.update_vocab_mask(prefill_done_idxs)


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

        self.guided_decoding = GuidedDecoding(fd_config)
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
        self.guided_decoding.apply_reasoning_parser(reasoning_parser)

    def apply_logits_processor(
        self, ids: int, future: Future[LogitsProcessorBase] = None, prefill_tokens: List[int] = []
    ):
        """apply logits processor to sampler"""
        self.guided_decoding.add_logits_processor(ids, future, prefill_tokens)

    def pre_process(self, prefill_done_idxs: List[int] = []):
        """pre process before running"""
        self.guided_decoding.pre_process(prefill_done_idxs)

    def post_process(self, next_tokens: paddle.Tensor):
        """post process after running"""
        self.guided_decoding.update_output_tokens(next_tokens)

    def compute_logprobs(
        self,
        logits: paddle.Tensor,
        sampling_metadata: Optional[SamplingMetadata] = None,
    ) -> paddle.Tensor:
        """ """
        if sampling_metadata is None:
            return F.log_softmax(logits, axis=-1)
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
        if len(token_ids.shape) < len(logprobs.shape):
            token_ids = token_ids.unsqueeze(-1)
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
        indices = indices.cpu()
        top_logprobs = top_logprobs.cpu()
        token_ranks = token_ranks.cpu()
        return LogprobsTensors(indices, top_logprobs, token_ranks)

    def forward_cuda(
        self,
        logits: paddle.Tensor,
        sampling_metadata: SamplingMetadata,
        p_done_idxs: List[int] = [],
    ) -> SamplerOutput:
        """ """
        logits = self.guided_decoding.apply_token_mask(logits, p_done_idxs)

        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits, sampling_metadata)
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = logits.clone()

        for proc in sampling_metadata.logits_processors or []:
            logits = proc.apply(logits)

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
        batch_token_num = share_inputs["accept_num"][:real_bsz]

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
        reject_all_drafts: bool = False,
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

        top_p, top_k = padding_sampling_params(
            sampling_metadata.top_p,
            sampling_metadata.top_k,
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
        )
        _, sampled_token_ids = top_k_top_p_sampling(probs, top_p=top_p, top_k=top_k, seed=sampling_metadata.seed[0, 0])

        verify_scores, verify_tokens, actual_candidate_len = top_p_candidates(
            probs,
            sampling_metadata.top_p,
            share_inputs["output_padding_offset"],
            self.speculative_max_candidate_len,
            max_model_len,
        )

        speculate_verify(
            sampled_token_ids,
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
            (self.speculative_benchmark_mode or reject_all_drafts),
            accept_all_drafts,
        )

        num_logprobs = sampling_metadata.max_num_logprobs
        batch_token_num = None
        if num_logprobs is not None:
            real_bsz = share_inputs["seq_lens_this_time"].shape[0]
            batch_token_num = paddle.where(
                share_inputs["seq_lens_encoder"][:real_bsz] != 0,
                paddle.ones_like(share_inputs["seq_lens_encoder"][:real_bsz]),
                share_inputs["seq_lens_this_time"],
            ).squeeze(1)
            share_inputs["batch_token_num"] = batch_token_num
            ori_cu_batch_token_offset = paddle.concat([paddle.to_tensor([0]), paddle.cumsum(batch_token_num)]).astype(
                "int32"
            )
            cu_batch_token_offset = paddle.concat(
                [paddle.to_tensor([0]), paddle.cumsum(share_inputs["accept_num"][:real_bsz])]
            ).astype("int32")
            share_inputs["cu_batch_token_offset"] = cu_batch_token_offset
            target_logits = paddle.empty(
                [share_inputs["accept_num"][:real_bsz].sum(), logits.shape[1]], dtype=logits.dtype
            )
            speculate_get_target_logits(
                target_logits,
                logits,
                cu_batch_token_offset,
                ori_cu_batch_token_offset,
                share_inputs["seq_lens_this_time"],
                share_inputs["seq_lens_encoder"],
                share_inputs["accept_num"],
            )
            if self.logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(target_logits, sampling_metadata)
            elif self.logprobs_mode == "raw_logits":
                raw_logprobs = target_logits.clone()

        logprobs_tensors = None
        token_ids = share_inputs["accept_tokens"]
        if num_logprobs is not None:
            token_ids = paddle.concat(
                [share_inputs["accept_tokens"][i, : share_inputs["accept_num"][i]] for i in range(real_bsz)]
            )
            logprobs_tensors = self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=token_ids)

        sampler_output = SamplerOutput(
            sampled_token_ids=token_ids,
            logprobs_tensors=logprobs_tensors,
            token_num_per_batch=share_inputs["accept_num"],
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

        top_p, top_k = padding_sampling_params(
            sampling_metadata.top_p,
            sampling_metadata.top_k,
            share_inputs["seq_lens_this_time"],
            share_inputs["seq_lens_encoder"],
        )
        _, next_tokens = top_k_top_p_sampling(probs, top_p=top_p, top_k=top_k, seed=sampling_metadata.seed[0, 0])

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
