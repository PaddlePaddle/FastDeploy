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

import copy
import threading
import time
import traceback
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Union

import numpy as np
import paddle

from fastdeploy import envs
from fastdeploy.cache_manager.multimodal_cache_manager import (
    EncoderCacheManager,
    ProcessorCacheManager,
)
from fastdeploy.config import ErnieArchitectures
from fastdeploy.engine.request import (
    ImagePosition,
    Request,
    RequestOutput,
    RequestStatus,
    RequestType,
)
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.input.utils import IDS_TYPE_FLAG
from fastdeploy.inter_communicator import IPCSignal
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.multimodal.hasher import MultimodalHasher
from fastdeploy.platforms import current_platform
from fastdeploy.utils import download_from_bos, init_bos_client, llm_logger


@dataclass
class ScheduledDecodeTask:
    """
    Task for allocating new blocks to decode.
    """

    idx: int
    request_id: str
    block_tables: list[int]
    task_type: RequestType = RequestType.DECODE


@dataclass
class ScheduledPreemptTask:
    """
    Task for terminating inference to recycle resource.
    """

    idx: int
    request_id: str
    task_type: RequestType = RequestType.PREEMPTED


@dataclass
class ScheduledExtendBlocksTask:
    """
    Task for allocating new blocks to extend.
    """

    idx: int
    request_id: str
    extend_block_tables: list[int]
    task_type: RequestType = RequestType.EXTEND


class SignalConsumer:
    """
    A class that consumes a signal value up to a specified limit.

    This class maintains an internal signal value and allows controlled consumption
    of that signal. The signal can be watched at any time, but can only be consumed
    a limited number of times before being reset to zero.
    """

    def __init__(self, signal, consume_limit):
        """
        Initialize the SignalConsumer with a signal value and consumption limit.

        Args:
            signal: The initial signal value to be consumed.
            consume_limit (int): The maximum number of times the signal can be consumed
                                before being reset to 0. Must be a positive integer.

        Raises:
            AssertionError: If consume_limit is not greater than 0.
        """
        assert consume_limit > 0

        self._signal = signal
        self._consume_limit = consume_limit

    def watch(self):
        """
        Get the current signal value without consuming it.

        This method allows reading the signal value any number of times without
        affecting the consumption limit or the signal value itself.

        Returns:
            The current signal value.
        """
        return self._signal

    def consume(self):
        """
        Consume the signal value, decrementing the consumption limit.

        This method returns the current signal value and decrements the consumption
        counter. When the consumption limit reaches zero, the signal is automatically
        reset to 0. The consumption happens in a finally block to ensure the limit is
        decremented even if an exception occurs while processing the signal.

        Returns:
            The current signal value before consumption.

        Note:
            After the consumption limit is reached, this method will continue to
            return 0 on subsequent calls.
        """
        try:
            return self._signal
        finally:
            if self._consume_limit > 0:
                self._consume_limit -= 1
            if self._consume_limit == 0:
                self._signal = 0


class ResourceManagerV1(ResourceManager):
    """
    Resource manager for scheduler v1.
    In scheduler v1, all gpu blocks are managed by PrefixCacheManager.
    Tasks sent to worker are divided into 3 types, PREFILL、DECODE and PREEMPTED.
    For prefill task, the worker infer with one step and then stopped for this query if not all prompt tokens are computed.
    For decode task, the work continues to decode until allocated blocks are exhausted.
    For preempted task, the work reset all inputs to terminate the inference.
    """

    def __init__(self, max_num_seqs, config, tensor_parallel_size, splitwise_role, local_data_parallel_id=0):
        super(ResourceManagerV1, self).__init__(
            max_num_seqs, config, tensor_parallel_size, splitwise_role, local_data_parallel_id
        )
        # req_id -> Request
        self.config = config
        self.requests: dict[str, Request] = {}
        # Priority queues for requests.
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []
        self.preallocated_reqs: dict[str, Request] = {}
        self.enable_max_prefill = envs.FD_ENABLE_MAX_PREFILL
        self.finish_execution_pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.to_be_rescheduled_request_id_set = set()
        main_process_metrics.max_batch_size.set(max_num_seqs)

        self.using_extend_tables_req_id = set()
        self.reuse_block_num_map = dict()
        self.abort_req_ids_set = set()

        # need block nums
        need_block_num_data = np.zeros([max_num_seqs], dtype=np.int32)
        self.need_block_num_signal = IPCSignal(
            name="need_block_num_signal",
            array=need_block_num_data,
            dtype=np.int32,
            suffix=self.config.parallel_config.local_engine_worker_queue_port,
            create=True,
        )

        self.need_block_num_map = dict()

        self.encoder_cache = None
        if config.model_config.enable_mm and config.cache_config.max_encoder_cache > 0:
            self.encoder_cache = EncoderCacheManager(config.cache_config.max_encoder_cache)

        self.processor_cache = None
        if config.model_config.enable_mm and config.cache_config.max_processor_cache > 0:
            max_processor_cache_in_bytes = int(config.cache_config.max_processor_cache * 1024 * 1024 * 1024)
            self.processor_cache = ProcessorCacheManager(max_processor_cache_in_bytes)

        self.bos_client = None
        self.async_preprocess_pool = ThreadPoolExecutor(max_workers=4)

        # New token ratio mechanism for dynamic decode reservation (inspired by SGLang)
        # This replaces the fixed per-request block reservation with a ratio-based approach
        schedule_conservativeness = 1.0  # Can be made configurable via SchedulerConfig later
        self.init_new_token_ratio = min(
            envs.FD_INIT_NEW_TOKEN_RATIO * schedule_conservativeness,
            1.0,
        )
        self.min_new_token_ratio = min(
            self.init_new_token_ratio * envs.FD_MIN_NEW_TOKEN_RATIO_FACTOR,
            1.0,
        )
        self.new_token_ratio_decay = (
            self.init_new_token_ratio - self.min_new_token_ratio
        ) / envs.FD_NEW_TOKEN_RATIO_DECAY_STEPS
        self.current_new_token_ratio = self.init_new_token_ratio
        self.clip_max_new_tokens_estimation = envs.FD_CLIP_MAX_NEW_TOKENS_ESTIMATION

        llm_logger.info(
            f"NewTokenRatio initialized: init={self.init_new_token_ratio:.3f}, "
            f"min={self.min_new_token_ratio:.3f}, decay_per_step={self.new_token_ratio_decay:.6f}, "
            f"clip_max_tokens={self.clip_max_new_tokens_estimation}"
        )

    def allocated_slots(self, request: Request):
        return len(request.block_tables) * self.config.cache_config.block_size

    def get_new_block_nums(self, request: Request, num_new_tokens: int):
        block_num = (
            request.num_computed_tokens + num_new_tokens + self.config.cache_config.block_size - 1
        ) // self.config.cache_config.block_size - len(request.block_tables)

        if self.config.speculative_config.method is not None:
            block_num = min(block_num + 1, self.config.cache_config.max_block_num_per_seq)
        return block_num

    def _prepare_prefill_task(self, request, new_token_num):
        request.prefill_start_index = request.num_computed_tokens
        request.prefill_end_index = request.num_computed_tokens + new_token_num
        request.task_type = RequestType.PREFILL
        return request

    def _prepare_decode_task(self, request):
        return ScheduledDecodeTask(idx=request.idx, request_id=request.request_id, block_tables=request.block_tables)

    def _prepare_preempt_task(self, request):
        return ScheduledPreemptTask(idx=request.idx, request_id=request.request_id)

    def reschedule_preempt_task(self, request_id, process_func=None):
        with self.lock:
            llm_logger.debug(f"reschedule {request_id} into waiting queue")
            if request_id in self.to_be_rescheduled_request_id_set and request_id in self.requests:
                request = self.requests[request_id]
                if process_func is not None:
                    process_func(request)
                llm_logger.debug(f"self.waiting append request to end:{request.request_id},req.type:{request.status}")
                self.waiting.append(request)  # Append to end of queue (FIFO order)
                self.to_be_rescheduled_request_id_set.remove(request_id)

    def _info_each_block(self):
        """
        print each req block
        """
        for req in self.running:
            llm_logger.debug(
                f"req idx {req.idx} occupy {len(req.block_tables)} block_tables and {len(req.extend_block_tables)} extend_block_tables"
            )

    def _can_preempt(self):
        """
        cannot preempt request which use extend block
        """
        for req in self.running:
            if not req.use_extend_tables:
                return True
        return False

    def preempted_all(self):
        with self.lock:
            preempted_reqs = []
            for i in range(len(self.running)):
                req = self.running.pop()
                # txt2image: req.use_extend_tables is True, req can not be preempted. txt2image is not used in RL.
                if req.use_extend_tables:
                    self.running.insert(0, req)
                    continue
                req.status = RequestStatus.PREEMPTED
                req.num_computed_tokens = 0
                self._free_blocks(req)
                req.cached_block_num = 0
                self.to_be_rescheduled_request_id_set.add(req.request_id)
                preempted_reqs.append(self._prepare_preempt_task(req))
            return preempted_reqs

    def wait_worker_inflight_requests_finish(self, timeout=60):
        count = 0
        while count < timeout * 1000:
            # wait ongoing running and rescheduled requests finished in worker
            running_reqs_count = len(self.to_be_rescheduled_request_id_set) + len(self.running)
            if running_reqs_count == 0:
                break

            count += 1
            time.sleep(0.001)
        if count >= timeout * 1000:
            llm_logger.info(
                f"wait_inflight_requests_finish timeout after {timeout} seconds, "
                f"still {len(self.to_be_rescheduled_request_id_set)} requests running"
            )

    def _trigger_preempt(self, request, num_new_blocks, preempted_reqs, scheduled_reqs):
        """
        If the request cannot be scheduled, preempt running decode requests one by one until it can be scheduled.
        Only preempt decode requests (num_computed_tokens >= need_prefill_tokens).

        SGLang-aligned strategy:
        - Sort by (output_len asc, input_len desc) to prioritize retracting short-output, long-input requests
        - Keep at least 1 request in running
        - After preemption, update current_new_token_ratio based on remaining requests
        """
        # Collect decode requests (num_computed_tokens >= need_prefill_tokens)
        decode_requests = [
            req for req in self.running
            if req.num_computed_tokens >= req.need_prefill_tokens
        ]

        # SGLang-aligned sort: prioritize retracting requests with shorter output
        # If output_len is equal, prioritize retracting requests with longer input
        decode_requests.sort(
            key=lambda r: (len(r.output_token_ids), -r.prompt_token_ids_len),
            reverse=True,  # pop from end: shorter output first
        )

        # If only 1 decode request, cannot preempt (need to keep at least 1)
        # Return False to let scheduler handle this gracefully
        if len(decode_requests) <= 1:
            can_schedule = self.cache_manager.can_allocate_gpu_blocks(num_new_blocks)
            return can_schedule

        preempted_count = 0
        remaining_req_count = len(decode_requests) - 1  # Count for KV eviction (decreases after each preempt)

        for preempted_req in decode_requests[:-1]:  # Skip last one to keep at least 1
            if self.cache_manager.can_allocate_gpu_blocks(num_new_blocks):
                break

            # Remove from running list
            self.running.remove(preempted_req)
            preempted_req.status = RequestStatus.PREEMPTED
            preempted_req.num_computed_tokens = 0

            # Mark as retracted for SGLang alignment
            preempted_req.is_retracted = True

            if self.config.scheduler_config.splitwise_role == "decode":
                self.tasks_list[preempted_req.idx] = None
                self.stop_flags[preempted_req.idx] = True
                if preempted_req.request_id in self.requests:
                    del self.requests[preempted_req.request_id]
                if preempted_req.request_id in self.req_dict:
                    del self.req_dict[preempted_req.request_id]
                self._free_blocks(preempted_req)
                llm_logger.info(f"Preemption is triggered! Preempted request id: {preempted_req.request_id}")
            else:
                self._free_blocks(preempted_req)
                preempted_req.num_cached_blocks = 0
                self.to_be_rescheduled_request_id_set.add(preempted_req.request_id)
                llm_logger.info(f"Preemption is triggered! Preempted request id: {preempted_req.request_id}")

            preempted_reqs.append(preempted_req)
            scheduled_reqs.append(self._prepare_preempt_task(preempted_req))
            preempted_count += 1

            # Evict KV cache from tree (SGLang-aligned: retract_decode_steps * remaining_req_count)
            self._evict_decode_kv_cache(remaining_req_count)
            remaining_req_count -= 1

            llm_logger.debug(
                f"preempt {preempted_req.request_id} in idx {preempted_req.idx} "
                f"with output_len={len(preempted_req.output_token_ids)}, "
                f"input_len={preempted_req.prompt_token_ids_len}"
            )

        if preempted_count > 0:
            llm_logger.debug(self.info())
            self._info_each_block()

            # Update new_token_ratio based on remaining requests (SGLang style)
            self._update_new_token_ratio_after_preemption()

        # Check if we can schedule now
        can_schedule = self.cache_manager.can_allocate_gpu_blocks(num_new_blocks)
        return can_schedule

    def _evict_decode_kv_cache(self, remaining_req_count: int):
        """
        Evict KV cache from tree cache after retracting a decode request.

        SGLang-aligned strategy:
        - Each retraction triggers eviction of retract_decode_steps * remaining_req_count tokens
        - This frees up space for new requests immediately

        Args:
            remaining_req_count: Number of requests that will remain in running after eviction
        """
        # Default retract_decode_steps = 20 (SGLang default)
        retract_decode_steps = getattr(self, 'retract_decode_steps', 20)

        num_tokens_to_evict = remaining_req_count * retract_decode_steps

        llm_logger.debug(
            f"Evicting {num_tokens_to_evict} KV cache tokens "
            f"(retract_decode_steps={retract_decode_steps}, remaining_req_count={remaining_req_count})"
        )

        # Convert tokens to blocks for FD's cache manager
        # FD's PrefixCacheManager uses block-based eviction
        if self.cache_manager is not None:
            block_size = self.config.cache_config.block_size
            num_blocks_to_evict = (num_tokens_to_evict + block_size - 1) // block_size

            llm_logger.debug(
                f"Evicting {num_blocks_to_evict} blocks "
                f"(={num_tokens_to_evict} tokens) from GPU cache "
                f"(block_size={block_size})"
            )

            # Use FD's existing cache eviction API
            # free_block_ids (sync version) will handle the LRU eviction logic
            # including GPU -> CPU swap and GPU -> Storage persistence
            # Use sync version to ensure blocks are freed before checking can_allocate
            self.cache_manager.free_block_ids(num_blocks_to_evict)
        else:
            llm_logger.warning("cache_manager is None, cannot evict KV cache")

    def _update_new_token_ratio_after_preemption(self):
        """
        Update current_new_token_ratio based on remaining running decode requests.
        Mimics SGLang's logic:
        new_ratio = (total_decoded_tokens + retract_decode_steps * num_decode_reqs) / (total_max_new_tokens + 1)

        Note: Only count decode requests (num_computed_tokens >= need_prefill_tokens),
        not prefill requests.

        SGLang-aligned behavior:
        - If no decode requests running, keep the current ratio unchanged
          (matching SGLang: ratio remains continuous when only prefill requests are running)
        - Ratio reset only happens when the system is completely idle
          (no prefill and no decode requests)
        """
        # Filter to only decode requests (matching SGLang's self.reqs)
        decode_reqs = [
            req for req in self.running
            if req.num_computed_tokens >= req.need_prefill_tokens
        ]

        if len(decode_reqs) == 0:
            # No decode requests running, keep current ratio unchanged
            # (matching SGLang: ratio remains continuous when only prefill requests are running)
            llm_logger.debug(
                f"No decode requests after preemption, keeping current_new_token_ratio={self.current_new_token_ratio:.3f}"
            )
            return

        total_decoded_tokens = 0
        total_max_new_tokens = 0

        for req in decode_reqs:
            # Count decoded tokens
            already_decoded = len(req.output_token_ids)
            total_decoded_tokens += already_decoded

            # Get max_new_tokens for this request
            if req.sampling_params and req.sampling_params.max_tokens is not None:
                max_new_tokens = req.sampling_params.max_tokens
            else:
                max_new_tokens = self.config.model_config.max_model_len - req.need_prefill_tokens
            total_max_new_tokens += max_new_tokens

        # SGLang's formula with retract_decode_steps (SGLang default: 20)
        retract_decode_steps = getattr(self, 'retract_decode_steps', 20)
        num_decode_reqs = len(decode_reqs)

        new_ratio = (
            total_decoded_tokens + retract_decode_steps * num_decode_reqs
        ) / (total_max_new_tokens + 1)

        # Clamp to [min_ratio, init_ratio]
        new_ratio = max(self.min_new_token_ratio, min(self.init_new_token_ratio, new_ratio))

        llm_logger.debug(
            f"Update new_token_ratio after preemption: "
            f"decode_reqs={num_decode_reqs}, decoded={total_decoded_tokens}, "
            f"max_new={total_max_new_tokens}, ratio={new_ratio:.3f} "
            f"(was {self.current_new_token_ratio:.3f})"
        )

        self.current_new_token_ratio = new_ratio

    def reset_new_token_ratio_on_idle(self):
        """
        Reset new_token_ratio to init_new_token_ratio when system is completely idle.

        SGLang alignment: Only reset when both running and waiting queues are empty.
        This mimics SGLang's self_check_during_idle behavior.
        """
        if len(self.running) == 0 and len(self.waiting) == 0:
            if self.current_new_token_ratio != self.init_new_token_ratio:
                llm_logger.debug(
                    f"System completely idle, resetting new_token_ratio "
                    f"from {self.current_new_token_ratio:.3f} to {self.init_new_token_ratio:.3f}"
                )
                self.current_new_token_ratio = self.init_new_token_ratio

    def _calculate_decode_reserved_tokens_by_ratio(self):
        """
        Calculate total reserved tokens for all running decode requests based on current_new_token_ratio.

        For each request in decode phase, calculate:
            remaining_tokens = min(max_new_tokens - already_decoded, clip_estimation)
            reserved_tokens = remaining_tokens * current_new_token_ratio

        Returns:
            int: Total number of tokens to reserve for decode requests
        """
        total_reserved_tokens = 0
        num_decode_reqs = 0

        for req in self.running:
            # Only calculate reservation for requests in decode phase
            if req.num_computed_tokens < req.need_prefill_tokens:
                continue  # Still in prefill, skip

            num_decode_reqs += 1

            # Get max_new_tokens for this request
            if req.sampling_params and req.sampling_params.max_tokens is not None:
                max_new_tokens = req.sampling_params.max_tokens
            else:
                # Fallback: use max_model_len - prompt_len
                max_new_tokens = self.config.model_config.max_model_len - req.need_prefill_tokens

            # Calculate remaining tokens to generate
            already_decoded = len(req.output_token_ids)
            remaining_tokens = max(0, max_new_tokens - already_decoded)

            # Clip to reasonable upper bound to avoid single long request dominating budget
            remaining_tokens = min(remaining_tokens, self.clip_max_new_tokens_estimation)

            # Calculate reservation based on current ratio (no rounding here, keep precision)
            reserved_tokens = remaining_tokens * self.current_new_token_ratio
            total_reserved_tokens += reserved_tokens

        llm_logger.debug(
            f"Decode reservation: {num_decode_reqs} decode reqs, "
            f"{total_reserved_tokens:.1f} tokens, ratio={self.current_new_token_ratio:.3f}"
        )

        return total_reserved_tokens

    def _get_can_schedule_prefill_threshold_block(self, request, num_chunk_new_block):
        """
        Calculate the total tokens needed for scheduling a new prefill request.

        Total tokens includes:
        1. Tokens needed for current prefill
        2. Tokens reserved for this request's future decode (max_new_tokens)
        3. Tokens reserved for all existing decode requests (ratio-based)

        Returns:
            int: Total blocks needed (ceiled once at the end) to safely admit this request
        """
        # 1. Tokens needed for current prefill
        required_tokens_for_prefill = request.need_prefill_tokens

        # 2. Tokens reserved for this request's future decode (full max_new_tokens)
        if hasattr(request, 'sampling_params') and request.sampling_params and request.sampling_params.max_tokens:
            max_new_tokens_for_request = request.sampling_params.max_tokens
        else:
            max_new_tokens_for_request = self.config.model_config.max_model_len - request.need_prefill_tokens
        max_new_tokens_for_request = min(max_new_tokens_for_request, self.clip_max_new_tokens_estimation)

        # 3. Tokens reserved for all existing decode requests (ratio-based)
        decode_reserved_tokens = self._calculate_decode_reserved_tokens_by_ratio()

        # Sum all tokens and convert to blocks once at the end
        total_tokens = required_tokens_for_prefill + max_new_tokens_for_request + decode_reserved_tokens
        can_schedule_block_num_threshold = (
            total_tokens + self.config.cache_config.block_size - 1
        ) // self.config.cache_config.block_size

        if self.config.speculative_config.method is not None:
            can_schedule_block_num_threshold = min(
                can_schedule_block_num_threshold + 1, self.config.cache_config.max_block_num_per_seq
            )

        llm_logger.debug(
            f"Prefill threshold: tokens={total_tokens:.1f} -> blocks={can_schedule_block_num_threshold} "
            f"(prefill={required_tokens_for_prefill}, future_decode={max_new_tokens_for_request:.1f}, "
            f"decode_reserved={decode_reserved_tokens:.1f})"
        )

        return can_schedule_block_num_threshold

    def _update_mm_hashes(self, request):
        if request.multimodal_inputs is None:
            return

        inputs = request.multimodal_inputs
        if (
            inputs.get("images", None) is not None
            and inputs.get("image_patch_id", None) is not None
            and inputs.get("grid_thw", None) is not None
            and len(inputs["grid_thw"]) != 0
        ):
            grid_thw = []
            new_mm_positions, new_mm_hashes = [], []
            image_st = 0
            for idx, one in enumerate(inputs["grid_thw"]):
                t, h, w = one[0], one[1], one[2]
                if t == 1:
                    grid_thw.append(one)
                    new_mm_positions.append(inputs["mm_positions"][idx])
                    new_mm_hashes.append(inputs["mm_hashes"][idx])
                    image_st += h * w
                else:
                    grid_thw.extend([[2, h, w]] * (t // 2))
                    token_st = inputs["mm_positions"][idx].offset
                    for _ in range(t // 2):
                        mm_num_token = inputs["mm_num_token_func"](grid_thw=[2, h, w])
                        new_mm_positions.append(ImagePosition(token_st, mm_num_token))
                        # videos are split into patches every 2 frames, need to rehash
                        new_mm_hashes.append(
                            MultimodalHasher.hash_features(inputs["images"][image_st : image_st + 2 * h * w])
                        )
                        image_st += 2 * h * w
                        token_st += mm_num_token
            inputs["mm_positions"] = new_mm_positions
            inputs["mm_hashes"] = new_mm_hashes
        elif inputs.get("mm_positions", None) is None or inputs.get("mm_hashes", None) is None:
            inputs["mm_positions"] = []
            inputs["mm_hashes"] = []

    def _is_mm_request(self, request):
        inputs = request.multimodal_inputs
        if inputs is None or len(inputs) == 0:
            return False

        if (
            (inputs.get("video_feature_urls") is not None and len(inputs["video_feature_urls"]) > 0)
            or (inputs.get("image_feature_urls") is not None and len(inputs["image_feature_urls"]) > 0)
            or (inputs.get("audio_feature_urls") is not None and len(inputs["audio_feature_urls"]) > 0)
        ):
            return True
        elif (
            inputs.get("images", None) is not None
            and inputs.get("image_patch_id", None) is not None
            and inputs.get("grid_thw", None) is not None
        ):
            return True

        return False

    def revert_chunked_mm_input(self, mm_inputs, matched_token_num):
        """
        revert mm_inputs that is chunked
        """
        if mm_inputs is None or "mm_positions" not in mm_inputs or len(mm_inputs["mm_positions"]) == 0:
            return matched_token_num

        position_idx = len(mm_inputs["mm_positions"]) - 1
        while matched_token_num > 0 and position_idx >= 0:
            position = mm_inputs["mm_positions"][position_idx]
            if position.offset < matched_token_num < position.offset + position.length:
                matched_token_num = (
                    position.offset // self.config.cache_config.block_size
                ) * self.config.cache_config.block_size
                position_idx -= 1
            elif matched_token_num <= position.offset:
                position_idx -= 1
            elif matched_token_num >= position.offset + position.length:
                break
            else:
                llm_logger.error(
                    f"revert_chunked_mm_input error, matched_token_num:{matched_token_num} position:{position}, {mm_inputs['mm_positions']}"
                )
                break
        return matched_token_num

    def _get_num_new_tokens(self, request, chunked_prefill_size, token_budget):
        # SGLang-aligned: use min(chunked_prefill_size, token_budget) as the limit
        # chunked_prefill_size is the max tokens for a single request (like SGLang's rem_chunk_tokens)
        # token_budget (max_num_batched_tokens) is the total batch budget (like SGLang's rem_total_tokens)
        num_new_tokens = request.need_prefill_tokens - request.num_computed_tokens
        # SGLang logic: _rem_tokens = min(rem_chunk_tokens, rem_total_tokens)
        num_new_tokens = min(num_new_tokens, chunked_prefill_size, token_budget)
        if (
            current_platform.is_intel_hpu()
            and request.need_prefill_tokens - request.num_computed_tokens > min(chunked_prefill_size, token_budget)
            and min(chunked_prefill_size, token_budget) > self.config.cache_config.block_size
        ):
            num_new_tokens = min(chunked_prefill_size, token_budget) // self.config.cache_config.block_size * self.config.cache_config.block_size
        request.with_image = False

        if not self.config.model_config.enable_mm:
            return num_new_tokens

        inputs = request.multimodal_inputs
        if inputs.get("patch_idx", None) is not None and inputs.get("patch_map", None) is not None:
            pre_end_idx = request.num_computed_tokens
            new_end_idx = pre_end_idx + num_new_tokens

            prompt_token_ids_len = len(request.prompt_token_ids)
            if not inputs.get("tts", False):
                assert prompt_token_ids_len == len(inputs["patch_idx"]), (
                    prompt_token_ids_len,
                    len(inputs["patch_idx"]),
                )

            def _compute_audio_prefix_count(end_idx, end_patch_idx):
                audio_prefix_count = 0
                pre_patch_end_idx = 0
                for patch_idx in range(end_patch_idx + 1):
                    patch_map = inputs["patch_map"][patch_idx]
                    modal_id = patch_map["modal_id"]
                    if modal_id == IDS_TYPE_FLAG["audio"]:
                        if patch_idx != end_patch_idx:
                            audio_prefix_count += patch_map["end_idx"] - pre_patch_end_idx
                        else:
                            audio_prefix_count += end_idx - pre_patch_end_idx
                    pre_patch_end_idx = patch_map["end_idx"]
                return audio_prefix_count

            # start
            if pre_end_idx >= prompt_token_ids_len:
                start_patch_idx = inputs["patch_idx"][-1]
            else:
                start_patch_idx = inputs["patch_idx"][pre_end_idx]
                if (
                    pre_end_idx > 0
                    and request.prompt_token_ids[pre_end_idx]
                    in [
                        inputs["image_patch_id"],
                        inputs["video_patch_id"],
                        inputs["audio_patch_id"],
                    ]
                    and request.prompt_token_ids[pre_end_idx] != request.prompt_token_ids[pre_end_idx - 1]
                ):
                    # It just hit the starting position of the image / video / audio
                    start_patch_idx -= 1
            start_patch_map = inputs["patch_map"][start_patch_idx]
            request.image_start = start_patch_map["image_num"]
            request.video_start = start_patch_map["video_num"]
            request.audio_start = _compute_audio_prefix_count(pre_end_idx, start_patch_idx)

            # end
            if new_end_idx >= prompt_token_ids_len:
                end_patch_idx = inputs["patch_idx"][-1]
            else:
                end_patch_idx = inputs["patch_idx"][new_end_idx]
                if request.prompt_token_ids[new_end_idx] in [
                    inputs["image_end_id"],
                    inputs["video_end_id"],
                    inputs["audio_end_id"],
                ]:
                    end_patch_idx -= 1
            end_patch_map = inputs["patch_map"][end_patch_idx]
            end_modal_id = end_patch_map["modal_id"]
            if end_modal_id == IDS_TYPE_FLAG["image"]:
                new_end_idx = end_patch_map["end_idx"]  # 当前模态结束位置

            if end_modal_id == IDS_TYPE_FLAG["video"] and "can_split_idx_list" in inputs:
                can_split_idx_list = inputs["can_split_idx_list"]
                for i in range(len(can_split_idx_list)):
                    if can_split_idx_list[i] >= new_end_idx:
                        new_end_idx = can_split_idx_list[i]
                        break
            num_new_tokens = new_end_idx - pre_end_idx

            request.image_end = end_patch_map["image_num"]
            request.video_end = end_patch_map["video_num"]
            request.audio_end = _compute_audio_prefix_count(new_end_idx, end_patch_idx)
        elif (
            inputs.get("images", None) is not None
            and inputs.get("image_patch_id", None) is not None
            and inputs.get("grid_thw", None) is not None
        ):
            input_ids_lst = request.prompt_token_ids + request.output_token_ids
            input_ids = paddle.to_tensor(input_ids_lst, dtype="int64")
            image_patch_id = inputs["image_patch_id"]

            if request.multimodal_img_boundaries is None:
                grid_thw = []
                for idx, one in enumerate(inputs["grid_thw"]):
                    t, h, w = one[0], one[1], one[2]
                    if t == 1:
                        grid_thw.append(one)
                    else:
                        grid_thw.extend([[2, h, w]] * (t // 2))

                if current_platform.is_xpu():
                    from fastdeploy.model_executor.ops.xpu import get_img_boundaries
                elif current_platform.is_iluvatar():
                    from fastdeploy.model_executor.ops.iluvatar import (
                        get_img_boundaries,
                    )
                else:
                    from fastdeploy.model_executor.ops.gpu import get_img_boundaries

                mm_num_token = inputs["mm_num_token_func"](grid_thw=grid_thw)
                mm_num_token = paddle.to_tensor(mm_num_token, dtype="int64")
                request.multimodal_img_boundaries = get_img_boundaries(
                    task_input_ids=input_ids, mm_num_token=mm_num_token, image_patch_id=image_patch_id
                ).numpy()

                grid_thw = np.array(grid_thw).reshape([-1, 3])
                inputs["grid_thw"] = grid_thw

            grid_thw = inputs["grid_thw"]
            img_boundaries_idx = request.multimodal_img_boundaries[0]
            img_num_per_boundary = request.multimodal_img_boundaries[1]
            ori_prompt_len = img_boundaries_idx[-1].item()
            pre_end_idx = request.num_computed_tokens
            new_end_idx = pre_end_idx + num_new_tokens
            if new_end_idx < ori_prompt_len and input_ids[new_end_idx - 1] == image_patch_id:
                boundary_idx = np.searchsorted(img_boundaries_idx, new_end_idx, side="left").item()
                if boundary_idx == len(img_boundaries_idx):
                    new_end_idx = ori_prompt_len
                else:
                    new_end_idx = img_boundaries_idx[boundary_idx].item()
            elif new_end_idx >= ori_prompt_len and paddle.sum(input_ids[pre_end_idx:new_end_idx] == image_patch_id):
                new_end_idx = ori_prompt_len
            num_new_tokens = new_end_idx - pre_end_idx

            image_mask = input_ids[pre_end_idx:new_end_idx] == image_patch_id
            request.with_image = image_mask.any()
            if request.with_image:
                pre_boundary_idx = np.searchsorted(img_boundaries_idx, pre_end_idx, side="left").item()
                if pre_boundary_idx == len(img_boundaries_idx):
                    request.num_image_start = img_num_per_boundary[-1]
                else:
                    pre_boundary_idx = (
                        pre_boundary_idx
                        if pre_end_idx == img_boundaries_idx[pre_boundary_idx]
                        else pre_boundary_idx - 1
                    )
                    request.num_image_start = img_num_per_boundary[pre_boundary_idx]

                new_boundary_idx = np.searchsorted(img_boundaries_idx, new_end_idx, side="left").item()
                if new_boundary_idx == len(img_boundaries_idx):
                    request.num_image_end = img_num_per_boundary[-1]
                else:
                    new_boundary_idx = (
                        new_boundary_idx
                        if new_end_idx == img_boundaries_idx[new_boundary_idx]
                        else new_boundary_idx - 1
                    )
                    request.num_image_end = img_num_per_boundary[new_boundary_idx]

                request.image_type_ids_start = np.sum(grid_thw[: request.num_image_start, 0])
                request.image_type_ids_end = np.sum(grid_thw[: request.num_image_end, 0])
                request.image_start = np.sum(np.prod(grid_thw[: request.num_image_start], axis=1))
                request.image_end = np.sum(np.prod(grid_thw[: request.num_image_end], axis=1))

                if self.encoder_cache:
                    cur_mm_hashes = inputs["mm_hashes"][request.num_image_start : request.num_image_end]
                    cur_mm_positions = inputs["mm_positions"][request.num_image_start : request.num_image_end]
                    request.evict_mm_hashes = self.encoder_cache.apply_cache(cur_mm_hashes, cur_mm_positions)

        # Compatible with scenarios without images and videos.
        return num_new_tokens

    def exist_mm_prefill(self, scheduled_reqs):
        for request in scheduled_reqs:
            if request.task_type == RequestType.PREFILL and self._is_mm_request(request):
                return True
        return False

    def exist_prefill(self, scheduled_reqs):
        for request in scheduled_reqs:
            if request.task_type == RequestType.PREFILL:
                return True
        return False

    def cache_output_tokens(self, request):
        if self.config.cache_config.enable_prefix_caching and self.config.cache_config.enable_output_caching:
            with self.lock:
                if request.num_computed_tokens >= request.need_prefill_tokens:  # request is decoding
                    self.cache_manager.cache_output_blocks(request, self.config.cache_config.block_size)

    def schedule(self):
        """
        Try to pull a batch of requests from the waiting queue and schedule them.
        """

        def get_enough_request(request, scheduled_reqs):
            return (
                ErnieArchitectures.is_ernie5_arch(self.config.model_config.architectures)
                and self._is_mm_request(request)
                and self.exist_mm_prefill(scheduled_reqs)
            )

        with self.lock:
            scheduled_reqs: list[Request] = []
            preempted_reqs: list[Request] = []
            error_reqs: list[tuple[str, str]] = []
            token_budget = self.config.scheduler_config.max_num_batched_tokens
            # SGLang-aligned: chunked_prefill_size is per-request limit, token_budget is batch limit
            chunked_prefill_size = self.config.scheduler_config.chunked_prefill_size

            # First, schedule the RUNNING requests.
            req_index = 0
            num_decoding_req_nums = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                need_block_num = self.need_block_num_signal.value[request.idx]
                if need_block_num != 0:
                    self.need_block_num_map[request.request_id] = SignalConsumer(need_block_num, 1)
                    self.need_block_num_signal.value[request.idx] = 0

                # NOTE: The decode scheduling logic should be OUTSIDE of `if need_block_num != 0:`
                # The need_block_num signal is only for handling worker-side block requests,
                # but decode scheduling should always happen regardless of this signal
                if request.num_computed_tokens >= request.need_prefill_tokens:  # to be decoding
                    if (
                        self.config.scheduler_config.splitwise_role == "prefill"
                    ):  # do not need to schedule for decoding
                        req_index += 1
                        continue
                    if request.num_total_tokens > request.need_prefill_tokens:  # has generated tokens
                        request.num_computed_tokens = request.num_total_tokens - 1

                    # SGLang-aligned: dynamically calculate how many blocks are needed for next decode
                    # Similar to SGLang's new_page_count_next_decode:
                    # If (num_total_tokens - 1) % block_size == 0, need 1 more block
                    block_size = self.config.cache_config.block_size
                    num_new_blocks_needed = 1 if (request.num_total_tokens - 1) % block_size == 0 else 0

                    # SGLang-aligned: schedule decode task without threshold check
                    # The pre-allocation is decoupled from scheduling
                    if num_new_blocks_needed > 0:
                        # Need to allocate new blocks, check if we can allocate
                        if self.cache_manager.can_allocate_gpu_blocks(num_new_blocks_needed):
                            llm_logger.debug(
                                f"schedule decoding task: {request} request.num_total_tokens {request.num_total_tokens} request.num_computed_tokens {request.num_computed_tokens}"
                            )
                            request.block_tables.extend(
                                self.cache_manager.allocate_gpu_blocks(
                                    num_new_blocks_needed, request.request_id
                                )
                            )
                            # Prepare decoding task
                            scheduled_reqs.append(self._prepare_decode_task(request))
                        else:
                            # Not enough blocks, trigger preemption
                            # SGLang-aligned: first try to evict decode KV cache
                            self._evict_decode_kv_cache(len(self.running))

                            # Check again after eviction
                            if self.cache_manager.can_allocate_gpu_blocks(num_new_blocks_needed):
                                request.block_tables.extend(
                                    self.cache_manager.allocate_gpu_blocks(
                                        num_new_blocks_needed, request.request_id
                                    )
                                )
                                scheduled_reqs.append(self._prepare_decode_task(request))
                            else:
                                # Cannot allocate even after preemption, use SGLang-aligned behavior
                                # Try to preempt other requests
                                can_schedule = self._trigger_preempt(
                                    request, num_new_blocks_needed, preempted_reqs, scheduled_reqs
                                )
                                if not can_schedule:
                                    # Cannot preempt (e.g., only 1 decode request left),
                                    # skip this request and continue to avoid system hang
                                    llm_logger.warning(
                                        f"Cannot allocate {num_new_blocks_needed} blocks "
                                        f"for decode request {request.request_id} (idx={request.idx}) "
                                        f"even after preemption attempt. Request will wait for more resources."
                                    )
                                    # Do NOT schedule this request - it will wait in the queue
                                    # But still consume token_budget to avoid infinite loop
                                    req_index += 1
                                    token_budget -= 1
                                    continue

                                # Allocation for next decoding blocks after preemption
                                request.block_tables.extend(
                                    self.cache_manager.allocate_gpu_blocks(
                                        num_new_blocks_needed, request.request_id
                                    )
                                )
                                # Prepare decoding task
                                scheduled_reqs.append(self._prepare_decode_task(request))

                    # No new blocks needed (num_new_blocks_needed == 0), but still schedule decode task
                    # SGLang-aligned: always schedule decode for each iteration
                    else:
                        scheduled_reqs.append(self._prepare_decode_task(request))

                    num_decoding_req_nums += 1
                    token_budget -= 1
                    if (
                        request.use_extend_tables
                        and request.request_id not in self.using_extend_tables_req_id
                        and self.need_block_num_map[request.request_id].watch() > 0
                    ):

                        def _allocate_decode_and_extend():
                            allocate_block_num = self.need_block_num_map[request.request_id].consume()
                            # Prepare decoding task
                            request.block_tables.extend(
                                self.cache_manager.allocate_gpu_blocks(allocate_block_num, request.request_id)
                            )
                            scheduled_reqs.append(self._prepare_decode_task(request))

                            # Prepare extend task
                            reuse_block_num = request.num_total_tokens // self.config.cache_config.block_size
                            llm_logger.info(
                                f"req {request.request_id} at batch id {request.idx} with reuse_block_num {reuse_block_num} is going to enable extend tables,"
                                f"need_block_num {allocate_block_num}"
                            )
                            self.using_extend_tables_req_id.add(request.request_id)
                            self.reuse_block_num_map[request.request_id] = reuse_block_num

                            request.extend_block_tables = request.block_tables[:reuse_block_num]  # copy prompt cache
                            request.extend_block_tables.extend(
                                self.cache_manager.allocate_gpu_blocks(allocate_block_num, request.request_id)
                            )
                            scheduled_reqs.append(
                                ScheduledExtendBlocksTask(
                                    idx=request.idx,
                                    request_id=request.request_id,
                                    extend_block_tables=request.extend_block_tables,
                                )
                            )
                            llm_logger.debug(f"extend blocks is {request.extend_block_tables}")

                        if self.cache_manager.can_allocate_gpu_blocks(
                            2 * self.need_block_num_map[request.request_id].watch()
                        ):
                            _allocate_decode_and_extend()
                        else:
                            llm_logger.info(
                                f"{request.idx} using extend block need {2 * self.need_block_num_map[request.request_id].watch()} blocks but got not enough blocks, ready to preempt"
                            )
                            can_schedule = self._trigger_preempt(
                                request,
                                2 * self.need_block_num_map[request.request_id].watch(),
                                preempted_reqs,
                                scheduled_reqs,
                            )

                            if can_schedule:
                                _allocate_decode_and_extend()
                            else:
                                break
                else:  # need to prefill
                    llm_logger.debug(
                        f"scheduler prefill task in running queue: {request.request_id}, "
                        f"request.need_prefill_tokens {request.need_prefill_tokens},"
                        f"request.num_computed_tokens {request.num_computed_tokens}"
                    )
                    if (
                        current_platform.is_intel_hpu()
                        and request.need_prefill_tokens - request.num_computed_tokens
                        >= self.config.cache_config.block_size
                        and token_budget < self.config.cache_config.block_size
                    ):
                        req_index += 1
                        continue
                    if get_enough_request(request, scheduled_reqs):
                        req_index += 1
                        continue
                    num_new_tokens = self._get_num_new_tokens(request, chunked_prefill_size, token_budget)
                    num_new_block = self.get_new_block_nums(request, num_new_tokens)
                    # Allocate blocks to prefill
                    if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                        request.block_tables.extend(
                            self.cache_manager.allocate_gpu_blocks(num_new_block, request.request_id)
                        )
                        # Prepare prefill task
                        scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                    else:  # Not enough blocks to allocate, trigger preemption
                        if self.config.scheduler_config.enable_priority_scheduling:
                            can_schedule = self._trigger_preempt(request, num_new_block, preempted_reqs, scheduled_reqs)
                        else:
                            can_schedule = False                        
                        if not can_schedule:
                            break
                        request.block_tables.extend(
                            self.cache_manager.allocate_gpu_blocks(num_new_block, request.request_id)
                        )
                        # Prepare prefill task
                        scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                    token_budget -= num_new_tokens
                    request.num_computed_tokens += num_new_tokens
                    if self.config.cache_config.enable_prefix_caching:
                        self.cache_manager.update_cache_blocks(
                            request, self.config.cache_config.block_size, request.num_computed_tokens
                        )
                req_index += 1

            # Second, schedule the WAITING requests.
            if not preempted_reqs:
                skip_requests: list[Request] = []
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_seqs:
                        break

                    request = self.waiting[0]
                    if get_enough_request(request, scheduled_reqs):
                        break
                    if request.status == RequestStatus.WAITING:
                        result = self.waiting_async_process(request)
                        if result is None:
                            error_reqs.append((request.request_id, request.error_message))
                            self.waiting.popleft()
                            continue
                        elif result is True:
                            # skip current request, try next request
                            skip_requests.append(request)
                            self.waiting.popleft()
                            continue

                        self._update_mm_hashes(request)
                        # Enable prefix caching
                        if self.config.cache_config.enable_prefix_caching:
                            if (
                                self.cache_manager.num_cpu_blocks > 0
                                or self.config.cache_config.kvcache_storage_backend
                            ):
                                if not self.cache_manager.can_allocate_gpu_blocks(
                                    (request.need_prefill_tokens + self.config.cache_config.block_size - 1)
                                    // self.config.cache_config.block_size
                                ):  # to prevent block allocation for matching in hierarchical cache and cause dead lock
                                    break
                            success = self.get_prefix_cached_blocks(request)
                            if not success:
                                self._free_blocks(request)
                                break

                        if (
                            current_platform.is_intel_hpu()
                            and request.need_prefill_tokens - request.num_computed_tokens
                            >= self.config.cache_config.block_size
                            and token_budget < self.config.cache_config.block_size
                        ):
                            continue
                        # Allocate blocks for the tokens that does not hit cache
                        num_new_tokens = self._get_num_new_tokens(request, chunked_prefill_size, token_budget)
                        num_new_block = self.get_new_block_nums(request, num_new_tokens)
                        can_schedule_block_num_threshold = self._get_can_schedule_prefill_threshold_block(
                            request, num_new_block
                        )
                        # Allocate blocks to prefill
                        if self.cache_manager.can_allocate_gpu_blocks(can_schedule_block_num_threshold):
                            if not request.get("skip_allocate", False):
                                extra_gpu_block_ids = self.cache_manager.allocate_gpu_blocks(
                                    num_new_block, request.request_id
                                )
                                request.block_tables.extend(extra_gpu_block_ids)
                            self.waiting.popleft()
                            self.running.append(request)
                            scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            if self.config.cache_config.enable_prefix_caching:
                                self.cache_manager.update_cache_blocks(
                                    request, self.config.cache_config.block_size, request.num_computed_tokens
                                )
                            request.status = RequestStatus.RUNNING
                            if self.config.scheduler_config.splitwise_role == "mixed":
                                allocated_position = self.get_available_position()
                                request.idx = allocated_position
                                self.tasks_list[allocated_position] = request
                                self.stop_flags[allocated_position] = False
                                self.req_dict[request.request_id] = allocated_position
                                llm_logger.debug(f"req_id:{request.request_id} allocate pos end")
                        else:
                            if self.config.cache_config.enable_prefix_caching:
                                self._free_blocks(request)
                            break
                    elif request.status == RequestStatus.PREEMPTED:
                        request.need_prefill_tokens = (
                            request.num_total_tokens
                        )  # Before preempted task rescheduled, preempted task has been sent to engine, no more tokens are output, here num_total_tokens should be static and correct
                        if self.config.cache_config.enable_prefix_caching:
                            if (
                                self.cache_manager.num_cpu_blocks > 0
                                or self.config.cache_config.kvcache_storage_backend
                            ):
                                if not self.cache_manager.can_allocate_gpu_blocks(
                                    (request.need_prefill_tokens + self.config.cache_config.block_size - 1)
                                    // self.config.cache_config.block_size
                                ):  # to prevent block allocation for matching in hierarchical cache and cause dead lock
                                    break
                            success = self.get_prefix_cached_blocks(request)
                            if not success:
                                self._free_blocks(request)
                                break

                        # Allocate blocks for the tokens that does not hit cache
                        num_new_tokens = self._get_num_new_tokens(request, chunked_prefill_size, token_budget)
                        num_new_block = self.get_new_block_nums(request, num_new_tokens)
                        can_schedule_block_num_threshold = self._get_can_schedule_prefill_threshold_block(
                            request, num_new_block
                        )
                        # Allocate blocks to prefill
                        if self.cache_manager.can_allocate_gpu_blocks(can_schedule_block_num_threshold):
                            if not request.get("skip_allocate", False):
                                extra_gpu_block_ids = self.cache_manager.allocate_gpu_blocks(
                                    num_new_block, request.request_id
                                )
                                request.block_tables.extend(extra_gpu_block_ids)
                            self.waiting.popleft()
                            self.running.append(request)
                            scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            if self.config.cache_config.enable_prefix_caching:
                                self.cache_manager.update_cache_blocks(
                                    request, self.config.cache_config.block_size, request.num_computed_tokens
                                )
                            request.status = RequestStatus.RUNNING
                        else:
                            if self.config.cache_config.enable_prefix_caching:
                                self._free_blocks(request)
                            break
                    else:
                        llm_logger.info(f"Unknown request status type:{request.status}, req_id:{request.request_id}")

                for req in skip_requests:
                    # move waiting request to end of the deque
                    self.waiting.append(req)

            if scheduled_reqs:
                llm_logger.debug(f"schedued_reqs: {scheduled_reqs}")

                # New mechanism: decay new_token_ratio only when there are decode requests
                has_decode_reqs = any(getattr(r, "task_type", None) == RequestType.DECODE for r in scheduled_reqs)
                if has_decode_reqs:
                    self.current_new_token_ratio = max(
                        self.current_new_token_ratio - self.new_token_ratio_decay,
                        self.min_new_token_ratio,
                    )
                    llm_logger.debug(f"NewTokenRatio decayed to {self.current_new_token_ratio:.4f}")

            if (
                hasattr(self, "scheduler_metrics_logger")
                and self.scheduler_metrics_logger is not None
                and envs.FD_CONSOLE_SCHEDULER_METRICS
            ):
                total_blocks = self.total_block_number()
                free_blocks = self.available_block_num()
                used_blocks = max(total_blocks - free_blocks, 0)
                tokens_used = used_blocks * self.config.cache_config.block_size
                token_usage = used_blocks / total_blocks if total_blocks > 0 else 0.0
                running_cnt = len(self.running)
                queue_cnt = len(self.waiting)

                prefill_reqs = [
                    r for r in scheduled_reqs if isinstance(r, Request) and r.task_type == RequestType.PREFILL
                ]
                has_decode = any(getattr(r, "task_type", None) == RequestType.DECODE for r in scheduled_reqs)

                self.scheduler_metrics_logger.log_prefill_batch(
                    prefill_reqs=prefill_reqs,
                    running_cnt=running_cnt,
                    queue_cnt=queue_cnt,
                    tokens_used=tokens_used,
                    token_usage=token_usage,
                )
                if has_decode:
                    has_prefill = len(prefill_reqs) > 0
                    graph_opt_cfg = self.config.graph_opt_config
                    use_cudagraph_cfg = bool(getattr(graph_opt_cfg, "use_cudagraph", False))
                    graph_opt_level = int(getattr(graph_opt_cfg, "graph_opt_level", 0) or 0)
                    full_cuda_graph = bool(getattr(graph_opt_cfg, "full_cuda_graph", True))
                    cudagraph_only_prefill = bool(getattr(graph_opt_cfg, "cudagraph_only_prefill", False))
                    use_decode_cudagraph = (
                        has_decode
                        and use_cudagraph_cfg
                        and (
                            # Reference PR https://github.com/PaddlePaddle/FastDeploy/pull/6196
                            # Static split graph mode: Prefill+Mixed and Decode can use CUDAGraph.
                            (graph_opt_level > 0 and not full_cuda_graph)
                            # Dynamic / static-full modes: decode-only can use CUDAGraph.
                            or (not has_prefill and not cudagraph_only_prefill)
                        )
                    )
                    self.scheduler_metrics_logger.log_decode_batch(
                        running_cnt=running_cnt,
                        queue_cnt=queue_cnt,
                        tokens_used=tokens_used,
                        token_usage=token_usage,
                        use_cudagraph=use_decode_cudagraph,
                    )

            # SGLang-aligned: reset new_token_ratio when completely idle
            if not scheduled_reqs:
                self.reset_new_token_ratio_on_idle()

            self.update_metrics()

            return scheduled_reqs, error_reqs

    def waiting_async_process(self, request: Request) -> None:
        """
        Check if async preprocessing is complete for a request.
        Args:
            request: The request to check
        Returns:
            None: If an error occurred during preprocessing
            True: If preprocessing is still in progress (request should be skipped)
            False: If preprocessing is complete (request can be scheduled)
        """
        for future in request.async_process_futures:
            if future.done():
                if request.get("error_message") is not None:
                    return None
            else:
                return True
        request.async_process_futures = []
        return False

    def apply_async_preprocess(self, request: Request) -> None:
        request.async_process_futures.append(self.async_preprocess_pool.submit(self._download_features, request))

    def _has_features_info(self, task):
        inputs = task.multimodal_inputs
        if inputs is None or len(inputs) == 0:
            return False

        if (
            (inputs.get("video_feature_urls") is not None and len(inputs["video_feature_urls"]) > 0)
            or (inputs.get("image_feature_urls") is not None and len(inputs["image_feature_urls"]) > 0)
            or (inputs.get("audio_feature_urls") is not None and len(inputs["audio_feature_urls"]) > 0)
        ):
            return True
        return False

    def _download_features(self, request: Request) -> None:
        """
        download multimodal features from bos
        Note:
            1. this function will be add features for request.multimodal_inputs
            2. this function maybe update request.error_message and request.error_code
        Args:
            request (Request): request object
        """

        def download_bos_features(bos_client, features_urls):
            result_list = []
            for status, feature in download_from_bos(self.bos_client, features_urls, retry=1):
                if status:
                    start_download_time = time.time()
                    if isinstance(feature, np.ndarray):
                        feature_info = f"type=np.ndarray, shape={feature.shape}, dtype={feature.dtype}"
                    elif isinstance(feature, list):
                        feature_info = f"type=list, len={len(feature)}"
                    else:
                        feature_info = f"type={type(feature).__name__}"

                    elapsed_time = round((time.time() - start_download_time) * 1000, 2)
                    llm_logger.info(
                        f"request {request.request_id} async download feature success: {feature_info}, "
                        f"elapsed time: {elapsed_time} ms"
                    )

                    result_list.append(feature)
                else:
                    error_msg = f"request {request.request_id} download features error: {feature}"
                    llm_logger.error(error_msg)
                    return error_msg
            return result_list

        if not self._has_features_info(request):
            return None

        if self.bos_client is None:
            try:
                self.bos_client = init_bos_client()
            except Exception as e:
                error_msg = f"request {request.request_id} init bos client error: {str(e)}"
                llm_logger.error(error_msg)
                request.error_message = error_msg
                request.error_code = 540
                return None

        inputs = request.multimodal_inputs
        if inputs.get("video_feature_urls") is not None and len(inputs["video_feature_urls"]) > 0:
            result = download_bos_features(self.bos_client, inputs["video_feature_urls"])
            if isinstance(result, str):  # download error
                request.error_message = result
                request.error_code = 530
                return None
            inputs["video_features"] = result
        if inputs.get("image_feature_urls") is not None and len(inputs["image_feature_urls"]) > 0:
            result = download_bos_features(self.bos_client, inputs["image_feature_urls"])
            if isinstance(result, str):  # download error
                request.error_message = result
                request.error_code = 530
                return None
            inputs["image_features"] = result
        if inputs.get("audio_feature_urls") is not None and len(inputs["audio_feature_urls"]) > 0:
            result = download_bos_features(self.bos_client, inputs["audio_feature_urls"])
            if isinstance(result, str):  # download error
                request.error_message = result
                request.error_code = 530
                return None
            inputs["audio_features"] = result

    def get_available_position(self) -> int:
        position = 0
        while position < self.max_num_seqs:
            if self.stop_flags[position] is True:
                return position
            position += 1
        raise RuntimeError("No available position is available for new request")

    def get_real_bsz(self) -> int:
        for i in range(self.max_num_seqs - 1, -1, -1):
            if not self.stop_flags[i]:
                self.real_bsz = i + 1
                break
        return self.real_bsz

    def get_prefix_cached_blocks(self, request: Request):
        """
        Match and fetch cache for a task.
        """
        try:
            (common_block_ids, matched_token_num, metrics) = self.cache_manager.request_match_blocks(
                request, self.config.cache_config.block_size
            )

            matched_block_num = len(common_block_ids)
            no_cache_block_num = self.cache_manager.get_required_block_num(
                request.need_prefill_tokens - matched_token_num,
                self.config.cache_config.block_size,
            )

            request.cache_info = [matched_block_num, no_cache_block_num]
            request.block_tables = common_block_ids
            request.skip_allocate = False
            request.num_cached_tokens = matched_token_num
            if self.config.cache_config.disable_chunked_mm_input:
                if matched_token_num == request.need_prefill_tokens:
                    matched_token_num = matched_token_num - self.config.cache_config.block_size
                    request.skip_allocate = True
                request.num_computed_tokens = self.revert_chunked_mm_input(
                    request.multimodal_inputs, matched_token_num
                )
            else:
                if matched_token_num == request.need_prefill_tokens:
                    request.num_computed_tokens = matched_token_num - self.config.cache_config.block_size
                    request.skip_allocate = True
                else:
                    request.num_computed_tokens = matched_token_num

            if request.num_cached_tokens != request.num_computed_tokens:
                revert_tokens_num = request.num_cached_tokens - request.num_computed_tokens
                llm_logger.info(
                    f"request {request.request_id} num_cached_tokens: {request.num_cached_tokens}, revert_tokens_num: {revert_tokens_num}"
                )

                revert_block_idx = len(common_block_ids) - revert_tokens_num // self.config.cache_config.block_size - 1
                for block_idx in range(len(common_block_ids) - 1, revert_block_idx, -1):
                    if common_block_ids[block_idx] in metrics["match_gpu_block_ids"]:
                        metrics["gpu_match_token_num"] -= self.config.cache_config.block_size
                    elif common_block_ids[block_idx] in metrics["gpu_recv_block_ids"]:
                        metrics["cpu_match_token_num"] -= self.config.cache_config.block_size
                    elif common_block_ids[block_idx] in metrics["match_storage_block_ids"]:
                        metrics["storage_match_token_num"] -= self.config.cache_config.block_size

            request.metrics.gpu_cache_token_num = metrics["gpu_match_token_num"]
            request.metrics.cpu_cache_token_num = metrics["cpu_match_token_num"]
            request.metrics.storage_cache_token_num = metrics["storage_match_token_num"]
            request.metrics.cpu_cache_prepare_time = metrics["cpu_cache_prepare_time"]
            request.metrics.storage_cache_prepare_time = metrics["storage_cache_prepare_time"]

            # Report the number of cached tokens to Prometheus metrics
            main_process_metrics.prefix_cache_token_num.inc(request.num_computed_tokens)
            main_process_metrics.prefix_gpu_cache_token_num.inc(request.metrics.gpu_cache_token_num)
            main_process_metrics.prefix_cpu_cache_token_num.inc(request.metrics.cpu_cache_token_num)

            return True
        except Exception as e:
            llm_logger.error(f"prefix match blocks error: {e}, {str(traceback.format_exc())} waiting reschedule...")
            return False

    def add_request(self, request: Request) -> None:
        with self.lock:
            self.apply_async_preprocess(request)
            llm_logger.debug(f"self.waiting append request:{request.request_id},req.type:{request.status}")
            self.waiting.append(request)
            self.requests[request.request_id] = request

    def pre_recycle_resource(self, request_id: str):
        """
        Recycle resource in P or D before finished due to unexpected error.
        """
        with self.lock:
            if request_id not in self.requests:
                return
            req = self.requests[request_id]
            self.tasks_list[req.idx] = None
            self.stop_flags[req.idx] = True
            self._free_blocks(req)
            del self.requests[request_id]
            if request_id in self.req_dict:
                del self.req_dict[request_id]

    def add_request_in_p(self, requests: list[Request]):
        with self.lock:
            for request in requests:
                self.running.append(request)

    def preallocate_resource_in_p(self, request: Request):
        """
        In P/D aggregated deployment, preallocate resource for P.
        If can allocate, allocate resources and return True
        If can not, return False
        """
        assert self.config.scheduler_config.splitwise_role == "prefill", "Only P instance can call this method"
        with self.lock:
            if self.available_batch() == 0:
                return False
            request.need_prefill_tokens = len(request.prompt_token_ids)
            need_prealloc_prefill_blocks = (
                request.need_prefill_tokens + self.config.cache_config.block_size - 1
            ) // self.config.cache_config.block_size + self.config.cache_config.enc_dec_block_num  # consider for mtp, plus enc_dec_block_num
            if self.config.cache_config.enable_prefix_caching:
                # Enable prefix caching
                if self.cache_manager.num_cpu_blocks > 0:
                    if not self.cache_manager.can_allocate_gpu_blocks(
                        need_prealloc_prefill_blocks
                    ):  # to prevent block allocation for matching in hierarchical cache and cause dead lock
                        return False
                success = self.get_prefix_cached_blocks(request)
                if not success:
                    self._free_blocks(request)
                    return False

                need_extra_prefill_blocks = need_prealloc_prefill_blocks - request.cache_info[0]
                if self.cache_manager.can_allocate_gpu_blocks(need_extra_prefill_blocks):
                    extra_gpu_block_ids = self.cache_manager.allocate_gpu_blocks(
                        need_extra_prefill_blocks, request.request_id
                    )
                    request.block_tables.extend(extra_gpu_block_ids)
                    allocated_position = self.get_available_position()
                    request.idx = allocated_position
                    self.tasks_list[request.idx] = request
                    self.stop_flags[request.idx] = False
                    self.requests[request.request_id] = request
                    self.req_dict[request.request_id] = allocated_position
                    return True
                else:
                    self._free_blocks(request)
                    return False

            else:
                if self.cache_manager.can_allocate_gpu_blocks(need_prealloc_prefill_blocks):
                    request.block_tables.extend(
                        self.cache_manager.allocate_gpu_blocks(need_prealloc_prefill_blocks, request.request_id)
                    )
                    request.num_computed_tokens = 0
                    allocated_position = self.get_available_position()
                    request.idx = allocated_position
                    self.tasks_list[request.idx] = request
                    self.stop_flags[request.idx] = False
                    self.requests[request.request_id] = request
                    self.req_dict[request.request_id] = allocated_position
                    return True

                return False

    def preallocate_resource_in_d(self, request: Request):
        """
        In P/D aggregated deployment, D should preallocate resource for P.
        If can allocate, allocate resources and return True
        If can not, return False
        """
        assert self.config.scheduler_config.splitwise_role == "decode", "Only D instance can call this method"
        if request.reasoning_max_tokens is not None:
            request.reasoning_max_tokens -= 1
        request.need_prefill_tokens = len(request.prompt_token_ids)
        need_prealloc_prefill_blocks = (
            request.need_prefill_tokens + self.config.cache_config.block_size - 1
        ) // self.config.cache_config.block_size + self.config.cache_config.enc_dec_block_num

        with self.lock:
            if len(self.waiting) > 0:
                return False
            if self.available_batch() == 0:
                return False
            if not self.cache_manager.can_allocate_gpu_blocks(need_prealloc_prefill_blocks):
                return False

            request.block_tables = self.cache_manager.allocate_gpu_blocks(
                need_prealloc_prefill_blocks, request.request_id
            )
            request.num_computed_tokens = request.need_prefill_tokens
            request.disaggregate_info["block_tables"] = request.block_tables
            allocated_position = self.get_available_position()
            request.idx = allocated_position
            self.tasks_list[request.idx] = request
            self.stop_flags[request.idx] = False
            self.requests[request.request_id] = request
            self.req_dict[request.request_id] = allocated_position
        return True

    def has_resource_for_prefilled_req(self, request_id: str):
        """
        Check whether there are enough slot and gpu resource for the prefilled request,
        of which the cache is saved in cpu buffer.
        """
        with self.lock:
            assert self.config.scheduler_config.splitwise_role == "decode", "Only D instance can call this method"
            assert request_id in self.preallocated_reqs, "request_id must be in preallocate"
            need_blocks_num = len(self.preallocated_reqs[request_id].disaggregate_info["block_tables"])
            return self.available_batch() > 0 and self.cache_manager.can_allocate_gpu_blocks(need_blocks_num)

    def add_prefilled_request(self, request_output: RequestOutput):
        """
        In P/D aggregated deployment, D should continue to decode after receiving first token and cache from P.
        NOTE: GPU resources should be checked in advance to ensure they are sufficient for the prefilled request.
        """
        with self.lock:
            assert self.config.scheduler_config.splitwise_role == "decode", "Only D instance can call this method"
            if request_output.request_id not in self.requests:
                llm_logger.error(f"Request {request_output.request_id} not found in requests")
                return
            request = self.requests[request_output.request_id]

            # update request and insert to running
            request.output_token_ids.append(request_output.outputs.token_ids[0])
            request.num_cached_tokens = request_output.num_cached_tokens
            if (
                self.config.speculative_config.method in ["mtp"]
                and self.config.scheduler_config.splitwise_role == "decode"
            ):
                request.draft_token_ids = copy.deepcopy(request_output.outputs.draft_token_ids)
            request.need_prefill_tokens = len(request.prompt_token_ids) + 1

            request_output.metrics.decode_recv_req_time = request.metrics.decode_recv_req_time
            request_output.metrics.decode_preallocate_req_time = request.metrics.decode_preallocate_req_time
            request.metrics = request_output.metrics
            self.running.append(request)

    def _free_blocks(self, request: Request):
        if self.config.cache_config.enable_prefix_caching:
            self.cache_manager.release_block_ids(request)
            self.cache_manager.recycle_gpu_blocks(
                request.block_tables[request.num_cached_blocks :], request.request_id
            )
        else:
            self.cache_manager.recycle_gpu_blocks(request.block_tables, request.request_id)
        request.block_tables = []

        if request.request_id in self.using_extend_tables_req_id:
            reuse_block_num = self.reuse_block_num_map[request.request_id]

            self.using_extend_tables_req_id.remove(request.request_id)
            self.cache_manager.recycle_gpu_blocks(request.extend_block_tables[reuse_block_num:], request.request_id)
            llm_logger.info(
                f"req {request.request_id} recycle extend blocks {request.extend_block_tables[reuse_block_num:]}"
            )
            request.extend_block_tables = []
            del self.reuse_block_num_map[request.request_id]
            del self.need_block_num_map[request.request_id]

    def finish_requests_async(self, request_ids: Union[str, Iterable[str]]):
        return self.finish_execution_pool.submit(self.finish_requests, request_ids)

    def finish_requests(self, request_ids: Union[str, Iterable[str]]):
        llm_logger.info(f"recycle resources for requests: {request_ids}")
        try:
            if isinstance(request_ids, str):
                request_ids = (request_ids,)
            else:
                request_ids = set(request_ids)

            need_postprocess_reqs = []
            with self.lock:
                for req_id in request_ids:
                    request = self.requests.get(req_id)
                    if request is None:
                        llm_logger.error(f"invalid request id: {req_id} self.requests: {self.requests}")
                        continue
                    if request in self.waiting:
                        llm_logger.error(f"request {request.request_id} scheduled into waiting list, after finished")
                        continue
                    if request in self.running:
                        llm_logger.info(f"finish running request: {req_id}")
                        self.running.remove(request)
                        request.status = RequestStatus.FINISHED
                        need_postprocess_reqs.append(request)
                    if request.request_id in self.to_be_rescheduled_request_id_set:
                        # finished after preempted, blocks have been recycled.
                        llm_logger.info(f"finish preempeted request: {req_id}")
                        self.to_be_rescheduled_request_id_set.remove(request.request_id)

                    self.tasks_list[request.idx] = None
                    self.stop_flags[request.idx] = True
                    del self.requests[req_id]
                    if req_id in self.req_dict:
                        del self.req_dict[req_id]

            # Do not block the main thread here
            for req in need_postprocess_reqs:
                self.cache_manager.write_cache_to_storage(req)

            with self.lock:
                for req in need_postprocess_reqs:
                    try:
                        self._free_blocks(req)
                        llm_logger.debug(f"req_id:{req.request_id} free pos:{req.idx}")
                    except Exception as e:
                        llm_logger.warning(f"release block failed {req.request_id}: {e}")
        except Exception as e:
            llm_logger.error(f"finish_request err: {e}, {str(traceback.format_exc())}")
        finally:
            self.update_metrics(verbose=True)

    def clear_data(self):
        self.waiting: deque[Request] = deque()
        self.to_be_rescheduled_request_id_set = set()
        self.update_metrics(verbose=True)

    def update_metrics(self, verbose=False):
        # Update metrics
        num_tasks = sum([1 if task else 0 for task in self.tasks_list])
        blocks_used_by_tasks = set()
        for task in self.tasks_list:
            if task is not None:
                blocks_used_by_tasks.update(task.block_tables)
        main_process_metrics.available_gpu_block_num.set(self.total_block_number() - len(blocks_used_by_tasks))
        main_process_metrics.batch_size.set(self.max_num_seqs - self.available_batch())
        main_process_metrics.gpu_cache_usage_perc.set(self.get_gpu_cache_usage_perc())
        main_process_metrics.num_requests_running.set(len(self.running))
        main_process_metrics.num_requests_waiting.set(num_tasks - len(self.running))
        if verbose:
            llm_logger.info(f"update metrics: running={len(self.running)}, waiting={num_tasks - len(self.running)}")

    def log_status(self):
        llm_logger.info(
            f"ResourceManagerV1( "
            f"waiting={len(self.waiting)}, "
            f"running={len(self.running)}, "
            f"preempted={len(self.to_be_rescheduled_request_id_set)}, "
            f"tasks_list={self.tasks_list}, "
            f"stop_flags={self.stop_flags}, "
            f"req_dict={self.req_dict}, "
            f"requests={self.requests}, "
            f")"
        )
