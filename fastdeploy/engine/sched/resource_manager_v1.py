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

        # need block nums
        need_block_num_data = np.zeros([max_num_seqs], dtype=np.int32)
        self.need_block_num_signal = IPCSignal(
            name="need_block_num_signal",
            array=need_block_num_data,
            dtype=np.int32,
            suffix=local_data_parallel_id,
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

    def reschedule_preempt_task(self, request_id):
        with self.lock:
            if request_id in self.to_be_rescheduled_request_id_set and request_id in self.requests:
                request = self.requests[request_id]
                self.waiting.appendleft(request)
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

    def _trigger_preempt(self, request, num_new_blocks, preempted_reqs, scheduled_reqs):
        """
        If the request cannot be scheduled, preempt the running request one by one until it can be scheduled. Last in, first out.
        """
        can_schedule = False
        while self._can_preempt():
            if not self.cache_manager.can_allocate_gpu_blocks(num_new_blocks):
                preempted_req = self.running.pop()
                if preempted_req.use_extend_tables:
                    self.running.insert(0, preempted_req)
                    continue
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
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
                    preempted_req.cached_block_num = 0
                    self.to_be_rescheduled_request_id_set.add(preempted_req.request_id)
                    llm_logger.info(f"Preemption is triggered! Preempted request id: {preempted_req.request_id}")
                preempted_reqs.append(preempted_req)
                scheduled_reqs.append(self._prepare_preempt_task(preempted_req))

                llm_logger.debug(
                    f"preempt {preempted_req.request_id} in idx {preempted_req.idx} with generated ids {preempted_req.output_token_ids}"
                )
                llm_logger.debug(self.info())
                self._info_each_block()

                if preempted_req == request:
                    # No more request to preempt.
                    can_schedule = False
                    break
            else:
                # The request can be scheduled.
                can_schedule = True
                break
        return can_schedule

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
                        new_mm_positions.append(ImagePosition(token_st, h * w // 4))
                        # videos are split into patches every 2 frames, need to rehash
                        new_mm_hashes.append(
                            MultimodalHasher.hash_features(inputs["images"][image_st : image_st + 2 * h * w])
                        )
                        image_st += 2 * h * w
                        token_st += h * w // 4
            inputs["mm_positions"] = new_mm_positions
            inputs["mm_hashes"] = new_mm_hashes
        else:
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

    def _get_num_new_tokens(self, request, token_budget):
        # TODO: set condition to new _get_num_new_tokens
        num_new_tokens = request.need_prefill_tokens - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)
        request.with_image = False

        if not self.config.model_config.enable_mm:
            return num_new_tokens

        inputs = request.multimodal_inputs
        if inputs.get("patch_idx", None) is not None and inputs.get("patch_map", None) is not None:
            pre_end_idx = request.num_computed_tokens
            new_end_idx = pre_end_idx + num_new_tokens

            prompt_token_ids_len = len(request.prompt_token_ids)
            assert prompt_token_ids_len == len(inputs["patch_idx"]), (prompt_token_ids_len, len(inputs["patch_idx"]))

            # start
            if pre_end_idx >= prompt_token_ids_len:
                start_patch_idx = inputs["patch_idx"][-1]
            else:
                start_patch_idx = inputs["patch_idx"][pre_end_idx]
            start_patch_map = inputs["patch_map"][start_patch_idx]
            request.image_start = start_patch_map["image_num"]
            request.video_start = start_patch_map["video_num"]
            request.audio_start = start_patch_map["audio_num"]

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
            if end_modal_id > 0 and end_modal_id != IDS_TYPE_FLAG["video"]:
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
            request.audio_end = end_patch_map["audio_num"]
        elif (
            inputs.get("images", None) is not None
            and inputs.get("image_patch_id", None) is not None
            and inputs.get("grid_thw", None) is not None
        ):
            input_ids_lst = request.prompt_token_ids + request.output_token_ids
            input_ids = paddle.to_tensor(input_ids_lst, dtype="int64")
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

                grid_thw = paddle.to_tensor(grid_thw, dtype="int64")
                if current_platform.is_xpu():
                    from fastdeploy.model_executor.ops.xpu import get_img_boundaries
                else:
                    from fastdeploy.model_executor.ops.gpu import get_img_boundaries

                request.multimodal_img_boundaries = get_img_boundaries(
                    task_input_ids=input_ids, grid_thw=grid_thw, image_patch_id=image_patch_id
                ).numpy()

                grid_thw = grid_thw.numpy().reshape([-1, 3])
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

                cur_mm_hashes = inputs["mm_hashes"][request.num_image_start : request.num_image_end]
                cur_mm_positions = inputs["mm_positions"][request.num_image_start : request.num_image_end]
                if self.encoder_cache:
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

    def schedule(self):
        """
        Try to pull a batch of requests from the waiting queue and schedule them.
        """
        with self.lock:
            scheduled_reqs: list[Request] = []
            preempted_reqs: list[Request] = []
            error_reqs: list[tuple[str, str]] = []
            token_budget = self.config.scheduler_config.max_num_batched_tokens

            # First, schedule the RUNNING requests.
            req_index = 0
            num_decoding_req_nums = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                need_block_num = self.need_block_num_signal.value[request.idx]
                if need_block_num != 0:
                    self.need_block_num_map[request.request_id] = SignalConsumer(need_block_num, 1)
                    self.need_block_num_signal.value[request.idx] = 0

                if request.num_computed_tokens >= request.need_prefill_tokens:  # to be decoding
                    if (
                        self.config.scheduler_config.splitwise_role == "prefill"
                    ):  # do not need to schedule for decoding
                        req_index += 1
                        continue
                    if request.num_total_tokens > request.need_prefill_tokens:  # has generated tokens
                        request.num_computed_tokens = request.num_total_tokens - 1
                    if (
                        self.allocated_slots(request) - request.num_total_tokens
                        <= self.config.cache_config.prealloc_dec_block_slot_num_threshold
                    ):
                        # Allocation for next decoding blocks
                        if self.cache_manager.can_allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num):
                            llm_logger.debug(
                                f"schedule decoding task: {request} request.num_total_tokens {request.num_total_tokens} request.num_computed_tokens {request.num_computed_tokens}"
                            )
                            request.block_tables.extend(
                                self.cache_manager.allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num)
                            )
                            # Prepare decoding task
                            scheduled_reqs.append(self._prepare_decode_task(request))
                        else:
                            # Not enough blocks to allocate, trigger preemption
                            can_schedule = self._trigger_preempt(
                                request, self.config.cache_config.enc_dec_block_num, preempted_reqs, scheduled_reqs
                            )
                            if not can_schedule:
                                break
                            # Allocation for next decoding blocks
                            request.block_tables.extend(
                                self.cache_manager.allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num)
                            )
                            # Prepare decoding task
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
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(allocate_block_num))
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
                                self.cache_manager.allocate_gpu_blocks(allocate_block_num)
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
                        f"scheduler prefill task: {request} request.need_prefill_tokens {request.need_prefill_tokens} request.num_computed_tokens {request.num_computed_tokens}"
                    )
                    num_new_tokens = self._get_num_new_tokens(request, token_budget)
                    num_new_block = self.get_new_block_nums(request, num_new_tokens)
                    # Allocate blocks to prefill
                    if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                        request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                        # Prepare prefill task
                        scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                    else:  # Not enough blocks to allocate, trigger preemption
                        can_schedule = self._trigger_preempt(request, num_new_block, preempted_reqs, scheduled_reqs)
                        if not can_schedule:
                            break
                        request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                        # Prepare prefill task
                        scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                    token_budget -= num_new_tokens
                    request.num_computed_tokens += num_new_tokens
                    if self.config.cache_config.enable_prefix_caching:
                        self.cache_manager.update_cache_blocks(
                            request, self.config.cache_config.block_size, request.num_computed_tokens
                        )
                req_index += 1
            # schedule the WAITING requests.
            if not preempted_reqs:
                skip_requests: list[Request] = []
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_seqs:
                        break

                    request = self.waiting[0]
                    if (self._is_mm_request(request) and self.exist_mm_prefill(scheduled_reqs)) or (
                        paddle.is_compiled_with_xpu() and self.exist_prefill(scheduled_reqs)
                    ):
                        break
                    if request.status == RequestStatus.WAITING:
                        result = self._waiting_async_process(request)
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
                                self.config.cache_config.enable_hierarchical_cache
                                and self.cache_manager.num_cpu_blocks > 0
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

                        num_new_tokens = self._get_num_new_tokens(request, token_budget)
                        num_new_block = self.get_new_block_nums(request, num_new_tokens)
                        # Allocate blocks to prefill
                        if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                            if not request.get("skip_allocate", False):
                                request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                            self.waiting.popleft()
                            self.running.append(request)
                            scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                            request.inference_start_time = time.time()
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
                                self.config.cache_config.enable_hierarchical_cache
                                and self.cache_manager.num_cpu_blocks > 0
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
                        num_new_tokens = self._get_num_new_tokens(request, token_budget)
                        num_new_block = self.get_new_block_nums(request, num_new_tokens)
                        # Allocate blocks to prefill
                        if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                            if not request.get("skip_allocate", False):
                                request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
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
                        llm_logger.error("Unknown request status type")

                for req in skip_requests:
                    # move waiting request to end of the deque
                    self.waiting.append(req)

            if scheduled_reqs:
                llm_logger.debug(f"schedued_reqs: {scheduled_reqs}")

            self.update_metrics()

            return scheduled_reqs, error_reqs

    def _waiting_async_process(self, request: Request) -> None:
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

    def _apply_async_preprocess(self, request: Request) -> None:
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
                    llm_logger.info(f"request {request.request_id} async download feature: {feature.shape}")
                    result_list.append(feature)
                else:
                    error_msg = f"request {request.request_id} download features error: {feature}"
                    llm_logger.error(error_msg)
                    return error_msg
            return result_list

        if not self._has_features_info(request):
            return None

        if self.bos_client is None:
            self.bos_client = init_bos_client()

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
        set prefix cached information for the given request
        """
        try:
            cache_prepare_time = time.time()
            (common_block_ids, matched_token_num, hit_info) = self.cache_manager.request_match_blocks(
                request, self.config.cache_config.block_size
            )

            matched_block_num = len(common_block_ids)
            no_cache_block_num = self.cache_manager.get_required_block_num(
                request.need_prefill_tokens - matched_token_num,
                self.config.cache_config.block_size,
            )

            request.num_cached_tokens = matched_token_num
            request.gpu_cache_token_num = hit_info["gpu_match_token_num"]
            request.cpu_cache_token_num = hit_info["cpu_match_token_num"]
            request.cache_info = (matched_block_num, no_cache_block_num)
            request.block_tables = common_block_ids
            request.skip_allocate = False

            # Report the number of cached tokens to Prometheus metrics
            main_process_metrics.prefix_cache_token_num.inc(matched_token_num)
            main_process_metrics.prefix_gpu_cache_token_num.inc(request.gpu_cache_token_num)
            main_process_metrics.prefix_cpu_cache_token_num.inc(request.cpu_cache_token_num)

            if matched_token_num == request.need_prefill_tokens:
                request.num_computed_tokens = matched_token_num - self.config.cache_config.block_size
                request.skip_allocate = True
            else:
                request.num_computed_tokens = matched_token_num
            request.cache_prepare_time = time.time() - cache_prepare_time
            return True
        except Exception as e:
            llm_logger.error(f"prefix match blocks error: {e}, {str(traceback.format_exc())} waiting reschedule...")
            return False

    def add_request(self, request: Request) -> None:
        with self.lock:
            self._apply_async_preprocess(request)
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
                request.inference_start_time = time.time()
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
                if self.config.cache_config.enable_hierarchical_cache and self.cache_manager.num_cpu_blocks > 0:
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
                    request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(need_extra_prefill_blocks))
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
                    request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(need_prealloc_prefill_blocks))
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

            request.block_tables = self.cache_manager.allocate_gpu_blocks(need_prealloc_prefill_blocks)
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
        assert self.config.scheduler_config.splitwise_role == "decode", "Only D instance can call this method"
        assert request_id in self.preallocated_reqs, "request_id must be in preallocate"
        need_blocks_num = len(self.preallocated_reqs[request_id].disaggregate_info["block_tables"])
        return self.available_batch() > 0 and self.cache_manager.can_allocate_gpu_blocks(need_blocks_num)

    def add_prefilled_request(self, request_output: RequestOutput):
        """
        In P/D aggregated deployment, D should continue to decode after receiving first token and cache from P.
        NOTE: GPU resources should be checked in advance to ensure they are sufficient for the prefilled request.
        """
        assert self.config.scheduler_config.splitwise_role == "decode", "Only D instance can call this method"
        if request_output.request_id not in self.requests:
            self.logger.error(f"Request {request_output.request_id} not found in requests")
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
        request.inference_start_time = time.time()
        request.schedule_start_time = time.time()
        self.running.append(request)

    def _free_blocks(self, request: Request):
        if self.config.cache_config.enable_prefix_caching:
            self.cache_manager.release_block_ids(request)
            self.cache_manager.recycle_gpu_blocks(request.block_tables[request.cached_block_num :])
        else:
            self.cache_manager.recycle_gpu_blocks(request.block_tables)
        request.block_tables = []

        if request.request_id in self.using_extend_tables_req_id:
            reuse_block_num = self.reuse_block_num_map[request.request_id]

            self.using_extend_tables_req_id.remove(request.request_id)
            self.cache_manager.recycle_gpu_blocks(request.extend_block_tables[reuse_block_num:])
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
            with self.lock:
                if isinstance(request_ids, str):
                    request_ids = (request_ids,)
                else:
                    request_ids = set(request_ids)
                for req_id in request_ids:
                    request = self.requests.get(req_id)
                    if request is None:
                        # Invalid request ID.
                        continue
                    if request in self.running:  # normally run and finished
                        self.running.remove(request)
                        request.status = RequestStatus.FINISHED
                        try:
                            self._free_blocks(request)
                        except Exception as e:
                            llm_logger.warning(f"release block failed {req_id}: {e}")
                    if (
                        request.request_id in self.to_be_rescheduled_request_id_set
                    ):  # finished after preempted, blocks have been recycled.
                        self.to_be_rescheduled_request_id_set.remove(
                            request.request_id
                        )  # just remove from to_be_rescheduled_request_id_set
                    if (
                        request in self.waiting
                    ):  # after finished, this request still scheduled from preempted to waiting, unexpected error, should not be here
                        raise RuntimeError(f"request {request.request_id} scheduled into waiting list, after finished")

                    self.tasks_list[request.idx] = None
                    self.stop_flags[request.idx] = True
                    del self.requests[req_id]
                    if req_id in self.req_dict:
                        del self.req_dict[req_id]
        except Exception as e:
            llm_logger.error(f"finish_request err: {e}, {str(traceback.format_exc())}")
        finally:
            self.update_metrics()

    def clear_data(self):
        self.waiting: deque[Request] = deque()
        self.to_be_rescheduled_request_id_set = set()

    def update_metrics(self):
        # Update metrics
        num_tasks = sum([1 if task else 0 for task in self.tasks_list])
        num_blocks_used_by_tasks = sum([len(task.block_tables) if task else 0 for task in self.tasks_list])
        main_process_metrics.available_gpu_block_num.set(self.total_block_number() - num_blocks_used_by_tasks)
        main_process_metrics.batch_size.set(self.max_num_seqs - self.available_batch())
        main_process_metrics.gpu_cache_usage_perc.set(self.get_gpu_cache_usage_perc())
        main_process_metrics.num_requests_running.set(len(self.running))
        main_process_metrics.num_requests_waiting.set(num_tasks - len(self.running))
