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

import threading
import time
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Union

import numpy as np
import paddle

from fastdeploy.engine.request import Request, RequestStatus, RequestType
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.utils import llm_logger


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
        self.finish_execution_pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()

    def allocated_slots(self, request: Request):
        return len(request.block_tables) * self.config.cache_config.block_size

    def get_new_block_nums(self, request: Request, num_new_tokens: int):
        return (
            request.num_computed_tokens + num_new_tokens + self.config.cache_config.block_size - 1
        ) // self.config.cache_config.block_size - len(request.block_tables)

    def _prepare_prefill_task(self, request, new_token_num):
        request.prefill_start_index = request.num_computed_tokens
        request.prefill_end_index = request.num_computed_tokens + new_token_num
        request.task_type = RequestType.PREFILL
        return request

    def _prepare_decode_task(self, request):
        return ScheduledDecodeTask(idx=request.idx, request_id=request.request_id, block_tables=request.block_tables)

    def _prepare_preempt_task(self, request):
        return ScheduledPreemptTask(idx=request.idx, request_id=request.request_id)

    def _trigger_preempt(self, request, num_new_blocks, preempted_reqs, scheduled_reqs):
        can_schedule = True
        while True:
            if not self.cache_manager.can_allocate_gpu_blocks(num_new_blocks):
                preempted_req = self.running.pop()
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
                self._free_blocks(preempted_req)
                self.waiting.appendleft(preempted_req)
                preempted_reqs.append(preempted_req)
                scheduled_reqs.append(self._prepare_preempt_task(preempted_req))
                if preempted_req == request:
                    # No more request to preempt.
                    can_schedule = False
                    break
            else:
                # The request can be scheduled.
                can_schedule = True
                break
        return can_schedule

    def _get_num_new_tokens(self, request, token_budget, schedule_waiting=False):
        if schedule_waiting:
            num_new_tokens = request.num_total_tokens - request.num_computed_tokens
        else:
            num_new_tokens = request.prompt_token_ids_len - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        if not self.config.enable_mm:
            return num_new_tokens

        inputs = request.multimodal_inputs
        request.with_image = False
        # Compatible with scenarios without images and videos.
        if inputs["images"] is None:
            return num_new_tokens

        input_ids_lst = request.prompt_token_ids + request.output_token_ids
        input_ids = paddle.to_tensor(input_ids_lst, dtype="int64")
        grid_thw = []
        for one in inputs["grid_thw"]:
            if one[0] == 1:
                grid_thw.append(one)
            else:
                grid_thw.extend([[2, one[1], one[2]]] * (one[0] // 2))

        image_patch_id = inputs["image_patch_id"]
        grid_thw = paddle.to_tensor(grid_thw, dtype="int64")
        if request.multimodal_img_boundaries is None:
            from fastdeploy.model_executor.ops.gpu import get_img_boundaries

            request.multimodal_img_boundaries = get_img_boundaries(
                task_input_ids=input_ids, grid_thw=grid_thw, image_patch_id=image_patch_id
            ).numpy()

        img_boundaries_idx = request.multimodal_img_boundaries[0]
        img_num_per_boundary = request.multimodal_img_boundaries[1]
        ori_prompt_len = img_boundaries_idx[-1].item()
        grid_thw = grid_thw.numpy().reshape([-1, 3])
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
                    pre_boundary_idx if pre_end_idx == img_boundaries_idx[pre_boundary_idx] else pre_boundary_idx - 1
                )
                request.num_image_start = img_num_per_boundary[pre_boundary_idx]

            new_boundary_idx = np.searchsorted(img_boundaries_idx, new_end_idx, side="left").item()
            if new_boundary_idx == len(img_boundaries_idx):
                request.num_image_end = img_num_per_boundary[-1]
            else:
                new_boundary_idx = (
                    new_boundary_idx if new_end_idx == img_boundaries_idx[new_boundary_idx] else new_boundary_idx - 1
                )
                request.num_image_end = img_num_per_boundary[new_boundary_idx]

            request.num_image_end = img_num_per_boundary[new_boundary_idx]
            request.image_type_ids_start = np.sum(grid_thw[: request.num_image_start, 0])
            request.image_type_ids_end = np.sum(grid_thw[: request.num_image_end, 0])
            request.image_start = np.sum(np.prod(grid_thw[: request.num_image_start], axis=1))
            request.image_end = np.sum(np.prod(grid_thw[: request.num_image_end], axis=1))
        return num_new_tokens

    def exist_prefill(self, scheduled_reqs):
        for request in scheduled_reqs:
            if request.task_type == RequestType.PREFILL:
                return True
        return False

    def schedule(self):
        with self.lock:
            scheduled_reqs: list[Request] = []
            preempted_reqs: list[Request] = []
            token_budget = self.config.max_num_batched_tokens

            # First, schedule the RUNNING requests.
            req_index = 0
            num_decoding_req_nums = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                if request.num_computed_tokens >= request.prompt_token_ids_len:  # to be decoding
                    if request.num_total_tokens > request.prompt_token_ids_len:  # has generated tokens
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
                else:  # need to prefill
                    llm_logger.debug(
                        f"scheduler prefill task: {request} request.prompt_token_ids_len {request.prompt_token_ids_len} request.num_computed_tokens {request.num_computed_tokens}"
                    )
                    num_new_tokens = self._get_num_new_tokens(request, token_budget)
                    num_new_block = self.get_new_block_nums(request, num_new_tokens)
                    # Allocate blocks to prefill
                    if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                        request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                        # Prepare prefill task
                        scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                    else:
                        can_schedule = self._trigger_preempt(request, num_new_block, preempted_reqs, scheduled_reqs)
                        if not can_schedule:
                            break
                        request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                        # Prepare prefill task
                        scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                    token_budget -= num_new_tokens
                    request.num_computed_tokens += num_new_tokens
                req_index += 1
            # schedule the WAITING requests.
            if not preempted_reqs:
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_seqs:
                        break
                    if self.config.enable_mm and self.exist_prefill(scheduled_reqs):
                        break
                    request = self.waiting[0]
                    if request.status == RequestStatus.WAITING:
                        num_new_tokens = self._get_num_new_tokens(request, token_budget, True)
                        num_new_block = self.get_new_block_nums(request, num_new_tokens)
                        # Allocate blocks to prefill
                        if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                            self.waiting.popleft()
                            self.running.append(request)
                            scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                            request.inference_start_time = time.time()
                            request.schedule_start_time = time.time()
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            request.status = RequestStatus.RUNNING
                            allocated_position = self.get_available_position()
                            request.idx = allocated_position
                            self.tasks_list[allocated_position] = request
                            self.stop_flags[allocated_position] = False
                            self.req_dict[request.request_id] = allocated_position
                        else:
                            break
                    elif request.status == RequestStatus.PREEMPTED:
                        num_new_tokens = self._get_num_new_tokens(request, token_budget, True)
                        num_new_block = self.get_new_block_nums(request, num_new_tokens)
                        # Allocate blocks to prefill
                        if self.cache_manager.can_allocate_gpu_blocks(num_new_block):
                            request.block_tables.extend(self.cache_manager.allocate_gpu_blocks(num_new_block))
                            self.waiting.popleft()
                            self.running.append(request)
                            scheduled_reqs.append(self._prepare_prefill_task(request, num_new_tokens))
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            request.status = RequestStatus.RUNNING
                        else:
                            break
                    else:
                        llm_logger.error("Unknown request status type")
            if scheduled_reqs:
                llm_logger.debug(f"schedued_reqs: {scheduled_reqs}")
            return scheduled_reqs

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

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)
        self.requests[request.request_id] = request

    def _free_blocks(self, request: Request):
        self.cache_manager.recycle_gpu_blocks(request.block_tables)
        request.block_tables = []

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
                    request.status = RequestStatus.FINISHED
                    self.running.remove(request)
                    self._free_blocks(request)
                    self.tasks_list[request.idx] = None
                    self.stop_flags[request.idx] = True
                    del self.requests[req_id]
        except Exception as e:
            llm_logger.error(e)
