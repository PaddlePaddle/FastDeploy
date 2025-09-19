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
import traceback
from collections import deque
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Union

import numpy as np
import paddle

from fastdeploy.engine.request import Request, RequestStatus, RequestType
from fastdeploy.engine.resource_manager import ResourceManager
from fastdeploy.metrics.metrics import main_process_metrics
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


@dataclass
class ScheduledExtendBlocksTask:
    """
    Task for allocating new blocks to extend.
    """

    idx: int
    request_id: str
    extend_block_tables: list[int]
    task_type: RequestType = RequestType.EXTEND


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
        self.to_be_rescheduled_request_id_set = set()
        main_process_metrics.max_batch_size.set(max_num_seqs)

        self.using_extend_tables_req_id = set()

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

    def _trigger_preempt(self, request, num_new_blocks, preempted_reqs, scheduled_reqs):
        """
        If the request cannot be scheduled, preempt the running request one by one until it can be scheduled. Last in, first out.
        """
        can_schedule = True
        while True:
            if not self.cache_manager.can_allocate_gpu_blocks(num_new_blocks):
                preempted_req = self.running.pop()
                preempted_req.status = RequestStatus.PREEMPTED
                preempted_req.num_computed_tokens = 0
                self._free_blocks(preempted_req)
                preempted_req.cached_block_num = 0
                self.to_be_rescheduled_request_id_set.add(preempted_req.request_id)
                preempted_reqs.append(preempted_req)
                scheduled_reqs.append(self._prepare_preempt_task(preempted_req))
                main_process_metrics.num_requests_waiting.inc(1)
                main_process_metrics.num_requests_running.dec(1)
                if preempted_req == request:
                    # No more request to preempt.
                    can_schedule = False
                    break
            else:
                # The request can be scheduled.
                can_schedule = True
                break
        return can_schedule

    def _get_num_new_tokens(self, request, token_budget):
        # TODO: set condition to new _get_num_new_tokens
        num_new_tokens = request.need_prefill_tokens - request.num_computed_tokens
        num_new_tokens = min(num_new_tokens, token_budget)

        if not self.config.model_config.enable_mm:
            return num_new_tokens

        request.with_image = False
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
            if end_modal_id > 0:
                new_end_idx = end_patch_map["end_idx"]  # 当前模态结束位置
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
                for one in inputs["grid_thw"]:
                    if one[0] == 1:
                        grid_thw.append(one)
                    else:
                        grid_thw.extend([[2, one[1], one[2]]] * (one[0] // 2))

                grid_thw = paddle.to_tensor(grid_thw, dtype="int64")
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

        # Compatible with scenarios without images and videos.
        return num_new_tokens

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
            token_budget = self.config.scheduler_config.max_num_batched_tokens

            # First, schedule the RUNNING requests.
            req_index = 0
            num_decoding_req_nums = 0
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                if request.num_computed_tokens >= request.need_prefill_tokens:  # to be decoding
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
                while self.waiting and token_budget > 0:
                    if len(self.running) == self.max_num_seqs:
                        break
                    if (self.config.model_config.enable_mm or paddle.is_compiled_with_xpu()) and self.exist_prefill(
                        scheduled_reqs
                    ):
                        break
                    request = self.waiting[0]
                    if request.status == RequestStatus.WAITING:
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
                            request.schedule_start_time = time.time()
                            token_budget -= num_new_tokens
                            request.num_computed_tokens += num_new_tokens
                            if self.config.cache_config.enable_prefix_caching:
                                self.cache_manager.update_cache_blocks(
                                    request, self.config.cache_config.block_size, request.num_computed_tokens
                                )
                            request.status = RequestStatus.RUNNING
                            main_process_metrics.num_requests_waiting.dec(1)
                            main_process_metrics.num_requests_running.inc(1)
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
                            main_process_metrics.num_requests_waiting.dec(1)
                            main_process_metrics.num_requests_running.inc(1)
                        else:
                            if self.config.cache_config.enable_prefix_caching:
                                self._free_blocks(request)
                            break
                    else:
                        llm_logger.error("Unknown request status type")

            # schedule when extend block tables is needed
            for req in self.running:
                num_prefill_blocks = req.need_prefill_tokens // self.config.cache_config.block_size
                # allocate
                if req.use_extend_tables and req.request_id not in self.using_extend_tables_req_id:
                    llm_logger.info(
                        f"req {req.request_id} at batch id {req.idx} with num_prefill_blocks {num_prefill_blocks} is going to enable extend tables"
                    )
                    self.using_extend_tables_req_id.add(req.request_id)
                    if self.cache_manager.can_allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num):
                        req.extend_block_tables = req.block_tables[:num_prefill_blocks]  # copy prompt cache
                        req.extend_block_tables.extend(
                            self.cache_manager.allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num)
                        )
                        scheduled_reqs.append(
                            ScheduledExtendBlocksTask(
                                idx=req.idx, request_id=req.request_id, extend_block_tables=req.extend_block_tables
                            )
                        )
                        llm_logger.info(f"extend blocks is {req.extend_block_tables}")
                    else:
                        continue
                # recycle
                elif not req.use_extend_tables and req.request_id in self.using_extend_tables_req_id:
                    llm_logger.info(f"req {req.request_id} is going to disable extend tables")
                    self.using_extend_tables_req_id.remove(req.request_id)
                    self.cache_manager.recycle_gpu_blocks(req.extend_block_tables[num_prefill_blocks:])
                    req.extend_block_tables = []

                # allocate extend blocks when blocks is going to exhaust
                elif req.request_id in self.using_extend_tables_req_id:
                    if (
                        self.allocated_slots(req) - req.num_total_tokens
                        <= self.config.cache_config.prealloc_dec_block_slot_num_threshold
                    ):
                        llm_logger.info(
                            f"req {req.request_id} is going to allocate more extend tables because allocated_slots {self.allocated_slots(req)} and prealloc_dec_block_slot_num_threshold {self.config.cache_config.prealloc_dec_block_slot_num_threshold} req.num_total_tokens {req.num_total_tokens}"
                        )
                        if self.cache_manager.can_allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num):
                            req.extend_block_tables.extend(
                                self.cache_manager.allocate_gpu_blocks(self.config.cache_config.enc_dec_block_num)
                            )
                            scheduled_reqs.append(
                                ScheduledExtendBlocksTask(
                                    idx=req.idx, request_id=req.request_id, extend_block_tables=req.extend_block_tables
                                )
                            )
                        else:
                            continue

            if scheduled_reqs:
                task_used_block_num = sum([len(task.block_tables) if task else 0 for task in self.tasks_list])
                main_process_metrics.available_gpu_block_num.set(self.total_block_number() - task_used_block_num)
                main_process_metrics.batch_size.set(self.max_num_seqs - self.available_batch())
                main_process_metrics.gpu_cache_usage_perc.set(self.get_gpu_cache_usage_perc())
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
            request.gpu_cache_token_num = hit_info["gpu_cache_blocks"] * self.config.cache_config.block_size
            request.cpu_cache_token_num = hit_info["cpu_cache_blocks"] * self.config.cache_config.block_size
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
            self.waiting.append(request)
            self.requests[request.request_id] = request

    def _free_blocks(self, request: Request):
        if self.config.cache_config.enable_prefix_caching:
            self.cache_manager.release_block_ids(request)
            self.cache_manager.recycle_gpu_blocks(request.block_tables[request.cached_block_num :])
        else:
            self.cache_manager.recycle_gpu_blocks(request.block_tables)
        request.block_tables = []

        if request.request_id in self.using_extend_tables_req_id:
            num_prefill_blocks = request.need_prefill_tokens // self.config.cache_config.block_size
            self.using_extend_tables_req_id.remove(request.request_id)
            self.cache_manager.recycle_gpu_blocks(request.extend_block_tables[num_prefill_blocks:])
            llm_logger.info(
                f"req {request.request_id} recycle extend blocks {request.extend_block_tables[num_prefill_blocks:]}"
            )
            request.extend_block_tables = []

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
                        self._free_blocks(request)
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
        except Exception as e:
            llm_logger.error(f"finish_request err: {e}, {str(traceback.format_exc())}")
