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
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import paddle
import zmq

import fastdeploy.metrics.trace as tracing
from fastdeploy import envs
from fastdeploy.config import PREEMPTED_TOKEN_ID
from fastdeploy.engine.request import (
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    Request,
    RequestMetrics,
    RequestOutput,
    RequestStatus,
    SpeculateMetrics,
)
from fastdeploy.inter_communicator import ZmqIpcServer
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.platforms import current_platform
from fastdeploy.spec_decode import SpecMethod
from fastdeploy.trace.constants import LoggingEventName
from fastdeploy.trace.trace_logger import print as trace_print
from fastdeploy.utils import llm_logger, spec_logger
from fastdeploy.worker.output import LogprobsLists

RECOVERY_STOP_SIGNAL = -3
MAX_DRAFT_TOKENS = 6
SPECULATE_MAX_BSZ = 256


MAX_BSZ = 512
K = 20


class TokenProcessor:
    """
    get Token/Score from Paddle inference engine
    """

    def __init__(self, cfg, cached_generated_tokens, engine_worker_queue, split_connector):
        paddle.device.set_device("cpu")
        self.cfg = cfg
        self.cached_generated_tokens = cached_generated_tokens
        self.resource_manager = None
        self.scheduler_metrics_logger = None
        self._benchmark_logger = None
        self.engine_worker_queue = engine_worker_queue
        self.tokens_counter = Counter()
        self.split_connector = split_connector

        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            port = self.cfg.parallel_config.local_engine_worker_queue_port
            llm_logger.debug(
                f"create zmq get_save_output_rank{self.cfg.parallel_config.local_data_parallel_id}_{port}"
            )
            self.zmq_server = ZmqIpcServer(
                name=f"get_save_output_rank{self.cfg.parallel_config.local_data_parallel_id}_{port}", mode=zmq.PULL
            )

        self.speculative_decoding = self.cfg.speculative_config.method is not None
        self.use_logprobs = self.cfg.model_config.enable_logprob
        self.use_sampling_mask = getattr(self.cfg.model_config, "enable_keep_sampling_mask", False)
        if not envs.FD_USE_GET_SAVE_OUTPUT_V1 and self.use_sampling_mask:
            rank_id = self.cfg.parallel_config.local_data_parallel_id
            port = self.cfg.parallel_config.engine_worker_queue_port[rank_id]
            self.sampling_mask_zmq_server = ZmqIpcServer(
                name=f"sampling_mask_output_rank_{rank_id}_{port}", mode=zmq.PULL
            )
            llm_logger.info(f"create zmq sampling_mask_output_rank_{rank_id}_{port}")
        self.enable_draft_logprob = self.cfg.speculative_config.enable_draft_logprob

        if self.speculative_decoding:
            if self.use_logprobs:
                self.output_tokens = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS * (K + 1) + MAX_BSZ + 3, 1], fill_value=2, dtype="int64"
                )
                self.output_scores = paddle.full(
                    shape=[MAX_BSZ * MAX_DRAFT_TOKENS * (K + 1), 1], fill_value=0.0, dtype="float32"
                )
                self.output_ranks = paddle.full(shape=[MAX_BSZ * MAX_DRAFT_TOKENS], fill_value=0, dtype="int64")
            else:
                self.output_tokens = paddle.full(
                    shape=[SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2],
                    fill_value=2,
                    dtype="int64",
                )
        elif self.use_logprobs:
            self.output_tokens = paddle.full(shape=[MAX_BSZ * (K + 1) + 2, 1], fill_value=2, dtype="int64")
            self.output_scores = paddle.full(shape=[MAX_BSZ * (K + 1), 1], fill_value=0.0, dtype="float32")
            self.output_ranks = paddle.full(shape=[MAX_BSZ], fill_value=0, dtype="int64")
        else:
            self.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1], fill_value=2, dtype="int64")
        self.worker = None

        self.statics_start_time = time.time()
        self.number_of_tasks = 0
        self.number_of_input_tokens = 0
        self.number_of_output_tokens = 0
        self.total_step = 0
        self.speculative_stats_step = 0
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.num_emitted_tokens = 0
        self.max_num_emitted_tokens = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefill_result_status = dict()
        self._finalizer = weakref.finalize(self, self._cleanup_resources)
        self._batch_result_buffer = None
        self.total_step_per_request = {}
        self.accept_token_num_per_head_per_request = {}
        self.accept_token_num_per_head = [0] * MAX_DRAFT_TOKENS

        # health monitor
        self.timestamp_for_alive_before_handle_batch = None
        self.timestamp_for_alive_after_handle_batch = None
        self.health_lock = threading.Lock()
        self.engine_output_token_hang = False

        # Routing replay: attach to SharedMemory routing_host_buffer (lazy init after profiling)
        self.routing_host_view = None
        self._routing_host_view_init_attempted = False
        self.routing_cache_manager = None  # Set by Engine after profiling for local/rdma store dispatch

    def _init_routing_host_view(self):
        """Attach to SharedMemory routing_host_buffer created by Engine. Called lazily."""
        self._routing_host_view_init_attempted = True
        if not self.cfg.routing_replay_config.enable_routing_replay:
            return
        try:
            from fastdeploy.cache_manager.routing_cache_manager import (
                RoutingHostBufferView,
            )

            rrc = self.cfg.routing_replay_config
            cache_config = self.cfg.cache_config

            dp_suffix = str(self.cfg.parallel_config.local_engine_worker_queue_port)
            shm_name = f"routing_host_buffer.{dp_suffix}"
            num_gpu_blocks = cache_config.total_block_num
            max_num_kv_tokens = num_gpu_blocks * cache_config.block_size
            shape = (max_num_kv_tokens, rrc.num_moe_layers, rrc.moe_top_k)

            self.routing_host_view = RoutingHostBufferView(shape=shape, dtype=rrc.routing_dtype, shm_name=shm_name)
            self._routing_block_size = cache_config.block_size
            llm_logger.info(f"[R3] TokenProcessor attached to RoutingHostBuffer: {shm_name}")
        except FileNotFoundError:
            llm_logger.warning("[R3] RoutingHostBuffer SharedMemory not found, routing gather disabled.")
        except Exception as e:
            llm_logger.warning(f"[R3] Failed to attach to RoutingHostBuffer: {e}")

    def _gather_routing_for_finished_request(self, task, seq_len: int):
        """
        Gather complete routing data for a finished request from routing_host_buffer.

        Args:
            task: Request task with block_tables
            seq_len: Total sequence length

        Returns:
            numpy array [seq_len, num_moe_layers, top_k] or None
        """
        if self.routing_host_view is None and not self._routing_host_view_init_attempted:
            self._init_routing_host_view()
        if self.routing_host_view is None:
            return None

        import math

        block_size = self._routing_block_size
        block_ids = task.block_tables[: math.ceil(seq_len / block_size)]
        positions = np.arange(seq_len)
        block_indices = positions // block_size
        offsets = positions % block_size
        slot_mapping = np.array(block_ids)[block_indices] * block_size + offsets

        return self.routing_host_view.gather(slot_mapping)

    def healthy(self):
        """
        whether token processor is healthy
        """
        with self.health_lock:
            if self.timestamp_for_alive_after_handle_batch is None:  # has entered handle batch
                if (
                    self.timestamp_for_alive_before_handle_batch is not None
                    and time.time() - self.timestamp_for_alive_before_handle_batch
                    > envs.FD_TOKEN_PROCESSOR_HEALTH_TIMEOUT
                ):
                    return False
                else:
                    return True
            if self.engine_output_token_hang:
                return False
            return True

    def _cleanup_resources(self):
        """Cleaning up shared memory resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)

    def set_resource_manager(self, resource_manager):
        """
        set ResourceManager

        Args:
            resource_manager (ResourceManager)
        """
        assert self.resource_manager is None, "The resource manager is not None, cannot set again."
        self.resource_manager = resource_manager

    def set_scheduler_metrics_logger(self, scheduler_metrics_logger):
        self.scheduler_metrics_logger = scheduler_metrics_logger

    def set_benchmark_logger(self, benchmark_logger):
        self._benchmark_logger = benchmark_logger

    def _is_decode_stage(self, task):
        if task is None:
            return False
        if task.need_prefill_tokens is None:
            return False
        return task.num_computed_tokens >= task.need_prefill_tokens

    def run(self):
        """
        start thread to get tokens
        """
        assert self.resource_manager is not None, "The resource manager is None, cannot run."
        if self.worker is not None:
            raise Exception("Worker is already running!")

        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            self.worker = threading.Thread(target=self.process_sampling_results_use_zmq)
        else:
            self.worker = threading.Thread(target=self.process_sampling_results)

        self.worker.daemon = True
        self.worker.start()

    def _reschedule_preempt_task_use_zmq(self, datas):
        """reschedule when real batch size is smaller than the insert position of preemted_task"""
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            need_to_be_reschedule_req_ids = list(self.resource_manager.to_be_rescheduled_request_id_set)
            if len(need_to_be_reschedule_req_ids) > 0:
                batch_id_set = set()
                for data in datas:
                    batch_id_set.add(data.batch_id)
                llm_logger.debug(f"_reschedule_preempt_task_use_zmq batch_id_set {batch_id_set}")
            for request_id in need_to_be_reschedule_req_ids:
                if (
                    self.resource_manager.requests[request_id].idx not in batch_id_set
                ):  # No more token generated for preempted request
                    llm_logger.debug(
                        f"reschedule_preempt_task request_id {request_id} at {self.resource_manager.requests[request_id].idx}"
                    )
                    self.resource_manager.reschedule_preempt_task(request_id)
                    llm_logger.debug(
                        f"finish reschedule_preempt_task request_id {request_id} at {self.resource_manager.requests[request_id].idx}"
                    )

    def _process_per_token(self, task, batch_id: int, token_ids: np.ndarray, result: RequestOutput, is_prefill: bool):
        """
        process output token by token
        """
        current_time = time.time()
        task_id = task.request_id
        token_id_list = token_ids.tolist()

        self._record_metrics(task, current_time, token_id_list)
        for token_id in token_id_list:
            recovery_stop = token_id == RECOVERY_STOP_SIGNAL
            if recovery_stop:
                llm_logger.info(f"recovery stop signal found at task {task_id}")
            self.tokens_counter[task_id] += 1
            if token_id != RECOVERY_STOP_SIGNAL:
                result.outputs.token_ids.append(token_id)
                task.output_token_ids.append(token_id)

            if token_id in task.eos_token_ids or is_prefill or recovery_stop:
                result.finished = True
                if recovery_stop:
                    result.error_msg = "Recover is not supported, the result is incomplete!"

                # Calculate statistics for the combined log
                is_decode = self.cfg.scheduler_config.splitwise_role == "decode"
                inference_start_time = task.metrics.get_inference_start_time(is_decode)
                task.metrics.cal_cost_time()
                e2e_time = current_time - inference_start_time
                token_ratio = self.tokens_counter[task_id] / e2e_time

                # Get cache information
                gpu_cache = getattr(task.metrics, "gpu_cache_token_num", 0)
                cpu_cache = getattr(task.metrics, "cpu_cache_token_num", 0)
                total_cached = gpu_cache + cpu_cache

                # Build cached detail dict
                cached_detail = f'{{"CachedToken": {total_cached}, "GPU": {gpu_cache}, "CPU": {cpu_cache}}}'

                # Print combined log with all required information
                ttft = task.metrics.first_token_time if task.metrics.first_token_time else 0
                llm_logger.info(
                    f"Request={task_id}, InputToken={task.prompt_token_ids_len}, "
                    f"CachedDetail={cached_detail}, OutputToken={self.tokens_counter[task_id]}, "
                    f"TokenRatio={token_ratio:.2f}, TTFT={ttft:.2f}, "
                    f"E2E={e2e_time:.2f}, IsPrefill={is_prefill}, RecoveryStop={recovery_stop}, "
                    f"PreemptedCount={getattr(task.metrics, 'preempted_count', 0)}"
                )

                main_process_metrics.request_token_ratio.observe(token_ratio)
                llm_logger.info(f"{self.resource_manager.info()}")
                if self.cfg.speculative_config.method:
                    self._compute_speculative_status()
                self._record_completion_metrics(task, current_time)
                self._finalize_routing(task_id, task, result, is_prefill)
                self._recycle_resources(task_id, batch_id, task, result, is_prefill)
                break
        return result

    def _process_batch_output_use_zmq(self, receive_datas):
        """
        process output sample by sample
        """
        batch_result = list()
        for _, stream_data in enumerate(receive_datas):
            i = stream_data.batch_id
            if self.resource_manager.stop_flags[i]:
                continue

            task: Request = self.resource_manager.tasks_list[i]
            task_id = task.request_id
            token_ids = stream_data.tokens  # numpy.array
            if token_ids is not None and token_ids[-1] < 0:
                if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                    if (
                        task_id in self.resource_manager.to_be_aborted_req_id_set
                        and token_ids[-1] == PREEMPTED_TOKEN_ID
                    ):
                        llm_logger.info(f"start to recycle abort request_id {task_id}")
                        self.resource_manager.recycle_abort_task(task_id)
                        self._put_abort_results(task)
                    if (
                        task_id in self.resource_manager.to_be_rescheduled_request_id_set
                        and token_ids[-1] == PREEMPTED_TOKEN_ID
                    ):
                        llm_logger.info(f"sync preemption for request_id {task_id} done.")
                        self.resource_manager.reschedule_preempt_task(task_id)
                continue
            if self.cfg.scheduler_config.splitwise_role == "decode":
                # In D instance, if preempted, error has been reported and resource recycled, tokens generated async not need to be handled
                if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                    if task_id in self.resource_manager.to_be_aborted_req_id_set:
                        continue
                    if task_id in self.resource_manager.to_be_rescheduled_request_id_set:
                        continue

            current_time = time.time()
            if self.tokens_counter[task_id] == 0:
                task.metrics.record_recv_first_token()
                task.metrics.cal_cost_time()
                metrics = copy.copy(task.metrics)
                self._record_first_token_metrics(task, current_time)
            else:
                task.metrics.record_recv_token()
                if self.tokens_counter[task_id] == 1 and self.cfg.scheduler_config.splitwise_role == "decode":
                    task.metrics.record_decode_recv_second_token()
                metrics = copy.copy(task.metrics)

            if task.pooling_params is not None:
                pooler_output = stream_data.pooler_output
                if isinstance(pooler_output, np.ndarray):
                    pooler_output = pooler_output.tolist()
                result = PoolingRequestOutput(
                    request_id=task_id,
                    finished=True,
                    metrics=metrics,
                    prompt_token_ids=task.prompt_token_ids,
                    outputs=PoolingOutput(data=pooler_output),
                )
                self._finalize_routing(task_id, task, result, False)
                self._recycle_resources(task_id, i, task, result, False)
                batch_result.append(result)
            else:
                result = RequestOutput(
                    request_id=task_id,
                    outputs=CompletionOutput(
                        index=i,
                        send_idx=self.tokens_counter[task_id],
                        token_ids=[],
                        draft_token_ids=[],
                    ),
                    finished=False,
                    metrics=metrics,
                    ic_req_data=task.ic_req_data,
                )
                if self.use_logprobs:
                    if getattr(stream_data, "logprobs", None) is not None:
                        try:
                            logprobs_list: LogprobsLists = stream_data.logprobs.tolists()
                            result.outputs.logprob = float(logprobs_list.logprobs[0][0])
                            result.outputs.top_logprobs = logprobs_list
                        except Exception as e:
                            llm_logger.warning(f"Failed to parse logprobs from StreamTransferData: {e}")
                    if getattr(stream_data, "prompt_logprobs", None) is not None:
                        try:
                            result.prompt_logprobs = stream_data.prompt_logprobs
                        except Exception as e:
                            llm_logger.warning(f"Failed to parse prompt_logprobs from StreamTransferData: {e}")
                if getattr(stream_data, "sampling_mask", None) is not None:
                    result.outputs.sampling_mask = stream_data.sampling_mask.tolist()
                if self.tokens_counter[task_id] == 0:
                    if task.messages is not None:
                        result.prompt = task.messages
                    result.num_cached_tokens = task.num_cached_tokens
                    if task.get("multimodal_inputs", None):
                        result.num_input_image_tokens = task.multimodal_inputs.get("num_input_image_tokens", 0)
                        result.num_input_video_tokens = task.multimodal_inputs.get("num_input_video_tokens", 0)

                is_prefill = task.disaggregate_info is not None and task.disaggregate_info["role"] == "prefill"
                result = self._process_per_token(task, i, token_ids, result, is_prefill)
                if not is_prefill or self.cfg.scheduler_config.name == "splitwise":
                    batch_result.append(result)

        return batch_result

    def process_sampling_results_use_zmq(self):
        """
        use zmq to receive outputs from worker and process them
        """
        if self.speculative_decoding:
            raise NotImplementedError("GET_SAVE_OUTPUT_V1 does not support speculative decoding")
        rank_id = self.cfg.parallel_config.local_data_parallel_id
        while True:
            try:
                if (
                    self.cfg.parallel_config.enable_expert_parallel and self.cfg.parallel_config.data_parallel_size > 1
                ) or (rank_id == 0):
                    receive_datas = self.zmq_server.recv_pyobj()
                    assert isinstance(receive_datas, list)
                    if envs.FD_DEBUG:
                        llm_logger.debug(f"token_processor receive_data {receive_datas}")

                    self._reschedule_preempt_task_use_zmq(receive_datas)

                    batch_result = self._process_batch_output_use_zmq(receive_datas)
                    self.postprocess(batch_result)
            except Exception as e:
                llm_logger.error(f"Receive message:{receive_datas}, error:{e}")
                continue

    def process_sampling_results(self):
        """
        read tokens from paddle inference engine and process
        """
        tracing.trace_set_thread_info("Token Processor")

        if current_platform.is_xpu():
            from fastdeploy.model_executor.ops.xpu import (
                get_output,
                get_output_ep,
                get_output_topk,
                speculate_get_output,
            )
        elif current_platform.is_iluvatar():
            from fastdeploy.model_executor.ops.iluvatar import get_output, get_output_ep
        elif current_platform.is_gcu():
            from fastdeploy.model_executor.ops.gcu import get_output
        elif current_platform.is_intel_hpu():
            from fastdeploy.model_executor.ops.intel_hpu import get_output
        else:
            from fastdeploy.model_executor.ops.gpu import (
                get_output,
                get_output_ep,
                get_output_topk,
                speculate_get_output,
                speculate_get_output_topk,
            )
        rank_id = self.cfg.parallel_config.local_data_parallel_id

        while True:
            try:
                is_blocking = True
                if self.speculative_decoding:
                    if self.use_logprobs:
                        speculate_get_output_topk(
                            self.output_tokens,
                            self.output_scores,
                            self.output_ranks,
                            K,
                            rank_id,
                            is_blocking,
                        )
                        if self.output_tokens[0, 0] == -2:
                            continue
                    else:
                        if (
                            self.cfg.parallel_config.enable_expert_parallel
                            and self.cfg.parallel_config.data_parallel_size > 1
                        ):
                            speculate_get_output(self.output_tokens, rank_id, is_blocking, True)
                        else:
                            speculate_get_output(self.output_tokens, rank_id, is_blocking, False)
                            if self.output_tokens[0] == -2:
                                continue
                else:
                    if self.use_logprobs:
                        get_output_topk(
                            self.output_tokens,
                            self.output_scores,
                            self.output_ranks,
                            K,
                            rank_id,
                            is_blocking,
                        )
                    elif self.cfg.parallel_config.data_parallel_size > 1:
                        get_output_ep(self.output_tokens, rank_id, is_blocking)
                    else:
                        get_output(self.output_tokens, rank_id, is_blocking)

                    if self.output_tokens[0, 0] == -2:
                        continue
                    llm_logger.debug(f"rank_id {rank_id} self.output_tokens[0, 0] {self.output_tokens[0, 0]}")
                with self.health_lock:
                    self.timestamp_for_alive_before_handle_batch = time.time()
                    self.timestamp_for_alive_after_handle_batch = None
                self._process_batch_output()
                with self.health_lock:
                    self.timestamp_for_alive_before_handle_batch = None
                    self.timestamp_for_alive_after_handle_batch = time.time()

            except Exception as e:
                llm_logger.info(f"while get input_data error: {e} {traceback.format_exc()!s}")

    def postprocess(self, batch_result: List[RequestOutput], mtype=3):
        """
        single post-processing function

        Args:
            batch_result (list): batch results
        """
        try:
            if self.cfg.speculative_config.method and self.use_logprobs and self.enable_draft_logprob:
                if mtype == 3:  # target
                    finished_batch_result, unfinished_batch_result = [], []
                    for r in batch_result:
                        (finished_batch_result if r.finished else unfinished_batch_result).append(r)
                    if finished_batch_result:
                        self.cached_generated_tokens.put_results(batch_result)
                    else:
                        self._batch_result_buffer = unfinished_batch_result
                elif mtype == 4:  # draft
                    target_batch_result = []
                    draft_batch_result = batch_result
                    if self._batch_result_buffer is not None:
                        for target, decode in zip(self._batch_result_buffer, draft_batch_result):
                            target.outputs.draft_top_logprobs = decode.outputs.draft_top_logprobs
                            target_batch_result.append(target)
                        self._batch_result_buffer = None
                    self.cached_generated_tokens.put_results(target_batch_result)
                else:
                    self.cached_generated_tokens.put_results(batch_result)
            else:
                self.cached_generated_tokens.put_results(batch_result)
        except Exception as e:
            llm_logger.error(f"Error in TokenProcessor's postprocess: {e}, {str(traceback.format_exc())}")

    def _finalize_routing(self, task_id, task, result, is_prefill=False):
        """
        Gather routing data before blocks are freed.
        Must be called before _recycle_resources so that block_tables are still valid.

        - PD P node (is_prefill=True): gather prefill-only routing, attach to result for sending to D.
        - Non-PD / D node (result.finished): gather full routing (prompt + output),
          either attach to result ("response" mode) or dispatch to store ("local"/"rdma" mode).
        """
        if not self.cfg.routing_replay_config.enable_routing_replay:
            return
        if result is None:
            return

        try:
            if is_prefill:
                if result.error_code == 200:
                    seq_len = task.prompt_token_ids_len
                    routing_data = self._gather_routing_for_finished_request(task, seq_len)
                    if routing_data is not None:
                        result.routing_data = routing_data
            elif result.finished:
                store_type = self.cfg.routing_replay_config.routing_store_type
                seq_len = (
                    task.prompt_token_ids_len + len(task.output_token_ids)
                    if hasattr(task, "output_token_ids")
                    else task.prompt_token_ids_len
                )
                if store_type == "response":
                    routing_data = self._gather_routing_for_finished_request(task, seq_len)
                    if routing_data is not None:
                        result.routing_data = routing_data
                elif self.routing_cache_manager is not None:
                    self.routing_cache_manager.on_request_finished(
                        request_id=task_id,
                        block_table=task.block_tables,
                        seq_len=seq_len,
                    )
        except Exception as e:
            llm_logger.warning(f"[R3] Failed to finalize routing for {task_id}: {e}")

    def _recycle_resources(self, task_id, index, task, result=None, is_prefill=False):
        """
        recycle resources
        """
        if is_prefill:
            start_time = time.time()
            result.metrics.wait_for_sending_cache_time = time.time()
            trace_print(LoggingEventName.CHECK_CACHE_TRANSFER_START, task_id, getattr(task, "user", ""))

            while True:
                finished_task_ids = self.engine_worker_queue.get_finished_req()
                if len(finished_task_ids) > 0:
                    for finished_task_id in finished_task_ids:
                        llm_logger.info(f"finished_task_id: {finished_task_id}")
                        self.prefill_result_status[finished_task_id[0]] = finished_task_id[1]
                if task_id in self.prefill_result_status:
                    if self.prefill_result_status[task_id] != "finished":
                        result.error_code = 501
                        result.error_msg = (
                            f"PD Error: prefill failed to send cache to decode, "
                            f"{task_id}, {self.prefill_result_status[task_id]}"
                        )
                    self.prefill_result_status.pop(task_id)
                    llm_logger.info(
                        f"wait for sending cache, request_id: {task_id}, cost seconds: {time.time()-start_time:.5f}"
                    )
                    trace_print(LoggingEventName.CHECK_CACHE_TRANSFER_END, task_id, getattr(task, "user", ""))
                    result.metrics.send_request_output_to_decode_time = time.time()
                    self.split_connector.send_first_token(task.disaggregate_info, [result])
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                        self.resource_manager.finish_requests_async(task_id)
                    else:
                        self.resource_manager.stop_flags[index] = True
                        self.resource_manager.tasks_list[index] = None
                        self.resource_manager._recycle_block_tables(task)
                        if task_id in self.resource_manager.req_dict:
                            del self.resource_manager.req_dict[task_id]
                    break
                else:
                    # TODO: Refine checking sending cache and do not keep waiting
                    if time.time() - start_time > 30:
                        llm_logger.warning(f"wait for sending cache, {task_id}")
                    time.sleep(0.005)
        else:
            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                self.resource_manager.finish_requests_async(task_id)
            else:
                self.resource_manager.stop_flags[index] = True
                self.resource_manager.tasks_list[index] = None
                self.resource_manager._recycle_block_tables(task)
                if task_id in self.resource_manager.req_dict:
                    del self.resource_manager.req_dict[task_id]

        # Update block metrics
        num_blocks_used_by_tasks = sum(
            [len(task.block_tables) if task else 0 for task in self.resource_manager.tasks_list]
        )
        main_process_metrics.available_gpu_block_num.set(
            self.resource_manager.total_block_number() - num_blocks_used_by_tasks
        )
        main_process_metrics.batch_size.set(
            self.resource_manager.max_num_seqs - self.resource_manager.available_batch()
        )
        main_process_metrics.available_batch_size.set(self.resource_manager.available_batch())

        if task_id in self.tokens_counter:
            del self.tokens_counter[task_id]

    def _compute_speculative_status(self, result: RequestOutput):
        # TODO(liuzichang): Supplement more statistics
        interval = 1
        if self.speculative_stats_step % interval == 0:
            accept_ratio = 1 - self.total_step * 1.0 / self.number_of_output_tokens
            spec_logger.info(
                f"Speculate global accept ratio(Accept draft_tokens/Generated tokens): {accept_ratio}"
                f" total step: {self.total_step}. total output token num: {self.number_of_output_tokens}"
                f" average accept len: {self.number_of_output_tokens / self.total_step}"
            )

            if self.cfg.speculative_config.method == SpecMethod.MTP:
                single_head_acceptance_rates = []
                for i in range(1, self.cfg.speculative_config.num_speculative_tokens + 1):
                    if self.accept_token_num_per_head[i - 1] != 0:
                        single_head_acceptance_rates.append(
                            self.accept_token_num_per_head[i] / self.accept_token_num_per_head[i - 1]
                        )
                spec_logger.info(f" Single head accept ratio: {single_head_acceptance_rates}")

            if self.number_of_output_tokens > 1000000:
                self.number_of_output_tokens = 0
                self.total_step = 0
        self.speculative_stats_step += 1

        # For result
        req_id = result.request_id
        accept_num_list = self.accept_token_num_per_head_per_request[req_id]
        req_total_step = self.total_step_per_request[req_id]
        req_total_draft_tokens = req_total_step * (self.cfg.speculative_config.num_speculative_tokens + 1)
        req_accepted_tokens = sum(accept_num_list)
        req_rejected_tokens = req_total_draft_tokens - req_accepted_tokens
        req_accept_ratio = 1 - req_total_step / req_accepted_tokens
        req_avg_accept_length = req_accepted_tokens / req_total_step

        accept_ratio_per_head = []
        for i in range(1, len(accept_num_list)):
            if accept_num_list[i - 1] != 0:
                accept_ratio_per_head.append(accept_num_list[i] / accept_num_list[i - 1])
            else:
                accept_ratio_per_head.append(0)

        result.metrics.speculate_metrics = SpeculateMetrics(
            accepted_tokens=req_accepted_tokens,
            rejected_tokens=req_rejected_tokens,
            accept_ratio=req_accept_ratio,
            average_accept_length=req_avg_accept_length,
            accepted_tokens_per_head=accept_num_list[: self.cfg.speculative_config.num_speculative_tokens + 1],
            accept_ratio_per_head=accept_ratio_per_head[: self.cfg.speculative_config.num_speculative_tokens],
        )

        # Log
        spec_logger.info(
            f"req_id: {result.request_id}, total_step: {req_total_step}, "
            f"accept_ratio: {accept_ratio}, average_accept_length: {req_avg_accept_length}, "
            f"accepted_tokens: {req_accepted_tokens}, rejected_tokens: {req_rejected_tokens}, "
            f"accepted_tokens_per_head: {accept_num_list[: self.cfg.speculative_config.num_speculative_tokens + 1]}, "
            f"accept_ratio_per_head: {accept_ratio_per_head[: self.cfg.speculative_config.num_speculative_tokens]}"
        )

        # Clear request record
        self.accept_token_num_per_head_per_request.pop(req_id)
        self.total_step_per_request.pop(req_id)

    def _process_batch_draft_tokens(self, mtype, batch, accept_num, tokens, scores, ranks):
        """
        Process batch draft tokens and generate corresponding request outputs

        Args:
            mtype (int): Message type (3=target token, 4=draft token)
            batch (int): Batch size
            accept_num (list): List of accepted token counts per request
            tokens (paddle.Tensor): Generated draft token IDs tensor
            scores (paddle.Tensor): Token scores tensor
            ranks (paddle.Tensor): Token sampling ranks tensor

        Returns:
            list[RequestOutput]: List containing processed results for all requests
        """
        batch_result = list()
        for i in range(batch):
            if self.resource_manager.stop_flags[i]:
                continue
            task = self.resource_manager.tasks_list[i]
            task_id = task.request_id
            result = RequestOutput(
                request_id=task_id,
                output_type=mtype,
                outputs=CompletionOutput(
                    index=i,
                    send_idx=None,
                    token_ids=[],
                    draft_token_ids=[],
                ),
                finished=False,
                metrics=None,
            )

            tokens_i = tokens[i].tolist()
            scores_i = scores[i].tolist()
            ranks_i = ranks[i].tolist()
            token_ids = [row[0] for row in tokens_i[: accept_num[i]]]
            for batch_token_index in range(len(token_ids)):
                result.outputs.logprob = scores_i[batch_token_index][0]
                topk_token_ids = tokens_i[batch_token_index]
                topk_logprobs = scores_i[batch_token_index]
                sampled_rank = ranks_i[batch_token_index]

                if result.outputs.draft_top_logprobs is None:
                    result.outputs.draft_top_logprobs = LogprobsLists(
                        logprob_token_ids=[topk_token_ids],
                        logprobs=[topk_logprobs],
                        sampled_token_ranks=[sampled_rank],
                    )
                else:
                    result.outputs.draft_top_logprobs.logprob_token_ids.extend([topk_token_ids])
                    result.outputs.draft_top_logprobs.logprobs.extend([topk_logprobs])
                    result.outputs.draft_top_logprobs.sampled_token_ranks.extend([sampled_rank])
            batch_result.append(result)
        return batch_result

    def _process_batch_output(self):
        """
        batch post-processing function
        """

        tokens = self.output_tokens.numpy()
        scores = None
        ranks = None
        # target:3, draft:4
        mtype = 3
        if self.cfg.speculative_config.method:
            if self.use_logprobs:
                # meta[1] packs message_flag (low 8 bits) and actual_topk (high 24 bits).
                packed_meta1 = int(self.output_tokens[1, 0].item())
                mtype = packed_meta1 & 0xFF
                actual_topk = packed_meta1 >> 8
                batch = self.output_tokens[2, 0]
                accept_num = [int(num[0]) for num in self.output_tokens[3 : batch + 3]]
                tokens = tokens[3 + MAX_BSZ : 3 + MAX_BSZ + batch * MAX_DRAFT_TOKENS * (K + 1)].reshape(
                    [batch, MAX_DRAFT_TOKENS, K + 1]
                )[:, :, :actual_topk]
                scores = (
                    self.output_scores[: batch * MAX_DRAFT_TOKENS * (K + 1)]
                    .numpy()
                    .reshape([batch, MAX_DRAFT_TOKENS, K + 1])[:, :, :actual_topk]
                )
                ranks = self.output_ranks[: batch * MAX_DRAFT_TOKENS].numpy().reshape([batch, MAX_DRAFT_TOKENS])

                # split draft_tokens into standalone post-processing path for MTP + logprobs
                if mtype == 4:
                    batch_result = self._process_batch_draft_tokens(mtype, batch, accept_num, tokens, scores, ranks)
                    self.postprocess(batch_result, mtype)
                    return
                # Pre-convert full arrays to Python lists once for MTP target token path.
                tokens_lists = tokens.tolist()
                scores_lists = scores.tolist()
                ranks_list = ranks.tolist()
            else:
                batch = self.output_tokens[1]
                accept_num = tokens[2 : batch + 2]
        elif self.use_logprobs:
            # mtext[1] packs bsz (low 16 bits) and actual_topk (high 16 bits).
            # actual_topk = max_num_logprobs written by save_output_topk, which
            # equals the actual number of logprob columns in this step's message
            # (top_logprobs+1 across the batch). Using actual_topk as stride
            # avoids processing the K+1=21 fixed-size slots when fewer are needed.
            packed = int(self.output_tokens[1, 0])
            batch = packed & 0xFFFF
            actual_topk = (packed >> 16) & 0xFFFF
            tokens = tokens[2 : batch * actual_topk + 2].reshape([batch, actual_topk])
            scores = self.output_scores[: batch * actual_topk].numpy().reshape([batch, actual_topk])
            ranks = self.output_ranks[:batch].numpy()
            # Pre-convert the full [batch, actual_topk] arrays to Python lists once,
            # avoiding per-row .tolist() calls inside the loop below.
            tokens_lists = tokens.tolist()
            scores_lists = scores.tolist()
            ranks_list = ranks.tolist()
        else:
            batch = self.output_tokens[1, 0]
            tokens = tokens[2 : batch + 2]

        # Receive sampling constraints per request from ZMQ side-channel (if enabled).
        # The worker sends a dict {batch_id: sparse_vocab_indices} each step,
        # where the value is a list[int] or list[list[int]] of allowed token ids
        sampling_masks_per_request = {}
        if self.use_sampling_mask and not envs.FD_USE_GET_SAVE_OUTPUT_V1 and hasattr(self, "sampling_mask_zmq_server"):
            _, mask_data = self.sampling_mask_zmq_server.receive_pyobj_once(block=True)
            if mask_data is not None and isinstance(mask_data, dict):
                sampling_masks_per_request = mask_data

        batch_result = list()
        # reschedule
        for i in range(batch):
            if self.resource_manager.stop_flags[i] or self.resource_manager.tasks_list[i] is None:
                continue

            recovery_stop = False
            task = self.resource_manager.tasks_list[i]
            task_id = task.request_id
            is_prefill = task.disaggregate_info is not None and self.cfg.scheduler_config.splitwise_role == "prefill"
            is_decode = task.disaggregate_info is not None and self.cfg.scheduler_config.splitwise_role == "decode"

            rid = task_id.split("_")[0]
            trace_carrier = task.trace_carrier
            metrics = task.metrics
            t = metrics.inference_start_time
            ts = int(t * 1_000_000_000) if t is not None else 0
            tracing.trace_set_proc_propagate_context(rid, trace_carrier, ts)
            if self.cfg.speculative_config.method:
                self._record_speculative_decoding_accept_num_per_request(task_id, accept_num[i])
                if accept_num[i] == PREEMPTED_TOKEN_ID:  # in MTP, means preemption has happened in worker
                    llm_logger.info(f"sync preemption for request_id {task_id} done.")
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                        if task_id in self.resource_manager.to_be_aborted_req_id_set:
                            self.resource_manager.recycle_abort_task(task_id)
                            self._put_abort_results(task)
                        if task_id in self.resource_manager.to_be_rescheduled_request_id_set:
                            self.resource_manager.reschedule_preempt_task(task_id)
                    continue
                if accept_num[i] == -3:
                    recovery_stop = True
                    if recovery_stop:
                        llm_logger.info(f"recovery stop signal found at task {task_id}")
                    token_ids = [RECOVERY_STOP_SIGNAL]
                elif self.use_logprobs:
                    token_ids = [row[0] for row in tokens_lists[i][: accept_num[i]]]
                else:
                    token_ids = tokens[
                        2
                        + SPECULATE_MAX_BSZ
                        + i * MAX_DRAFT_TOKENS : 2
                        + SPECULATE_MAX_BSZ
                        + i * MAX_DRAFT_TOKENS
                        + accept_num[i]
                    ].tolist()
                if accept_num[i] == 0:
                    continue
            else:
                token_id = int(tokens[i, 0])
                token_ids = [token_id]
                recovery_stop = token_id == RECOVERY_STOP_SIGNAL
                if recovery_stop:
                    llm_logger.info(f"recovery stop signal found at task {task_id}")
                if not recovery_stop and token_id < 0:
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                        if (
                            task_id in self.resource_manager.to_be_aborted_req_id_set
                            and token_id == PREEMPTED_TOKEN_ID
                        ):
                            self.resource_manager.recycle_abort_task(task_id)
                            self._put_abort_results(task)
                            llm_logger.info(f"sync abortion for request_id {task_id} done.")
                        if (
                            task_id in self.resource_manager.to_be_rescheduled_request_id_set
                            and token_id == PREEMPTED_TOKEN_ID
                        ):
                            llm_logger.info(f"sync preemption for request_id {task_id} done.")
                            self.resource_manager.reschedule_preempt_task(task_id)
                    continue
            if self.cfg.scheduler_config.splitwise_role == "decode":
                # In D instance, if preempted, error has been reported and resource recycled, tokens generated async not need to be handled
                if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                    if task_id in self.resource_manager.to_be_rescheduled_request_id_set:
                        continue
                    if task_id in self.resource_manager.to_be_aborted_req_id_set:
                        continue

            if self.scheduler_metrics_logger and self._is_decode_stage(task):
                self.scheduler_metrics_logger.on_decode_tokens(len(token_ids))

            if task.get("prefill_chunk_info", None) is not None:
                prefill_chunk_num = task.get("prefill_chunk_num", 0)
                task.prefill_chunk_num = prefill_chunk_num + 1

                if task.prefill_chunk_num < len(task.prefill_chunk_info):
                    continue

            self.total_step += 1
            if task.status == RequestStatus.RUNNING_PREFILL:
                task.status = RequestStatus.RUNNING_DECODE
            current_time = time.time()
            trace_carrier = None
            if self.tokens_counter[task_id] == 0:
                task.metrics.record_recv_first_token()
                task.metrics.cal_cost_time()
                metrics = copy.copy(task.metrics)
                llm_logger.info(f"task:{task.request_id} start recode first token")
                self._record_first_token_metrics(task, current_time)

                tracing.trace_report_span(
                    name=tracing.TraceSpanName.PREFILL,
                    rid=rid,
                    start_time_ns=int(task.metrics.inference_start_time * 1e9),
                    end_time_ns=int(time.time() * 1e9),
                    thread_finish_flag=False,
                )

            else:
                task.metrics.record_recv_token()
                if self.tokens_counter[task_id] == 1 and self.cfg.scheduler_config.splitwise_role == "decode":
                    task.metrics.record_decode_recv_second_token()
                metrics = copy.copy(task.metrics)

            self.number_of_output_tokens += len(token_ids)
            self._record_metrics(task, current_time, token_ids)
            result = RequestOutput(
                request_id=task_id,
                output_type=mtype,
                outputs=CompletionOutput(
                    index=i,
                    send_idx=self.tokens_counter[task_id],
                    token_ids=[],
                    draft_token_ids=[],
                ),
                finished=False,
                metrics=metrics,
                ic_req_data=task.ic_req_data,
                prompt_token_ids_len=task.prompt_token_ids_len,
                trace_carrier=trace_carrier,
            )
            if self.tokens_counter[task_id] == 0:
                if task.messages is not None:
                    result.prompt = task.messages
            result.num_cached_tokens = task.num_cached_tokens
            if task.get("multimodal_inputs", None):
                result.num_input_image_tokens = task.multimodal_inputs.get("num_input_image_tokens", 0)
                result.num_input_video_tokens = task.multimodal_inputs.get("num_input_video_tokens", 0)

            if self.use_sampling_mask and i in sampling_masks_per_request:
                result.outputs.sampling_mask = sampling_masks_per_request[i]

            if is_prefill and len(token_ids) > 1:
                result.outputs.draft_token_ids = copy.deepcopy(token_ids)

            for batch_token_index in range(len(token_ids)):
                token_id = token_ids[batch_token_index]
                self.tokens_counter[task_id] += 1
                if token_id != RECOVERY_STOP_SIGNAL:
                    if not (envs.FD_ENABLE_INTERNAL_ADAPTER and token_id in task.eos_token_ids):
                        result.outputs.token_ids.append(token_id)

                    task.output_token_ids.append(token_id)
                    if self.use_logprobs:
                        if self.cfg.speculative_config.method:
                            result.outputs.logprob = scores_lists[i][batch_token_index][0]
                            topk_token_ids = tokens_lists[i][batch_token_index]
                            topk_logprobs = scores_lists[i][batch_token_index]
                            sampled_rank = ranks_list[i][batch_token_index]
                        else:
                            # Use pre-converted lists (batch .tolist() done before the loop).
                            result.outputs.logprob = scores_lists[i][0]
                            topk_token_ids = tokens_lists[i]
                            topk_logprobs = scores_lists[i]
                            sampled_rank = ranks_list[i]

                        if result.outputs.top_logprobs is None:
                            result.outputs.top_logprobs = LogprobsLists(
                                logprob_token_ids=[topk_token_ids],
                                logprobs=[topk_logprobs],
                                sampled_token_ranks=[sampled_rank],
                            )
                        else:
                            result.outputs.top_logprobs.logprob_token_ids.extend([topk_token_ids])
                            result.outputs.top_logprobs.logprobs.extend([topk_logprobs])
                            result.outputs.top_logprobs.sampled_token_ranks.extend([sampled_rank])
                if token_id in task.eos_token_ids or is_prefill or recovery_stop:
                    result.finished = True
                    trace_carrier = tracing.trace_get_proc_propagate_context(rid=rid)
                    result.trace_carrier = trace_carrier
                    tracing.trace_report_span(
                        name=tracing.TraceSpanName.DECODE,
                        rid=rid,
                        start_time_ns=int(task.metrics.inference_start_time * 1e9),
                        end_time_ns=int(time.time() * 1e9),
                        thread_finish_flag=True,
                    )
                    if recovery_stop:
                        result.error_msg = "Recover is not supported, the result is incomplete!"

                    # Calculate statistics for the combined log
                    inference_start_time = task.metrics.get_inference_start_time(is_decode)
                    task.metrics.cal_cost_time()
                    e2e_time = current_time - inference_start_time
                    token_ratio = self.tokens_counter[task_id] / e2e_time

                    # Get cache information
                    gpu_cache = getattr(task.metrics, "gpu_cache_token_num", 0)
                    cpu_cache = getattr(task.metrics, "cpu_cache_token_num", 0)
                    total_cached = gpu_cache + cpu_cache

                    # Build cached detail dict
                    cached_detail = f'{{"CachedToken": {total_cached}, "GPU": {gpu_cache}, "CPU": {cpu_cache}}}'

                    # Print combined log with all required information
                    ttft = task.metrics.first_token_time if task.metrics.first_token_time else 0
                    ttft_s = ttft + task.metrics.time_in_queue
                    llm_logger.info(
                        f"Request={task_id}, InputToken={task.prompt_token_ids_len}, "
                        f"CachedDetail={cached_detail}, OutputToken={self.tokens_counter[task_id]}, "
                        f"TokenRatio={token_ratio:.2f}, TTFT={ttft:.2f}, TTFT_S={ttft_s:.2f}, "
                        f"E2E={e2e_time:.2f}, IsPrefill={is_prefill}, RecoveryStop={recovery_stop}, "
                        f"PreemptedCount={getattr(task.metrics, 'preempted_count', 0)}"
                    )

                    main_process_metrics.request_token_ratio.observe(token_ratio)
                    llm_logger.info(f"{self.resource_manager.info()}")
                    if self.cfg.speculative_config.method:
                        self._compute_speculative_status(result)
                    self._record_completion_metrics(task, current_time)
                    llm_logger.info(f"task {task_id} received eos token. Recycling.")
                    if (
                        envs.ENABLE_V1_KVCACHE_SCHEDULER
                        and self.cfg.cache_config.enable_prefix_caching
                        and self.cfg.cache_config.enable_output_caching
                    ):
                        self.resource_manager.cache_output_tokens(
                            task
                        )  # when enable prefix caching, cache kv cache for output tokens
                    self._finalize_routing(task_id, task, result, is_prefill)
                    self._recycle_resources(task_id, i, task, result, is_prefill)
                    llm_logger.info(f"eos token {task_id} Recycle end.")
                    break

            llm_logger.debug(f"get response from infer: {result}")
            batch_result.append(result)

        if self.cfg.speculative_config.method:
            self._record_speculative_decoding_metrics(accept_num)
        self.postprocess(batch_result, mtype)

    def _record_metrics(self, task, current_time, token_ids):
        """Record all metrics for a task"""
        if hasattr(task, "last_token_time") and task.last_token_time is not None:
            token_gen_time = current_time - task.last_token_time
            main_process_metrics.time_per_output_token.observe(token_gen_time)
            if self._benchmark_logger:
                if not hasattr(task, "_itl_samples"):
                    task._itl_samples = []
                task._itl_samples.append(token_gen_time)
        task.last_token_time = current_time

        # Record generation metrics
        main_process_metrics.generation_tokens_total.inc(len(token_ids))

    def _record_first_token_metrics(self, task, current_time):
        """Record metrics for first token"""
        metrics = task.metrics
        trace_print(LoggingEventName.FIRST_TOKEN_GENERATED, task.request_id, getattr(task, "user", ""))
        trace_print(LoggingEventName.DECODE_START, task.request_id, getattr(task, "user", ""))
        main_process_metrics.time_to_first_token.observe(current_time - metrics.arrival_time)
        main_process_metrics.request_queue_time.observe(metrics.inference_start_time - metrics.preprocess_end_time)
        main_process_metrics.request_prefill_time.observe(current_time - metrics.inference_start_time)

    def _record_completion_metrics(self, task, current_time):
        """Record metrics when request completes"""
        role = self.cfg.scheduler_config.splitwise_role
        metrics = task.metrics

        if role in ("mixed", "decode"):
            if metrics.engine_recv_first_token_time:
                decode_time = current_time - metrics.engine_recv_first_token_time
                main_process_metrics.request_decode_time.observe(decode_time)
            trace_print(LoggingEventName.INFERENCE_END, task.request_id, getattr(task, "user", ""))

        if role == "prefill":
            trace_print(LoggingEventName.PREFILL_INFERENCE_END, task.request_id, getattr(task, "user", ""))
        elif role == "decode":
            trace_print(LoggingEventName.DECODE_INFERENCE_END, task.request_id, getattr(task, "user", ""))

        trace_print(LoggingEventName.POSTPROCESSING_START, task.request_id, getattr(task, "user", ""))
        main_process_metrics.request_success_total.inc()
        main_process_metrics.request_inference_time.observe(current_time - metrics.inference_start_time)
        main_process_metrics.request_generation_tokens.observe(self.tokens_counter[task.request_id])

        if self._benchmark_logger:
            from fastdeploy.metrics.benchmark_metrics_logger import (
                CompletedRequestRecord,
            )

            record = CompletedRequestRecord(
                request_id=task.request_id,
                completion_time=current_time,
                arrival_time=metrics.arrival_time or 0.0,
                inference_start_time=metrics.inference_start_time or 0.0,
                first_token_time=metrics.engine_recv_first_token_time or 0.0,
                last_token_time=metrics.engine_recv_latest_token_time or current_time,
                input_len=getattr(task, "prompt_token_ids_len", 0) or 0,
                output_len=self.tokens_counter[task.request_id],
                num_cached_tokens=getattr(task, "num_cached_tokens", 0) or 0,
                itl_samples=getattr(task, "_itl_samples", []),
            )
            self._benchmark_logger.on_request_completed(record)

    def _record_speculative_decoding_metrics(self, accept_num):
        """Record metrics of speculative decoding"""
        if not hasattr(main_process_metrics, "spec_decode_draft_acceptance_rate"):
            main_process_metrics._init_speculative_metrics(
                self.cfg.speculative_config.method,
                self.cfg.speculative_config.num_speculative_tokens,
            )

        real_accept_num = [x for x in accept_num if x > 0]
        self.num_accepted_tokens = sum(self.accept_token_num_per_head[1:])
        self.num_emitted_tokens = sum(self.accept_token_num_per_head)
        if self.num_emitted_tokens == 0:
            return

        main_process_metrics.spec_decode_num_accepted_tokens_total.set(self.num_accepted_tokens)
        main_process_metrics.spec_decode_num_emitted_tokens_total.set(self.num_emitted_tokens)

        if self.cfg.speculative_config.method == SpecMethod.NGRAM:
            main_process_metrics.spec_decode_draft_acceptance_rate.set(
                self.num_accepted_tokens / self.num_emitted_tokens
            )

        if self.cfg.speculative_config.method == SpecMethod.MTP:
            num_draft_tokens = len(real_accept_num) * self.cfg.speculative_config.num_speculative_tokens
            self.num_draft_tokens += num_draft_tokens

            self.max_num_emitted_tokens += len(real_accept_num) * (
                self.cfg.speculative_config.num_speculative_tokens + 1
            )

            main_process_metrics.spec_decode_draft_acceptance_rate.set(
                self.num_accepted_tokens / self.num_draft_tokens
            )
            main_process_metrics.spec_decode_efficiency.set(self.num_emitted_tokens / self.max_num_emitted_tokens)
            main_process_metrics.spec_decode_num_draft_tokens_total.inc(num_draft_tokens)

            for i in range(1, self.cfg.speculative_config.num_speculative_tokens + 1):
                if self.accept_token_num_per_head[i - 1] != 0:
                    single_head_acceptance_rate = (
                        self.accept_token_num_per_head[i] / self.accept_token_num_per_head[i - 1]
                    )
                main_process_metrics.spec_decode_draft_single_head_acceptance_rate[i - 1].set(
                    single_head_acceptance_rate
                )

    def _record_speculative_decoding_accept_num_per_request(self, req_id, accept_num):
        if req_id not in self.total_step_per_request:
            self.total_step_per_request[req_id] = 0
        if req_id not in self.accept_token_num_per_head_per_request:
            self.accept_token_num_per_head_per_request[req_id] = [0] * MAX_DRAFT_TOKENS

        self.total_step_per_request[req_id] += 1
        for i in range(accept_num):
            self.accept_token_num_per_head_per_request[req_id][i] += 1
            self.accept_token_num_per_head[i] += 1

    def _put_abort_results(self, task):
        now = time.time()
        eos_token_ids = getattr(task, "eos_token_ids", [0])
        abort_metrics = copy.copy(task.metrics)
        for field in (
            "arrival_time",
            "inference_start_time",
            "engine_recv_latest_token_time",
            "engine_recv_first_token_time",
            "request_start_time",
        ):
            if not getattr(abort_metrics, field):
                setattr(abort_metrics, field, now)
        result = RequestOutput(
            request_id=task.request_id,
            finished=True,
            outputs=CompletionOutput(
                index=0,
                send_idx=self.tokens_counter.get(task.request_id),
                token_ids=[eos_token_ids[0]],
            ),
            metrics=abort_metrics,
            error_code=200,
            error_msg="Aborted",
        )
        self.cached_generated_tokens.put_results([result])

    def clear_data(self):
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.resource_manager.clear_data()
        for i in range(self.resource_manager.max_num_seqs):
            if self.resource_manager.stop_flags[i]:
                continue
            task = self.resource_manager.tasks_list[i]
            result = RequestOutput(
                request_id=task.request_id,
                outputs=CompletionOutput(
                    index=i,
                    send_idx=self.tokens_counter[task.request_id],
                    token_ids=task.eos_token_ids,
                    draft_token_ids=[],
                ),
                finished=True,
                metrics=RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.metrics.arrival_time,
                ),
            )
            is_prefill = task.disaggregate_info is not None and task.disaggregate_info["role"] == "prefill"
            self._finalize_routing(task.request_id, task, result, is_prefill)
            self._recycle_resources(task.request_id, i, task, result, is_prefill)
            llm_logger.warning(f"clear data for task {task.request_id}")


class WarmUpTokenProcessor(TokenProcessor):
    """
    Warmup Processor
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        self._is_running = True
        self._is_blocking = True

    def postprocess(self, batch_result):
        pass

    def process_sampling_results(self):
        """
        get output from model and process it
        """

        if current_platform.is_xpu():
            from fastdeploy.model_executor.ops.xpu import get_output
        elif current_platform.is_iluvatar():
            from fastdeploy.model_executor.ops.iluvatar import get_output
        else:
            from fastdeploy.model_executor.ops.gpu import (
                get_output,
                speculate_get_output,
            )

        while self._is_running:
            try:
                rank_id = 0
                if self.speculative_decoding:
                    speculate_get_output(self.output_tokens, rank_id, self._is_blocking)
                    if self.output_tokens[0] == -2:
                        continue
                else:
                    get_output(self.output_tokens, rank_id, self._is_blocking)

                    if self.output_tokens[0, 0] == -2:
                        continue
                self._process_batch_output()
            except Exception as e:
                llm_logger.info(f"while get input_data error: {e} {traceback.format_exc()!s}")

    def stop(self):
        """
        stop warm up thread
        """
        self._is_running = False
        self.worker.join()
        llm_logger.info("warm up thread stop")
        del self.worker
