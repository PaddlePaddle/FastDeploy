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
import os
import threading
import time
import traceback
import weakref
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import paddle
import zmq

from fastdeploy import envs
from fastdeploy.engine.request import CompletionOutput, RequestMetrics, RequestOutput
from fastdeploy.inter_communicator import IPCSignal, ZmqIpcServer
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.platforms import current_platform
from fastdeploy.utils import llm_logger, spec_logger
from fastdeploy.worker.output import LogprobsLists

RECOVERY_STOP_SIGNAL = -3
MAX_BSZ = 512
K = 20
MAX_DRAFT_TOKENS = 6
SPECULATE_MAX_BSZ = 256


class TokenProcessor:
    """
    get Token/Score from Paddle inference engine
    """

    def __init__(self, cfg, cached_generated_tokens, engine_worker_queue, split_connector):

        paddle.device.set_device("cpu")
        self.cfg = cfg
        self.cached_generated_tokens = cached_generated_tokens
        self.resource_manager = None
        self.engine_worker_queue = engine_worker_queue
        self.tokens_counter = Counter()
        self.split_connector = split_connector

        if envs.FD_USE_GET_SAVE_OUTPUT_V1:
            llm_logger.debug(f"create zmq get_save_output_rank{self.cfg.parallel_config.local_data_parallel_id}")
            self.zmq_server = ZmqIpcServer(
                name=f"get_save_output_rank{self.cfg.parallel_config.local_data_parallel_id}", mode=zmq.PULL
            )

        self.speculative_decoding = self.cfg.speculative_config.method is not None
        self.use_logprobs = self.cfg.model_config.enable_logprob

        if self.speculative_decoding:
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
        self.num_rest_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        self.num_accept_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        prefill_time_data = np.zeros([100], dtype=np.float32)
        self.prefill_time_signal = IPCSignal(
            name="prefill_time_signal",
            array=prefill_time_data,
            dtype=np.float32,
            suffix=os.getpid(),
            create=True,
        )
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefill_result_status = dict()
        self._finalizer = weakref.finalize(self, self._cleanup_resources)

    def _cleanup_resources(self):
        """Cleaning up shared memory resources"""
        if hasattr(self, "prefill_time_signal"):
            self.prefill_time_signal.clear()

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

    def _reschedule_preempt_task(self, batch_size):
        """reschedule when real batch size is smaller than the insert position of preemted_task"""
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            need_to_be_reschedule_req_ids = list(self.resource_manager.to_be_rescheduled_request_id_set)
            for request_id in need_to_be_reschedule_req_ids:
                if self.resource_manager.requests[request_id].idx >= (
                    batch_size - 1
                ):  # No more token generated for preempted request
                    self.resource_manager.reschedule_preempt_task(request_id)

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
                llm_logger.info(
                    f"Request: {task_id} finished, number of " f"generated tokens: {self.tokens_counter[task_id]}."
                )
                llm_logger.info(
                    f"Request: {task_id} token ratio: {self.tokens_counter[task_id] / (time.time() - task.inference_start_time)}"
                )
                llm_logger.info(f"{self.resource_manager.info()}")
                if self.cfg.speculative_config.method:
                    self._compute_speculative_status()
                if not is_prefill:
                    self._record_completion_metrics(task, current_time)
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

            task = self.resource_manager.tasks_list[i]

            task_id = task.request_id
            token_ids = stream_data.tokens  # numpy.array

            current_time = time.time()
            if self.tokens_counter[task_id] == 0:
                metrics = RequestMetrics(
                    arrival_time=task.arrival_time,
                    inference_start_time=task.inference_start_time,
                    first_token_time=time.time() - task.inference_start_time,
                    time_in_queue=task.schedule_start_time - task.preprocess_end_time,
                    preprocess_cost_time=task.preprocess_end_time - task.preprocess_start_time,
                    request_start_time=task.arrival_time,
                )
                self._record_first_token_metrics(task, current_time)

            else:
                metrics = RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.arrival_time,
                )

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
            )

            if self.tokens_counter[task_id] == 0:
                if task.messages is not None:
                    result.prompt = task.messages
                result.num_cached_tokens = task.num_cached_tokens

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
        if self.use_logprobs:
            raise NotImplementedError("GET_SAVE_OUTPUT_V1 does not support use_logprobs")
        rank_id = self.cfg.parallel_config.local_data_parallel_id
        while True:
            try:
                if (
                    self.cfg.parallel_config.enable_expert_parallel and self.cfg.parallel_config.data_parallel_size > 1
                ) or (rank_id == 0):
                    receive_datas = self.zmq_server.recv_pyobj()
                    assert isinstance(receive_datas, list)
                    llm_logger.debug(f"token_processor receive_data {receive_datas}")

                    batch_size = len(receive_datas)
                    self._reschedule_preempt_task(batch_size)

                    batch_result = self._process_batch_output_use_zmq(receive_datas)
                    self.postprocess(batch_result)
            except Exception as e:
                llm_logger.error(f"Recieve message error: {e}")
                continue

    def process_sampling_results(self):
        """
        read tokens from paddle inference engine and process
        """

        if current_platform.is_xpu():
            from fastdeploy.model_executor.ops.xpu import get_output, get_output_ep
        elif current_platform.is_iluvatar():
            from fastdeploy.model_executor.ops.iluvatar import get_output
        elif current_platform.is_gcu():
            from fastdeploy.model_executor.ops.gcu import get_output
        else:
            from fastdeploy.model_executor.ops.gpu import (
                get_output,
                get_output_ep,
                get_output_topk,
                speculate_get_output,
            )
        rank_id = self.cfg.parallel_config.local_data_parallel_id

        while True:
            try:
                is_blocking = True
                if self.speculative_decoding:
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
                    elif (
                        self.cfg.parallel_config.enable_expert_parallel
                        and self.cfg.parallel_config.data_parallel_size > 1
                    ):
                        get_output_ep(self.output_tokens, rank_id, is_blocking)

                    else:
                        get_output(self.output_tokens, rank_id, is_blocking)

                    if self.output_tokens[0, 0] == -2:
                        continue
                    llm_logger.debug(f"rank_id {rank_id} self.output_tokens[0, 0] {self.output_tokens[0, 0]}")
                self._process_prefill_metrics()
                self._process_batch_output()
            except Exception as e:
                llm_logger.info(f"while get input_data error: {e} {traceback.format_exc()!s}")

    def _process_prefill_metrics(self):
        """Asynchronous processing prefill time indicators"""

        def process_metrics():
            try:
                current_index = 0
                while current_index < len(self.prefill_time_signal.value):
                    prefill_time = self.prefill_time_signal.value[current_index]
                    if prefill_time > 0:
                        main_process_metrics.request_prefill_time.observe(prefill_time)
                        self.prefill_time_signal.value[current_index] = 0
                    current_index += 1
            except Exception as e:
                llm_logger.error(f"Error processing prefill metrics: {e}, {str(traceback.format_exc())}")

        self.executor.submit(process_metrics)

    def postprocess(self, batch_result):
        """
        single post-processing function

        Args:
            batch_result (list): batch results
        """
        try:
            self.cached_generated_tokens.put_results(batch_result)
        except Exception as e:
            llm_logger.error(f"Error in TokenProcessor's postprocess: {e}, {str(traceback.format_exc())}")

    def _recycle_resources(self, task_id, index, task, result=None, is_prefill=False):
        """
        recycle resources
        """
        if is_prefill:
            while True:
                finished_task_ids = self.engine_worker_queue.get_finished_req()
                if len(finished_task_ids) > 0:
                    for finished_task_id in finished_task_ids:
                        llm_logger.info(f"finished_task_id: {finished_task_id}")
                        self.prefill_result_status[finished_task_id[0]] = finished_task_id[1]
                if task_id in self.prefill_result_status:
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                        self.resource_manager.finish_requests_async(task_id)
                    else:
                        self.resource_manager.stop_flags[index] = True
                        self.resource_manager.tasks_list[index] = None
                        self.resource_manager._recycle_block_tables(task)
                        if task_id in self.resource_manager.req_dict:
                            del self.resource_manager.req_dict[task_id]
                    if self.prefill_result_status[task_id] != "finished":
                        result.error_code = 400
                        result.error_message = f"{task_id} failed to {self.prefill_result_status[task_id]}"
                    self.split_connector.send_first_token(task.disaggregate_info, [result])
                    break
                else:
                    time.sleep(0.002)
        else:
            if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                self.resource_manager.finish_requests_async(task_id)
            else:
                self.resource_manager.stop_flags[index] = True
                self.resource_manager.tasks_list[index] = None
                self.resource_manager._recycle_block_tables(task)
                if task_id in self.resource_manager.req_dict:
                    del self.resource_manager.req_dict[task_id]

        task_used_block_num = sum([len(task.block_tables) if task else 0 for task in self.resource_manager.tasks_list])
        main_process_metrics.available_gpu_block_num.set(
            self.resource_manager.total_block_number() - task_used_block_num
        )
        main_process_metrics.batch_size.set(
            self.resource_manager.max_num_seqs - self.resource_manager.available_batch()
        )
        main_process_metrics.available_batch_size.set(self.resource_manager.available_batch())

        if task_id in self.tokens_counter:
            del self.tokens_counter[task_id]

    def _compute_speculative_status(self):
        # TODO(liuzichang): Supplement more statistics
        interval = 10
        if self.speculative_stats_step % interval == 0:
            accept_ratio = 1 - self.total_step * 1.0 / self.number_of_output_tokens
            spec_logger.info(
                f"Speculate global accept ratio(Accept draft_tokens/Generated tokens): {accept_ratio}"
                f" total step: {self.total_step}. total output token num: {self.number_of_output_tokens}"
                f" average accept len: {self.number_of_output_tokens / self.total_step}"
            )

            if self.cfg.speculative_config.method in ["mtp"]:
                single_head_acceptance_rates = []
                for head in range(self.cfg.speculative_config.num_speculative_tokens):
                    if self.num_rest_requests_per_head[head] != 0:
                        single_head_acceptance_rates.append(
                            self.num_accept_requests_per_head[head] / self.num_rest_requests_per_head[head]
                        )
                    else:
                        single_head_acceptance_rates.append(0)
                spec_logger.info(f" Single head accept ratio: {single_head_acceptance_rates}")

            if self.number_of_output_tokens > 1000000:
                self.number_of_output_tokens = 0
                self.total_step = 0
        self.speculative_stats_step += 1

    def _process_batch_output(self):
        """
        batch post-processing function
        """

        tokens = self.output_tokens.numpy()
        scores = None
        ranks = None
        if self.cfg.speculative_config.method:
            batch = self.output_tokens[1]
            accept_num = tokens[2 : batch + 2]
            self._record_speculative_decoding_mertics(accept_num)
        elif self.use_logprobs:
            batch = self.output_tokens[1, 0]
            tokens = tokens[2 : batch * (K + 1) + 2].reshape([batch, K + 1])[:, : (K + 1)]
            scores = self.output_scores[: batch * (K + 1)].numpy().reshape([batch, K + 1])[:, : (K + 1)]
            ranks = self.output_ranks[:batch].numpy()
        else:
            batch = self.output_tokens[1, 0]
            tokens = tokens[2 : batch + 2]

        batch_result = list()
        # reschedule
        self._reschedule_preempt_task(batch)
        for i in range(batch):
            if self.resource_manager.stop_flags[i]:
                continue

            recovery_stop = False
            task = self.resource_manager.tasks_list[i]

            task_id = task.request_id
            if self.cfg.speculative_config.method:
                if accept_num[i] == -3:
                    recovery_stop = True
                    if recovery_stop:
                        llm_logger.info(f"recovery stop signal found at task {task_id}")
                    token_ids = [RECOVERY_STOP_SIGNAL]
                else:
                    token_ids = tokens[
                        2
                        + SPECULATE_MAX_BSZ
                        + i * MAX_DRAFT_TOKENS : 2
                        + SPECULATE_MAX_BSZ
                        + i * MAX_DRAFT_TOKENS
                        + accept_num[i]
                    ].tolist()
                    if (not recovery_stop) and (len(token_ids) == 0 or token_ids[-1] <= 0):
                        continue
            else:
                token_id = int(tokens[i, 0])
                token_ids = [token_id]
                recovery_stop = token_id == RECOVERY_STOP_SIGNAL
                if recovery_stop:
                    llm_logger.info(f"recovery stop signal found at task {task_id}")
                if not recovery_stop and token_id < 0:
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                        if task_id in self.resource_manager.to_be_rescheduled_request_id_set:
                            self.resource_manager.reschedule_preempt_task(task_id)
                    continue

            if task.get("prefill_chunk_info", None) is not None:
                prefill_chunk_num = task.get("prefill_chunk_num", 0)
                task.prefill_chunk_num = prefill_chunk_num + 1

                if task.prefill_chunk_num < len(task.prefill_chunk_info):
                    continue

            self.total_step += 1
            current_time = time.time()
            if self.tokens_counter[task_id] == 0:
                metrics = RequestMetrics(
                    arrival_time=task.arrival_time,
                    inference_start_time=task.inference_start_time,
                    model_execute_time=time.time() - task.inference_start_time,
                    first_token_time=time.time() - task.inference_start_time,
                    time_in_queue=task.schedule_start_time - task.preprocess_end_time,
                    preprocess_cost_time=task.preprocess_end_time - task.preprocess_start_time,
                    request_start_time=task.arrival_time,
                )

                self._record_first_token_metrics(task, current_time)

            else:
                metrics = RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.arrival_time,
                    model_execute_time=time.time() - task.inference_start_time,
                )
            self.number_of_output_tokens += len(token_ids)
            self._record_metrics(task, current_time, token_ids)
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
            )
            if self.tokens_counter[task_id] == 0:
                if task.messages is not None:
                    result.prompt = task.messages
            result.num_cached_tokens = task.num_cached_tokens

            is_prefill = task.disaggregate_info is not None and task.disaggregate_info["role"] == "prefill"

            if is_prefill and len(token_ids) > 1:
                result.outputs.draft_token_ids = copy.deepcopy(token_ids)

            for token_id in token_ids:
                self.tokens_counter[task_id] += 1
                if token_id != RECOVERY_STOP_SIGNAL:
                    if not (envs.FD_ENABLE_INTERNAL_ADAPTER and token_id in task.eos_token_ids):
                        result.outputs.token_ids.append(token_id)
                    task.output_token_ids.append(token_id)
                    if self.use_logprobs:
                        result.outputs.logprob = float(scores[i, 0])
                        # Construct top_logprobs
                        topk_token_ids = tokens[i, :].tolist()
                        topk_logprobs = scores[i, :].tolist()
                        sampled_rank = ranks[i].item()
                        result.outputs.top_logprobs = LogprobsLists(
                            logprob_token_ids=[topk_token_ids],
                            logprobs=[topk_logprobs],
                            sampled_token_ranks=[sampled_rank],
                        )
                if token_id in task.eos_token_ids or is_prefill or recovery_stop:
                    result.finished = True
                    if recovery_stop:
                        result.error_msg = "Recover is not supported, the result is incomplete!"
                    llm_logger.info(
                        f"Request: {task_id} finished, number of " f"generated tokens: {self.tokens_counter[task_id]}."
                    )
                    llm_logger.info(
                        f"Request: {task_id} token ratio: {self.tokens_counter[task_id] / (time.time() - task.inference_start_time)}"
                    )
                    llm_logger.info(f"{self.resource_manager.info()}")
                    if self.cfg.speculative_config.method:
                        self._compute_speculative_status()
                    if not is_prefill:
                        self._record_completion_metrics(task, current_time)
                    self._recycle_resources(task_id, i, task, result, is_prefill)
                    break
            if (
                not is_prefill
                or self.cfg.scheduler_config.name == "splitwise"
                or self.cfg.scheduler_config.name == "dp"
            ):
                batch_result.append(result)

        self.postprocess(batch_result)

    def _record_metrics(self, task, current_time, token_ids):
        """Record all metrics for a task"""
        if hasattr(task, "last_token_time") and task.last_token_time is not None:
            token_gen_time = current_time - task.last_token_time
            main_process_metrics.time_per_output_token.observe(token_gen_time)
        task.last_token_time = current_time

        # Record generation metrics
        main_process_metrics.generation_tokens_total.inc(len(token_ids))

    def _record_first_token_metrics(self, task, current_time):
        """Record metrics for first token"""
        task.first_token_time = current_time
        main_process_metrics.first_token_latency.set(current_time - task.inference_start_time)
        main_process_metrics.time_to_first_token.observe(current_time - task.inference_start_time)
        main_process_metrics.request_queue_time.observe(task.schedule_start_time - task.preprocess_end_time)

    def _record_completion_metrics(self, task, current_time):
        """Record metrics when request completes"""
        if hasattr(task, "first_token_time"):
            decode_time = current_time - task.first_token_time
            main_process_metrics.request_decode_time.observe(decode_time)

        main_process_metrics.num_requests_running.dec(1)
        main_process_metrics.request_success_total.inc()
        main_process_metrics.infer_latency.set(current_time - task.inference_start_time)
        main_process_metrics.request_inference_time.observe(current_time - task.inference_start_time)
        main_process_metrics.request_generation_tokens.observe(self.tokens_counter[task.request_id])

    def _record_speculative_decoding_mertics(self, accept_num):
        """Record metrics of speculative decoding"""
        if not hasattr(main_process_metrics, "spec_decode_draft_acceptance_rate"):
            main_process_metrics._init_speculative_metrics(
                self.cfg.speculative_config.method,
                self.cfg.speculative_config.num_speculative_tokens,
            )

        real_accept_num = [x for x in accept_num if x > 0]
        num_accepted_tokens = sum([x - 1 for x in real_accept_num])
        self.num_accepted_tokens += num_accepted_tokens
        num_emitted_tokens = sum(real_accept_num)
        self.num_emitted_tokens += num_emitted_tokens

        main_process_metrics.spec_decode_num_accepted_tokens_total.inc(num_accepted_tokens)
        main_process_metrics.spec_decode_num_emitted_tokens_total.inc(num_emitted_tokens)

        if self.cfg.speculative_config.method in ["ngram"]:
            main_process_metrics.spec_decode_draft_acceptance_rate.set(
                self.num_accepted_tokens / self.num_emitted_tokens
            )

        if self.cfg.speculative_config.method in ["mtp"]:
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

            num_rest_requests = len(real_accept_num)
            for head in range(self.cfg.speculative_config.num_speculative_tokens):
                num_accept_requests = len([x for x in real_accept_num if x >= head + 2])
                # Accumulate the number of requests for each head
                self.num_accept_requests_per_head[head] += num_accept_requests
                self.num_rest_requests_per_head[head] += num_rest_requests
                # Update the rest requests for each head
                num_rest_requests = num_accept_requests
                # Calculate the acceptance rate for each head
                if self.num_rest_requests_per_head[head] != 0:
                    single_head_acceptance_rate = (
                        self.num_accept_requests_per_head[head] / self.num_rest_requests_per_head[head]
                    )
                else:
                    single_head_acceptance_rate = 0
                main_process_metrics.spec_decode_draft_single_head_acceptance_rate[head].set(
                    single_head_acceptance_rate
                )

    def clear_data(self):
        if envs.ENABLE_V1_KVCACHE_SCHEDULER:
            self.resource_manager.clear_data()
        for i in range(self.cfg.max_num_seqs):
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
                    request_start_time=task.arrival_time,
                ),
            )
            is_prefill = task.disaggregate_info is not None and task.disaggregate_info["role"] == "prefill"
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
