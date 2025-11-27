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

from fastdeploy import envs
from fastdeploy.engine.request import (
    CompletionOutput,
    PoolingOutput,
    PoolingRequestOutput,
    Request,
    RequestMetrics,
    RequestOutput,
)
from fastdeploy.inter_communicator import ZmqIpcServer
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.platforms import current_platform
from fastdeploy.trace.constants import LoggingEventName
from fastdeploy.trace.trace_logger import print as trace_print
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
        self.num_rest_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        self.num_accept_requests_per_head = [
            0,
        ] * MAX_DRAFT_TOKENS
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.prefill_result_status = dict()
        self._finalizer = weakref.finalize(self, self._cleanup_resources)
        self._batch_result_buffer = None

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

            task: Request = self.resource_manager.tasks_list[i]

            task_id = task.request_id
            token_ids = stream_data.tokens  # numpy.array
            if token_ids is not None and token_ids[-1] <= 0:
                if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                    if task_id in self.resource_manager.to_be_rescheduled_request_id_set:
                        self.resource_manager.reschedule_preempt_task(task_id)
                continue

            current_time = time.time()
            if self.tokens_counter[task_id] == 0:
                metrics = RequestMetrics(
                    arrival_time=task.arrival_time,
                    inference_start_time=task.inference_start_time,
                    first_token_time=time.time() - task.inference_start_time,
                    time_in_queue=task.inference_start_time - task.preprocess_end_time,
                    preprocess_cost_time=task.preprocess_end_time - task.preprocess_start_time,
                    request_start_time=task.arrival_time,
                )
                self._record_first_token_metrics(task, current_time)

            else:
                metrics = RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.arrival_time,
                )

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
                            result.prompt_logprobs_tensors = stream_data.prompt_logprobs
                        except Exception as e:
                            llm_logger.warning(f"Failed to parse prompt_logprobs from StreamTransferData: {e}")
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
                self._process_batch_output()
            except Exception as e:
                llm_logger.info(f"while get input_data error: {e} {traceback.format_exc()!s}")

    def postprocess(self, batch_result: List[RequestOutput], mtype=3):
        """
        single post-processing function

        Args:
            batch_result (list): batch results
        """
        try:
            if self.cfg.speculative_config.method and self.use_logprobs:
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

    def _recycle_resources(self, task_id, index, task, result=None, is_prefill=False):
        """
        recycle resources
        """
        if is_prefill:
            start_time = time.time()
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
                    llm_logger.info(
                        f"wait for sending cache, request_id: {task_id}, cost seconds: {time.time()-start_time:.5f}"
                    )
                    self.split_connector.send_first_token(task.disaggregate_info, [result])
                    break
                else:
                    # TODO: Refine checking sending cache and do not keep waiting
                    if time.time() - start_time > 30:
                        llm_logger.warning(f"wait for sending cache, {task_id}")
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

    def _compute_speculative_status(self):
        # TODO(liuzichang): Supplement more statistics
        interval = 1
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

            token_ids = tokens[i][:, 0].tolist()[: accept_num[i]]
            for batch_token_index in range(len(token_ids)):
                result.outputs.logprob = float(scores[i, batch_token_index, 0])
                topk_token_ids = tokens[i, batch_token_index, :].tolist()
                topk_logprobs = scores[i, batch_token_index, :].tolist()
                sampled_rank = ranks[i, batch_token_index].item()

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
                mtype = int(self.output_tokens[1, 0].item())
                batch = self.output_tokens[2, 0]
                accept_num = [int(num[0]) for num in self.output_tokens[3 : batch + 3]]
                tokens = tokens[3 + MAX_BSZ : 3 + MAX_BSZ + batch * MAX_DRAFT_TOKENS * (K + 1)].reshape(
                    [batch, MAX_DRAFT_TOKENS, K + 1]
                )
                scores = (
                    self.output_scores[: batch * MAX_DRAFT_TOKENS * (K + 1)]
                    .numpy()
                    .reshape([batch, MAX_DRAFT_TOKENS, K + 1])
                )
                ranks = self.output_ranks[: batch * MAX_DRAFT_TOKENS].numpy().reshape([batch, MAX_DRAFT_TOKENS])

                # split draft_tokens into standalone post-processing path for MTP + logprobs
                if mtype == 4:
                    batch_result = self._process_batch_draft_tokens(mtype, batch, accept_num, tokens, scores, ranks)
                    self.postprocess(batch_result, mtype)
                    return
            else:
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
                elif self.use_logprobs:
                    token_ids = tokens[i][:, 0].tolist()[: accept_num[i]]
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
                    if envs.ENABLE_V1_KVCACHE_SCHEDULER:
                        if task_id in self.resource_manager.to_be_rescheduled_request_id_set:
                            self.resource_manager.reschedule_preempt_task(task_id)
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
                    time_in_queue=task.inference_start_time - task.preprocess_end_time,
                    preprocess_cost_time=task.preprocess_end_time - task.preprocess_start_time,
                    request_start_time=task.arrival_time,
                    llm_engine_recv_req_timestamp=task.llm_engine_recv_req_timestamp,
                    llm_engine_send_req_to_engine_timestamp=task.inference_start_time,
                    llm_engine_recv_token_timestamp=time.time(),
                )
                self._record_first_token_metrics(task, current_time)

            else:
                metrics = RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.arrival_time,
                    model_execute_time=time.time() - task.inference_start_time,
                    llm_engine_recv_req_timestamp=task.llm_engine_recv_req_timestamp,
                    llm_engine_send_req_to_engine_timestamp=task.inference_start_time,
                    llm_engine_recv_token_timestamp=time.time(),
                )
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
            )
            if self.tokens_counter[task_id] == 0:
                if task.messages is not None:
                    result.prompt = task.messages
            result.num_cached_tokens = task.num_cached_tokens
            if task.get("multimodal_inputs", None):
                result.num_input_image_tokens = task.multimodal_inputs.get("num_input_image_tokens", 0)
                result.num_input_video_tokens = task.multimodal_inputs.get("num_input_video_tokens", 0)

            is_prefill = task.disaggregate_info is not None and task.disaggregate_info["role"] == "prefill"

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
                            result.outputs.logprob = float(scores[i, batch_token_index, 0])
                            topk_token_ids = tokens[i, batch_token_index, :].tolist()
                            topk_logprobs = scores[i, batch_token_index, :].tolist()
                            sampled_rank = ranks[i, batch_token_index].item()
                        else:
                            result.outputs.logprob = float(scores[i, 0])
                            topk_token_ids = tokens[i, :].tolist()
                            topk_logprobs = scores[i, :].tolist()
                            sampled_rank = ranks[i].item()

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
                    if recovery_stop:
                        result.error_msg = "Recover is not supported, the result is incomplete!"
                    llm_logger.info(
                        f"Request: {task_id} finished, number of "
                        f"generated tokens: {self.tokens_counter[task_id]}, token_id:{token_id},is_prefill:{is_prefill},recovery_stop:{recovery_stop}"
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

            llm_logger.debug(f"get response from infer: {result}")
            batch_result.append(result)

        self.postprocess(batch_result, mtype)

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
        trace_print(LoggingEventName.FIRST_TOKEN_GENERATED, task.request_id, getattr(task, "user", ""))
        trace_print(LoggingEventName.DECODE_START, task.request_id, getattr(task, "user", ""))
        main_process_metrics.time_to_first_token.observe(current_time - task.arrival_time)
        main_process_metrics.request_queue_time.observe(task.inference_start_time - task.preprocess_end_time)
        main_process_metrics.request_prefill_time.observe(current_time - task.inference_start_time)

    def _record_completion_metrics(self, task, current_time):
        """Record metrics when request completes"""
        if hasattr(task, "first_token_time"):
            decode_time = current_time - task.first_token_time
            main_process_metrics.request_decode_time.observe(decode_time)
        trace_print(LoggingEventName.INFERENCE_END, task.request_id, getattr(task, "user", ""))
        trace_print(LoggingEventName.POSTPROCESSING_START, task.request_id, getattr(task, "user", ""))
        main_process_metrics.num_requests_running.dec(1)
        main_process_metrics.request_success_total.inc()
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
        if num_emitted_tokens == 0:
            return
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
