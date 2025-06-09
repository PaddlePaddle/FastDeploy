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
import os
import threading
import time
import traceback
from collections import Counter

from paddlenlp.utils.env import MAX_BSZ, MAX_DRAFT_TOKENS, SPECULATE_MAX_BSZ

from fastdeploy.engine.request import (CompletionOutput, RequestMetrics,
                                       RequestOutput)
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import llm_logger


class TokenProcessor(object):
    """
    get Token/Score from Paddle inference engine
    """

    def __init__(self, cfg, cached_generated_tokens):
        import paddle

        paddle.device.set_device("cpu")
        self.cfg = cfg
        self.cached_generated_tokens = cached_generated_tokens
        self.resource_manager = None

        self.tokens_counter = Counter()

        self.is_speculate_decoding = False
        if self.is_speculate_decoding:
            self.output_tokens = paddle.full(shape=[
                SPECULATE_MAX_BSZ * MAX_DRAFT_TOKENS + SPECULATE_MAX_BSZ + 2, 1
            ],
                                             fill_value=2,
                                             dtype="int64")
        else:
            self.output_tokens = paddle.full(shape=[MAX_BSZ + 2, 1],
                                             fill_value=2,
                                             dtype="int64")
        self.worker = None

        self.statics_start_time = time.time()
        self.number_of_tasks = 0
        self.number_of_input_tokens = 0
        self.number_of_output_tokens = 0
        self.total_step = 0

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

        self.worker = threading.Thread(target=self.process_sampling_results,
                                       args=())
        self.worker.daemon = True
        self.worker.start()

    def process_sampling_results(self):
        """
        read tokens from paddle inference engine and process
        """
        from fastdeploy.model_executor.models import \
            inference_runner_supported_models
        if self.cfg.model_config.architectures not in inference_runner_supported_models \
            and "ErnieMoEVLForCausalLM" not in self.cfg.model_config.architectures:
            from paddlenlp_ops import get_output, speculate_get_output
        else:
            os.environ["ELLM_LOG_LEVEL"] = "3"
            use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
            if use_pip_eff_llm is None:
                from fastdeploy.model_executor.ops.gpu import (
                    get_output, speculate_get_output)
            else:
                from efficientllm.ops.gpu import (get_output,
                                                  speculate_get_output)

        while True:
            try:
                rank_id = 0
                is_blocking = True
                if self.is_speculate_decoding:
                    speculate_get_output(self.output_tokens, rank_id,
                                         is_blocking)
                else:
                    get_output(self.output_tokens, rank_id, is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue

                self._process_batch_output()
            except Exception as e:
                llm_logger.info("while get input_data error: {0} {1}".format(
                    e, str(traceback.format_exc())))

    def postprocess(self, batch_result):
        """
        single post-processing function

        Args:
            batch_result (list): batch results
        """
        self.cached_generated_tokens.put_results(batch_result)

    def _recycle_resources(self, task_id, index, task):
        """
        recycle resources
        """
        self.resource_manager.stop_flags[index] = True
        self.resource_manager.tasks_list[index] = None
        self.resource_manager._recycle_block_tables(task.block_tables)
        if task_id in self.tokens_counter:
            del self.tokens_counter[task_id]

    def _process_batch_output(self):
        """
        batch post-processing function
        """
        tokens = self.output_tokens.numpy()
        batch = self.output_tokens[1, 0]
        if not self.is_speculate_decoding:
            tokens = tokens[2:batch + 2]
        else:
            accept_num = tokens[2:batch + 2]

        batch_result = list()
        for i in range(batch):
            if self.resource_manager.stop_flags[i]:
                continue

            if not self.is_speculate_decoding:
                token_ids = [int(tokens[i, 0])]
            else:
                token_ids = tokens[
                    2 + SPECULATE_MAX_BSZ + i * MAX_DRAFT_TOKENS:2 +
                    SPECULATE_MAX_BSZ + i * MAX_DRAFT_TOKENS +
                    accept_num[i, 0],
                    0,
                ].tolist()
            if any(token_id < 0 for token_id in token_ids):
                continue

            task = self.resource_manager.tasks_list[i]

            if self.cfg.enable_chunked_prefill:
                if task.get("prefill_token_num", None) is None:
                    task.set("prefill_token_num", task.token_chunk_size)
                else:
                    task.prefill_token_num += task.token_chunk_size
                if task.prompt_token_ids_len > task.prefill_token_num:
                    continue

            task_id = task.request_id

            self.total_step += 1
            current_time = time.time()
            if self.tokens_counter[task_id] == 0:
                metrics = RequestMetrics(
                    arrival_time=task.arrival_time,
                    inference_start_time=task.inference_start_time,
                    first_token_time=time.time() - task.inference_start_time,
                    time_in_queue=task.schedule_start_time -
                    task.preprocess_end_time,
                    preprocess_cost_time=task.preprocess_end_time -
                    task.preprocess_start_time)

                main_process_metrics.time_to_first_token.observe(
                    current_time - task.inference_start_time)
                main_process_metrics.request_queue_time.observe(
                    metrics.time_in_queue)

            else:
                if hasattr(task, 'last_token_time'
                           ) and task.last_token_time is not None:
                    token_gen_time = current_time - task.last_token_time
                    main_process_metrics.time_per_output_token.observe(
                        token_gen_time)

                task.last_token_time = current_time
                metrics = RequestMetrics(
                    arrival_time=time.time(),
                    request_start_time=task.arrival_time,
                )
            self.number_of_output_tokens += len(token_ids)
            result = RequestOutput(request_id=task_id,
                                   outputs=CompletionOutput(index=i,
                                                            token_ids=[]),
                                   finished=False,
                                   metrics=metrics)
            if self.tokens_counter[task_id] == 0:
                if task.messages is not None:
                    result.prompt = task.messages
                result.prompt_token_ids = task.prompt_token_ids

            for token_id in token_ids:
                self.tokens_counter[task_id] += 1
                result.outputs.token_ids.append(token_id)
                if token_id in task.eos_token_ids:
                    result.finished = True
                    result.prompt = task.prompt
                    result.prompt_token_ids = task.prompt_token_ids
                    llm_logger.info(
                        f"Request: {task_id} finished, number of "
                        f"generated tokens: {self.tokens_counter[task_id]}.")
                    llm_logger.info(
                        f"Request: {task_id} token ratio: {self.tokens_counter[task_id] / (time.time() - task.inference_start_time)}"
                    )
                    llm_logger.info(f"{self.resource_manager.info()}")
                    llm_logger.info(
                        f"Speculate accept ratio: {1 - self.total_step * 1.0 / self.number_of_output_tokens}"
                        f" total step: {self.total_step}. total_output_token_num: {self.number_of_output_tokens}"
                    )
                    self._recycle_resources(task_id, i, task)
                    main_process_metrics.num_requests_running.dec(1)
                    main_process_metrics.request_inference_time.observe(
                        current_time - task.inference_start_time)
                    break
            batch_result.append(result)

        self.postprocess(batch_result)


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
        from fastdeploy.model_executor.models import \
            inference_runner_supported_models
        if self.cfg.model_config.architectures not in inference_runner_supported_models \
            and "ErnieMoEVLForCausalLM" not in self.cfg.model_config.architectures:
            from paddlenlp_ops import get_output, speculate_get_output
        else:
            os.environ["ELLM_LOG_LEVEL"] = "3"
            use_pip_eff_llm = os.getenv('USE_PIP_EFF_LLM')
            if use_pip_eff_llm is None:
                from fastdeploy.model_executor.ops.gpu import (
                    get_output, speculate_get_output)
            else:
                from efficientllm.ops.gpu import (get_output,
                                                  speculate_get_output)

        while self._is_running:
            try:
                rank_id = 0
                if self.is_speculate_decoding:
                    speculate_get_output(self.output_tokens, rank_id,
                                         self._is_blocking)
                else:
                    get_output(self.output_tokens, rank_id, self._is_blocking)

                if self.output_tokens[0, 0] == -2:
                    continue
                self._process_batch_output()
            except Exception as e:
                llm_logger.info("while get input_data error: {0} {1}".format(
                    e, str(traceback.format_exc())))

    def stop(self):
        """
        stop warm up thread
        """
        self._is_running = False
        self.worker.join()
        llm_logger.info("warm up thread stop")
        del self.worker
