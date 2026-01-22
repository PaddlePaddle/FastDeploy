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

import asyncio
import inspect
import itertools
import time
import traceback
import uuid
from collections.abc import Iterable
from typing import List, Optional

import numpy as np

import fastdeploy.envs as envs
import fastdeploy.metrics.trace as tracing
from fastdeploy.engine.request import Request, RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    CompletionTokenUsageInfo,
    ErrorInfo,
    ErrorResponse,
    PromptTokenUsageInfo,
    UsageInfo,
)
from fastdeploy.trace.constants import LoggingEventName
from fastdeploy.trace.trace_logger import print as trace_print
from fastdeploy.utils import (
    ErrorCode,
    ErrorType,
    ParameterError,
    api_server_logger,
    clamp_prompt_logprobs,
    get_host_ip,
)
from fastdeploy.worker.output import (
    Logprob,
    LogprobsLists,
    LogprobsTensors,
    PromptLogprobs,
)

NONES = itertools.repeat(None)


class OpenAIServingCompletion:
    def __init__(self, engine_client, models, pid, ips, max_waiting_time):
        self.engine_client = engine_client
        self.models = models
        self.pid = pid
        self.max_waiting_time = max_waiting_time
        if ips is not None:
            if isinstance(ips, list):
                self.master_ip = ips[0]
            else:
                self.master_ip = ips.split(",")[0]
            self.is_master_ip = get_host_ip() == self.master_ip
        else:
            self.master_ip = "0.0.0.0"
            self.is_master_ip = True
        self._is_process_response_dict_async = None
        api_server_logger.info(f"master ip: {self.master_ip}")

    def _check_master(self):
        return self.engine_client.is_master or self.is_master_ip

    async def create_completion(self, request: CompletionRequest):
        """
        Create a completion for the given prompt.
        """
        tracing.trace_set_thread_info("API Server")
        if not self._check_master():
            err_msg = (
                f"Only master node can accept completion request, please send request to master node: {self.master_ip}"
            )
            api_server_logger.error(err_msg)
            return ErrorResponse(error=ErrorInfo(message=err_msg, type=ErrorType.INTERNAL_ERROR))
        if self.models:
            is_supported, request.model = self.models.is_supported_model(request.model)
            if not is_supported:
                err_msg = f"Unsupported model: [{request.model}], support [{', '.join([x.name for x in self.models.model_paths])}] or default"
                api_server_logger.error(err_msg)
                return ErrorResponse(
                    error=ErrorInfo(message=err_msg, type=ErrorType.INTERNAL_ERROR, code=ErrorCode.MODEL_NOT_SUPPORT)
                )
        created_time = int(time.time())
        if request.request_id is not None:
            request_id = request.request_id
            if not request_id.startswith("cmpl-"):
                request_id = f"cmpl-{request_id}"
        elif request.user is not None:
            request_id = f"cmpl-{request.user}-{uuid.uuid4()}"
        else:
            request_id = f"cmpl-{uuid.uuid4()}"
        api_server_logger.info(f"Initialize request {request_id}: {request}")
        tracing.trace_req_start(rid=request_id, trace_content=request.trace_context, role="FastDeploy")
        del request.trace_context
        request_prompt_ids = None
        request_prompts = None

        # Handle prompt and prompt_token_ids
        try:
            if request.prompt_token_ids is not None:  # let `prompt_token_ids` support batch inference
                assert len(request.prompt_token_ids) > 0, "prompt_token_ids should not be an empty list"
                if isinstance(request.prompt_token_ids[0], list):
                    request_prompt_ids = request.prompt_token_ids
                elif isinstance(request.prompt_token_ids[0], int):
                    request_prompt_ids = [request.prompt_token_ids]
                else:
                    raise ValueError(
                        "If prompt_token_ids is provided, its type should be one of: list[int], list[list[int]]"
                    )
                # reset `prompt_token_ids` to avoid data processor directly using it; let data processor fill it
                request.prompt_token_ids = None
            else:
                if isinstance(request.prompt, str):
                    request_prompts = [request.prompt]
                elif isinstance(request.prompt, list) and all(isinstance(item, int) for item in request.prompt):
                    request_prompt_ids = [request.prompt]
                elif isinstance(request.prompt, list) and all(isinstance(item, str) for item in request.prompt):
                    request_prompts = request.prompt
                elif isinstance(request.prompt, list):
                    for item in request.prompt:
                        if isinstance(item, list) and all(isinstance(x, int) for x in item):
                            continue
                        else:
                            raise ValueError("If prompt is a list, each item type must be one of: str, list[int]")
                    request_prompt_ids = request.prompt
                else:
                    raise ValueError("Prompt type must be one of: str, list[str], list[int], list[list[int]]")
        except Exception as e:
            error_msg = f"OpenAIServingCompletion create_completion: {e}, {str(traceback.format_exc())}"
            api_server_logger.error(error_msg)
            return ErrorResponse(error=ErrorInfo(message=error_msg, type=ErrorType.INTERNAL_ERROR))

        if request_prompt_ids is not None:
            request_prompts = request_prompt_ids

        num_choices = len(request_prompts) * (1 if request.n is None else request.n)
        api_server_logger.info(f"Start preprocessing request: req_id={request_id}), num_choices={num_choices}")
        prompt_batched_token_ids = []
        prompt_tokens_list = []
        max_tokens_list = []
        try:
            if self.max_waiting_time < 0:
                await self.engine_client.semaphore.acquire()
            else:
                await asyncio.wait_for(self.engine_client.semaphore.acquire(), timeout=self.max_waiting_time)
        except Exception as e:
            error_msg = (
                f"OpenAIServingCompletion waiting error: {e}, {str(traceback.format_exc())}, "
                f"max waiting time: {self.max_waiting_time}"
            )
            api_server_logger.error(error_msg)
            return ErrorResponse(
                error=ErrorInfo(message=error_msg, code=ErrorCode.TIMEOUT, type=ErrorType.TIMEOUT_ERROR)
            )

        try:
            try:
                for idx, prompt in enumerate(request_prompts):
                    request_id_idx = f"{request_id}_{idx}"
                    if not envs.ENABLE_V1_DATA_PROCESSOR:
                        current_req_dict = request.to_dict_for_infer(request_id_idx, prompt)
                    else:
                        current_req_dict = Request.from_generic_request(request, request_id=f"{request_id}_0")
                    current_req_dict["metrics"]["arrival_time"] = time.time()
                    prompt_token_ids = await self.engine_client.format_and_add_data(current_req_dict)  # tokenize
                    if isinstance(prompt_token_ids, np.ndarray):
                        prompt_token_ids = prompt_token_ids.tolist()
                    prompt_tokens_list.append(current_req_dict.get("prompt_tokens"))
                    prompt_batched_token_ids.append(prompt_token_ids)
                    max_tokens_list.append(current_req_dict.get("max_tokens"))
                    del current_req_dict
            except ParameterError as e:
                api_server_logger.error(f"OpenAIServingCompletion format error: {e}, {e.message}")
                self.engine_client.semaphore.release()
                return ErrorResponse(
                    error=ErrorInfo(code="400", message=str(e.message), type="invalid_request", param=e.param)
                )
            except Exception as e:
                error_msg = f"OpenAIServingCompletion format error: {e}, {str(traceback.format_exc())}"
                api_server_logger.error(error_msg)
                self.engine_client.semaphore.release()
                return ErrorResponse(
                    error=ErrorInfo(message=str(e), code=ErrorCode.INVALID_VALUE, type=ErrorType.INVALID_REQUEST_ERROR)
                )

            if request.stream:
                return self.completion_stream_generator(
                    request=request,
                    num_choices=num_choices,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model,
                    prompt_batched_token_ids=prompt_batched_token_ids,
                    prompt_tokens_list=prompt_tokens_list,
                    max_tokens_list=max_tokens_list,
                )
            else:
                try:
                    return await self.completion_full_generator(
                        request=request,
                        num_choices=num_choices,
                        request_id=request_id,
                        created_time=created_time,
                        model_name=request.model,
                        prompt_batched_token_ids=prompt_batched_token_ids,
                        prompt_tokens_list=prompt_tokens_list,
                        max_tokens_list=max_tokens_list,
                    )
                except Exception as e:
                    error_msg = (
                        f"OpenAIServingCompletion completion_full_generator error: {e}, {str(traceback.format_exc())}"
                    )
                    api_server_logger.error(error_msg)
                    return ErrorResponse(error=ErrorInfo(message=error_msg, type=ErrorType.INTERNAL_ERROR))
        except asyncio.CancelledError as e:
            await self.engine_client.abort(f"{request_id}_0", num_choices)
            error_msg = f"request[{request_id}_0] client disconnected: {str(e)}, {str(traceback.format_exc())}"
            api_server_logger.error(error_msg)
            return ErrorResponse(
                error=ErrorInfo(message=error_msg, type=ErrorType.INVALID_REQUEST_ERROR, code=ErrorCode.CLIENT_ABORTED)
            )
        except Exception as e:
            error_msg = f"OpenAIServingCompletion create_completion error: {e}, {str(traceback.format_exc())}"
            api_server_logger.error(error_msg)
            return ErrorResponse(error=ErrorInfo(message=error_msg, type=ErrorType.INTERNAL_ERROR))

    async def completion_full_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str,
        prompt_batched_token_ids: list(),
        prompt_tokens_list: list(),
        max_tokens_list: list(),
    ):
        """
        Process the full completion request with multiple choices.
        """
        dealer = None
        try:
            request_ids = [f"{request_id}_{i}" for i in range(num_choices)]
            # create dealer
            dealer, response_queue = await self.engine_client.connection_manager.get_connection(
                request_id, num_choices
            )

            for rid in request_ids:
                dealer.write([b"", rid.encode("utf-8")])

            valid_results = [dict()] * num_choices
            output_tokens = [0] * num_choices
            aggregated_top_logprobs = [[[], [], []] for _ in range(num_choices)]
            aggregated_draft_top_logprobs = [[[], [], []] for _ in range(num_choices)]
            aggregated_token_ids = [[] for _ in range(num_choices)]
            aggregated_prompt_logprobs_tensors = [None] * num_choices
            completion_batched_token_ids = [[] for _ in range(num_choices)]
            aggregated_speculate_metrics = [None] * num_choices
            current_waiting_time = 0
            while num_choices > 0:
                if self.engine_client.check_model_weight_status():
                    return ErrorResponse(
                        error=ErrorInfo(
                            message="Model weight cleared",
                            code=ErrorCode.INVALID_VALUE,
                            type=ErrorType.INVALID_REQUEST_ERROR,
                        )
                    )
                try:
                    response = await asyncio.wait_for(response_queue.get(), timeout=10)
                    current_waiting_time = 0
                except asyncio.TimeoutError:
                    current_waiting_time += 10
                    if current_waiting_time == 300:
                        status, msg = self.engine_client.check_health(
                            time_interval_threashold=envs.FD_WORKER_ALIVE_TIMEOUT
                        )
                        if not status:
                            raise ValueError(f"Engine is not healthy: {msg}")
                        else:
                            current_waiting_time = 0
                    await asyncio.sleep(0.1)
                    continue

                for data in response:
                    rid = int(data["request_id"].split("_")[-1])
                    if data.get("error_code", 200) != 200:
                        raise ValueError("{}".format(data["error_msg"]))

                    output = data["outputs"]
                    output_top_logprobs = output.get("top_logprobs") or None
                    output_draft_top_logprobs = output.get("draft_top_logprobs") or None
                    if output_top_logprobs is not None:
                        aggregated_top_logprobs[rid][0].extend(output_top_logprobs[0])
                        aggregated_top_logprobs[rid][1].extend(output_top_logprobs[1])
                        aggregated_top_logprobs[rid][2].extend(output_top_logprobs[2])

                        # draft logprobs
                        if request.include_draft_logprobs and output_draft_top_logprobs is not None:
                            aggregated_draft_top_logprobs[rid][0].extend(output_draft_top_logprobs[0])
                            aggregated_draft_top_logprobs[rid][1].extend(output_draft_top_logprobs[1])
                            aggregated_draft_top_logprobs[rid][2].extend(output_draft_top_logprobs[2])

                    output_prompt_logprobs_tensors = data.get("prompt_logprobs") or None
                    if output_prompt_logprobs_tensors is not None:
                        aggregated_prompt_logprobs_tensors[rid] = output_prompt_logprobs_tensors

                    aggregated_token_ids[rid].extend(data["outputs"]["token_ids"])
                    await self._call_process_response_dict(data, request, stream=False)
                    output_tokens[rid] += len(data["outputs"]["token_ids"])
                    completion_batched_token_ids[rid].extend(data["outputs"]["token_ids"])

                    output_speculate_metrics = data["metrics"].get("speculate_metrics", None)
                    if output_speculate_metrics is not None:
                        aggregated_speculate_metrics[rid] = output_speculate_metrics

                    if data.get("finished", False):
                        trace_carrier = data.get("trace_carrier")
                        if trace_carrier:
                            tracing.trace_set_proc_propagate_context(request_id, trace_carrier)
                            start_time = data["metrics"]["engine_recv_latest_token_time"]
                            tracing.trace_report_span(
                                tracing.TraceSpanName.POSTPROCESSING,
                                request_id,
                                int(start_time * 1e9),
                                int(time.time() * 1e9),
                                thread_finish_flag=True,
                            )
                            if "trace_carrier" in data:
                                del data["trace_carrier"]
                        data["output_token_ids"] = output_tokens[rid]
                        data["outputs"]["top_logprobs"] = aggregated_top_logprobs[rid]
                        data["outputs"]["draft_top_logprobs"] = aggregated_draft_top_logprobs[rid]
                        data["outputs"]["token_ids"] = aggregated_token_ids[rid]
                        data["prompt_logprobs_tensors"] = aggregated_prompt_logprobs_tensors[rid]
                        data["speculate_metrics"] = aggregated_speculate_metrics[rid]
                        valid_results[rid] = data
                        num_choices -= 1
                        break
            res = self.request_output_to_completion_response(
                final_res_batch=valid_results,
                request=request,
                request_id=request_id,
                created_time=created_time,
                model_name=model_name,
                prompt_batched_token_ids=prompt_batched_token_ids,
                completion_batched_token_ids=completion_batched_token_ids,
                prompt_tokens_list=prompt_tokens_list,
                max_tokens_list=max_tokens_list,
            )
            api_server_logger.info(f"Completion response: {res.model_dump_json()}")
            return res
        except Exception as e:
            api_server_logger.error(f"Error in completion_full_generator: {e}", exc_info=True)
        finally:
            tracing.trace_req_finish(request_id)
            trace_print(LoggingEventName.POSTPROCESSING_END, request_id, getattr(request, "user", ""))
            self.engine_client.semaphore.release()
            if dealer is not None:
                await self.engine_client.connection_manager.cleanup_request(request_id)

    def _echo_back_prompt(self, request, idx):
        """
        The echo pre-process of the smallest unit
        """
        if isinstance(request.prompt, str):
            prompt_text = request.prompt
        elif isinstance(request.prompt, list):
            if all(isinstance(item, str) for item in request.prompt):
                prompt_text = request.prompt[idx]
            elif all(isinstance(item, int) for item in request.prompt):
                prompt_text = self.engine_client.data_processor.tokenizer.decode(request.prompt)
            else:
                prompt_text = self.engine_client.data_processor.tokenizer.decode(request.prompt[idx])
        return prompt_text

    async def _process_echo_logic(self, request, idx, res_outputs):
        """
        Process the echo logic and return the modified text.
        """
        if request.echo and res_outputs.get("send_idx", -1) == 0:
            prompt_text = self._echo_back_prompt(request, idx // (1 if request.n is None else request.n))
            res_outputs["text"] = prompt_text + (res_outputs["text"] or "")
        return res_outputs

    def calc_finish_reason(self, max_tokens, token_num, output, tool_called):
        if max_tokens is None or token_num != max_tokens:
            if tool_called or output.get("tool_call"):
                return "tool_calls"
            else:
                return "stop"
        else:
            return "length"

    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str,
        prompt_batched_token_ids: list(),
        prompt_tokens_list: list(),
        max_tokens_list: list(),
    ):
        """
        Process the stream completion request.
        """
        try:
            dealer, response_queue = await self.engine_client.connection_manager.get_connection(
                request_id, num_choices
            )

            for i in range(num_choices):
                req_id = f"{request_id}_{i}"
                dealer.write([b"", req_id.encode("utf-8")])  # 发送多路请求
            output_tokens = [0] * num_choices
            num_cache_tokens = [0] * num_choices
            num_image_tokens = [0] * num_choices
            inference_start_time = [0] * num_choices
            reasoning_tokens = [0] * num_choices
            first_iteration = [True] * num_choices
            tool_called = [False] * num_choices
            max_streaming_response_tokens = (
                request.max_streaming_response_tokens
                if request.max_streaming_response_tokens is not None
                else (request.suffix or {}).get("max_streaming_response_tokens", 1)
            )  # dierctly passed & passed in suffix
            max_streaming_response_tokens = max(1, max_streaming_response_tokens)
            choices = []
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=choices,
            )
            current_waiting_time = 0
            while num_choices > 0:
                if self.engine_client.check_model_weight_status():
                    raise ValueError("Engine is clearing model weight")
                try:
                    response = await asyncio.wait_for(response_queue.get(), timeout=10)
                    current_waiting_time = 0
                except asyncio.TimeoutError:
                    current_waiting_time += 10
                    if current_waiting_time == 300:
                        status, msg = self.engine_client.check_health(
                            time_interval_threashold=envs.FD_WORKER_ALIVE_TIMEOUT
                        )
                        if not status:
                            raise ValueError(f"Engine is not healthy: {msg}")
                        else:
                            current_waiting_time = 0
                    await asyncio.sleep(0.1)
                    continue

                for res in response:
                    idx = int(res["request_id"].split("_")[-1])
                    if res.get("error_code", 200) != 200:
                        raise ValueError("{}".format(res["error_msg"]))
                    prompt_logprobs_res: Optional[PromptLogprobs] = None
                    if first_iteration[idx]:
                        prompt_logprobs_tensors = res.get("prompt_logprobs", None)
                        if request.prompt_logprobs is not None and prompt_logprobs_tensors is not None:
                            num_prompt_logprobs = (
                                request.prompt_logprobs
                                if request.prompt_logprobs != -1
                                else self.engine_client.ori_vocab_size
                            )
                            prompt_logprobs_res = self._build_prompt_logprobs(
                                prompt_logprobs_tensors, num_prompt_logprobs, request.include_logprobs_decode_token
                            )
                        if request.return_token_ids:
                            chunk = CompletionStreamResponse(
                                id=request_id,
                                created=created_time,
                                model=model_name,
                                choices=[
                                    CompletionResponseStreamChoice(
                                        index=idx,
                                        text="",
                                        prompt_token_ids=list(
                                            prompt_batched_token_ids[idx // (1 if request.n is None else request.n)]
                                        ),
                                        prompt_logprobs=clamp_prompt_logprobs(prompt_logprobs_res),
                                        prompt_tokens=prompt_tokens_list[
                                            idx // (1 if request.n is None else request.n)
                                        ],
                                        completion_token_ids=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                            api_server_logger.info(
                                f"Completion Streaming response send_idx 0: {chunk.model_dump_json()}"
                            )
                        first_iteration[idx] = False

                    await self._call_process_response_dict(res, request, stream=True)
                    if inference_start_time[idx] == 0:
                        arrival_time = res["metrics"]["first_token_time"]
                        inference_start_time[idx] = res["metrics"]["inference_start_time"]
                    else:
                        arrival_time = res["metrics"]["engine_recv_latest_token_time"] - inference_start_time[idx]

                    await self._process_echo_logic(request, idx, res["outputs"])
                    output = res["outputs"]
                    output_top_logprobs = output["top_logprobs"]
                    output_draft_top_logprobs = output["draft_top_logprobs"]
                    logprobs_res: Optional[CompletionLogprobs] = None
                    draft_logprobs_res: Optional[CompletionLogprobs] = None
                    if request.logprobs is not None and output_top_logprobs is not None:
                        num_logprobs = (
                            request.logprobs if request.logprobs != -1 else self.engine_client.ori_vocab_size
                        )
                        logprobs_res = self._create_completion_logprobs(output_top_logprobs, num_logprobs, 0)

                        # draft logprobs
                        if request.include_draft_logprobs and output_draft_top_logprobs is not None:
                            draft_logprobs_res = self._create_completion_logprobs(
                                output_draft_top_logprobs, num_logprobs, 0
                            )
                    output_tokens[idx] += len(output.get("token_ids", [])) or 0
                    num_cache_tokens[idx] += output.get("num_cache_tokens") or 0
                    if output.get("num_image_tokens"):
                        output_tokens[idx] += output.get("num_image_tokens")
                        num_image_tokens[idx] += output.get("num_image_tokens")
                    reasoning_tokens[idx] += output.get("reasoning_token_num", 0)
                    output_speculate_metrics = res["metrics"].get("speculate_metrics", None)
                    delta_message = CompletionResponseStreamChoice(
                        index=idx,
                        text=output["text"],
                        prompt_token_ids=None,
                        completion_token_ids=output.get("token_ids") if request.return_token_ids else None,
                        tool_calls=None,
                        completion_tokens=output.get("completion_tokens") if request.return_token_ids else None,
                        reasoning_content="",
                        arrival_time=arrival_time,
                        logprobs=logprobs_res,
                        prompt_logprobs=(
                            clamp_prompt_logprobs(prompt_logprobs_res) if not request.return_token_ids else None
                        ),
                        draft_logprobs=draft_logprobs_res,
                        speculate_metrics=output_speculate_metrics,
                    )
                    if not res["finished"] and output["enable_parser"]:
                        delta_message_output = output["delta_message"]
                        if delta_message_output is None:
                            continue
                        delta_message.text = delta_message_output.content or ""
                        delta_message.reasoning_content = delta_message_output.reasoning_content or ""
                        if delta_message_output.tool_calls:
                            delta_message.tool_calls = delta_message_output.tool_calls
                            tool_called[idx] = True

                    choices.append(delta_message)

                    if res["finished"]:
                        choices[-1].finish_reason = self.calc_finish_reason(
                            max_tokens_list[idx // (1 if request.n is None else request.n)],
                            output_tokens[idx],
                            output,
                            tool_called[idx],
                        )
                        inference_start_time[idx] = 0

                    send_idx = output.get("send_idx")
                    # 只有当 send_idx 明确为 0 时才记录日志
                    if send_idx == 0 and not request.return_token_ids:
                        chunk_temp = chunk
                        chunk_temp.choices = choices
                        api_server_logger.info(
                            f"Completion Streaming response send_idx 0: {chunk_temp.model_dump_json()}"
                        )
                        del chunk_temp

                    if len(choices) == max_streaming_response_tokens or res["finished"]:
                        chunk = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name,
                            choices=choices,
                            metrics=res["metrics"] if request.collect_metrics else None,
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                        choices = []

                    if res["finished"]:
                        trace_carrier = res.get("trace_carrier")
                        if trace_carrier:
                            tracing.trace_set_proc_propagate_context(request_id, trace_carrier)
                            start_time = res["metrics"]["engine_recv_latest_token_time"]
                            tracing.trace_report_span(
                                tracing.TraceSpanName.POSTPROCESSING,
                                request_id,
                                int(start_time * 1e9),
                                int(time.time() * 1e9),
                                thread_finish_flag=True,
                            )
                            if "trace_carrier" in res:
                                del res["trace_carrier"]
                        num_choices -= 1
                        if getattr(request, "stream_options", None) and request.stream_options.include_usage:
                            usage_chunk = CompletionStreamResponse(
                                id=request_id,
                                created=created_time,
                                model=model_name,
                                choices=[],
                                usage=UsageInfo(
                                    prompt_tokens=len(
                                        prompt_batched_token_ids[idx // (1 if request.n is None else request.n)]
                                    ),
                                    completion_tokens=output_tokens[idx],
                                    total_tokens=len(
                                        prompt_batched_token_ids[idx // (1 if request.n is None else request.n)]
                                    )
                                    + output_tokens[idx],
                                    prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=num_cache_tokens[idx]),
                                    completion_tokens_details=CompletionTokenUsageInfo(
                                        image_tokens=num_image_tokens[idx], reasoning_tokens=reasoning_tokens[idx]
                                    ),
                                ),
                                metrics=res["metrics"] if request.collect_metrics else None,
                            )
                            yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"
                        api_server_logger.info(f"Completion Streaming response last send: {chunk.model_dump_json()}")

        except asyncio.CancelledError as e:
            await self.engine_client.abort(f"{request_id}_0", num_choices)
            error_msg = f"request[{request_id}_0] client disconnected: {str(e)}, {str(traceback.format_exc())}"
            api_server_logger.error(error_msg)
        except Exception as e:
            api_server_logger.error(f"Error in completion_stream_generator: {e}, {str(traceback.format_exc())}")
            yield f"data: {ErrorResponse(error=ErrorInfo(message=str(e), code='400', type=ErrorType.INTERNAL_ERROR)).model_dump_json(exclude_unset=True)}\n\n"
        finally:

            tracing.trace_req_finish(request_id)
            trace_print(LoggingEventName.POSTPROCESSING_END, request_id, getattr(request, "user", ""))
            del request
            if dealer is not None:
                await self.engine_client.connection_manager.cleanup_request(request_id)
                self.engine_client.semaphore.release()
            yield "data: [DONE]\n\n"

    def request_output_to_completion_response(
        self,
        final_res_batch: List[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
        prompt_batched_token_ids: list(),
        completion_batched_token_ids: list(),
        prompt_tokens_list: list(),
        max_tokens_list: list(),
    ) -> CompletionResponse:
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0
        num_cache_tokens = 0
        num_image_tokens = 0
        num_reasoning_tokens = 0

        for idx in range(len(final_res_batch)):
            final_res = final_res_batch[idx]
            prompt_token_ids = prompt_batched_token_ids[idx // (1 if request.n is None else request.n)]
            assert prompt_token_ids is not None
            prompt_text = request.prompt
            completion_token_ids = completion_batched_token_ids[idx]

            output = final_res["outputs"]
            output_top_logprobs = output.get("top_logprobs") or None
            output_draft_top_logprobs = output.get("draft_top_logprobs") or None

            aggregated_logprobs: Optional[CompletionLogprobs] = None
            num_logprobs = request.logprobs if request.logprobs != -1 else self.engine_client.ori_vocab_size
            if output_top_logprobs is not None:
                aggregated_logprobs = self._create_completion_logprobs(output_top_logprobs, num_logprobs, 0)

            aggregated_draft_logprobs: Optional[CompletionLogprobs] = None
            if output_draft_top_logprobs is not None:
                aggregated_draft_logprobs = self._create_completion_logprobs(
                    output_draft_top_logprobs, num_logprobs, 0
                )
            prompt_logprobs_res: Optional[PromptLogprobs] = None
            prompt_logprobs_tensors = final_res.get("prompt_logprobs_tensors", None)
            if request.prompt_logprobs is not None and prompt_logprobs_tensors is not None:
                num_prompt_logprobs = (
                    request.prompt_logprobs if request.prompt_logprobs != -1 else self.engine_client.ori_vocab_size
                )
                prompt_logprobs_res = self._build_prompt_logprobs(
                    prompt_logprobs_tensors, num_prompt_logprobs, request.include_logprobs_decode_token
                )
            if request.echo:
                prompt_text = self._echo_back_prompt(request, idx // (1 if request.n is None else request.n))
                token_ids = [*prompt_token_ids, *output["token_ids"]]
                output_text = prompt_text + output["text"]
            else:
                token_ids = output["token_ids"]
                output_text = output["text"]
            finish_reason = self.calc_finish_reason(
                max_tokens_list[idx // (1 if request.n is None else request.n)],
                final_res["output_token_ids"],
                output,
                False,
            )

            choice_data = CompletionResponseChoice(
                token_ids=token_ids,
                index=len(choices),
                text=output_text,
                prompt_token_ids=prompt_token_ids if request.return_token_ids else None,
                completion_token_ids=completion_token_ids if request.return_token_ids else None,
                completion_tokens=output.get("completion_tokens") if request.return_token_ids else None,
                prompt_tokens=(
                    prompt_tokens_list[idx // (1 if request.n is None else request.n)]
                    if request.return_token_ids
                    else None
                ),
                reasoning_content=output.get("reasoning_content"),
                tool_calls=output.get("tool_call", None),
                logprobs=aggregated_logprobs,
                draft_logprobs=aggregated_draft_logprobs,
                prompt_logprobs=clamp_prompt_logprobs(prompt_logprobs_res),
                finish_reason=finish_reason,
                speculate_metrics=final_res["metrics"].get("speculate_metrics", None),
            )
            choices.append(choice_data)

            num_generated_tokens += final_res["output_token_ids"]

            num_prompt_tokens += len(prompt_token_ids)
            num_cache_tokens += output.get("num_cache_tokens") or 0
            if output.get("num_image_tokens"):
                num_generated_tokens += output.get("num_image_tokens")
                num_image_tokens += output.get("num_image_tokens")

            num_reasoning_tokens += output.get("reasoning_token_num", 0)

        num_prompt_tokens = num_prompt_tokens // (1 if request.n is None else request.n)
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=num_cache_tokens),
            completion_tokens_details=CompletionTokenUsageInfo(
                reasoning_tokens=num_reasoning_tokens, image_tokens=num_image_tokens
            ),
        )
        del request

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )

    async def _call_process_response_dict(self, res, request, stream):
        if self._is_process_response_dict_async is None:
            self._is_process_response_dict_async = inspect.iscoroutinefunction(
                self.engine_client.data_processor.process_response_dict
            )
        if self._is_process_response_dict_async:
            await self.engine_client.data_processor.process_response_dict(
                res, stream=stream, include_stop_str_in_output=request.include_stop_str_in_output
            )
        else:
            self.engine_client.data_processor.process_response_dict(
                res, stream=stream, include_stop_str_in_output=request.include_stop_str_in_output
            )

    def _create_completion_logprobs(
        self,
        output_top_logprobs,
        request_logprobs: Optional[int] = None,
        prompt_text_offset: Optional[int] = None,
    ) -> Optional[CompletionLogprobs]:
        """Create OpenAI-style logprobs for completions."""

        # Parameter validation
        if output_top_logprobs is None or len(output_top_logprobs) < 3 or any(not lst for lst in output_top_logprobs):
            return None

        logprobs_res: Optional[CompletionLogprobs] = None
        # Iterate over the top-k candidates for each token
        for logprob_token_ids, logprobs, sampled_token_ranks in zip(
            output_top_logprobs[0], output_top_logprobs[1], output_top_logprobs[2]
        ):
            top_logprobs = LogprobsLists(
                logprob_token_ids=[logprob_token_ids],
                logprobs=[logprobs],
                sampled_token_ranks=[sampled_token_ranks],
            )
            # Build the logprobs response
            step_logprobs_res = self._build_logprobs_response(
                response_logprobs=top_logprobs,
                request_top_logprobs=request_logprobs,
                prompt_text_offset=prompt_text_offset,
            )
            if logprobs_res is None:
                logprobs_res = step_logprobs_res
            else:
                # Append the new tokens to the existing logprobs response
                logprobs_res.tokens.extend(step_logprobs_res.tokens)
                logprobs_res.token_logprobs.extend(step_logprobs_res.token_logprobs)
                logprobs_res.top_logprobs.extend(step_logprobs_res.top_logprobs)

        return logprobs_res

    def _build_logprobs_response(
        self,
        response_logprobs: Optional[LogprobsLists] = None,
        request_top_logprobs: Optional[int] = None,
        prompt_text_offset: Optional[int] = None,
    ) -> Optional[CompletionLogprobs]:
        """
        Construct a logprobs response object in line with the OpenAI style.
        Retain the complete top-k candidates and avoid circular references.
        """

        # Parameter validation
        if response_logprobs is None or request_top_logprobs is None or request_top_logprobs < 0:
            return None

        try:
            # The top-k candidates for the current token
            topk_token_ids = []
            topk_logprobs = []

            if response_logprobs.logprob_token_ids and len(response_logprobs.logprob_token_ids) > 0:
                topk_token_ids = response_logprobs.logprob_token_ids[0][: request_top_logprobs + 1]

            if response_logprobs.logprobs and len(response_logprobs.logprobs) > 0:
                topk_logprobs = response_logprobs.logprobs[0][: request_top_logprobs + 1]

            # Construct the sampled token object (avoid sharing references with top_logprob_entries)
            tokens = []
            token_logprobs = []
            top_logprobs = {}
            idx = 0
            for tid, lp in zip(topk_token_ids, topk_logprobs):
                token_str = self.engine_client.data_processor.process_logprob_response(
                    [tid], clean_up_tokenization_spaces=False
                )
                if "\ufffd" in token_str:
                    raw_token = self.engine_client.data_processor.tokenizer.convert_ids_to_tokens(tid)
                    token_bytes = raw_token.encode("utf-8", errors="replace")
                    token_str = "bytes:" + "".join(f"\\x{byte:02x}" for byte in token_bytes)
                if idx == 0:
                    tokens.append(token_str)
                    token_logprobs.append(lp)
                top_logprobs[token_str] = lp
                idx += 1

            # Construct the sampled token object (avoid sharing references with top_logprob_entries)
            # text_offset = prompt_text_offset + len(tokens) - 1
            return CompletionLogprobs(
                tokens=tokens,
                token_logprobs=token_logprobs,
                top_logprobs=[top_logprobs],
                # text_offset=[text_offset],
            )

        except Exception as e:
            api_server_logger.error(f"Error in _build_logprobs_response: {str(e)}, {str(traceback.format_exc())}")
            return None

    def _build_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
        num_prompt_logprobs: int,
        include_logprobs_decode_token: bool,
    ):
        """Update with prompt logprobs from worker.
        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.
        """

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        if include_logprobs_decode_token:
            decoded_tokens = [
                self.engine_client.data_processor.process_logprob_response(token_id)
                for token_id in token_ids.flatten().tolist()
            ]
        else:
            decoded_tokens = None

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the paddle tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()
        result: Optional[PromptLogprobs] = [None]
        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = NONES if decoded_tokens is None else decoded_tokens[offset:offset_end]

            # Update with the Logprob dictionary for this pos.
            result.append(
                self._make_logprob_dict(
                    prompt_logprobs[pos],
                    token_ids[pos],
                    decoded_tokens_for_pos,
                    prompt_token_ranks[pos],
                    num_prompt_logprobs,
                )
            )
        return result

    @staticmethod
    def _make_logprob_dict(
        logprobs: list[float],
        logprob_token_ids: list[int],
        decoded_tokens: Iterable[str | None],
        rank: int,
        num_logprobs: int,
    ) -> dict[int, Logprob]:
        """Make a Logprob dictionary for a position.
        Args:
          logprobs: list of log probabilities
          logprob_token_ids: list of top token ids
          decoded_tokens: list of decoded top tokens
          rank: rank of the sampled token
          num_logprobs: number of logprobs requested
            by the user (in addition to sampled logprob)
        Returns:
          dict[token id, Logprob]
        """
        if num_logprobs == -1:
            num_logprobs = len(logprobs)
        # We do not need a special case for the sampled token
        # being in the topk, since inserting duplicated data
        # into a dictionary twice is the same as doing it once.
        topk_ranks = range(1, num_logprobs + 1)
        ranks = itertools.chain((rank,), topk_ranks)

        return {
            token_id: Logprob(
                logprob=logprob,
                rank=rank,
                decoded_token=token,
            )
            for token_id, logprob, rank, token in zip(logprob_token_ids, logprobs, ranks, decoded_tokens)
        }
