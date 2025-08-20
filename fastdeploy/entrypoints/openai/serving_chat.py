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
import time
import traceback
import uuid
from typing import List, Optional

import aiozmq
import msgpack
import numpy as np
from aiozmq import zmq

from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    DeltaMessage,
    ErrorResponse,
    LogProbEntry,
    LogProbs,
    PromptTokenUsageInfo,
    UsageInfo,
)
from fastdeploy.metrics.work_metrics import work_process_metrics
from fastdeploy.utils import api_server_logger, get_host_ip
from fastdeploy.worker.output import LogprobsLists


class OpenAIServingChat:
    """
    OpenAI-style chat completions serving
    """

    def __init__(self, engine_client, pid, ips, max_waiting_time, chat_template):
        self.engine_client = engine_client
        self.pid = pid
        self.master_ip = ips
        self.max_waiting_time = max_waiting_time
        self.host_ip = get_host_ip()
        self.chat_template = chat_template
        if self.master_ip is not None:
            if isinstance(self.master_ip, list):
                self.master_ip = self.master_ip[0]
            else:
                self.master_ip = self.master_ip.split(",")[0]

    def _check_master(self):
        if self.master_ip is None:
            return True
        if self.host_ip == self.master_ip:
            return True
        return False

    async def create_chat_completion(self, request: ChatCompletionRequest):
        """
        Create a new chat completion using the specified parameters.
        """

        if not self._check_master():
            err_msg = f"Only master node can accept completion request, please send request to master node: {self.pod_ips[0]}"
            api_server_logger.error(err_msg)
            return ErrorResponse(message=err_msg, code=400)
        try:
            if self.max_waiting_time < 0:
                await self.engine_client.semaphore.acquire()
            else:
                await asyncio.wait_for(self.engine_client.semaphore.acquire(), timeout=self.max_waiting_time)
            api_server_logger.info(f"current {self.engine_client.semaphore.status()}")

            if request.user is not None:
                request_id = f"chatcmpl-{request.user}-{uuid.uuid4()}"
            else:
                request_id = f"chatcmpl-{uuid.uuid4()}"
            api_server_logger.info(f"create chat completion request: {request_id}")
            text_after_process = None
            try:
                current_req_dict = request.to_dict_for_infer(request_id)
                if "chat_template" not in current_req_dict:
                    current_req_dict["chat_template"] = self.chat_template
                current_req_dict["arrival_time"] = time.time()
                prompt_token_ids = self.engine_client.format_and_add_data(current_req_dict)
                text_after_process = current_req_dict.get("text_after_process")
                if isinstance(prompt_token_ids, np.ndarray):
                    prompt_token_ids = prompt_token_ids.tolist()
            except Exception as e:
                error_msg = f"request[{request_id}] generator error: {str(e)}, {str(traceback.format_exc())}"
                api_server_logger.error(error_msg)
                return ErrorResponse(code=400, message=error_msg)

            del current_req_dict

            if request.stream:
                return self.chat_completion_stream_generator(
                    request, request_id, request.model, prompt_token_ids, text_after_process
                )
            else:
                try:
                    return await self.chat_completion_full_generator(
                        request, request_id, request.model, prompt_token_ids, text_after_process
                    )
                except Exception as e:
                    error_msg = f"request[{request_id}]full generator error: {str(e)}, {str(traceback.format_exc())}"
                    api_server_logger.error(error_msg)
                    return ErrorResponse(code=408, message=error_msg)
        except Exception as e:
            error_msg = (
                f"request[{request_id}] waiting error: {str(e)}, {str(traceback.format_exc())}, "
                f"max waiting time: {self.max_waiting_time}"
            )
            api_server_logger.error(error_msg)
            return ErrorResponse(code=408, message=error_msg)

    def _create_streaming_error_response(self, message: str) -> str:
        api_server_logger.error(message)
        error_response = ErrorResponse(
            code=400,
            message=message,
        )
        return error_response.model_dump_json()

    async def chat_completion_stream_generator(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        model_name: str,
        prompt_token_ids: list(),
        text_after_process: str,
    ):
        """
        Streaming chat completion generator.
        """
        created_time = int(time.time())
        chunk_object_type: str = "chat.completion.chunk"
        first_iteration = True
        previous_num_tokens = 0
        num_prompt_tokens = 0
        num_choices = 1
        tool_called = False
        max_streaming_response_tokens = (
            request.max_streaming_response_tokens
            if request.max_streaming_response_tokens is not None
            else (request.metadata or {}).get("max_streaming_response_tokens", 1)
        )  # dierctly passed & passed in metadata

        enable_thinking = request.chat_template_kwargs.get("enable_thinking") if request.chat_template_kwargs else None
        if enable_thinking is None:
            enable_thinking = request.metadata.get("enable_thinking") if request.metadata else None

        include_stop_str_in_output = request.include_stop_str_in_output

        stream_options = request.stream_options
        if stream_options is None:
            include_usage = False
            include_continuous_usage = False
        else:
            include_usage = stream_options.include_usage
            include_continuous_usage = stream_options.continuous_usage_stats
        chunk = ChatCompletionStreamResponse(
            id=request_id,
            object=chunk_object_type,
            created=created_time,
            choices=[],
            model=model_name,
        )
        try:
            dealer = await aiozmq.create_zmq_stream(zmq.DEALER, connect=f"ipc:///dev/shm/router_{self.pid}.ipc")
            dealer.write([b"", request_id.encode("utf-8")])
            choices = []
            current_waiting_time = 0
            while num_choices > 0:
                try:
                    raw_data = await asyncio.wait_for(dealer.read(), timeout=10)
                    current_waiting_time = 0
                except asyncio.TimeoutError:
                    current_waiting_time += 10
                    if current_waiting_time == 300:
                        status, msg = self.engine_client.check_health()
                        if not status:
                            if choices:
                                chunk.choices = choices
                                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                            raise ValueError(f"Engine is not healthy: {msg}")
                        else:
                            current_waiting_time = 0
                    await asyncio.sleep(0.01)
                    continue
                response = msgpack.unpackb(raw_data[-1])
                for res in response:
                    if res.get("error_code", 200) != 200:
                        raise ValueError("{}".format(res["error_msg"]))

                    self.engine_client.data_processor.process_response_dict(
                        res,
                        stream=True,
                        enable_thinking=enable_thinking,
                        include_stop_str_in_output=include_stop_str_in_output,
                    )

                    if res["metrics"]["first_token_time"] is not None:
                        arrival_time = res["metrics"]["first_token_time"]
                        inference_start_time = res["metrics"]["inference_start_time"]
                    else:
                        arrival_time = res["metrics"]["arrival_time"] - inference_start_time
                    if first_iteration:
                        num_prompt_tokens = len(prompt_token_ids)
                        num_cached_tokens = res.get("num_cached_tokens", 0)
                        for i in range(num_choices):
                            choice = ChatCompletionResponseStreamChoice(
                                index=i,
                                delta=DeltaMessage(
                                    role="assistant",
                                    content="",
                                    reasoning_content="",
                                    tool_calls=None,
                                    prompt_token_ids=None,
                                    completion_token_ids=None,
                                ),
                            )
                            if request.return_token_ids:
                                choice.delta.prompt_token_ids = list(prompt_token_ids)
                                choice.delta.text_after_process = text_after_process
                            chunk = ChatCompletionStreamResponse(
                                id=request_id,
                                object=chunk_object_type,
                                created=created_time,
                                choices=[choice],
                                model=model_name,
                            )
                            if include_continuous_usage:
                                chunk.usage = UsageInfo(
                                    prompt_tokens=num_prompt_tokens,
                                    completion_tokens=0,
                                    total_tokens=num_prompt_tokens,
                                    prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=num_cached_tokens),
                                )
                            yield f"data: {chunk.model_dump_json(exclude_unset=True)} \n\n"
                            api_server_logger.info(f"Chat Streaming response send_idx 0: {chunk.model_dump_json()}")
                        first_iteration = False

                    output = res["outputs"]
                    delta_text = output["text"]
                    output_top_logprobs = output["top_logprobs"]
                    previous_num_tokens += len(output["token_ids"])
                    logprobs_res: Optional[LogProbs] = None
                    if request.logprobs and output_top_logprobs is not None:
                        logprobs_res = self._create_chat_logprobs(
                            output_top_logprobs, request.logprobs, request.top_logprobs
                        )

                    if self.engine_client.data_processor.tool_parser_obj and not res["finished"]:
                        tool_delta_message = output["tool_delta_message"]
                        if tool_delta_message is None:
                            continue
                        delta_message = tool_delta_message
                        delta_message.reasoning_content = output.get("reasoning_content")
                        if delta_message.tool_calls:
                            tool_called = True
                    else:
                        delta_message = DeltaMessage(
                            content=delta_text,
                            reasoning_content=output.get("reasoning_content"),
                            prompt_token_ids=None,
                            completion_token_ids=None,
                            tool_calls=None,
                        )

                    choice = ChatCompletionResponseStreamChoice(
                        index=0,
                        delta=delta_message,
                        logprobs=logprobs_res,
                        arrival_time=arrival_time,
                    )
                    if res["finished"]:
                        num_choices -= 1
                        work_process_metrics.e2e_request_latency.observe(
                            time.time() - res["metrics"]["request_start_time"]
                        )
                        has_no_token_limit = request.max_tokens is None and request.max_completion_tokens is None
                        max_tokens = request.max_completion_tokens or request.max_tokens
                        if has_no_token_limit or previous_num_tokens != max_tokens:
                            choice.finish_reason = "stop"
                            if tool_called:
                                choice.finish_reason = "tool_calls"
                        else:
                            choice.finish_reason = "length"

                        if res.get("error_msg") is not None and "Recover" in res["error_msg"]:
                            choice.finish_reason = "recover_stop"

                    if request.return_token_ids:
                        choice.delta.completion_token_ids = list(output["token_ids"])
                        choice.delta.raw_prediction = output.get("raw_prediction")
                    if include_continuous_usage:
                        chunk.usage = UsageInfo(
                            prompt_tokens=num_prompt_tokens,
                            completion_tokens=previous_num_tokens,
                            total_tokens=num_prompt_tokens + previous_num_tokens,
                        )
                    choices.append(choice)

                    if len(choices) == max_streaming_response_tokens or res["finished"]:
                        chunk.choices = choices
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                        if res["finished"]:
                            api_server_logger.info(f"Chat Streaming response last send: {chunk.model_dump_json()}")
                        choices = []

                if choices:
                    chunk.choices = choices
                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    choices = []

            if include_usage:
                completion_tokens = previous_num_tokens
                usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens,
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=usage,
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        except Exception as e:
            error_data = self._create_streaming_error_response(
                f"request[{request_id}] generate stream error: {str(e)}, {str(traceback.format_exc())}"
            )
            yield f"data: {error_data}\n\n"
        finally:
            dealer.close()
            self.engine_client.semaphore.release()
            api_server_logger.info(f"release {self.engine_client.semaphore.status()}")
            yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        model_name: str,
        prompt_token_ids: list(),
        text_after_process: str,
    ):
        """
        Full chat completion generator.
        """
        created_time = int(time.time())
        final_res = None
        enable_thinking = request.chat_template_kwargs.get("enable_thinking") if request.chat_template_kwargs else None
        if enable_thinking is None:
            enable_thinking = request.metadata.get("enable_thinking") if request.metadata else None

        include_stop_str_in_output = request.include_stop_str_in_output

        try:
            dealer = await aiozmq.create_zmq_stream(zmq.DEALER, connect=f"ipc:///dev/shm/router_{self.pid}.ipc")
            dealer.write([b"", request_id.encode("utf-8")])
            final_res = None
            previous_num_tokens = 0
            current_waiting_time = 0
            logprob_contents = []
            completion_token_ids = []
            while True:
                try:
                    raw_data = await asyncio.wait_for(dealer.read(), timeout=10)
                    current_waiting_time = 0
                except asyncio.TimeoutError:
                    current_waiting_time += 10
                    if current_waiting_time == 300:
                        status, msg = self.engine_client.check_health()
                        if not status:
                            raise ValueError(f"Engine is not healthy: {msg}")
                        else:
                            current_waiting_time = 0
                    await asyncio.sleep(0.1)
                    continue

                response = msgpack.unpackb(raw_data[-1])
                task_is_finished = False
                for data in response:
                    if data.get("error_code", 200) != 200:
                        raise ValueError("{}".format(data["error_msg"]))
                    data = self.engine_client.data_processor.process_response_dict(
                        data,
                        stream=False,
                        enable_thinking=enable_thinking,
                        include_stop_str_in_output=include_stop_str_in_output,
                    )
                    # api_server_logger.debug(f"Client {request_id} received: {data}")
                    previous_num_tokens += len(data["outputs"]["token_ids"])
                    completion_token_ids.extend(data["outputs"]["token_ids"])
                    # The logprob for handling the response
                    output = data["outputs"]
                    output_top_logprobs = output["top_logprobs"]
                    if output_top_logprobs is not None:
                        logprobs_res = self._create_chat_logprobs(
                            output_top_logprobs, request.logprobs, request.top_logprobs
                        )
                        if logprobs_res and logprobs_res.content is not None:
                            logprob_contents.extend(logprobs_res.content)
                    if data["finished"]:
                        final_res = data
                        task_is_finished = True
                        break
                if task_is_finished:
                    break
        finally:
            dealer.close()
            self.engine_client.semaphore.release()
            api_server_logger.info(f"release {self.engine_client.semaphore.status()}")

        choices = []
        output = final_res["outputs"]
        message = ChatMessage(
            role="assistant",
            content=output["text"],
            reasoning_content=output.get("reasoning_content"),
            tool_calls=output.get("tool_call"),
            prompt_token_ids=prompt_token_ids if request.return_token_ids else None,
            completion_token_ids=completion_token_ids if request.return_token_ids else None,
            text_after_process=text_after_process if request.return_token_ids else None,
            raw_prediction=output.get("raw_prediction") if request.return_token_ids else None,
        )
        logprobs_full_res = None
        if logprob_contents:
            logprobs_full_res = LogProbs(content=logprob_contents)

        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            logprobs=logprobs_full_res,
            finish_reason=None,
        )
        has_no_token_limit = request.max_tokens is None and request.max_completion_tokens is None
        max_tokens = request.max_completion_tokens or request.max_tokens
        if has_no_token_limit or previous_num_tokens != max_tokens:
            choice.finish_reason = "stop"
            if self.engine_client.reasoning_parser == "ernie_x1" and output.get("finish_reason", "") == "tool_calls":
                choice.finish_reason = "tool_calls"
        else:
            choice.finish_reason = "length"

        if final_res.get("error_msg") is not None and "Recover" in final_res["error_msg"]:
            choice.finish_reason = "recover_stop"
        choices.append(choice)

        num_prompt_tokens = len(prompt_token_ids)
        num_generated_tokens = previous_num_tokens
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=final_res.get("num_cached_tokens", 0)),
        )
        work_process_metrics.e2e_request_latency.observe(time.time() - final_res["metrics"]["request_start_time"])
        res = ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )
        api_server_logger.info(f"Chat response: {res.model_dump_json()}")
        return res

    def _create_chat_logprobs(
        self,
        output_top_logprobs,
        request_logprobs: Optional[bool] = None,
        request_top_logprobs: Optional[int] = None,
    ) -> Optional[LogProbs]:
        """Create OpenAI-style logprobs for chat completions."""
        if output_top_logprobs is None or len(output_top_logprobs) < 3 or any(not lst for lst in output_top_logprobs):
            return None
        logprobs_res: Optional[LogProbs] = None
        for logprob_token_ids, logprobs, sampled_token_ranks in zip(
            output_top_logprobs[0], output_top_logprobs[1], output_top_logprobs[2]
        ):
            top_logprobs = LogprobsLists(
                logprob_token_ids=[logprob_token_ids],
                logprobs=[logprobs],
                sampled_token_ranks=[sampled_token_ranks],
            )
            step_logprobs_res = self._build_logprobs_response(
                request_logprobs=request_logprobs,
                response_logprobs=top_logprobs,
                request_top_logprobs=request_top_logprobs,
            )
            if logprobs_res is None:
                logprobs_res = step_logprobs_res
            else:
                logprobs_res.content.extend(step_logprobs_res.content)
        return logprobs_res

    def _build_logprobs_response(
        self,
        request_logprobs: bool,
        response_logprobs: Optional[LogprobsLists],
        request_top_logprobs: int,
    ) -> Optional[LogProbs]:
        """
        Construct a logprobs response object in line with the OpenAI style.
        Retain the complete top-k candidates and avoid circular references.
        """

        # Parameter validation
        if (
            response_logprobs is None
            or not request_logprobs
            or request_top_logprobs is None
            or request_top_logprobs < 0
        ):
            return None

        try:
            # The top-k candidates for the current token
            topk_token_ids = []
            topk_logprobs = []

            if response_logprobs.logprob_token_ids and len(response_logprobs.logprob_token_ids) > 0:
                topk_token_ids = response_logprobs.logprob_token_ids[0][: request_top_logprobs + 1]

            if response_logprobs.logprobs and len(response_logprobs.logprobs) > 0:
                topk_logprobs = response_logprobs.logprobs[0][: request_top_logprobs + 1]

            # Construct the candidate token structure (LogProbEntry) of topk
            top_logprob_entries: List[LogProbEntry] = []
            for tid, lp in zip(topk_token_ids, topk_logprobs):
                token_str = self.engine_client.data_processor.process_logprob_response(
                    [tid], clean_up_tokenization_spaces=False
                )
                token_bytes = token_str.encode("utf-8", errors="replace")
                if "\ufffd" in token_str:
                    token_str = "bytes:" + "".join(f"\\x{byte:02x}" for byte in token_bytes)
                entry = LogProbEntry(token=token_str, logprob=lp, bytes=list(token_bytes))
                top_logprob_entries.append(entry)
            # Construct the sampled token object (avoid sharing references with top_logprob_entries)
            sampled_entry = LogProbEntry(
                token=top_logprob_entries[0].token,
                logprob=top_logprob_entries[0].logprob,
                bytes=top_logprob_entries[0].bytes,
                top_logprobs=top_logprob_entries[1:],  # Here are the complete topk candidates
            )

            return LogProbs(content=[sampled_entry])

        except Exception as e:
            error_msg = f"Error in _build_logprobs_response: {e}, {str(traceback.format_exc())}"
            api_server_logger.error(error_msg)
            return None
