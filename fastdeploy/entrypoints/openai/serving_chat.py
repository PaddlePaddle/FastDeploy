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
import aiozmq
from aiozmq import zmq
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Callable, Optional, Union, List
import uuid

from fastapi import Request
from pydantic import BaseModel
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
    ChatCompletionResponseChoice,
    ChatCompletionStreamResponse,
    ChatCompletionResponseStreamChoice,
    ChatMessage,
    UsageInfo,
    PromptTokenUsageInfo,
    ChatCompletionResponse,
    ErrorResponse,
)
from fastdeploy.metrics.work_metrics import work_process_metrics

from fastdeploy.utils import api_server_logger

from fastdeploy.engine.request import RequestOutput



class OpenAIServingChat:
    """
    OpenAI-style chat completions serving
    """

    def __init__(self, engine_client, pid):
        self.engine_client = engine_client
        self.pid = pid

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest
    ):
        """
        Create a new chat completion using the specified parameters.
        """
        if request.user is not None:
            request_id = f"chatcmpl-{request.user}-{uuid.uuid4()}"
        else:
            request_id = f"chatcmpl-{uuid.uuid4()}"
        api_server_logger.info(f"create chat completion request: {request_id}")

        try:
            current_req_dict = request.to_dict_for_infer(request_id)
            current_req_dict["arrival_time"] = time.time()
            prompt_token_ids = self.engine_client.format_and_add_data(current_req_dict)
        except Exception as e:
            return ErrorResponse(code=400, message=str(e))

        del current_req_dict

        if request.stream:
            return self.chat_completion_stream_generator(
                request, request_id,
                request.model,
                prompt_token_ids)
        else:
            try:
                return await self.chat_completion_full_generator(
                    request, request_id,
                    request.model,
                    prompt_token_ids)
            except Exception as e:
                return ErrorResponse(code=400, message=str(e))

    def _create_streaming_error_response(self, message: str) -> str:
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
        prompt_token_ids: list()
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
        max_streaming_response_tokens = 1
        enable_thinking = None
        if request.metadata is not None and request.metadata.get("max_streaming_response_tokens", 1) > 1:
            max_streaming_response_tokens = request.metadata["max_streaming_response_tokens"]

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
            model=model_name
        )
        try:
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect=f"ipc:///dev/shm/router_{self.pid}.ipc"
            )
            dealer.write([b"", request_id.encode('utf-8')])
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
                    await asyncio.sleep(0.1)
                    continue
    
                res = json.loads(raw_data[-1].decode('utf-8'))
                if res.get("error_code", 200) != 200:
                    raise ValueError("{}".format(res["error_msg"]))
                if request.metadata is not None:
                    enable_thinking = request.metadata.get("enable_thinking")
                self.engine_client.data_processor.process_response_dict(
                    res, stream=True, enable_thinking=enable_thinking)

                if res['metrics']['first_token_time'] is not None:
                    arrival_time = res['metrics']['first_token_time']
                    inference_start_time = res['metrics']['inference_start_time']
                else:
                    arrival_time = res['metrics']['arrival_time'] - inference_start_time
                if first_iteration:
                    num_prompt_tokens = len(prompt_token_ids)
                    num_cached_tokens = res.get("num_cached_tokens", 0)
                    for i in range(num_choices):
                        choice = ChatCompletionResponseStreamChoice(
                            index=i,
                            delta=DeltaMessage(role="assistant", content="", reasoning_content="", tool_calls=None)
                        )
                        if request.metadata is not None and request.metadata.get("training", False):
                            choice.delta.token_ids = prompt_token_ids
                        chunk = ChatCompletionStreamResponse(
                            id=request_id,
                            object=chunk_object_type,
                            created=created_time,
                            choices=[choice],
                            model=model_name
                        )
                        if include_continuous_usage:
                            chunk.usage = UsageInfo(
                                prompt_tokens=num_prompt_tokens,
                                completion_tokens=0,
                                total_tokens=num_prompt_tokens,
                                prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=num_cached_tokens)
                            )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)} \n\n"
                    first_iteration = False

                output = res["outputs"]
                delta_text = output["text"]

                previous_num_tokens += len(output["token_ids"])
                delta_message = DeltaMessage(content=delta_text, reasoning_content=output.get("reasoning_content"), \
                    token_ids=output.get("token_ids"), tool_calls=output.get("tool_call_content", []))

                choice = ChatCompletionResponseStreamChoice(
                    index=0,
                    delta=delta_message,
                    arrival_time=arrival_time
                )
                if res["finished"]:
                    num_choices -= 1
                    work_process_metrics.e2e_request_latency.observe(time.time() - res["metrics"]["request_start_time"])
                    if request.max_tokens is None or previous_num_tokens != request.max_tokens:
                        choice.finish_reason = "stop"
                        if self.engine_client.reasoning_parser == "ernie_x1" and \
                                output.get("finish_reason", "") == "tool_calls":
                            choice.finish_reason = "tool_calls"
                    else:
                        choice.finish_reason = "length"

                if request.metadata is not None and request.metadata.get("training", False) and delta_text != "":
                    choice.delta.token_ids = output["token_ids"]
                if include_continuous_usage:
                    chunk.usage = UsageInfo(
                        prompt_tokens=num_prompt_tokens,
                        completion_tokens=previous_num_tokens,
                        total_tokens=num_prompt_tokens + previous_num_tokens
                    )
                choices.append(choice)

                if len(choices) == max_streaming_response_tokens or res["finished"]:
                    chunk.choices = choices
                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    choices = []


            if include_usage:
                completion_tokens = previous_num_tokens
                usage = UsageInfo(
                    prompt_tokens=num_prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=num_prompt_tokens + completion_tokens
                )
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    object=chunk_object_type,
                    created=created_time,
                    choices=[],
                    model=model_name,
                    usage=usage
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        except Exception as e:
            error_data = self._create_streaming_error_response(str(e))
            yield f"data: {error_data}\n\n"
        finally:
            dealer.close()
            yield "data: [DONE]\n\n"

    async def chat_completion_full_generator(
        self,
        request: ChatCompletionRequest,
        request_id: str,
        model_name: str,
        prompt_token_ids: list()
    ):
        """
        Full chat completion generator.
        """
        created_time = int(time.time())
        final_res = None
        enable_thinking = None
        try:
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect=f"ipc:///dev/shm/router_{self.pid}.ipc"
            )
            dealer.write([b"", request_id.encode('utf-8')])
            final_res = None
            previous_num_tokens = 0
            current_waiting_time = 0
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

                data = json.loads(raw_data[-1].decode('utf-8'))
                if data.get("error_code", 200) != 200:
                    raise ValueError("{}".format(data["error_msg"]))
                if request.metadata is not None:
                    enable_thinking = request.metadata.get("enable_thinking")
                data = self.engine_client.data_processor.process_response_dict(
                    data, stream=False, enable_thinking=enable_thinking)
                # api_server_logger.debug(f"Client {request_id} received: {data}")
                previous_num_tokens += len(data["outputs"]["token_ids"])
                if data["finished"]:
                    final_res = data
                    break
        finally:
            dealer.close()

        choices = []
        output = final_res["outputs"]
        message = ChatMessage(
            role="assistant",
            content=output["text"],
            reasoning_content=output.get("reasoning_content"),
            tool_calls=output.get("tool_call_content"),
            token_ids=output.get("token_ids")
        )

        choice = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason=None
        )
        if request.max_tokens is None or previous_num_tokens != request.max_tokens:
            choice.finish_reason = "stop"
            if self.engine_client.reasoning_parser == "ernie_x1" and \
                    output.get("finish_reason", "") == "tool_calls":
                choice.finish_reason = "tool_calls"
        else:
            choice.finish_reason = "length"
        choices.append(choice)

        num_prompt_tokens = len(prompt_token_ids)
        num_generated_tokens = previous_num_tokens
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
            prompt_tokens_details=PromptTokenUsageInfo(cached_tokens=final_res.get("num_cached_tokens", 0))
        )
        work_process_metrics.e2e_request_latency.observe(time.time() - final_res["metrics"]["request_start_time"])
        return ChatCompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage
        )
