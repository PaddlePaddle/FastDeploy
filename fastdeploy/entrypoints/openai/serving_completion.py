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
import json
from aiozmq import zmq
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Task
import time
from collections.abc import AsyncGenerator, AsyncIterator
from collections.abc import Sequence as GenericSequence
from typing import Optional, Union, cast, TypeVar, List
import uuid
from fastapi import Request

from fastdeploy.entrypoints.openai.protocol import ErrorResponse, CompletionRequest, CompletionResponse, CompletionStreamResponse, CompletionResponseStreamChoice, CompletionResponseChoice,UsageInfo
from fastdeploy.utils import api_server_logger
from fastdeploy.engine.request import RequestOutput


class OpenAIServingCompletion:
    """
    Implementation of OpenAI-compatible text completion API endpoints.
    
    Handles both streaming and non-streaming text completion requests.
    
    Attributes:
        engine_client: Client for communicating with the LLM engine
        pid: Process ID for ZMQ communication
    """
    def __init__(self, engine_client, pid):
        """
        Initialize the completion service.
        
        Args:
            engine_client: Client for engine communication
            pid: Process ID for ZMQ routing
        """
        self.engine_client = engine_client
        self.pid = pid

    async def create_completion(self, request: CompletionRequest):
        """
        Create text completion based on the given request.
        
        Args:
            request (CompletionRequest): Completion request parameters
            
        Returns:
            Union[AsyncGenerator, CompletionResponse, ErrorResponse]:
                - Streaming generator if request.stream=True
                - Full completion response if request.stream=False
                - ErrorResponse if validation fails
        """
        created_time = int(time.time())
        if request.user is not None:
            request_id = f"cmpl-{request.user}-{uuid.uuid4()}"
        else:
            request_id = f"cmpl-{uuid.uuid4()}"
        api_server_logger.info(f"initialize request {request_id}")
        request_prompt_ids = None
        request_prompts = None
        try:
            if isinstance(request.prompt, str):
                request_prompts = [request.prompt]
            elif isinstance(request.prompt, list) and all(isinstance(item,  int) for item in request.prompt):
                request_prompt_ids = [request.prompt]
            elif isinstance(request.prompt, list) and all(isinstance(item, str) for item in request.prompt):
                request_prompts = request.prompt
            elif isinstance(request.prompt, list):
                for item in request.prompt:
                    if isinstance(item, list) and all(isinstance(x, int) for x in item):
                        continue
                    else:
                        raise ValueError("Prompt must be a string, a list of strings or a list of integers.")
                request_prompt_ids = request.prompt
            else:
                raise ValueError("Prompt must be a string, a list of strings or a list of integers.")
        except Exception as e:
            return ErrorResponse(message=str(e), code=400)

        if request_prompt_ids is not None:
            request_prompts = request_prompt_ids
        num_choices = len(request_prompts)

        api_server_logger.info(f"start inference for request {num_choices}")

        try:
            for idx, prompt in enumerate(request_prompts):
                request_id_idx = f"{request_id}-{idx}"
                current_req_dict = request.to_dict_for_infer(request_id_idx, prompt)
                try:
                    current_req_dict["arrival_time"] = time.time()
                    self.engine_client.format_and_add_data(current_req_dict)
                except Exception as e:
                    return ErrorResponse(message=str(e), code=400)

                del current_req_dict

            if request.stream:
                return self.completion_stream_generator(
                    request=request,
                    num_choices = num_choices,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model
                )
            else:
                try:
                    return await self.completion_full_generator(
                        request=request,
                        num_choices=num_choices,
                        request_id=request_id,
                        created_time=created_time,
                        model_name=request.model
                    )
                except ValueError as e:
                    return ErrorResponse(code=400, message=str(e))

        except ValueError as e:
            return ErrorResponse(message=str(e), code=400)


    async def completion_full_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str,
    ):
        """
        Generate complete text response in one-shot mode.
        
        Args:
            request (CompletionRequest): Original request parameters
            num_choices (int): Number of prompt variations
            request_id (str): Unique request identifier
            created_time (int): Unix timestamp of creation
            model_name (str): Name of the model being used
            
        Returns:
            CompletionResponse: Complete text response with:
                - Generated text
                - Usage statistics
                - Finish reason
                
        Raises:
            ValueError: If engine communication fails or times out
        """
        dealer = None
        try:
            request_ids = [f"{request_id}-{i}" for i in range(num_choices)]
            # create dealer
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect=f"ipc:///dev/shm/router_{self.pid}.ipc"
            )

            for rid in request_ids:
                dealer.write([b"", rid.encode("utf-8")])

            valid_results = [dict()] * num_choices
            output_tokens = [0] * num_choices
            while num_choices > 0:
                try:
                    raw_data = await asyncio.wait_for(dealer.read(), timeout=300)
                except asyncio.TimeoutError:
                    status, msg = self.engine_client.check_health()
                    if not status:
                        raise ValueError(f"Engine is not healthy: {msg}")
                    else:
                        continue
                data = json.loads(raw_data[-1].decode("utf-8"))
                rid = int(data["request_id"].split("-")[-1])
                if data.get("error_code", 200) != 200:
                    raise ValueError("{}".format(data["error_msg"]))
                self.engine_client.data_processor.process_response_dict(
                    data, stream=False
                )
                output_tokens[rid] += len(data["outputs"]["token_ids"])
                if data.get("finished", False):
                    data["output_token_ids"] = output_tokens[rid]
                    valid_results[rid] = data
                    num_choices -= 1

            return self.request_output_to_completion_response(
                final_res_batch=valid_results,
                request=request,
                request_id=request_id,
                created_time=created_time,
                model_name=model_name
            )
        except Exception as e:
            api_server_logger.error(
                f"Error in completion_full_generator: {e}", exc_info=True
            )
            raise
        finally:
            if dealer is not None:
                dealer.close()


    async def completion_stream_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str
    ):
        """
        Generator for streaming text completion responses.
        
        Args:
            request (CompletionRequest): Original request parameters
            num_choices (int): Number of prompt variations
            request_id (str): Unique request identifier
            created_time (int): Unix timestamp of creation
            model_name (str): Name of the model being used
            
        Yields:
            str: Server-Sent Events (SSE) formatted chunks containing:
                - Partial completion results
                - Usage statistics (if enabled)
                - Error messages (if any)
                
        Note:
            Uses ZMQ for inter-process communication with the engine.
            Maintains streaming protocol compatibility with OpenAI API.
        """
        try:
            dealer = await aiozmq.create_zmq_stream(
                zmq.DEALER,
                connect=f"ipc:///dev/shm/router_{self.pid}.ipc"
            )

            for i in range(num_choices):
                req_id = f"{request_id}-{i}"
                dealer.write([b"", req_id.encode('utf-8')])  # 发送多路请求
            output_tokens = [0] * num_choices
            inference_start_time = [0] * num_choices
            first_iteration = [True] * num_choices
            max_streaming_response_tokens = 1
            if request.suffix is not None and request.suffix.get("max_streaming_response_tokens", 1) > 1:
                max_streaming_response_tokens = request.suffix["max_streaming_response_tokens"]
            choices = []


            while num_choices > 0:
                try:
                    raw_data = await asyncio.wait_for(dealer.read(), timeout=300)
                except asyncio.TimeoutError:
                    status, msg = self.engine_client.check_health()
                    if not status:
                        raise ValueError(f"Engine is not healthy: {msg}")
                    else:
                        continue


                res = json.loads(raw_data[-1].decode('utf-8'))
                idx = int(res["request_id"].split("-")[-1])
                if res.get("error_code", 200) != 200:
                    raise ValueError("{}".format(res["error_msg"]))

                if first_iteration[idx]:
                    if request.suffix is not None and request.suffix.get("training", False):
                        chunk = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name,
                            choices=[CompletionResponseStreamChoice(
                                index=idx,
                                text="",
                                token_ids=list(res["prompt_token_ids"])
                            )]
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    first_iteration[idx] = False


                self.engine_client.data_processor.process_response_dict(res, stream=True)
                if res['metrics'].get('first_token_time') is not None:
                    arrival_time = res['metrics']['first_token_time']
                    inference_start_time[idx] = res['metrics']['inference_start_time']
                else:
                    arrival_time = res['metrics']['arrival_time'] - inference_start_time[idx]
                # api_server_logger.info(f"{arrival_time}")

                output = res["outputs"]

                choices.append(CompletionResponseStreamChoice(
                    index=idx,
                    text=output["text"],
                    token_ids=output.get("token_ids"),
                    reasoning_content=output.get("reasoning_content"),
                    arrival_time=arrival_time
                ))
                if res["finished"]:
                    if request.max_tokens is None or output_tokens[idx] + 1 != request.max_tokens:
                        chunk.choices[0].finish_reason = "stop"
                    else:
                        chunk.choices[0].finish_reason = "length"

                output_tokens[idx] += 1

                if len(choices) == max_streaming_response_tokens or res["finished"]:
                    chunk = CompletionStreamResponse(
                        id=request_id,
                        created=created_time,
                        model=model_name,
                        choices=choices
                    )
                    choices = []

                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

                if res["finished"]:
                    num_choices -= 1
                    if getattr(request, "stream_options", None) and request.stream_options.include_usage:
                        usage_chunk = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name,
                            choices=[],
                            usage=UsageInfo(
                                prompt_tokens=len(res.get("prompt_token_ids", [])),
                                completion_tokens=output_tokens[idx]
                            )
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"


        except Exception as e:
            yield f"data: {ErrorResponse(message=str(e), code=400).model_dump_json(exclude_unset=True)}\n\n"
        finally:
            del request
            if dealer is not None:
                dealer.close()
            yield "data: [DONE]\n\n"


    def request_output_to_completion_response(
        self,
        final_res_batch: List[RequestOutput],
        request: CompletionRequest,
        request_id: str,
        created_time: int,
        model_name: str,
    ) -> CompletionResponse:
        """
        Convert raw engine outputs to OpenAI-compatible completion response.
        
        Args:
            final_res_batch (List[RequestOutput]): Batch of engine responses
            request (CompletionRequest): Original request parameters
            request_id (str): Unique request identifier
            created_time (int): Unix timestamp of creation
            model_name (str): Name of the model being used
            
        Returns:
            CompletionResponse: Formatted completion response with:
                - Generated text choices
                - Token usage statistics
        """
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for final_res in final_res_batch:
            prompt_token_ids = final_res["prompt_token_ids"]
            assert prompt_token_ids is not None
            prompt_text = final_res["prompt"]

            output = final_res["outputs"]
            if request.echo:
                assert prompt_text is not None
                if request.max_tokens == 0:
                    token_ids = prompt_token_ids
                    output_text = prompt_text
                else:
                    token_ids = [*prompt_token_ids, *output["token_ids"]]
                    output_text = prompt_text + output["text"]
            else:
                token_ids = output["token_ids"]
                output_text = output["text"]

            choice_data = CompletionResponseChoice(
                index=len(choices),
                text=output_text,
                reasoning_content=output.get('reasoning_content'),
                logprobs=None,
                finish_reason=None
            )
            choices.append(choice_data)

            num_generated_tokens += final_res["output_token_ids"]

            num_prompt_tokens += len(prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=num_generated_tokens,
            total_tokens=num_prompt_tokens + num_generated_tokens,
        )
        del request

        return CompletionResponse(
            id=request_id,
            created=created_time,
            model=model_name,
            choices=choices,
            usage=usage,
        )
