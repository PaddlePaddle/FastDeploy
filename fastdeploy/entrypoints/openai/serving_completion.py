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
import uuid
from typing import List

import aiozmq
import msgpack
from aiozmq import zmq

from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorResponse,
    UsageInfo,
)
from fastdeploy.utils import api_server_logger, get_host_ip


class OpenAIServingCompletion:
    def __init__(self, engine_client, pid, dist_init_ip):
        self.engine_client = engine_client
        self.pid = pid
        self.master_ip = dist_init_ip
        self.host_ip = get_host_ip()

    def _check_master(self):
        if self.master_ip is None:
            return True
        if self.host_ip == self.master_ip:
            return True
        return False

    async def create_completion(self, request: CompletionRequest):
        """
        Create a completion for the given prompt.
        """
        if not self._check_master():
            err_msg = f"Only master node can accept completion request, please send request to master node: {self.pod_ips[0]}"
            api_server_logger.error(err_msg)
            return ErrorResponse(message=err_msg, code=400)
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
            elif isinstance(request.prompt, list) and all(isinstance(item, int) for item in request.prompt):
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
        prompt_batched_token_ids = []
        try:
            for idx, prompt in enumerate(request_prompts):
                request_id_idx = f"{request_id}-{idx}"
                current_req_dict = request.to_dict_for_infer(request_id_idx, prompt)
                try:
                    current_req_dict["arrival_time"] = time.time()
                    prompt_batched_token_ids.append(self.engine_client.format_and_add_data(current_req_dict))
                except Exception as e:
                    return ErrorResponse(message=str(e), code=400)

                del current_req_dict

            if request.stream:
                return self.completion_stream_generator(
                    request=request,
                    num_choices=num_choices,
                    request_id=request_id,
                    created_time=created_time,
                    model_name=request.model,
                    prompt_batched_token_ids=prompt_batched_token_ids,
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
                    )
                except Exception as e:
                    return ErrorResponse(code=400, message=str(e))

        except Exception as e:
            return ErrorResponse(message=str(e), code=400)

    async def completion_full_generator(
        self,
        request: CompletionRequest,
        num_choices: int,
        request_id: str,
        created_time: int,
        model_name: str,
        prompt_batched_token_ids: list(),
    ):
        """
        Process the full completion request with multiple choices.
        """
        dealer = None
        try:
            request_ids = [f"{request_id}-{i}" for i in range(num_choices)]
            # create dealer
            dealer = await aiozmq.create_zmq_stream(zmq.DEALER, connect=f"ipc:///dev/shm/router_{self.pid}.ipc")

            for rid in request_ids:
                dealer.write([b"", rid.encode("utf-8")])

            valid_results = [dict()] * num_choices
            output_tokens = [0] * num_choices
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
                            raise ValueError(f"Engine is not healthy: {msg}")
                        else:
                            current_waiting_time = 0
                    await asyncio.sleep(0.1)
                    continue
                response = msgpack.unpackb(raw_data[-1])
                for data in response:
                    rid = int(data["request_id"].split("-")[-1])
                    if data.get("error_code", 200) != 200:
                        raise ValueError("{}".format(data["error_msg"]))

                    self.engine_client.data_processor.process_response_dict(data, stream=False)
                    output_tokens[rid] += len(data["outputs"]["token_ids"])
                    if data.get("finished", False):
                        data["output_token_ids"] = output_tokens[rid]
                        valid_results[rid] = data
                        num_choices -= 1
                        break

            return self.request_output_to_completion_response(
                final_res_batch=valid_results,
                request=request,
                request_id=request_id,
                created_time=created_time,
                model_name=model_name,
                prompt_batched_token_ids=prompt_batched_token_ids,
            )
        except Exception as e:
            api_server_logger.error(f"Error in completion_full_generator: {e}", exc_info=True)
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
        model_name: str,
        prompt_batched_token_ids: list(),
    ):
        """
        Process the stream completion request.
        """
        try:
            dealer = await aiozmq.create_zmq_stream(zmq.DEALER, connect=f"ipc:///dev/shm/router_{self.pid}.ipc")

            for i in range(num_choices):
                req_id = f"{request_id}-{i}"
                dealer.write([b"", req_id.encode("utf-8")])  # 发送多路请求
            output_tokens = [0] * num_choices
            inference_start_time = [0] * num_choices
            first_iteration = [True] * num_choices
            max_streaming_response_tokens = 1
            if request.suffix is not None and request.suffix.get("max_streaming_response_tokens", 1) > 1:
                max_streaming_response_tokens = request.suffix["max_streaming_response_tokens"]
            choices = []
            chunk = CompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=model_name,
                choices=choices,
            )

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
                            raise ValueError(f"Engine is not healthy: {msg}")
                        else:
                            current_waiting_time = 0
                    await asyncio.sleep(0.1)
                    continue

                response = msgpack.unpackb(raw_data[-1])
                for res in response:
                    idx = int(res["request_id"].split("-")[-1])
                    if res.get("error_code", 200) != 200:
                        raise ValueError("{}".format(res["error_msg"]))

                    if first_iteration[idx]:
                        if request.suffix is not None and request.suffix.get("training", False):
                            chunk = CompletionStreamResponse(
                                id=request_id,
                                created=created_time,
                                model=model_name,
                                choices=[
                                    CompletionResponseStreamChoice(
                                        index=idx,
                                        text="",
                                        token_ids=list(prompt_batched_token_ids[idx]),
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                        first_iteration[idx] = False

                    self.engine_client.data_processor.process_response_dict(res, stream=True)
                    if res["metrics"].get("first_token_time") is not None:
                        arrival_time = res["metrics"]["first_token_time"]
                        inference_start_time[idx] = res["metrics"]["inference_start_time"]
                    else:
                        arrival_time = res["metrics"]["arrival_time"] - inference_start_time[idx]

                    output = res["outputs"]

                    choices.append(
                        CompletionResponseStreamChoice(
                            index=idx,
                            text=output["text"],
                            token_ids=output.get("token_ids"),
                            tool_calls=output.get("tool_call_content"),
                            reasoning_content=output.get("reasoning_content"),
                            arrival_time=arrival_time,
                        )
                    )
                    if res["finished"]:
                        if request.max_tokens is None or output_tokens[idx] + 1 != request.max_tokens:
                            chunk.choices[0].finish_reason = "stop"
                            if (
                                self.engine_client.reasoning_parser == "ernie_x1"
                                and output.get("finish_reason", "") == "tool_calls"
                            ):
                                chunk.choices[0].finish_reason = "tool_calls"
                        else:
                            chunk.choices[0].finish_reason = "length"

                    output_tokens[idx] += 1

                    if len(choices) == max_streaming_response_tokens or res["finished"]:
                        chunk = CompletionStreamResponse(
                            id=request_id,
                            created=created_time,
                            model=model_name,
                            choices=choices,
                        )
                        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                        choices = []

                    if res["finished"]:
                        num_choices -= 1
                        if getattr(request, "stream_options", None) and request.stream_options.include_usage:
                            usage_chunk = CompletionStreamResponse(
                                id=request_id,
                                created=created_time,
                                model=model_name,
                                choices=[],
                                usage=UsageInfo(
                                    prompt_tokens=len(prompt_batched_token_ids[idx]),
                                    completion_tokens=output_tokens[idx],
                                ),
                            )
                            yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"
                if choices:
                    chunk.choices = choices
                    yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
                    choices = []

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
        prompt_batched_token_ids: list(),
    ) -> CompletionResponse:
        choices: List[CompletionResponseChoice] = []
        num_prompt_tokens = 0
        num_generated_tokens = 0

        for idx in range(len(final_res_batch)):
            final_res = final_res_batch[idx]
            prompt_token_ids = prompt_batched_token_ids[idx]
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
                token_ids=token_ids,
                index=len(choices),
                text=output_text,
                reasoning_content=output.get("reasoning_content"),
                tool_calls=output.get("tool_call_content"),
                logprobs=None,
                finish_reason=None,
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
