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

import uuid
from abc import abstractmethod
from typing import Any, AsyncGenerator, List, Literal, Optional, Union

from typing_extensions import override

from fastdeploy.config import FDConfig
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    CompletionTokenUsageInfo,
    ErrorResponse,
    PromptTokenUsageInfo,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import OpenAIServing, ServeContext
from fastdeploy.entrypoints.openai.serving_models import OpenAIServingModels
from fastdeploy.trace.constants import LoggingEventName
from fastdeploy.trace.trace_logger import print as trace_print
from fastdeploy.utils import api_server_logger, get_host_ip


class ServingResponseContext:
    def __init__(self):
        self.usage = UsageInfo()
        self.choice_completion_tokens_dict = {}
        self.inference_start_time_dict = {}
        self.remain_choices: Optional[int] = None


class OpenAiServingBase(OpenAIServing):
    """
    OpenAI-style chat completions serving
    """

    def __init__(
        self,
        engine_client: AsyncLLM,
        config: FDConfig,
        models: OpenAIServingModels,
        pid: int,
        ips,
        max_waiting_time: int,
    ) -> None:
        # Initialize parent class first to set up __semaphore
        super().__init__(models, config, pid, ips, max_waiting_time)
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
        self.eoi_token_id = 101032
        api_server_logger.info(f"master ip: {self.master_ip}")

    @override
    def _check_master(self) -> bool:
        return self.is_master_ip

    @override
    def _generate_request_id(self, request: Union[ChatCompletionRequest, CompletionRequest]) -> str:
        """Generate a unique request ID"""
        if request.request_id is not None:
            request_id = request.request_id
            if not request_id.startswith("chatcmpl-"):
                request_id = f"chatcmpl-{request_id}"
        elif request.user is not None:
            request_id = f"chatcmpl-{request.user}-{uuid.uuid4()}"
        else:
            request_id = f"chatcmpl-{uuid.uuid4()}"
        return request_id

    @override
    async def _preprocess(self, ctx: ServeContext[Union[ChatCompletionRequest, CompletionRequest]]) -> None:
        request = ctx.request
        request_id = ctx.request_id
        current_req_dict = request.to_dict_for_infer(f"{request_id}_0")
        ctx.preprocess_requests = [current_req_dict]

    @override
    async def _prepare_generators(self, ctx: ServeContext) -> AsyncGenerator[RequestOutput, None]:
        """Process engine response into final format"""
        if ctx.preprocess_requests is None:
            raise ValueError("preprocess_requests is None")
        for request_dict in ctx.preprocess_requests:
            kwargs = request_dict.pop("kwargs") if request_dict.get("kwargs") else {}
            generator: AsyncGenerator[RequestOutput, None] = self.engine_client.generate(
                request_dict, request_id=ctx.request_id, **kwargs
            )
            async for response in generator:
                yield response

    @override
    def _build_response(
        self,
        ctx: ServeContext[ChatCompletionRequest | CompletionRequest],
        request_output: RequestOutput,
    ) -> Any:
        """Generate the final response object"""
        return request_output

    async def handle(self, ctx: ServeContext[Any]) -> Union[AsyncGenerator, ErrorResponse]:
        if ctx.request.stream:
            return self.handle_stream(ctx)
        else:
            return await self.handle_non_stream(ctx)

    async def handle_stream(self, ctx: ServeContext) -> Union[AsyncGenerator, ErrorResponse]:
        """Handle incoming requests"""
        response_ctx: ServingResponseContext = ServingResponseContext()
        # 获取生成器 (假定 _pipeline 调用后返回的是一个 AsyncGenerator)
        try:
            generator: AsyncGenerator[RequestOutput] = self._pipeline(ctx)
            choice_accumulate_buffer: dict[int, RequestOutput] = {}
            async for request_output in generator:
                response_ctx.usage.add(self._calc_usage(request_output))
                outputs = request_output.outputs
                choice_completion_tokens = response_ctx.choice_completion_tokens_dict.get(outputs.index, 0)
                choice_completion_tokens += len(outputs.token_ids)
                response_ctx.choice_completion_tokens_dict[outputs.index] = choice_completion_tokens
                if request_output.finished:
                    if response_ctx.remain_choices is None:
                        response_ctx.remain_choices = len(ctx.preprocess_requests) * (
                            1 if ctx.request.n is None else ctx.request.n
                        )
                    response_ctx.remain_choices -= 1
                if outputs.decode_type == 1:
                    acc_output = choice_accumulate_buffer.get(outputs.index)
                    if acc_output is None:
                        choice_accumulate_buffer[outputs.index] = request_output
                        acc_output = request_output
                    else:
                        acc_output.accumulate(request_output)
                    continue
                elif (
                    self.eoi_token_id
                    and self.eoi_token_id in outputs.token_ids
                    and choice_accumulate_buffer.get(outputs.index)
                ):
                    acc_output = choice_accumulate_buffer.pop(outputs.index)
                    response_generator = self._build_stream_response(ctx, acc_output, response_ctx)
                    async for stream_response in response_generator:
                        yield stream_response
                response_generator = self._build_stream_response(ctx, request_output, response_ctx)
                async for stream_response in response_generator:
                    yield stream_response
        finally:
            trace_print(LoggingEventName.POSTPROCESSING_END, ctx.request_id, getattr(ctx.request, "user", ""))

    @abstractmethod
    async def _build_stream_response(
        self,
        ctx: ServeContext[ChatCompletionRequest],
        request_output: RequestOutput,
        response_ctx: ServingResponseContext,
    ) -> AsyncGenerator:
        pass

    async def handle_non_stream(self, ctx: ServeContext[ChatCompletionRequest | CompletionRequest]) -> Any:
        """Handle non-streaming requests"""
        accumula_output_map: dict[int, list[RequestOutput]] = {}
        response_ctx: ServingResponseContext = ServingResponseContext()
        try:
            generator: AsyncGenerator[RequestOutput] = self._pipeline(ctx)
            async for request_output in generator:
                choice_res_acc = accumula_output_map.get(request_output.outputs.index)
                if choice_res_acc is None:
                    accumula_output_map[request_output.outputs.index] = [request_output]
                else:
                    last_acc = choice_res_acc[-1]
                    if last_acc.outputs.decode_type == request_output.outputs.decode_type:
                        last_acc.accumulate(request_output)
                    else:
                        accumula_output_map[request_output.outputs.index].append(request_output)
                response_ctx.usage.add(self._calc_usage(request_output))
            return await self._build_full_response(ctx, accumula_output_map, response_ctx)
        finally:
            trace_print(LoggingEventName.POSTPROCESSING_END, ctx.request_id, getattr(ctx.request, "user", ""))

    async def _build_full_response(
        self,
        ctx: ServeContext[ChatCompletionRequest | CompletionRequest],
        accumula_output_map: dict[int, List[RequestOutput]],
        response_ctx: ServingResponseContext,
    ) -> Any:
        pass

    def _calc_finish_reason(
        self, request_output: RequestOutput, max_tokens: Optional[int], token_nums: int
    ) -> Literal["stop", "length", "tool_calls", "recover_stop"]:
        finish_reason = "stop"
        if request_output.outputs.tool_calls:
            finish_reason = "tool_calls"
        if max_tokens is not None and token_nums >= max_tokens:
            finish_reason = "length"
        if request_output.error_msg is not None and "Recover" in request_output.error_msg:
            finish_reason = "recover_stop"
        return finish_reason

    def _calc_usage(self, request_output: RequestOutput) -> UsageInfo:
        outputs = request_output.outputs
        num_prompt_tokens = (
            len(request_output.prompt_token_ids) if request_output.prompt_token_ids and outputs.send_idx == 0 else 0
        )
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            completion_tokens=len(outputs.token_ids),
            total_tokens=num_prompt_tokens + len(outputs.token_ids),
            prompt_tokens_details=PromptTokenUsageInfo(
                cached_tokens=request_output.num_cached_tokens,
                image_tokens=request_output.num_input_image_tokens,
                video_tokens=request_output.num_input_video_tokens,
            ),
            completion_tokens_details=CompletionTokenUsageInfo(
                reasoning_tokens=outputs.reasoning_token_num,
                image_tokens=len(outputs.token_ids) if outputs.decode_type == 1 else 0,
            ),
        )
        return usage
