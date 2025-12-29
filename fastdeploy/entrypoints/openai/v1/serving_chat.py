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

import time
import traceback
from typing import Any, AsyncGenerator, List, Optional

from typing_extensions import override

from fastdeploy.config import FDConfig
from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse,
    ChatMessage,
    CompletionTokenUsageInfo,
    DeltaMessage,
    ErrorResponse,
    LogProbEntry,
    LogProbs,
    PromptTokenUsageInfo,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ServeContext
from fastdeploy.entrypoints.openai.serving_models import OpenAIServingModels
from fastdeploy.entrypoints.openai.v1.serving_base import (
    OpenAiServingBase,
    ServingResponseContext,
)
from fastdeploy.input.tokenzier_client import AsyncTokenizerClient, ImageDecodeRequest
from fastdeploy.metrics.metrics import main_process_metrics
from fastdeploy.utils import api_server_logger
from fastdeploy.worker.output import LogprobsLists


class OpenAIServingChat(OpenAiServingBase):
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
        chat_template,
        enable_mm_output: Optional[bool] = False,
        tokenizer_base_url: Optional[str] = None,
    ) -> None:
        super().__init__(engine_client, config, models, pid, ips, max_waiting_time)
        self.chat_template = chat_template
        self.enable_mm_output = enable_mm_output
        self.tokenizer_base_url = tokenizer_base_url
        if tokenizer_base_url is not None:
            self.decoder_client = AsyncTokenizerClient(base_url=tokenizer_base_url)
        else:
            self.decoder_client = None

    def _get_thinking_status(self, request: ChatCompletionRequest) -> bool:
        """
        Get the thinking status from the request.
        """
        enable_thinking = request.chat_template_kwargs.get("enable_thinking") if request.chat_template_kwargs else None
        if enable_thinking is None:
            enable_thinking = request.metadata.get("enable_thinking") if request.metadata else None
        options = request.chat_template_kwargs.get("options") if request.chat_template_kwargs else None
        if options:
            thinking_mode = options.get("thinking_mode")
            if thinking_mode:
                if thinking_mode == "close" or thinking_mode == "false":
                    enable_thinking = False
                else:
                    enable_thinking = True
        if enable_thinking is None:
            enable_thinking = True
        return enable_thinking

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
            if logprobs_res is None or logprobs_res.content is None:
                logprobs_res = step_logprobs_res
            elif step_logprobs_res is not None and step_logprobs_res.content is not None:
                logprobs_res.content.extend(step_logprobs_res.content)
        return logprobs_res

    def _build_logprobs_response(
        self,
        request_logprobs: Optional[bool],
        response_logprobs: Optional[LogprobsLists],
        request_top_logprobs: Optional[int],
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

    @override
    async def _preprocess(self, ctx: ServeContext[ChatCompletionRequest]) -> None:
        request = ctx.request
        request_id = ctx.request_id
        current_req_dict = request.to_dict_for_infer(f"{request_id}_0")
        current_req_dict["kwargs"] = {}
        current_req_dict["kwargs"]["enable_thinking"] = self._get_thinking_status(request)
        if "chat_template" not in current_req_dict:
            current_req_dict["chat_template"] = self.chat_template
        ctx.preprocess_requests = [current_req_dict]

    async def _build_stream_response(
        self,
        ctx: ServeContext[ChatCompletionRequest],
        request_output: RequestOutput,
        response_ctx: ServingResponseContext,
    ) -> AsyncGenerator:
        request: ChatCompletionRequest = ctx.request
        stream_options = request.stream_options
        if stream_options is None:
            include_usage = False
            include_continuous_usage = False
        else:
            include_usage = stream_options.include_usage
            include_continuous_usage = stream_options.continuous_usage_stats
        request_id = ctx.request_id
        output = request_output.outputs

        metrics = request_output.metrics
        arrival_time = None
        if metrics and metrics.first_token_time:
            arrival_time = metrics.first_token_time
            response_ctx.inference_start_time_dict[output.index] = metrics.inference_start_time
        else:
            arrival_time = metrics.arrival_time - response_ctx.inference_start_time_dict[output.index]
        choice = ChatCompletionResponseStreamChoice(
            index=output.index,
            delta=DeltaMessage(
                role="assistant",
                reasoning_content="",
                content="",
                completion_token_ids=None,
                prompt_token_ids=None,
            ),
            arrival_time=arrival_time,
        )
        chunk: ChatCompletionStreamResponse = ChatCompletionStreamResponse(
            id=request_id,
            choices=[choice],
            model=ctx.request.model,
            created=ctx.created_time,
        )
        if output.send_idx == 0:
            if request.return_token_ids:
                choice.delta.prompt_token_ids = request_output.prompt_token_ids
                choice.delta.prompt_tokens = request_output.prompt
            if include_continuous_usage:
                chunk.usage = UsageInfo(
                    prompt_tokens=response_ctx.usage.prompt_tokens,
                    completion_tokens=0,
                    total_tokens=response_ctx.usage.prompt_tokens,
                    prompt_tokens_details=PromptTokenUsageInfo(
                        cached_tokens=response_ctx.usage.prompt_tokens_details.cached_tokens,
                        image_tokens=response_ctx.usage.prompt_tokens_details.image_tokens,
                        video_tokens=response_ctx.usage.prompt_tokens_details.video_tokens,
                    ),
                    completion_tokens_details=CompletionTokenUsageInfo(reasoning_tokens=0),
                )
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

        if request.return_token_ids:
            choice.delta.prompt_token_ids = None
            choice.delta.prompt_tokens = None
            if self.enable_mm_output:
                if choice.delta.multimodal_content:
                    choice.delta.multimodal_content[0]["completion_token_ids"] = output.token_ids
            else:
                choice.delta.completion_token_ids = output.token_ids if output.token_ids else []
            choice.delta.completion_tokens = output.completion_tokens
        if include_continuous_usage:
            chunk.usage = response_ctx.usage

        if self.enable_mm_output:
            if output.decode_type == 1:
                image = {"type": "image"}
                if self.decoder_client:
                    req_id = ctx.request_id
                    image_ret = await self.decoder_client.decode_image(
                        request=ImageDecodeRequest(req_id=req_id, data=output.token_ids)
                    )
                    if image_ret is not None:
                        image["url"] = image_ret["http_url"]
                choice.delta.multimodal_content = [image]
            else:
                choice.delta.multimodal_content = [
                    {
                        "type": "text",
                        "text": output.text,
                    }
                ]
        else:
            choice.delta.content = output.text or ""

        choice.delta.reasoning_content = output.reasoning_content or ""
        choice.delta.tool_calls = [output.tool_calls] if output.tool_calls else None

        if request.logprobs and output.top_logprobs is not None:
            choice.logprobs = self._create_chat_logprobs(output.top_logprobs, request.logprobs, request.top_logprobs)
            if request.include_draft_logprobs and output.draft_top_logprobs is not None:
                choice.draft_logprobs = self._create_chat_logprobs(
                    output.draft_top_logprobs, request.logprobs, request.top_logprobs
                )

        if request_output.finished:
            if request_output.metrics and request_output.metrics.request_start_time:
                main_process_metrics.e2e_request_latency.observe(
                    time.time() - request_output.metrics.request_start_time
                )
            max_tokens = request.max_completion_tokens or request.max_tokens
            choice_completion_tokens = response_ctx.choice_completion_tokens_dict[output.index]
            choice.finish_reason = self._calc_finish_reason(request_output, max_tokens, choice_completion_tokens)
            api_server_logger.info(f"Chat Streaming response last send: {chunk.model_dump_json()}")

        yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
        if request_output.finished and response_ctx.remain_choices == 0:
            if getattr(request, "stream_options", None) and include_usage:
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=ctx.created_time,
                    choices=[],
                    model=ctx.request.model,
                    usage=response_ctx.usage,
                )
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"
            yield "data: [DONE]\n\n"

    async def _build_full_response(
        self,
        ctx: ServeContext[ChatCompletionRequest],
        accumula_output_map: dict[int, list[RequestOutput]],
        response_ctx: ServingResponseContext,
    ) -> Any:
        """
        Full chat completion generator.
        """
        request = ctx.request

        # api_server_logger.debug(f"Client {ctx.request_id} received: {data}")
        # The logprob for handling the response
        choices: list[ChatCompletionResponseChoice] = []
        for choice_index, respose in accumula_output_map.items():
            choice = await self._create_chat_completion_choice(respose, ctx)
            choices.append(choice)

        choices = sorted(choices, key=lambda x: x.index)
        res = ChatCompletionResponse(
            id=ctx.request_id, model=request.model, choices=choices, created=ctx.created_time, usage=response_ctx.usage
        )
        api_server_logger.info(f"Chat response: {res.model_dump_json()}")
        return res

    async def _create_chat_completion_choice(
        self,
        request_output_list: list[RequestOutput],
        ctx: ServeContext[ChatCompletionRequest],
    ) -> ChatCompletionResponseChoice:
        request = ctx.request
        message = ChatMessage(
            role="assistant",
        )
        request_output = None
        if self.enable_mm_output:
            multipart = []
            for current_output in request_output_list:
                if current_output.outputs.decode_type == 0:
                    multipart.append({"type": "text", "text": current_output.outputs.text})
                elif current_output.outputs.decode_type == 1:
                    image = {"type": "image"}
                    image_ret = await self.decoder_client.decode_image(
                        request=ImageDecodeRequest(req_id=ctx.request_id, data=current_output.outputs.token_ids)
                    )
                    if image_ret and image_ret["http_url"]:
                        image["url"] = image_ret["http_url"]
                    multipart.append(image_ret)
                if request_output:
                    request_output.accumulate(current_output)
                else:
                    request_output = current_output
            message.multimodal_content = multipart
        else:
            request_output = request_output_list[-1]
            message.content = request_output.outputs.text
        output = request_output.outputs
        message.reasoning_content = output.reasoning_content
        message.tool_calls = request_output.accumulate_tool_calls if request_output.accumulate_tool_calls else None
        if output is not None and request_output.metrics and request_output.metrics.request_start_time:
            main_process_metrics.e2e_request_latency.observe(time.time() - request_output.metrics.request_start_time)

        if request.return_token_ids:
            message.prompt_token_ids = request_output.prompt_token_ids
            message.completion_token_ids = output.token_ids
            message.prompt_tokens = request_output.prompt
            message.completion_tokens = output.completion_tokens
        max_tokens = request.max_completion_tokens or request.max_tokens
        finish_reason = self._calc_finish_reason(request_output, max_tokens, len(output.token_ids))
        choice = ChatCompletionResponseChoice(
            index=output.index,
            message=message,
            finish_reason=finish_reason,
        )

        if output.top_logprobs is not None:
            # logprobs
            choice.logprobs = self._create_chat_logprobs(output.top_logprobs, request.logprobs, request.top_logprobs)
            if request.include_draft_logprobs and output.draft_top_logprobs is not None:
                choice.draft_logprobs = self._create_chat_logprobs(
                    output.draft_top_logprobs, request.logprobs, request.top_logprobs
                )
        return choice

    async def create_chat_completion(
        self, request: ChatCompletionRequest
    ) -> ErrorResponse | AsyncGenerator[str, Any] | ChatCompletionResponse:
        """
        Create a new chat completion using the specified parameters.
        """
        ctx = ServeContext[ChatCompletionRequest](
            request=request,
            model_name=request.model,
            request_id=self._generate_request_id(request),
        )
        return await self.handle(ctx)
