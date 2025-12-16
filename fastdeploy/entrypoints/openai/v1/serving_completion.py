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

from fastdeploy.engine.async_llm import AsyncLLM
from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    CompletionLogprobs,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    ErrorInfo,
    ErrorResponse,
)
from fastdeploy.entrypoints.openai.serving_models import OpenAIServingModels
from fastdeploy.entrypoints.openai.v1.serving_base import (
    OpenAiServingBase,
    ServeContext,
    ServingResponseContext,
)
from fastdeploy.utils import ErrorType, api_server_logger
from fastdeploy.worker.output import LogprobsLists


class OpenAIServingCompletion(OpenAiServingBase):
    def __init__(
        self,
        engine_client: AsyncLLM,
        config,
        models: OpenAIServingModels,
        pid: int,
        ips,
        max_waiting_time: int,
    ) -> None:
        # Initialize parent class first to set up __semaphore
        super().__init__(engine_client, config, models, pid, ips, max_waiting_time)

    @override
    async def _preprocess(self, ctx: ServeContext[CompletionRequest]) -> None:
        request = ctx.request
        request_id = ctx.request_id

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
        ctx.preprocess_requests = []
        for idx, prompt in enumerate(request_prompts):
            request_id_idx = f"{request_id}_{idx}"
            current_req_dict = request.to_dict_for_infer(request_id_idx, prompt)
            current_req_dict["arrival_time"] = time.time()
            ctx.preprocess_requests.append(current_req_dict)

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
                    token_bytes = token_str.encode("utf-8", errors="replace")
                    token_str = "bytes:" + "".join(f"\\x{byte:02x}" for byte in token_bytes)
                if idx == 0:
                    tokens.append(token_str)
                    token_logprobs.append(lp)
                else:
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

    async def _build_stream_response(
        self,
        ctx: ServeContext[CompletionRequest],
        request_output: RequestOutput,
        response_ctx: ServingResponseContext,
    ) -> Any:
        request: CompletionRequest = ctx.request
        stream_options = request.stream_options
        if stream_options is None:
            include_usage = False
        else:
            include_usage = stream_options.include_usage
        request_id = ctx.request_id
        output = request_output.outputs
        try:
            if request_output.error_code != 200:
                raise ValueError("{}".format(request_output.error_msg))
            metrics = request_output.metrics
            arrival_time = None
            if metrics and metrics.first_token_time:
                arrival_time = metrics.first_token_time
                response_ctx.inference_start_time_dict[output.index] = metrics.inference_start_time
            else:
                arrival_time = metrics.arrival_time - response_ctx.inference_start_time_dict[output.index]

            send_idx = output.send_idx

            choice = CompletionResponseStreamChoice(
                index=output.index,
                text="",
                arrival_time=arrival_time,
                completion_token_ids=None,
                prompt_token_ids=None,
            )
            choices = [choice]
            chunk = CompletionStreamResponse(
                id=request_id,
                created=ctx.created_time,
                model=ctx.request.model,
                choices=choices,
            )
            choice.index = output.index
            if output.send_idx == 0:
                if request.return_token_ids:
                    choice.prompt_token_ids = request_output.prompt_token_ids
                    choice.prompt_tokens = request_output.prompt
                yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"

            if request.return_token_ids:
                choice.completion_token_ids = output.token_ids if output.token_ids else []
                choice.completion_tokens = output.completion_tokens
                choice.prompt_token_ids = None
                choice.prompt_tokens = None
            if request.logprobs and output.top_logprobs is not None:
                choice.logprobs = self._create_completion_logprobs(output.top_logprobs, request.logprobs, 0)
                # draft logprobs
                if request.include_draft_logprobs and output.draft_top_logprobs is not None:
                    choice.draft_logprobs = self._create_completion_logprobs(
                        output.draft_top_logprobs, request.logprobs, 0
                    )
            choice.text = output.text or ""
            choice.reasoning_content = output.reasoning_content or ""
            choice.tool_calls = request_output.accumulate_tool_calls if request_output.accumulate_tool_calls else None

            if request_output.finished:
                choice_completion_tokens = response_ctx.choice_completion_tokens_dict[output.index]
                choice.finish_reason = self._calc_finish_reason(
                    request_output, request.max_tokens, choice_completion_tokens
                )
                api_server_logger.info(f"Completion Streaming response last send: {chunk.model_dump_json()}")
            if send_idx == 0 and not request.return_token_ids:
                api_server_logger.info(f"Completion Streaming response send_idx 0: {chunk.model_dump_json()}")
            yield f"data: {chunk.model_dump_json()}\n\n"
            if request_output.finished and response_ctx.remain_choices == 0:
                if include_usage:
                    usage_chunk = CompletionStreamResponse(
                        id=request_id,
                        created=ctx.created_time,
                        model=ctx.model_name,
                        choices=[],
                        usage=response_ctx.usage,
                    )
                    yield f"data: {usage_chunk.model_dump_json(exclude_unset=True)}\n\n"
                yield "data: [DONE]\n\n"
        except Exception as e:
            api_server_logger.error(f"Error in completion_stream_generator: {e}, {str(traceback.format_exc())}")
            yield f"data: {ErrorResponse(error=ErrorInfo(message=str(e), code='400', type=ErrorType.INTERNAL_ERROR)).model_dump_json(exclude_unset=True)}\n\n"

    async def _build_full_response(
        self,
        ctx: ServeContext[CompletionRequest],
        accumula_output_map: dict[int, List[RequestOutput]],
        response_ctx: ServingResponseContext,
    ) -> CompletionResponse | None:
        """
        Process the full completion request with multiple choices.
        """
        try:
            choices: List[CompletionResponseChoice] = []

            for choice_index, respose_list in accumula_output_map.items():
                response: RequestOutput | None = None
                for response_current in respose_list:
                    if response_current.error_code != 200:
                        raise ValueError("{}".format(response_current.error_msg))
                    if response is None:
                        response = response_current
                    else:
                        response.accumulate(response_current)
                choice = self.build_completion_choice(choice_index, response, ctx)
                choices.append(choice)

            res = CompletionResponse(
                id=ctx.request_id,
                created=ctx.created_time,
                model=ctx.request.model,
                choices=choices,
                usage=response_ctx.usage,
            )
            api_server_logger.info(f"Completion response: {res.model_dump_json()}")
            return res
        except Exception as e:
            api_server_logger.error(f"Error in completion_full_generator: {e}", exc_info=True)
            return self._create_error_response(str(e))

    def build_completion_choice(
        self, index: int, final_res: RequestOutput, ctx: ServeContext[CompletionRequest]
    ) -> CompletionResponseChoice:

        output = final_res.outputs
        request = ctx.request
        if request.echo:
            output.token_ids = [*final_res.prompt_token_ids, *output.token_ids]
            final_res.prompt = final_res.prompt + (output.text or "")
        finish_reason = self._calc_finish_reason(final_res, request.max_tokens, len(output.token_ids))
        choice_data = CompletionResponseChoice(
            index=index,
            text=output.text or "",
            reasoning_content=output.reasoning_content or "",
            tool_calls=[output.tool_calls] if output.tool_calls else None,
            finish_reason=finish_reason,
        )

        if output.top_logprobs is not None:
            choice_data.logprobs = self._create_completion_logprobs(output.top_logprobs, request.logprobs, 0)

        if output.draft_top_logprobs is not None:
            choice_data.draft_logprobs = self._create_completion_logprobs(
                output.draft_top_logprobs, request.logprobs, 0
            )
        if request.return_token_ids:
            choice_data.prompt_tokens = final_res.prompt
            choice_data.prompt_token_ids = final_res.prompt_token_ids
            choice_data.completion_token_ids = output.token_ids
            choice_data.completion_tokens = output.completion_tokens
        return choice_data

    async def create_completion(
        self, request: CompletionRequest
    ) -> ErrorResponse | AsyncGenerator[str, Any] | CompletionResponse:
        """
        Create a new chat completion using the specified parameters.
        """
        ctx = ServeContext[CompletionRequest](
            request=request,
            model_name=request.model,
            request_id=self._generate_request_id(request),
        )
        return await self.handle(ctx)
