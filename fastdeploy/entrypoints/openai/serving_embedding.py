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

import base64
from collections.abc import AsyncGenerator
from typing import Literal, Union

import numpy as np
from typing_extensions import assert_never, override

from fastdeploy.engine.pooling_params import PoolingParams
from fastdeploy.engine.request import (
    EmbeddingOutput,
    EmbeddingRequestOutput,
    PoolingRequestOutput,
)
from fastdeploy.entrypoints.openai.protocol import (
    EmbeddingCompletionRequest,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ServeContext, ZmqOpenAIServing
from fastdeploy.utils import api_server_logger


def _get_embedding(
    output: EmbeddingOutput,
    encoding_format: Literal["float", "base64"],
) -> Union[list[float], str]:
    if encoding_format == "float":
        return output.embedding
    elif encoding_format == "base64":
        # Force to use float32 for base64 encoding
        # to match the OpenAI python client behavior
        embedding_bytes = np.array(output.embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")

    assert_never(encoding_format)


class OpenAIServingEmbedding(ZmqOpenAIServing):
    request_id_prefix = "embd"

    """
    OpenAI-style embedding serving using pipeline pattern
    """

    def __init__(self, engine_client, models, cfg, pid, ips, max_waiting_time, chat_template):
        super().__init__(engine_client, models, cfg, pid, ips, max_waiting_time, chat_template)

    @override
    def _request_to_dict(self, ctx: ServeContext):
        request: EmbeddingRequest = ctx.request
        request_dict = super()._request_to_dict(ctx)
        if hasattr(request, "to_pooling_params"):
            pooling_params: PoolingParams = request.to_pooling_params()
            pooling_params.verify("embed", self.cfg.model_config)
            request_dict["pooling_params"] = pooling_params.to_dict()
        return request_dict

    @override
    def _request_to_batch_dicts(self, ctx: ServeContext):
        """
        Convert the request into dictionary format that can be sent to the inference server
        """
        request_dicts = []
        if isinstance(ctx.request, EmbeddingCompletionRequest):
            # Union[list[int], list[list[int]], str, list[str]]
            request: EmbeddingCompletionRequest = ctx.request
            if isinstance(request.input, str):
                request_prompts = [request.input]
            elif isinstance(request.input, list) and all(isinstance(item, int) for item in request.input):
                request_prompts = [request.input]
            elif isinstance(request.input, list) and all(isinstance(item, str) for item in request.input):
                request_prompts = request.input
            elif isinstance(request.input, list):
                for item in request.input:
                    if isinstance(item, list) and all(isinstance(x, int) for x in item):
                        continue
                    else:
                        raise ValueError("If prompt is a list, each item type must be one of: str, list[int]")
                request_prompts = request.input
            else:
                raise ValueError("Prompt type must be one of: str, list[str], list[int], list[list[int]]")

            for idx, prompt in enumerate(request_prompts):
                request_dict = self._request_to_dict(ctx)
                request_dict["request_id"] = f"{ctx.request_id}_{idx}"
                request_dict["prompt"] = prompt
                request_dicts.append(request_dict)
        else:
            request_dict = self._request_to_dict(ctx)
            request_dict["request_id"] = f"{ctx.request_id}_0"
            request_dicts = [request_dict]
        return request_dicts

    async def create_embedding(self, request: EmbeddingRequest):
        """
        Create embeddings for the input texts using the pipeline pattern
        """
        request_id = self._generate_request_id(getattr(request, "user", None))

        ctx = ServeContext[EmbeddingRequest](
            request=request,
            model_name=request.model,
            request_id=request_id,
        )

        idx = 0
        response: EmbeddingResponse = None
        generators: AsyncGenerator[EmbeddingResponse, None] = self.handle(ctx)
        async for r in generators:
            r.data[0].index = idx
            idx += 1
            if response is None:
                response = r
            else:
                response.data.append(r.data[0])
                response.usage.prompt_tokens += r.usage.prompt_tokens
                response.usage.total_tokens += r.usage.total_tokens

        return response

    @override
    def _build_response(self, ctx: ServeContext):
        """Generate final embedding response"""
        api_server_logger.info(f"[{ctx.request_id}] Embedding RequestOutput received:{ctx.request_output}")

        base = PoolingRequestOutput.from_dict(ctx.request_output)
        embedding_res = EmbeddingRequestOutput.from_base(base)

        data = EmbeddingResponseData(
            index=0,
            embedding=_get_embedding(embedding_res.outputs, ctx.request.encoding_format),
        )

        num_prompt_tokens = 0
        if embedding_res.prompt_token_ids:
            num_prompt_tokens = len(embedding_res.prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return EmbeddingResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=[data],
            usage=usage,
        )
