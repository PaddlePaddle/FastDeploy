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
from typing import Literal, Union

import numpy as np
from typing_extensions import assert_never, override

from fastdeploy.engine.request import EmbeddingOutput, EmbeddingRequestOutput
from fastdeploy.entrypoints.openai.protocol import (
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

    def __init__(self, engine_client, models, pid, ips, max_waiting_time):
        super().__init__(engine_client, models, pid, ips, max_waiting_time)

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

        generation = self.handle(ctx)
        async for response in generation:
            return response

    @override
    def _build_response(self, ctx: ServeContext):
        """Generate final embedding response"""

        embedding_res = EmbeddingRequestOutput.from_base(ctx.request_output)

        data = EmbeddingResponseData(
            index=0,
            embedding=_get_embedding(embedding_res.outputs, ctx.request.encoding_format),
        )

        api_server_logger.info(f"[{ctx.request_id}] Embedding response generated:{ctx.request_output}")

        num_prompt_tokens = 0
        if ctx.request_output.prompt_token_ids:
            num_prompt_tokens = len(ctx.request_output.prompt_token_ids)

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
