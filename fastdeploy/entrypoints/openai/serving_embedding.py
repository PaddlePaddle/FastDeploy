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

from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ZmqOpenAIServing
from fastdeploy.utils import api_server_logger


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
        yield self.handle(request)

    async def _build_final_response(self, request_id: str, request_output: RequestOutput):
        """Generate final embedding response"""

        api_server_logger.info(f"[{request_id}] Embedding response generated:{request_output}")

        num_prompt_tokens = 0
        if request_output["prompt_ids"]:
            num_prompt_tokens = len(request_output["prompt_ids"])
        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return EmbeddingResponse(
            id=request_id,
            created=None,
            model=None,
            data=[],
            usage=usage,
        )
