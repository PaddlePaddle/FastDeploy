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

import numpy as np

from fastdeploy.entrypoints.openai.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingResponseData,
    InvalidParameterException,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import OpenAIServing


class OpenAIServingEmbedding(OpenAIServing):
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
        return await self.handle(request)

    async def _preprocess_request(self, request: EmbeddingRequest) -> dict:
        """Preprocess embedding request"""
        # Validate encoding format
        encoding_format = request.encoding_format if request.encoding_format else "float"
        if encoding_format not in ["float", "base64"]:
            raise InvalidParameterException(f"Unsupported encoding format: {encoding_format}", "encoding_format")

        # Convert input to list if it's a single string
        input_texts = [request.input] if isinstance(request.input, str) else request.input

        return {"input_texts": input_texts, "encoding_format": encoding_format, "request": request}

    async def _process_response(self, response: dict, request: dict) -> dict:
        """Process engine response for embedding"""
        input_texts = request["input_texts"]
        encoding_format = request["encoding_format"]
        embeddings = []
        for text in input_texts:
            # req_dict = {"input": text, "arrival_time": time.time()}
            embedding = [1, 2]
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            if encoding_format == "base64":
                embedding = base64.b64encode(np.array(embedding).astype(np.float32)).decode("utf-8")

            embeddings.append(embedding)

        return {
            "embeddings": embeddings,
            "num_prompt_tokens": sum(len(text) for text in input_texts),  # Simplified token counting
            "encoding_format": encoding_format,
            "model": request.model,
        }

    async def _build_final_response(self, processed_data: dict, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate final embedding response"""
        data = [
            EmbeddingResponseData(index=idx, embedding=embedding)
            for idx, embedding in enumerate(processed_data["embeddings"])
        ]

        usage = UsageInfo(
            prompt_tokens=processed_data["num_prompt_tokens"],
            completion_tokens=0,
            total_tokens=processed_data["num_prompt_tokens"],
        )

        return EmbeddingResponse(object="list", data=data, model=processed_data["model"], usage=usage)
