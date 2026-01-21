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
from collections.abc import AsyncGenerator

from typing_extensions import override

import fastdeploy.envs as envs
from fastdeploy.engine.pooling_params import PoolingParams
from fastdeploy.engine.request import PoolingRequestOutput, Request, RewardRequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    ChatRewardData,
    ChatRewardRequest,
    ChatRewardResponse,
    UsageInfo,
)
from fastdeploy.entrypoints.openai.serving_engine import ServeContext, ZmqOpenAIServing
from fastdeploy.utils import api_server_logger


class OpenAIServingReward(ZmqOpenAIServing):
    request_id_prefix = "reward"

    """
    OpenAI-style reward serving using pipeline pattern
    """

    def __init__(self, engine_client, models, cfg, pid, ips, max_waiting_time, chat_template):
        super().__init__(engine_client, models, cfg, pid, ips, max_waiting_time, chat_template)

    @override
    def _request_to_dict(self, ctx: ServeContext):
        request: ChatRewardRequest = ctx.request
        if not envs.ENABLE_V1_DATA_PROCESSOR:
            request_dict = super()._request_to_dict(ctx)
            if hasattr(request, "to_pooling_params"):
                pooling_params: PoolingParams = request.to_pooling_params()
                pooling_params.verify("reward", self.cfg.model_config)
                request_dict["pooling_params"] = pooling_params.to_dict()
                request_dict["metrics"] = {}
            return request_dict
        else:
            request_obj = None
            if hasattr(request, "to_pooling_params"):
                pooling_params: PoolingParams = request.to_pooling_params()
                pooling_params.verify("reward", self.cfg.model_config)
                request_obj = Request.from_generic_request(
                    req=request, request_id=ctx.request_id, pooling_params=pooling_params
                )
                request_obj.metrics.arrival_time = time.time()
                super()._process_chat_template_kwargs(request_obj)
            return request_obj

    @override
    def _request_to_batch_dicts(self, ctx: ServeContext):
        """
        Convert the request into dictionary format that can be sent to the inference server
        """
        request_dict = self._request_to_dict(ctx)
        request_dict["request_id"] = f"{ctx.request_id}_0"
        request_dicts = [request_dict]
        return request_dicts

    async def create_reward(self, request: ChatRewardRequest):
        """
        Create embeddings for the input texts using the pipeline pattern
        """
        request_id = self._generate_request_id(request)

        ctx = ServeContext[ChatRewardRequest](
            request=request,
            model_name=request.model,
            request_id=request_id,
        )
        idx = 0
        response: ChatRewardResponse = None
        generators: AsyncGenerator[ChatRewardResponse, None] = self.handle(ctx)
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
    def _build_response(self, ctx: ServeContext, request_output: dict):
        """Generate final reward response"""
        api_server_logger.info(f"[{ctx.request_id}] Reward RequestOutput received:{request_output}")

        base = PoolingRequestOutput.from_dict(request_output)
        reward_res = RewardRequestOutput.from_base(base)

        data = ChatRewardData(
            index=0,
            score=reward_res.outputs.score,
        )

        num_prompt_tokens = 0
        if reward_res.prompt_token_ids:
            num_prompt_tokens = len(reward_res.prompt_token_ids)

        usage = UsageInfo(
            prompt_tokens=num_prompt_tokens,
            total_tokens=num_prompt_tokens,
        )

        return ChatRewardResponse(
            id=ctx.request_id,
            created=ctx.created_time,
            model=ctx.model_name,
            data=[data],
            usage=usage,
        )
