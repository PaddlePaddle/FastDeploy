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
import traceback
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, ClassVar, Generic, Optional, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from fastdeploy.engine.request import PoolingRequestOutput, RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    InvalidParameterException,
)
from fastdeploy.utils import ErrorCode, ErrorType, api_server_logger

RequestT = TypeVar("RequestT")


class ServeContext(
    BaseModel,
    Generic[RequestT],
):
    # Shared across all requests
    request: RequestT
    model_name: str
    request_id: str
    created_time: int = Field(default_factory=lambda: int(time.time()))
    preprocess_requests: Optional[list[dict]] = None
    request_output: Optional[Union[RequestOutput, PoolingRequestOutput]] = None

    # `protected_namespaces` resolves Pydantic v2's warning
    # on conflict with protected namespace "model_"
    model_config = ConfigDict(
        protected_namespaces=(),
        arbitrary_types_allowed=True,
    )


class OpenAIServing(ABC, Generic[RequestT]):
    request_id_prefix: ClassVar[str]
    """
    Base pipeline for OpenAI-style serving implementations
    """

    def __init__(self, engine_client, models, cfg, pid, ips, max_waiting_time):
        self.engine_client = engine_client
        self.models = models
        self.cfg = cfg
        self.pid = pid
        self.max_waiting_time = max_waiting_time

        # Parse master IP
        if ips is not None:
            if isinstance(ips, list):
                self.master_ip = ips[0]
            else:
                self.master_ip = ips.split(",")[0]
        else:
            self.master_ip = "0.0.0.0"

        api_server_logger.info(f"master ip: {self.master_ip}")

    def _check_master(self) -> bool:
        """Check if current node is master"""
        return self.engine_client.is_master

    def _check_supported_model(self, model_name: str) -> tuple[bool, str]:
        """Check if model is supported and return adjusted model name"""
        if not self.models:
            return True, model_name
        is_supported, adjusted_name = self.models.is_supported_model(model_name)
        if not is_supported:
            err_msg = f"Unsupported model: [{model_name}]"
            api_server_logger.error(err_msg)
        return is_supported, adjusted_name

    async def _acquire_semaphore(self, request_id: str) -> bool:
        """Acquire engine client semaphore with timeout"""
        try:
            api_server_logger.info(f"Acquire request:{request_id} status:{self.engine_client.semaphore.status()}")
            if self.max_waiting_time < 0:
                await self.engine_client.semaphore.acquire()
            else:
                await asyncio.wait_for(self.engine_client.semaphore.acquire(), timeout=self.max_waiting_time)
            return True
        except asyncio.TimeoutError:
            self._release_semaphore(request_id)
            error_msg = f"Request waiting timeout, request:{request_id} max waiting time:{self.max_waiting_time}"
            api_server_logger.error(error_msg)
            return False

    def _release_semaphore(self, request_id: str) -> None:
        """Release engine client semaphore"""
        self.engine_client.semaphore.release()
        api_server_logger.info(f"Release request:{request_id} status:{self.engine_client.semaphore.status()}")

    def _create_error_response(
        self,
        message: str,
        error_type: ErrorType = ErrorType.INTERNAL_ERROR,
        code: Optional[ErrorCode] = ErrorCode.INTERNAL_ERROR,
        param: Optional[str] = None,
    ) -> ErrorResponse:
        """Create standardized error response"""
        traceback.print_exc()
        api_server_logger.error(message)
        return ErrorResponse(error=ErrorInfo(message=message, type=error_type, code=code, param=param))

    def _generate_request_id(self, user: Optional[str] = None) -> str:
        """Generate a unique request ID"""
        if user is not None:
            return f"{self.request_id_prefix}-{user}-{uuid.uuid4()}"
        return f"{self.request_id_prefix}-{uuid.uuid4()}"

    def _validate_request(self, ctx: ServeContext):
        """Validate the request before processing"""
        pass

    @abstractmethod
    async def _preprocess(self, ctx: ServeContext):
        """Preprocess the request into engine format"""
        pass

    @abstractmethod
    async def _prepare_generators(self, ctx: ServeContext) -> Any:
        """Process engine response into final format"""
        # 此函数是一个异步方法，用于处理引擎响应并将其转换为最终格式
        pass

    @abstractmethod
    def _build_response(self, ctx: ServeContext) -> Any:
        """Generate the final response object"""
        pass

    async def handle(self, ctx: ServeContext) -> Union[Any, ErrorResponse]:
        """Handle incoming requests"""
        generation = self._pipeline(ctx)

        async for response in generation:
            yield response

    async def _pipeline(self, ctx: ServeContext) -> Union[Any, ErrorResponse]:
        """
        Pipeline for handling requests
        Args:
            reqeust: The request to be handled
        Returns:
            A generator that yields responses
        """
        # Step 1: Request validation
        # Step 1.1: Check if current node is master
        if not self._check_master():
            yield self._create_error_response(
                f"Only master node can accept request, please send to master node: {self.master_ip}"
            )

        request = ctx.request
        # Step 1.2: Check supported model
        is_supported, request.model = self._check_supported_model(ctx.model_name)
        if not is_supported:
            yield self._create_error_response(
                f"Unsupported model: [{request.model}]", ErrorType.API_CONNECTION_ERROR, ErrorCode.MODEL_NOT_SUPPORT
            )

        # Step 1.3: Validate request
        self._validate_request(ctx)

        request_id = self._generate_request_id(getattr(request, "user", None))
        api_server_logger.info(f"Initialize request {request_id}: {request}")

        # Step 2: Semaphore acquisition
        if not await self._acquire_semaphore(request_id):
            yield self._create_error_response("Request waiting timeout", ErrorType.TIMEOUT_ERROR, ErrorCode.TIMEOUT)

        try:
            # Step 3: Preprocessing
            await self._preprocess(ctx)

            # Step 4: Response processing
            generators = self._prepare_generators(ctx)

            # Step 5: Final response build
            async for request_output in generators:
                ctx.request_output = request_output
                yield self._build_response(ctx)

        except InvalidParameterException as e:
            traceback.print_exc()
            yield self._create_error_response(str(e.message), ErrorType.INVALID_REQUEST_ERROR, param=e.param)
        except Exception as e:
            traceback.print_exc()
            yield self._create_error_response(str(e))
        finally:
            self._release_semaphore(request_id)


class ZmqOpenAIServing(OpenAIServing):
    """
    OpenAI-style service architecture using ZeroMQ as the communication mechanism.
    """

    def __init__(self, engine_client, models, cfg, pid, ips, max_waiting_time, chat_template):
        super().__init__(engine_client, models, cfg, pid, ips, max_waiting_time)
        self.chat_template = chat_template

    def _request_to_dict(self, ctx: ServeContext):
        request = ctx.request
        if hasattr(request, "to_dict_for_infer"):
            request_dict = request.to_dict_for_infer(ctx.request_id)
        else:
            request_dict = request.dict()
        request_dict["request_id"] = ctx.request_id
        request_dict["arrival_time"] = time.time()

        self._process_chat_template_kwargs(request_dict)
        return request_dict

    def _request_to_batch_dicts(self, ctx: ServeContext):
        """Convert multiple requests to dictionary form"""
        return [self._request_to_dict(ctx)]

    @override
    async def _preprocess(self, ctx: ServeContext):
        """Preprocess the request into engine format"""
        request_dicts = self._request_to_batch_dicts(ctx)
        ctx.preprocess_requests = request_dicts
        for request_dict in request_dicts:
            api_server_logger.info(f"batch add request_id: {request_dict['request_id']}, request: {request_dict}")
            await self.engine_client.format_and_add_data(request_dict)

    def _process_chat_template_kwargs(self, request_dict):
        """Add default values to chat template kwargs"""
        if "chat_template" not in request_dict:
            request_dict["chat_template"] = self.chat_template
        chat_template_kwargs = request_dict.get("chat_template_kwargs") or {}
        chat_template_kwargs.update(
            {
                "chat_template": request_dict.get("chat_template"),
                "add_generation_prompt": request_dict.get("add_generation_prompt"),
                "add_stop_sequences": request_dict.get("add_stop_sequences"),
            }
        )
        request_dict["chat_template_kwargs"] = chat_template_kwargs

    @override
    async def _prepare_generators(self, ctx: ServeContext) -> AsyncGenerator[dict]:
        """Prepare a generator of responses"""
        request_id = ctx.request_id
        try:
            num_choices = len(ctx.preprocess_requests)
            dealer, request_output_queue = await self.engine_client.connection_manager.get_connection(
                request_id, num_choices
            )
            for pr in ctx.preprocess_requests:
                dealer.write([b"", pr["request_id"].encode("utf-8")])
            # if self.engine_client.check_model_weight_status():
            #     raise ValueError("Engine is clearing model weight")
            while num_choices > 0:
                request_output_dicts = await asyncio.wait_for(request_output_queue.get(), timeout=60)
                for request_output_dict in request_output_dicts:
                    api_server_logger.debug(f"Received RequestOutput: {request_output_dict}")
                    if request_output_dict["finished"] is True:
                        num_choices -= 1
                    yield request_output_dict

        except Exception as e:
            raise ValueError(f"Error processing response: {str(e)}")
        finally:
            await self.engine_client.connection_manager.cleanup_request(request_id)
