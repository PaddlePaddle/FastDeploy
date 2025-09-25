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
from typing import Any, ClassVar, Dict, Generic, Optional, TypeVar, Union

from fastdeploy.engine.request import RequestOutput
from fastdeploy.entrypoints.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    InvalidParameterException,
)
from fastdeploy.utils import ErrorCode, ErrorType, api_server_logger

RequestT = TypeVar("RequestT")


class OpenAIServing(ABC, Generic[RequestT]):
    request_id_prefix: ClassVar[str]
    """
    Base pipeline for OpenAI-style serving implementations
    """

    def __init__(self, engine_client, models, pid, ips, max_waiting_time):
        self.engine_client = engine_client
        self.models = models
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
            error_msg = f"Request waiting timeout, request:{request_id} max waiting time:{self.max_waiting_time}"
            api_server_logger.error(error_msg)
            return False

    async def _release_semaphore(self, request_id: str) -> None:
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
        api_server_logger.error(message)
        return ErrorResponse(error=ErrorInfo(message=message, type=error_type, code=code, param=param))

    def _generate_request_id(self, user: Optional[str] = None) -> str:
        """Generate a unique request ID"""
        if user is not None:
            return f"{self.request_id_prefix}-{user}-{uuid.uuid4()}"
        return f"{self.request_id_prefix}-{uuid.uuid4()}"

    def _validate_request():
        """Validate the request before processing"""
        pass

    @abstractmethod
    async def _preprocess(self, request_id: str, request: RequestT) -> Dict:
        """Preprocess the request into engine format"""
        pass

    @abstractmethod
    async def _prepare_generators(self, request_id: str, request: dict) -> Any:
        """Process engine response into final format"""
        pass

    @abstractmethod
    async def _build_final_response(self, request_id: str, request_output: RequestOutput) -> Any:
        """Generate the final response object"""
        pass

    async def handle(self, reqeust: RequestT) -> Union[Any, ErrorResponse]:
        """Handle incoming requests"""
        yield self._pipeline(reqeust)

    async def _pipeline(self, request: RequestT) -> Union[Any, ErrorResponse]:
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

        # Step 1.2: Check supported model
        is_supported, request.model = self._check_supported_model(request.model)
        if not is_supported:
            yield self._create_error_response(
                f"Unsupported model: [{request.model}]", ErrorType.API_CONNECTION_ERROR, ErrorCode.MODEL_NOT_SUPPORT
            )

        # Step 1.3: Validate request
        self._validate_request(request)

        request_id = self._generate_request_id(getattr(request, "user", None))
        api_server_logger.info(f"Initialize request {request_id}: {request}")

        # Step 2: Semaphore acquisition
        if not await self._acquire_semaphore(request_id):
            yield self._create_error_response("Request waiting timeout", ErrorType.TIMEOUT_ERROR, ErrorCode.TIMEOUT)

        try:
            # Step 3: Preprocessing
            request_dict = await self._preprocess(request_id, request)
            request_dict["request_id"] = request_id

            # Step 4: Response processing
            generators = await self._prepare_generators(request_id, request_dict)

            # Step 5: Final response build
            async for request_output in generators:
                yield self._build_final_response(request_id, request_output)

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

    def __init__(self, engine_client, models, pid, ips, max_waiting_time):
        super().__init__(engine_client, models, pid, ips, max_waiting_time)

    async def _preprocess(self, request_id: str, request: Any) -> Dict:
        """Preprocess the request into engine format"""
        request_dict = request.to_dict_for_infer(request_id)
        if "chat_template" not in request_dict:
            request_dict["chat_template"] = self.chat_template
        request_dict["arrival_time"] = time.time()
        await self.engine_client.format_and_add_data(request_dict)
        return request_dict

    async def _prepare_generators(self, request_id: str, request: dict) -> RequestOutput:
        try:
            dealer, response_queue = await self.engine_client.connection_manager.get_connection(request_id)
            dealer.write([b"", request_id.encode("utf-8")])
            if self.engine_client.check_model_weight_status():
                raise ValueError("Engine is clearing model weight")
            responses = await asyncio.wait_for(response_queue.get(), timeout=60)
            for response in responses:
                yield response
        except Exception as e:
            raise ValueError(f"Error processing response: {str(e)}")
        finally:
            await self.engine_client.connection_manager.cleanup_request(request_id)
