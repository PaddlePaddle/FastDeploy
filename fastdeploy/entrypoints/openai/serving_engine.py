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

from pydantic import BaseModel

from fastdeploy.entrypoints.openai.protocol import (
    ErrorInfo,
    ErrorResponse,
    InvalidParameterException,
)
from fastdeploy.utils import ErrorCode, ErrorType, api_server_logger

RequestT = TypeVar("RequestT")


class ServeContext(BaseModel, Generic[RequestT]):
    """
    Context class for OpenAI serving
    """

    request_id: str
    model_name: str
    request: RequestT


class OpenAIServing(ABC):
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

    async def _acquire_semaphore(self) -> bool:
        """Acquire engine client semaphore with timeout"""
        try:
            if self.max_waiting_time < 0:
                await self.engine_client.semaphore.acquire()
            else:
                await asyncio.wait_for(self.engine_client.semaphore.acquire(), timeout=self.max_waiting_time)
                pass
            return True
        except asyncio.TimeoutError:
            error_msg = f"Request waiting timeout, max waiting time: {self.max_waiting_time}"
            api_server_logger.error(error_msg)
            return False

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

    @abstractmethod
    def _validate_request():
        """Validate the request before processing"""
        pass

    async def _preprocess_request(self, request_id: str, request: Any) -> Dict:
        """Preprocess the request into engine format"""
        request_dict = request.to_dict_for_infer(request_id)
        if "chat_template" not in request_dict:
            request_dict["chat_template"] = self.chat_template
        request_dict["arrival_time"] = time.time()
        await self.engine_client.format_and_add_data(request_dict)
        return request_dict

    @abstractmethod
    async def _process_response(self, response: Dict, request: Any) -> Any:
        """Process engine response into final format"""
        pass

    @abstractmethod
    async def _build_final_response(self, processed_data: Any, request: Any) -> Any:
        """Generate the final response object"""
        pass

    async def handle(self, ctx: Any) -> Union[Any, ErrorResponse]:
        """Handle incoming requests"""
        return await self._pipeline(ctx.request)

    async def _pipeline(self, ctx: ServeContext) -> Union[Any, ErrorResponse]:
        """
        Execute the full serving pipeline:
        1. Request validation
        2. Preprocessing
        3. Execution
        4. Response processing
        5. Final response generation
        """
        request = ctx.request
        # Step 1: Request validation

        # Step 1.1: Check if current node is master
        if not self._check_master():
            return self._create_error_response(
                f"Only master node can accept request, please send to master node: {self.master_ip}"
            )

        # Step 1.2: Check supported model
        is_supported, request.model = self._check_supported_model(request.model)
        if not is_supported:
            return self._create_error_response(
                f"Unsupported model: [{request.model}]", ErrorType.API_CONNECTION_ERROR, ErrorCode.MODEL_NOT_SUPPORT
            )

        # Step 1.3: Validate request
        self._validate_request(request)

        # Step 2: Semaphore acquisition
        if not await self._acquire_semaphore():
            return self._create_error_response("Request waiting timeout", ErrorType.TIMEOUT_ERROR, ErrorCode.TIMEOUT)

        request_id = self._generate_request_id(getattr(request, "user", None))
        api_server_logger.info(f"Initialize request {request_id}: {request}")

        try:
            # Step 3: Preprocessing
            request_dict = await self._preprocess_request(request_id, request)
            request_dict["request_id"] = request_id

            # Step 4: Response processing
            processed_data = await self._process_response(request_dict)

            # Step 5: Final response build
            return await self._build_final_response(processed_data, request_dict)

        except InvalidParameterException as e:
            traceback.print_exc()
            return self._create_error_response(str(e.message), ErrorType.INVALID_REQUEST_ERROR, param=e.param)
        except Exception as e:
            traceback.print_exc()
            return self._create_error_response(str(e))
        finally:
            api_server_logger.info(f"Release request {request_id}")
            self.engine_client.semaphore.release()
