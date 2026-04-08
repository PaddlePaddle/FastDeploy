"""
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import json
import time
from datetime import datetime

from fastdeploy.engine.request import Request, RequestOutput


class ScheduledRequest:
    """
    A wrapper class for Request objects with scheduling metadata.

    This class extends Request objects with:
    - Queue information for distributed scheduling
    - Timestamp tracking
    - Serialization capabilities
    """

    def __init__(
        self,
        request: Request,
        request_queue_name: str = "",
        response_queue_name: str = "",
    ):
        """
        Initialize a ScheduledRequest instance.

        Args:
            request: The original Request object
            request_queue_name: Name of the request queue
            response_queue_name: Name of the response queue
        """
        self.raw: Request = request
        self.request_queue_name = request_queue_name
        self.response_queue_name = response_queue_name
        self.schedule_time = time.time()

    def __repr__(self) -> str:
        local_time = datetime.fromtimestamp(self.schedule_time)
        formatted_time = local_time.strftime("%Y-%m-%d %H:%M:%S") + f"{local_time.microsecond // 1000:03d}"
        return (
            f"request_id:{self.request_id} request_queue:{self.request_queue_name} "
            f"response_queue:{self.response_queue_name} "
            f"schedule_time:{formatted_time}"
        )

    @property
    def request_id(self) -> str:
        """
        Get the request ID.

        Returns:
            The unique request identifier
        """
        return self.raw.request_id

    @request_id.setter
    def request_id(self, id: str):
        """
        Set the request ID.

        Args:
            id: New request identifier
        """
        self.raw.request_id = id

    @property
    def prompt_tokens_ids_len(self) -> int:
        """
        Get the length of prompt token IDs.

        Returns:
            Number of tokens in the prompt
        """
        return len(self.raw.prompt_token_ids)

    def serialize(self) -> bytes:
        """
        Serialize the request to bytes for storage/transmission.

        Returns:
            Serialized request data as bytes
        """
        data = {
            "request_queue_name": self.request_queue_name,
            "response_queue_name": self.response_queue_name,
            "schedule_time": self.schedule_time,
            "raw": self.raw.to_dict(),
        }
        serialized_data = json.dumps(data, ensure_ascii=False)
        return serialized_data.encode()

    @classmethod
    def unserialize(cls, serialized_data: bytes) -> "ScheduledRequest":
        """
        Deserialize bytes back into a ScheduledRequest.

        Args:
            serialized_data: Serialized request data

        Returns:
            Reconstructed ScheduledRequest object
        """
        data = json.loads(serialized_data)
        request = Request.from_dict(data["raw"])
        scheduled_request = cls(request)
        scheduled_request.schedule_time = data["schedule_time"]
        scheduled_request.request_queue_name = data["request_queue_name"]
        scheduled_request.response_queue_name = data["response_queue_name"]
        return scheduled_request


class ScheduledResponse:
    """
    A wrapper class for RequestOutput objects with scheduling metadata.

    This class extends RequestOutput objects with:
    - Timestamp tracking
    - Serialization capabilities
    - Status checking
    """

    def __init__(self, response: RequestOutput):
        """
        Initialize a ScheduledResponse instance.

        Args:
            response: The original RequestOutput object
        """
        self.raw: RequestOutput = response
        self.schedule_time = time.time()

    def __repr__(self):
        return f"request_id:{self.request_id} index:{self.index} finished:{self.finished}"

    @property
    def request_id(self) -> str:
        """
        Get the request ID.

        Returns:
            The unique request identifier
        """
        return self.raw.request_id

    @request_id.setter
    def request_id(self, id: str):
        """
        Set the request ID.

        Args:
            id: New request identifier
        """
        self.raw.request_id = id

    @property
    def index(self) -> int:
        """
        Get the output index.

        Returns:
            Position index of this response in the sequence
        """
        return self.raw.outputs.index

    @property
    def finished(self) -> bool:
        """
        Check if the request is complete.

        Returns:
            True if this is the final response for the request
        """
        return self.raw.finished

    def serialize(self) -> bytes:
        """
        Serialize the response to bytes for storage/transmission.

        Returns:
            Serialized response data as bytes
        """
        data = {
            "schedule_time": self.schedule_time,
            "raw": self.raw.to_dict(),
        }
        serialized_data = json.dumps(data, ensure_ascii=False)
        return serialized_data.encode()

    @classmethod
    def unserialize(cls, serialized_data: bytes) -> "ScheduledResponse":
        """
        Deserialize bytes back into a ScheduledResponse.

        Args:
            serialized_data: Serialized response data

        Returns:
            Reconstructed ScheduledResponse object
        """
        data = json.loads(serialized_data)
        response = RequestOutput.from_dict(data["raw"])
        scheduled_response = cls(response)
        scheduled_response.schedule_time = data["schedule_time"]
        return scheduled_response
