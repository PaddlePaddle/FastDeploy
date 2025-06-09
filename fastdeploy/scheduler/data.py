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

import time
import json
from fastdeploy.engine.request import Request, RequestOutput


class ScheduledRequest(object):
    """
        ScheduledRequest class
    """

    def __init__(self, request: Request):
        self.raw: Request = request
        self.id = request.request_id
        self.scheduled_time = time.time()
        self.size = len(request.prompt_token_ids)

    def serialize(self) -> bytes:
        """serialize to bytes"""
        data = {
            "scheduled_time": self.scheduled_time,
            "raw": self.raw.to_dict()
        }
        serialized_data = json.dumps(data, ensure_ascii=False)
        return serialized_data.encode()

    @classmethod
    def unserialize(cls, serialized_data: bytes) -> 'ScheduledRequest':
        """unserialize to Request"""
        data = json.loads(serialized_data)
        request = Request.from_dict(data["raw"])
        scheduled_request = cls(request)
        scheduled_request.scheduled_time = data["scheduled_time"]
        return scheduled_request


class ScheduledResponse(object):
    """
        ScheduledResponse class
    """

    def __init__(self, response: RequestOutput):
        self.raw: RequestOutput = response
        self.id = response.request_id
        self.index = response.outputs.index
        self.finished = response.finished

    def serialize(self) -> bytes:
        """serialize to bytes"""
        data = self.raw.to_dict()
        serialized_data = json.dumps(data, ensure_ascii=False)
        return serialized_data.encode()

    @classmethod
    def unserialize(cls, serialized_data: bytes) -> 'ScheduledResponse':
        """unserialize to RequestOutput"""
        data = json.loads(serialized_data)
        request_output = RequestOutput.from_dict(data)
        scheduled_response = ScheduledResponse(request_output)
        return scheduled_response
