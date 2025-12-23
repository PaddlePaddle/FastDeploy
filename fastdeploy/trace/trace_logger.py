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

from fastdeploy.trace.constants import EVENT_TO_STAGE_MAP
from fastdeploy.utils import trace_logger


def print(event, request_id, user):
    """
    Records task tracking log information, including task name, start time, end time, etc.
    Args:
        task (Task): Task object to be recorded.
    """
    try:
        trace_logger.info(
            "",
            extra={
                "attributes": {
                    "request_id": f"{request_id}",
                    "user_id": f"{user}",
                    "event": event.value,
                    "stage": EVENT_TO_STAGE_MAP.get(event).value,
                }
            },
        )
    except:
        pass
