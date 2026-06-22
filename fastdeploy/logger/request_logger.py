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
Request logging module

Provides leveled request logging with L0-L3 levels:
- L0: Critical lifecycle events (creation, completion, abort)
- L1: Processing stage details
- L2: Request/response content
- L3: Full data
"""

from enum import IntEnum

from fastdeploy import envs


class RequestLogLevel(IntEnum):
    """Request log level"""

    LIFECYCLE = 0  # Lifecycle start/end: creation, completion, abort
    STAGES = 1  # Processing stages: semaphore, first token, signal handling
    CONTENT = 2  # Content and scheduling: request params, scheduling, response
    FULL = 3  # Complete raw data


def _should_log(level: int) -> bool:
    """Check if this level should be logged"""
    if int(envs.FD_LOG_REQUESTS) == 0:
        return False
    return int(level) <= int(envs.FD_LOG_REQUESTS_LEVEL)


def log_request(level: int, message: str, **fields):
    """
    Log request message

    Args:
        level: Log level (0-3)
        message: Log message template, supports {field} formatting
        **fields: Message fields
    """
    if not _should_log(level):
        return

    from fastdeploy.logger import _request_logger

    if not fields:
        _request_logger.info(message, stacklevel=2)
        return

    payload = fields
    _request_logger.info(message.format(**payload), stacklevel=2)


def log_request_error(message: str, **fields):
    """
    Log request error message

    Args:
        message: Log message template, supports {field} formatting
        **fields: Message fields
    """
    from fastdeploy.logger import _request_logger

    if fields:
        _request_logger.error(message.format(**fields), stacklevel=2)
    else:
        _request_logger.error(message, stacklevel=2)
