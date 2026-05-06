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
Log configuration parsing module
"""

import os


def resolve_log_level(raw_level=None, debug_enabled=None) -> str:
    """
    Resolve log level configuration

    Priority: FD_LOG_LEVEL > FD_DEBUG
    """
    raw = os.getenv("FD_LOG_LEVEL") if raw_level is None else raw_level
    # Handle None or string "None" case
    if raw and str(raw).upper() != "NONE":
        level = raw.upper()
        if level not in {"INFO", "DEBUG"}:
            raise ValueError(f"Unsupported FD_LOG_LEVEL: {raw}")
        return level
    debug = os.getenv("FD_DEBUG", "0") if debug_enabled is None else str(debug_enabled)
    return "DEBUG" if debug == "1" else "INFO"


def resolve_request_logging_defaults() -> dict[str, int]:
    """
    Resolve request logging default configuration
    """
    return {
        "enabled": int(os.getenv("FD_LOG_REQUESTS", "1")),
        "level": int(os.getenv("FD_LOG_REQUESTS_LEVEL", "2")),
    }
