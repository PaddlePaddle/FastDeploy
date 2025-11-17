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

import logging
from unittest.mock import patch

from fastdeploy.trace.constants import LoggingEventName, StageName
from fastdeploy.trace.trace_logger import print as trace_print


class TestTraceLogging:
    def test_trace_print(self, caplog):
        request_id = "test123"
        user = "test_user"
        event = LoggingEventName.PREPROCESSING_START

        with caplog.at_level(logging.INFO):
            trace_print(event, request_id, user)

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert f"[request_id={request_id}]" in record.message
        assert f"[user_id={user}]" in record.message
        assert f"[event={event.value}]" in record.message
        assert f"[stage={StageName.PREPROCESSING.value}]" in record.message

    def test_trace_print_with_logger_error(self, caplog):
        request_id = "test123"
        user = "test_user"
        event = LoggingEventName.PREPROCESSING_START

        with patch("logging.Logger.info", side_effect=Exception("Logger error")):
            with caplog.at_level(logging.INFO):
                trace_print(event, request_id, user)

        assert len(caplog.records) == 0
