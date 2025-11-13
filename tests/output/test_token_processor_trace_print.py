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
import time
from unittest.mock import MagicMock

from fastdeploy.engine.request import Request
from fastdeploy.output.token_processor import TokenProcessor


class TestTokenProcessorMetrics:
    def setup_method(self):
        self.mock_cfg = MagicMock()
        self.mock_cached_tokens = MagicMock()
        self.mock_engine_queue = MagicMock()
        self.mock_split_connector = MagicMock()

        self.processor = TokenProcessor(
            cfg=self.mock_cfg,
            cached_generated_tokens=self.mock_cached_tokens,
            engine_worker_queue=self.mock_engine_queue,
            split_connector=self.mock_split_connector,
        )

        # Create a complete Request object with all required parameters
        self.task = Request(
            request_id="test123",
            prompt="test prompt",
            prompt_token_ids=[1, 2, 3],
            prompt_token_ids_len=3,
            messages=["test message"],
            history=[],
            tools=[],
            system="test system",
            eos_token_ids=[0],
            arrival_time=time.time(),
        )
        self.task.inference_start_time = time.time()
        self.task.schedule_start_time = self.task.inference_start_time - 0.1
        self.task.preprocess_end_time = self.task.schedule_start_time - 0.05
        self.task.preprocess_start_time = self.task.preprocess_end_time - 0.05
        self.task.arrival_time = self.task.preprocess_start_time - 0.1

    def test_record_first_token_metrics(self, caplog):
        current_time = time.time()

        with caplog.at_level(logging.INFO):
            self.processor._record_first_token_metrics(self.task, current_time)

        assert len(caplog.records) == 2
        assert "[request_id=test123]" in caplog.text
        assert "[event=FIRST_TOKEN_GENERATED]" in caplog.text
        assert "[event=DECODE_START]" in caplog.text

        # Verify metrics are set
        assert hasattr(self.task, "first_token_time")
        assert self.task.first_token_time == current_time

    def test_record_completion_metrics(self, caplog):
        current_time = time.time()
        self.task.first_token_time = current_time - 0.5

        with caplog.at_level(logging.INFO):
            self.processor._record_completion_metrics(self.task, current_time)

        assert len(caplog.records) == 2
        assert "[request_id=test123]" in caplog.text
        assert "[event=INFERENCE_END]" in caplog.text
        assert "[event=POSTPROCESSING_START]" in caplog.text

        # Verify metrics are updated
        assert self.processor.tokens_counter["test123"] == 0  # Just checking counter exists
