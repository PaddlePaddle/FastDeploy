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

"""
metrics
"""

from prometheus_client import Counter, Histogram

from fastdeploy.metrics.metrics import build_1_2_5_buckets


class WorkMetricsManager:
    """Prometheus Metrics Manager handles all metric updates"""

    _initialized = False

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""

        if self._initialized:
            return

        self.e2e_request_latency = Histogram(
            "fastdeploy:e2e_request_latency_seconds",
            "End-to-end request latency (from request arrival to final response)",
            buckets=[
                0.3,
                0.5,
                0.8,
                1.0,
                1.5,
                2.0,
                2.5,
                5.0,
                10.0,
                15.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                120.0,
                240.0,
                480.0,
                960.0,
                1920.0,
                7680.0,
            ],
        )
        self.request_params_max_tokens = Histogram(
            name="fastdeploy:request_params_max_tokens",
            documentation="Histogram of max_tokens parameter in request parameters",
            buckets=build_1_2_5_buckets(33792),
        )
        self.prompt_tokens_total = Counter(
            name="fastdeploy:prompt_tokens_total",
            documentation="Total number of prompt tokens processed",
        )
        self.request_prompt_tokens = Histogram(
            name="fastdeploy:request_prompt_tokens",
            documentation="Number of prefill tokens processed.",
            buckets=build_1_2_5_buckets(33792),
        )

        self._initialized = True


work_process_metrics = WorkMetricsManager()
