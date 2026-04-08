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

import os
import unittest
from unittest.mock import patch

from prometheus_client import Gauge

from fastdeploy.metrics.metrics import get_filtered_metrics, main_process_metrics


class TestGetFilteredMetrics(unittest.TestCase):
    def _collect_metrics_with_mocked_multiprocess(self, metric_name, multiprocess_value):
        def fake_multiprocess_collector(registry):
            gauge = Gauge(metric_name, f"fake metric for {metric_name}", ["pid"], registry=registry)
            gauge.labels(pid="10001").set(multiprocess_value)

        with (
            patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": "/tmp/fake-prometheus-multiproc-dir"}, clear=False),
            patch(
                "fastdeploy.metrics.metrics.multiprocess.MultiProcessCollector",
                side_effect=fake_multiprocess_collector,
            ),
        ):
            return get_filtered_metrics()

    def _assert_unique_metric_value(self, metrics_text, metric_name, expected_value):
        metric_lines = [line for line in metrics_text.splitlines() if line.startswith(f"{metric_name} ")]
        self.assertEqual(metric_lines, [f"{metric_name} {expected_value}"])
        self.assertNotIn("pid=", metrics_text)

    def test_regular_gauge_returns_single_value_without_pid(self):
        metric = main_process_metrics.batch_size
        metric.set(8.0)

        result = self._collect_metrics_with_mocked_multiprocess(metric._name, multiprocess_value=1008.0)

        self._assert_unique_metric_value(result, metric._name, 8.0)

    def test_speculative_gauge_returns_single_value_without_pid(self):
        if not hasattr(main_process_metrics, "spec_decode_draft_acceptance_rate"):
            main_process_metrics._init_speculative_metrics("mtp", 2)

        metric = main_process_metrics.spec_decode_draft_acceptance_rate
        metric.set(0.75)

        result = self._collect_metrics_with_mocked_multiprocess(metric._name, multiprocess_value=1000.75)

        self._assert_unique_metric_value(result, metric._name, 0.75)

    def test_speculative_single_head_gauge_returns_single_value_without_pid(self):
        if not hasattr(main_process_metrics, "spec_decode_draft_acceptance_rate"):
            main_process_metrics._init_speculative_metrics("mtp", 2)

        metric = main_process_metrics.spec_decode_draft_single_head_acceptance_rate[0]
        metric.set(0.6)

        result = self._collect_metrics_with_mocked_multiprocess(metric._name, multiprocess_value=1000.6)

        self._assert_unique_metric_value(result, metric._name, 0.6)


if __name__ == "__main__":
    unittest.main()
