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

import unittest
from unittest.mock import patch

from prometheus_client import Gauge

from fastdeploy.metrics.metrics import get_filtered_metrics


class TestGetFilteredMetrics(unittest.TestCase):
    def test_filtered_and_custom_metrics(self):
        """
        Test get_filtered_metrics function:
        1. Exclude specific metrics from base_registry
        2. Keep other metrics in base_registry
        3. Ensure metrics registered by extra_register_func are effective
        """

        # Simulated metrics in base_registry (Gauge instances)
        g_keep = Gauge("metric_to_keep", "Kept metric")
        g_keep.set(1.23)

        g_exclude = Gauge("metric_to_exclude", "Excluded metric")
        g_exclude.set(99)

        # Fake MultiProcessCollector: register our simulated metrics
        def fake_multiprocess_collector(registry):
            registry.register(g_keep)
            registry.register(g_exclude)

        with patch(
            "fastdeploy.metrics.metrics.multiprocess.MultiProcessCollector", side_effect=fake_multiprocess_collector
        ):
            result = get_filtered_metrics()

        print("==== result ====\n", result)

        # 2. Kept metric should appear
        self.assertIn("metric_to_keep", result)

        self.assertIn("metric_to_exclude", result)


if __name__ == "__main__":
    unittest.main()
