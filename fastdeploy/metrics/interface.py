"""
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
MetricsManagerInterface provides a unified interface for metric operations.
When FD_DEFAULT_METRIC_LABEL_VALUES is set to a valid JSON dict, metric labels
(e.g. model_id) are automatically applied. Otherwise, operations fall back to
the raw prometheus_client calls.
"""

from abc import ABC, abstractmethod


class MetricsManagerInterface(ABC):
    """Abstract base class that defines the unified metrics interface."""

    @abstractmethod
    def set_value(self, name: str, value, labelvalues: dict = None):
        """Set a Gauge metric to the given value.

        Args:
            name: The attribute name of the metric on the MetricsManager.
            value: The value to set.
            labelvalues: Optional dict of label key-value pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def inc_value(self, name: str, value=1, labelvalues: dict = None):
        """Increment a Counter or Gauge metric by the given value.

        Args:
            name: The attribute name of the metric on the MetricsManager.
            value: The amount to increment by (default 1).
            labelvalues: Optional dict of label key-value pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def dec_value(self, name: str, value=1, labelvalues: dict = None):
        """Decrement a Gauge metric by the given value.

        Args:
            name: The attribute name of the metric on the MetricsManager.
            value: The amount to decrement by (default 1).
            labelvalues: Optional dict of label key-value pairs.
        """
        raise NotImplementedError

    @abstractmethod
    def obs_value(self, name: str, value, labelvalues: dict = None):
        """Observe a value on a Histogram metric.

        Args:
            name: The attribute name of the metric on the MetricsManager.
            value: The value to observe.
            labelvalues: Optional dict of label key-value pairs.
        """
        raise NotImplementedError
