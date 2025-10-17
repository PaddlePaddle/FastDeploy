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


from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)
from fastdeploy import envs

class MetricsManagerInterface:

    def set_value(self, name, value, labelvalues=None):
        metric = getattr(self, name, None)
        if isinstance(metric, Gauge):
            if envs.FD_ENABLE_METRIC_LABELS:
                if labelvalues is None:
                    labelvalues = {ln: "" for ln in metric._labelnames}
                metric.labels(**labelvalues).set(value)
            else:
                metric.set(value)
        return
    
    def inc_value(self, name, value=1, labelvalues=None):
        metric = getattr(self, name, None)
        if isinstance(metric, Gauge) or isinstance(metric, Counter):
            if envs.FD_ENABLE_METRIC_LABELS:
                if labelvalues is None:
                    labelvalues = {ln: "" for ln in metric._labelnames}
                metric.labels(**labelvalues).inc(value)
            else:
                metric.inc(value)
        return

    def dec_value(self, name, value=1, labelvalues=None):
        metric = getattr(self, name, None)
        if isinstance(metric, Gauge):
            if envs.FD_ENABLE_METRIC_LABELS:
                if labelvalues is None:
                    labelvalues = {ln: "" for ln in metric._labelnames}
                metric.labels(**labelvalues).dec(value)
            else:
                metric.dec(value)
        return

    def obs_value(self, name, value, labelvalues=None):
        metric = getattr(self, name, None)
        if isinstance(metric, Histogram):
            if envs.FD_ENABLE_METRIC_LABELS:
                if labelvalues is None:
                    labelvalues = {ln: "" for ln in metric._labelnames}
                metric.labels(**labelvalues).observe(value)
            else:
                metric.observe(value)
        return

