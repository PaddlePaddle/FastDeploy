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
import os
import shutil
from typing import Set, TYPE_CHECKING

from prometheus_client import Gauge, Histogram, multiprocess, CollectorRegistry, generate_latest
from prometheus_client.registry import Collector

from fastdeploy.metrics.work_metrics import work_process_metrics
from fastdeploy.utils import api_server_logger

if TYPE_CHECKING:
    from prometheus_client import Gauge, Histogram


def cleanup_prometheus_files(is_main):
    """
       Cleans and recreates the Prometheus multiprocess directory.

       Depending on whether it's the main process or a worker, this function removes the corresponding
       Prometheus multiprocess directory (/tmp/prom_main or /tmp/prom_worker) and recreates it as an empty directory.

       Args:
           is_main (bool): Indicates whether the current process is the main process.

       Returns:
           str: The path to the newly created Prometheus multiprocess directory.
    """
    PROM_DIR = "/tmp/prom_main" if is_main else "/tmp/prom_worker"
    if os.path.exists(PROM_DIR):
        shutil.rmtree(PROM_DIR)
    os.makedirs(PROM_DIR, exist_ok=True)
    return PROM_DIR


class SimpleCollector(Collector):
    """
        A custom Prometheus collector that filters out specific metrics by name.

        This collector wraps an existing registry and yields only those metrics
        whose names are not in the specified exclusion set.
    """

    def __init__(self, base_registry, exclude_names: Set[str]):
        """
            Initializes the SimpleCollector.

            Args:
                base_registry (CollectorRegistry): The source registry from which metrics are collected.
                exclude_names (Set[str]): A set of metric names to exclude from collection.
        """
        self.base_registry = base_registry
        self.exclude_names = exclude_names

    def collect(self):
        """
                Collects and yields metrics not in the exclusion list.

                Yields:
                    Metric: Prometheus Metric objects that are not excluded.
                """
        for metric in self.base_registry.collect():
            if metric.name not in self.exclude_names:
                yield metric


def get_filtered_metrics(exclude_names: Set[str], extra_register_func=None) -> str:
    """
    Get the merged metric text (specified metric name removed)
    :param exclude_names: metric.name set to be excluded
    :param extra_register_func: optional, main process custom metric registration method
    :return: filtered metric text (str)
    """
    base_registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(base_registry)

    filtered_registry = CollectorRegistry()
    filtered_registry.register(SimpleCollector(base_registry, exclude_names))

    if extra_register_func:
        extra_register_func(filtered_registry)

    return generate_latest(filtered_registry).decode("utf-8")


REQUEST_LATENCY_BUCKETS = [
    0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
    40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
]


class MetricsManager:
    """Prometheus Metrics Manager handles all metric updates """

    _instance = None

    num_requests_running: 'Gauge'
    num_requests_waiting: 'Gauge'
    time_to_first_token: 'Histogram'
    time_per_output_token: 'Histogram'
    request_inference_time: 'Histogram'
    request_queue_time: 'Histogram'

    # 定义所有指标配置
    METRICS = {
        'num_requests_running': {
            'type': Gauge,
            'name': 'fastdeploy:num_requests_running',
            'description': 'Number of requests currently running',
            'kwargs': {}
        },
        'num_requests_waiting': {
            'type': Gauge,
            'name': 'fastdeploy:num_requests_waiting',
            'description': 'Number of requests currently waiting',
            'kwargs': {}
        },
        'time_to_first_token': {
            'type': Histogram,
            'name': 'fastdeploy:time_to_first_token_seconds',
            'description': 'Time to first token in seconds',
            'kwargs': {
                'buckets': [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0]
            }
        },
        'time_per_output_token': {
            'type': Histogram,
            'name': 'fastdeploy:time_per_output_token_seconds',
            'description': 'Time per output token in seconds',
            'kwargs': {
                'buckets': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0]
            }
        },

        'request_inference_time': {
            'type': Histogram,
            'name': 'fastdeploy:request_inference_time_seconds',
            'description': 'Time spent in inference phase (from inference start to last token)',
            'kwargs': {
                'buckets': REQUEST_LATENCY_BUCKETS
            }
        },
        'request_queue_time': {
            'type': Histogram,
            'name': 'fastdeploy:request_queue_time_seconds',
            'description': 'Time spent in waiting queue (from preprocess end to inference start)',
            'kwargs': {
                'buckets': REQUEST_LATENCY_BUCKETS
            }
        }
    }

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""
        # 动态创建所有指标
        for metric_name, config in self.METRICS.items():
            setattr(self, metric_name, config['type'](
                config['name'],
                config['description'],
                **config['kwargs']
            ))

    def register_all(self, registry: CollectorRegistry, workers: int = 1):
        """Register all metrics to the specified registry"""
        for metric_name in self.METRICS:
            registry.register(getattr(self, metric_name))
        if workers == 1:
            registry.register(work_process_metrics.e2e_request_latency)

    @classmethod
    def get_excluded_metrics(cls) -> Set[str]:
        """Get the set of indicator names that need to be excluded"""
        return {config['name'] for config in cls.METRICS.values()}


main_process_metrics = MetricsManager()

EXCLUDE_LABELS = MetricsManager.get_excluded_metrics()
