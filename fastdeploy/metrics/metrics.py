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
import uuid
from typing import Set

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from prometheus_client.registry import Collector

from fastdeploy.metrics import build_1_2_5_buckets
from fastdeploy.metrics.work_metrics import work_process_metrics


def cleanup_prometheus_files(is_main: bool, instance_id: str = None):
    """
    Cleans and recreates the Prometheus multiprocess directory.
    """
    base_dir = "/tmp/prom_main" if is_main else "/tmp/prom_worker"
    if instance_id is None:
        instance_id = str(uuid.uuid4())
    prom_dir = f"{base_dir}_{instance_id}"

    if os.path.exists(prom_dir):
        shutil.rmtree(prom_dir, ignore_errors=True)
    os.makedirs(prom_dir, exist_ok=True)

    return prom_dir


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
            if not any(name.startswith(metric.name) for name in self.exclude_names):
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
]


class MetricsManager:
    """Prometheus Metrics Manager handles all metric updates"""

    _instance = None

    num_requests_running: "Gauge"
    num_requests_waiting: "Gauge"
    time_to_first_token: "Histogram"
    time_per_output_token: "Histogram"
    request_inference_time: "Histogram"
    request_queue_time: "Histogram"
    gpu_cache_usage_perc: "Gauge"
    generation_tokens_total: "Counter"
    request_prefill_time: "Histogram"
    request_decode_time: "Histogram"
    request_generation_tokens: "Histogram"
    request_success_total: "Counter"
    spec_decode_draft_acceptance_rate: "Gauge"
    spec_decode_efficiency: "Gauge"
    spec_decode_num_accepted_tokens_total: "Counter"
    spec_decode_num_draft_tokens_total: "Counter"
    spec_decode_num_emitted_tokens_total: "Counter"
    spec_decode_draft_single_head_acceptance_rate: "list[Gauge]"

    # for YIYAN Adapter
    prefix_cache_token_num: "Counter"
    prefix_gpu_cache_token_num: "Counter"
    prefix_cpu_cache_token_num: "Counter"
    prefix_ssd_cache_token_num: "Counter"
    batch_size: "Gauge"
    max_batch_size: "Gauge"
    available_gpu_block_num: "Gauge"
    free_gpu_block_num: "Gauge"
    max_gpu_block_num: "Gauge"
    available_gpu_resource: "Gauge"
    requests_number: "Counter"
    send_cache_failed_num: "Counter"
    first_token_latency: "Gauge"
    infer_latency: "Gauge"
    cache_config_info: "Gauge"
    available_batch_size: "Gauge"
    hit_req_rate: "Gauge"
    hit_token_rate: "Gauge"
    cpu_hit_token_rate: "Gauge"
    gpu_hit_token_rate: "Gauge"
    # 定义所有指标配置
    METRICS = {
        "num_requests_running": {
            "type": Gauge,
            "name": "fastdeploy:num_requests_running",
            "description": "Number of requests currently running",
            "kwargs": {},
        },
        "num_requests_waiting": {
            "type": Gauge,
            "name": "fastdeploy:num_requests_waiting",
            "description": "Number of requests currently waiting",
            "kwargs": {},
        },
        "time_to_first_token": {
            "type": Histogram,
            "name": "fastdeploy:time_to_first_token_seconds",
            "description": "Time to first token in seconds",
            "kwargs": {
                "buckets": [
                    0.001,
                    0.005,
                    0.01,
                    0.02,
                    0.04,
                    0.06,
                    0.08,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                ]
            },
        },
        "time_per_output_token": {
            "type": Histogram,
            "name": "fastdeploy:time_per_output_token_seconds",
            "description": "Time per output token in seconds",
            "kwargs": {
                "buckets": [
                    0.01,
                    0.025,
                    0.05,
                    0.075,
                    0.1,
                    0.15,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.75,
                    1.0,
                ]
            },
        },
        "request_inference_time": {
            "type": Histogram,
            "name": "fastdeploy:request_inference_time_seconds",
            "description": "Time spent in inference phase (from inference start to last token)",
            "kwargs": {"buckets": REQUEST_LATENCY_BUCKETS},
        },
        "request_queue_time": {
            "type": Histogram,
            "name": "fastdeploy:request_queue_time_seconds",
            "description": "Time spent in waiting queue (from preprocess end to inference start)",
            "kwargs": {"buckets": REQUEST_LATENCY_BUCKETS},
        },
        "gpu_cache_usage_perc": {
            "type": Gauge,
            "name": "fastdeploy:gpu_cache_usage_perc",
            "description": "GPU KV-cache usage. 1 means 100 percent usage",
            "kwargs": {},
        },
        "generation_tokens_total": {
            "type": Counter,
            "name": "fastdeploy:generation_tokens_total",
            "description": "Total number of generation tokens processed",
            "kwargs": {},
        },
        "request_prefill_time": {
            "type": Histogram,
            "name": "fastdeploy:request_prefill_time_seconds",
            "description": "Time spent in prefill phase (from preprocess start to preprocess end)",
            "kwargs": {"buckets": REQUEST_LATENCY_BUCKETS},
        },
        "request_decode_time": {
            "type": Histogram,
            "name": "fastdeploy:request_decode_time_seconds",
            "description": "Time spent in decode phase (from first token to last token)",
            "kwargs": {"buckets": REQUEST_LATENCY_BUCKETS},
        },
        "request_generation_tokens": {
            "type": Histogram,
            "name": "fastdeploy:request_generation_tokens",
            "description": "Number of generation tokens processed.",
            "kwargs": {"buckets": build_1_2_5_buckets(33792)},
        },
        "request_success_total": {
            "type": Counter,
            "name": "fastdeploy:request_success_total",
            "description": "Total number of successfully processed requests",
            "kwargs": {},
        },
        # for YIYAN Adapter
        "prefix_cache_token_num": {
            "type": Counter,
            "name": "fastdeploy:prefix_cache_token_num",
            "description": "Total number of cached tokens",
            "kwargs": {},
        },
        "prefix_gpu_cache_token_num": {
            "type": Counter,
            "name": "fastdeploy:prefix_gpu_cache_token_num",
            "description": "Total number of cached tokens on GPU",
            "kwargs": {},
        },
        "prefix_cpu_cache_token_num": {
            "type": Counter,
            "name": "fastdeploy:prefix_cpu_cache_token_num",
            "description": "Total number of cached tokens on CPU",
            "kwargs": {},
        },
        "prefix_ssd_cache_token_num": {
            "type": Counter,
            "name": "fastdeploy:prefix_ssd_cache_token_num",
            "description": "Total number of cached tokens on SSD",
            "kwargs": {},
        },
        "batch_size": {
            "type": Gauge,
            "name": "fastdeploy:batch_size",
            "description": "Real batch size during inference",
            "kwargs": {},
        },
        "max_batch_size": {
            "type": Gauge,
            "name": "fastdeploy:max_batch_size",
            "description": "Maximum batch size determined when service started",
            "kwargs": {},
        },
        "available_gpu_block_num": {
            "type": Gauge,
            "name": "fastdeploy:available_gpu_block_num",
            "description": "Number of available gpu blocks in cache, including blocks in LRU list",
            "kwargs": {},
        },
        "free_gpu_block_num": {
            "type": Gauge,
            "name": "fastdeploy:free_gpu_block_num",
            "description": "Number of free blocks in cache",
            "kwargs": {},
        },
        "max_gpu_block_num": {
            "type": Gauge,
            "name": "fastdeploy:max_gpu_block_num",
            "description": "Number of total blocks determined when service started",
            "kwargs": {},
        },
        "available_gpu_resource": {
            "type": Gauge,
            "name": "fastdeploy:available_gpu_resource",
            "description": "Available blocks percentage, i.e. available_gpu_block_num / max_gpu_block_num",
            "kwargs": {},
        },
        "requests_number": {
            "type": Counter,
            "name": "fastdeploy:requests_number",
            "description": "Total number of requests received",
            "kwargs": {},
        },
        "send_cache_failed_num": {
            "type": Counter,
            "name": "fastdeploy:send_cache_failed_num",
            "description": "Total number of failures of sending cache",
            "kwargs": {},
        },
        "first_token_latency": {
            "type": Gauge,
            "name": "fastdeploy:first_token_latency",
            "description": "Latest time to first token in seconds",
            "kwargs": {},
        },
        "infer_latency": {
            "type": Gauge,
            "name": "fastdeploy:infer_latency",
            "description": "Latest time to generate one token in seconds",
            "kwargs": {},
        },
        "available_batch_size": {
            "type": Gauge,
            "name": "fastdeploy:available_batch_size",
            "description": "Number of requests that can still be inserted during the Decode phase",
            "kwargs": {},
        },
        "hit_req_rate": {
            "type": Gauge,
            "name": "fastdeploy:hit_req_rate",
            "description": "Request-level prefix cache hit rate",
            "kwargs": {},
        },
        "hit_token_rate": {
            "type": Gauge,
            "name": "fastdeploy:hit_token_rate",
            "description": "Token-level prefix cache hit rate",
            "kwargs": {},
        },
        "cpu_hit_token_rate": {
            "type": Gauge,
            "name": "fastdeploy:cpu_hit_token_rate",
            "description": "Token-level CPU prefix cache hit rate",
            "kwargs": {},
        },
        "gpu_hit_token_rate": {
            "type": Gauge,
            "name": "fastdeploy:gpu_hit_token_rate",
            "description": "Token-level GPU prefix cache hit rate",
            "kwargs": {},
        },
    }
    SPECULATIVE_METRICS = {}

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""
        # 动态创建所有指标
        for metric_name, config in self.METRICS.items():
            setattr(
                self,
                metric_name,
                config["type"](config["name"], config["description"], **config["kwargs"]),
            )

    def _init_speculative_metrics(self, speculative_method, num_speculative_tokens):
        self.SPECULATIVE_METRICS = {
            "spec_decode_draft_acceptance_rate": {
                "type": Gauge,
                "name": "fastdeploy:spec_decode_draft_acceptance_rate",
                "description": "Acceptance rate of speculative decoding",
                "kwargs": {},
            },
            "spec_decode_num_accepted_tokens_total": {
                "type": Counter,
                "name": "fastdeploy:spec_decode_num_accepted_tokens_total",
                "description": "Total number of tokens accepted by the scoring model and verification program",
                "kwargs": {},
            },
            "spec_decode_num_emitted_tokens_total": {
                "type": Counter,
                "name": "fastdeploy:spec_decode_num_emitted_tokens_total",
                "description": "Total number of tokens output by the entire system",
                "kwargs": {},
            },
        }
        if speculative_method == "mtp":
            self.SPECULATIVE_METRICS["spec_decode_efficiency"] = {
                "type": Gauge,
                "name": "fastdeploy:spec_decode_efficiency",
                "description": "Efficiency of speculative decoding",
                "kwargs": {},
            }
            self.SPECULATIVE_METRICS["spec_decode_num_draft_tokens_total"] = {
                "type": Counter,
                "name": "fastdeploy:spec_decode_num_draft_tokens_total",
                "description": "Total number of speculative tokens generated by the proposal method",
                "kwargs": {},
            }
            self.SPECULATIVE_METRICS["spec_decode_draft_single_head_acceptance_rate"] = {
                "type": list[Gauge],
                "name": "fastdeploy:spec_decode_draft_single_head_acceptance_rate",
                "description": "Single head acceptance rate of speculative decoding",
                "kwargs": {},
            }
        for metric_name, config in self.SPECULATIVE_METRICS.items():
            if metric_name == "spec_decode_draft_single_head_acceptance_rate":
                gauges = []
                for i in range(num_speculative_tokens):
                    gauges.append(
                        Gauge(
                            f"{config['name']}_{i}",
                            f"{config['description']} (head {i})",
                        )
                    )
                    setattr(self, metric_name, gauges)
            else:
                setattr(
                    self,
                    metric_name,
                    config["type"](
                        config["name"],
                        config["description"],
                        **config["kwargs"],
                    ),
                )

    def set_cache_config_info(self, obj) -> None:
        if hasattr(self, "cache_config_info") and isinstance(self.cache_config_info, Gauge):
            metrics_info = obj.metrics_info()
            if metrics_info:
                self.cache_config_info.labels(**metrics_info).set(1)
            return

        metrics_info = obj.metrics_info()
        if not metrics_info:
            return

        self.cache_config_info = Gauge(
            name="fastdeploy:cache_config_info",
            documentation="Information of the engine's CacheConfig",
            labelnames=list(metrics_info.keys()),
            multiprocess_mode="mostrecent",
        )

        self.cache_config_info.labels(**metrics_info).set(1)

    def register_speculative_metrics(self, registry: CollectorRegistry):
        """Register all speculative metrics to the specified registry"""
        for metric_name in self.SPECULATIVE_METRICS:
            if metric_name == "spec_decode_draft_single_head_acceptance_rate":
                for gauge in getattr(self, metric_name):
                    registry.register(gauge)
            else:
                registry.register(getattr(self, metric_name))

    def register_all(self, registry: CollectorRegistry, workers: int = 1):
        """Register all metrics to the specified registry"""
        for metric_name in self.METRICS:
            registry.register(getattr(self, metric_name))
        if self.cache_config_info is not None:
            registry.register(self.cache_config_info)
        if workers == 1:
            registry.register(work_process_metrics.e2e_request_latency)
            registry.register(work_process_metrics.request_params_max_tokens)
            registry.register(work_process_metrics.prompt_tokens_total)
            registry.register(work_process_metrics.request_prompt_tokens)
        if hasattr(main_process_metrics, "spec_decode_draft_acceptance_rate"):
            self.register_speculative_metrics(registry)

    @classmethod
    def get_excluded_metrics(cls) -> Set[str]:
        """Get the set of indicator names that need to be excluded"""
        return {config["name"] for config in cls.METRICS.values()}


main_process_metrics = MetricsManager()

EXCLUDE_LABELS = MetricsManager.get_excluded_metrics()
