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

from fastdeploy import envs
from fastdeploy.metrics import build_1_2_5_buckets
from fastdeploy.metrics.prometheus_multiprocess_setup import (
    setup_multiprocess_prometheus,
)
from fastdeploy.metrics.stats import ZMQMetricsStats


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


def get_filtered_metrics() -> str:
    """
    Get the merged metric text (specified metric name removed)
    :return: filtered metric text (str)
    """

    base_registry = CollectorRegistry()

    # 判断是否多进程
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        # multiprocess 会将当前共享目录中的所有指标收集到base_registry中
        multiprocess.MultiProcessCollector(base_registry)

        filtered_registry = CollectorRegistry()
        # 注册一个新的colletor，过滤gauge指标
        filtered_registry.register(SimpleCollector(base_registry, EXCLUDE_LABELS))

        # 将gauge指标重新注册到filtered_registry中，从内存中读取
        main_process_metrics.re_register_gauge(filtered_registry)

        return generate_latest(filtered_registry).decode("utf-8")

    else:
        # 非多进程直接注册所有指标，从内存中读取
        main_process_metrics.register_all(base_registry)
        return generate_latest(base_registry).decode("utf-8")


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
    _collect_zmq_metrics = False
    cache_config_info = None

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
    cache_config_info: "Gauge"
    available_batch_size: "Gauge"
    hit_req_rate: "Gauge"
    hit_token_rate: "Gauge"
    cpu_hit_token_rate: "Gauge"
    gpu_hit_token_rate: "Gauge"

    # for http request
    http_requests_total: "Counter"
    http_request_duration_seconds: "Histogram"

    # for zmq
    msg_send_total: "Counter"
    msg_send_failed_total: "Counter"
    msg_bytes_send_total: "Counter"
    msg_recv_total: "Counter"
    msg_bytes_recv_total: "Counter"
    zmq_latency: "Histogram"
    # for request metrics
    e2e_request_latency: "Histogram"
    request_params_max_tokens: "Histogram"
    prompt_tokens_total: "Counter"
    request_prompt_tokens: "Histogram"

    # 定义所有指标配置

    # gauge指标在多进程中，会有pid隔离，需要特殊处理，因此手动定义出来
    GAUGE_METRICS = {
        "num_requests_running": {
            "type": Gauge,
            "name": "fastdeploy:num_requests_running",
            "description": "Number of requests currently running",
            "kwargs": {"multiprocess_mode": "sum"},
        },
        "num_requests_waiting": {
            "type": Gauge,
            "name": "fastdeploy:num_requests_waiting",
            "description": "Number of requests currently waiting",
            "kwargs": {},
        },
        "gpu_cache_usage_perc": {
            "type": Gauge,
            "name": "fastdeploy:gpu_cache_usage_perc",
            "description": "GPU KV-cache usage. 1 means 100 percent usage",
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

    METRICS = {
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
        # for http
        "http_requests_total": {
            "type": Counter,
            "name": "http_requests_total",
            "description": "Total number of requests by method, status and handler.",
            "kwargs": {"labelnames": ["method", "path", "status_code"]},
        },
        "http_request_duration_seconds": {
            "type": Histogram,
            "name": "http_request_duration_seconds",
            "description": "Duration of HTTP requests in seconds",
            "kwargs": {
                "labelnames": ["method", "path"],
                "buckets": [
                    0.01,
                    0.025,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    0.75,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                    7.5,
                    10,
                    30,
                    60,
                ],
            },
        },
    }

    SPECULATIVE_METRICS = {}

    ZMQ_METRICS = {
        "msg_send_total": {
            "type": Counter,
            "name": "fastdeploy:zmq:msg_send_total",
            "description": "Total number of zmq messages sent",
            "kwargs": {"labelnames": ["address"]},
        },
        "msg_send_failed_total": {
            "type": Counter,
            "name": "fastdeploy:zmq:msg_send_failed_total",
            "description": "Total number of zmq messages send failed",
            "kwargs": {"labelnames": ["address"]},
        },
        "msg_bytes_send_total": {
            "type": Counter,
            "name": "fastdeploy:zmq:msg_bytes_send_total",
            "description": "Total number of bytes sent over zmq",
            "kwargs": {"labelnames": ["address"]},
        },
        "msg_recv_total": {
            "type": Counter,
            "name": "fastdeploy:zmq:msg_recv_total",
            "description": "Total number of zmq messages recieved",
            "kwargs": {"labelnames": ["address"]},
        },
        "msg_bytes_recv_total": {
            "type": Counter,
            "name": "fastdeploy:zmq:msg_bytes_recv_total",
            "description": "Total number of bytes recieved over zmq",
            "kwargs": {"labelnames": ["address"]},
        },
        "zmq_latency": {
            "type": Histogram,
            "name": "fastdeploy:zmq:latency",
            "description": "Latency of zmq message (in millisecond)",
            "kwargs": {
                "labelnames": ["address"],
                "buckets": [
                    0.001,
                    0.01,
                    0.02,
                    0.05,
                    0.1,
                    0.25,
                    0.5,
                    1.0,
                    2.0,
                    5.0,
                    10.0,
                    20.0,
                    50.0,
                    100.0,
                    200.0,
                    500.0,
                    1000.0,
                ],
            },
        },
    }

    SERVER_METRICS = {
        "e2e_request_latency": {
            "type": Histogram,
            "name": "fastdeploy:e2e_request_latency_seconds",
            "description": "End-to-end request latency (from request arrival to final response)",
            "kwargs": {
                "buckets": [
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
            },
        },
        "request_params_max_tokens": {
            "type": Histogram,
            "name": "fastdeploy:request_params_max_tokens",
            "description": "Histogram of max_tokens parameter in request parameters",
            "kwargs": {"buckets": build_1_2_5_buckets(33792)},
        },
        "prompt_tokens_total": {
            "type": Counter,
            "name": "fastdeploy:prompt_tokens_total",
            "description": "Total number of prompt tokens processed",
            "kwargs": {},
        },
        "request_prompt_tokens": {
            "type": Histogram,
            "name": "fastdeploy:request_prompt_tokens",
            "description": "Number of prefill tokens processed",
            "kwargs": {"buckets": build_1_2_5_buckets(33792)},
        },
    }

    def __init__(self):
        """Initializes the Prometheus metrics and starts the HTTP server if not already initialized."""

        # 在模块加载，指标注册先设置Prometheus环境变量
        setup_multiprocess_prometheus()

        # 动态创建所有指标
        for metric_name, config in self.METRICS.items():
            setattr(
                self,
                metric_name,
                config["type"](config["name"], config["description"], **config["kwargs"]),
            )
        # 动态创建所有指标
        for metric_name, config in self.GAUGE_METRICS.items():
            setattr(
                self,
                metric_name,
                config["type"](config["name"], config["description"], **config["kwargs"]),
            )
        # 动态创建server metrics
        for metric_name, config in self.SERVER_METRICS.items():
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

    def init_zmq_metrics(self):
        # 动态创建所有指标
        for metric_name, config in self.ZMQ_METRICS.items():
            setattr(
                self,
                metric_name,
                config["type"](config["name"], config["description"], **config["kwargs"]),
            )
        self._collect_zmq_metrics = True

    def record_zmq_stats(self, zmq_metrics_stats: ZMQMetricsStats, address: str = "unknown"):
        """
        Recording zmq statistics.
        """
        # 判断是否开启了zmq指标收集
        if not self._collect_zmq_metrics:
            return

        # 记录zmq统计信息
        self.msg_send_total.labels(address=address).inc(zmq_metrics_stats.msg_send_total)
        self.msg_send_failed_total.labels(address=address).inc(zmq_metrics_stats.msg_send_failed_total)
        self.msg_bytes_send_total.labels(address=address).inc(zmq_metrics_stats.msg_bytes_send_total)
        self.msg_recv_total.labels(address=address).inc(zmq_metrics_stats.msg_recv_total)
        self.msg_bytes_recv_total.labels(address=address).inc(zmq_metrics_stats.msg_bytes_recv_total)
        if zmq_metrics_stats.zmq_latency > 0.0:
            # trans to millisecond
            self.zmq_latency.labels(address=address).observe(zmq_metrics_stats.zmq_latency * 1000)

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

    def re_register_gauge(self, registry: CollectorRegistry):
        """Re-register gauge to the specified registry"""
        for metric_name in self.GAUGE_METRICS:
            registry.register(getattr(self, metric_name))

    def register_all(self, registry: CollectorRegistry):
        """Register all metrics to the specified registry"""

        for metric_name in self.METRICS:
            registry.register(getattr(self, metric_name))

        for metric_name in self.GAUGE_METRICS:
            registry.register(getattr(self, metric_name))

        for metric_name in self.SERVER_METRICS:
            registry.register(getattr(self, metric_name))

        if self.cache_config_info is not None:
            registry.register(self.cache_config_info)

        if hasattr(main_process_metrics, "spec_decode_draft_acceptance_rate"):
            self.register_speculative_metrics(registry)

    @classmethod
    def get_excluded_metrics(cls) -> Set[str]:
        """Get the set of indicator names that need to be excluded"""
        return {config["name"] for config in cls.GAUGE_METRICS.values()}


main_process_metrics = MetricsManager()

# 由于zmq指标记录比较耗时，默认不开启，通过DEBUG参数开启
if envs.FD_DEBUG:
    main_process_metrics.init_zmq_metrics()

EXCLUDE_LABELS = MetricsManager.get_excluded_metrics()
