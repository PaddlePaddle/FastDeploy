"""
Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import os
import time

from fastdeploy.config import BenchmarkMetricsConfig
from fastdeploy.metrics.benchmark_metrics_logger import (
    BenchmarkMetricsLogger,
    CompletedRequestRecord,
)


def test_config_defaults():
    config = BenchmarkMetricsConfig(None)
    assert config.window_size == 0
    assert config.percentile_values == [50.0, 90.0, 95.0, 99.0]
    assert config.selected_metrics == set(BenchmarkMetricsConfig._ALL_METRICS)


def test_config_custom():
    config = BenchmarkMetricsConfig({"window_size": 200, "percentiles": "50,99", "metrics": "ttft,e2el"})
    assert config.window_size == 200
    assert config.percentile_values == [50.0, 99.0]
    assert config.selected_metrics == {"ttft", "e2el"}


def test_config_empty_dict():
    config = BenchmarkMetricsConfig({})
    assert config.window_size == 0
    assert config.percentile_values == [50.0, 90.0, 95.0, 99.0]


def test_logger_writes_jsonl(tmp_path):
    config = BenchmarkMetricsConfig({"window_size": 0, "percentiles": "50,99", "metrics": "ttft,e2el"})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    now = time.time()
    for i in range(5):
        record = CompletedRequestRecord(
            request_id=f"req-{i}",
            completion_time=now + i * 0.1,
            arrival_time=now + i * 0.1 - 0.05,
            inference_start_time=now + i * 0.1 - 0.04,
            first_token_time=now + i * 0.1 - 0.02,
            last_token_time=now + i * 0.1,
            input_len=100,
            output_len=50,
            itl_samples=[0.02, 0.021, 0.019],
        )
        logger.on_request_completed(record)

    time.sleep(0.5)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    assert os.path.exists(jsonl_path)

    with open(jsonl_path) as f:
        lines = f.readlines()

    assert len(lines) == 5

    last_record = json.loads(lines[-1])
    assert last_record["completed"] == 5
    assert "ttft_ms" in last_record
    assert "e2el_ms" in last_record
    assert "tpot_ms" not in last_record
    assert last_record["ttft_ms"]["mean"] > 0


def test_logger_sliding_window(tmp_path):
    config = BenchmarkMetricsConfig({"window_size": 3, "percentiles": "50", "metrics": "all"})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    now = time.time()
    for i in range(5):
        record = CompletedRequestRecord(
            request_id=f"req-{i}",
            completion_time=now + i,
            arrival_time=now + i - 0.1,
            inference_start_time=now + i - 0.08,
            first_token_time=now + i - 0.05,
            last_token_time=now + i,
            input_len=100 + i * 10,
            output_len=50 + i * 5,
            itl_samples=[0.02] * 10,
        )
        logger.on_request_completed(record)

    time.sleep(0.5)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    with open(jsonl_path) as f:
        lines = f.readlines()

    assert len(lines) == 5

    last_record = json.loads(lines[-1])
    assert last_record["completed"] == 3
    assert last_record["window_size"] == 3


def test_logger_no_output_when_no_requests(tmp_path):
    config = BenchmarkMetricsConfig({})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    time.sleep(0.3)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    assert os.path.exists(jsonl_path)
    with open(jsonl_path) as f:
        content = f.read()
    assert content == ""


def test_stats_computation():
    stats = BenchmarkMetricsLogger._stats([10.0, 20.0, 30.0, 40.0, 50.0], [50.0, 99.0])
    assert stats["mean"] == 30.0
    assert stats["median"] == 30.0
    assert "p50" in stats
    assert "p99" in stats
    assert stats["p50"] == 30.0


def test_stats_empty_list():
    stats = BenchmarkMetricsLogger._stats([], [50.0])
    assert stats == {}
