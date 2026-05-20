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
from unittest.mock import MagicMock, patch

import pytest

from fastdeploy.config import BenchmarkMetricsConfig, FDConfig
from fastdeploy.metrics.benchmark_metrics_logger import (
    BenchmarkMetricsLogger,
    CompletedRequestRecord,
)


def _make_record(request_id, now, offset, input_len=100, output_len=50):
    return CompletedRequestRecord(
        request_id=request_id,
        completion_time=now + offset,
        arrival_time=now + offset - 0.05,
        inference_start_time=now + offset - 0.04,
        first_token_time=now + offset - 0.02,
        last_token_time=now + offset,
        input_len=input_len,
        output_len=output_len,
        itl_samples=[0.02, 0.021, 0.019],
    )


def test_config_defaults():
    config = BenchmarkMetricsConfig(None)
    assert config.enable is False
    assert config.window_size == 0
    assert config.window_mode == "sliding"
    assert config.percentile_values == [50.0, 90.0, 95.0, 99.0]
    assert config.selected_metrics == set(BenchmarkMetricsConfig._ALL_METRICS)


def test_config_custom():
    config = BenchmarkMetricsConfig(
        {"enable": True, "window_size": 200, "window_mode": "tumbling", "percentiles": "50,99", "metrics": "ttft,e2el"}
    )
    assert config.enable is True
    assert config.window_size == 200
    assert config.window_mode == "tumbling"
    assert config.percentile_values == [50.0, 99.0]
    assert config.selected_metrics == {"ttft", "e2el"}


def test_config_empty_dict():
    config = BenchmarkMetricsConfig({})
    assert config.enable is False
    assert config.window_size == 0
    assert config.window_mode == "sliding"
    assert config.percentile_values == [50.0, 90.0, 95.0, 99.0]


def test_config_enable_only():
    config = BenchmarkMetricsConfig({"enable": True})
    assert config.enable is True
    assert config.window_mode == "sliding"


def test_logger_writes_jsonl(tmp_path):
    config = BenchmarkMetricsConfig({"enable": True, "window_size": 0, "percentiles": "50,99", "metrics": "ttft,e2el"})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    now = time.time()
    for i in range(5):
        logger.on_request_completed(_make_record(f"req-{i}", now, i * 0.1))

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
    """Sliding window: keeps the last N records, never clears."""
    config = BenchmarkMetricsConfig(
        {"enable": True, "window_size": 3, "window_mode": "sliding", "percentiles": "50", "metrics": "all"}
    )
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    now = time.time()
    for i in range(5):
        logger.on_request_completed(_make_record(f"req-{i}", now, i))

    time.sleep(0.5)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    with open(jsonl_path) as f:
        lines = f.readlines()

    assert len(lines) == 5

    # After 5 records with window_size=3, the window always has at most 3
    rec3 = json.loads(lines[2])  # 3rd record: window full (3 records)
    assert rec3["completed"] == 3

    rec4 = json.loads(lines[3])  # 4th record: still 3 (oldest dropped)
    assert rec4["completed"] == 3

    last_record = json.loads(lines[-1])
    assert last_record["completed"] == 3
    assert last_record["window_size"] == 3
    assert last_record["window_mode"] == "sliding"


def test_logger_tumbling_window(tmp_path):
    """Tumbling window: clears after reaching window_size, then restarts."""
    config = BenchmarkMetricsConfig(
        {"enable": True, "window_size": 3, "window_mode": "tumbling", "percentiles": "50", "metrics": "all"}
    )
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    now = time.time()
    for i in range(5):
        logger.on_request_completed(_make_record(f"req-{i}", now, i))

    time.sleep(0.5)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    with open(jsonl_path) as f:
        lines = f.readlines()

    assert len(lines) == 5

    # Records 1,2,3 accumulate then clear; records 4,5 start fresh
    rec1 = json.loads(lines[0])
    assert rec1["completed"] == 1

    rec3 = json.loads(lines[2])  # 3rd record: window full (3 records), then clears
    assert rec3["completed"] == 3

    rec4 = json.loads(lines[3])  # 4th record: window restarted, 1 record
    assert rec4["completed"] == 1

    rec5 = json.loads(lines[4])  # 5th record: 2 records in new window
    assert rec5["completed"] == 2
    assert rec5["window_mode"] == "tumbling"


def test_logger_no_output_when_no_requests(tmp_path):
    config = BenchmarkMetricsConfig({"enable": True})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    time.sleep(0.3)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    assert os.path.exists(jsonl_path)
    with open(jsonl_path) as f:
        content = f.read()
    assert content == ""


def test_logger_enabled_flag(tmp_path):
    """Logger with enable=False should have enabled=False."""
    config = BenchmarkMetricsConfig({"enable": False})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)
    assert logger.enabled is False
    logger.shutdown()


def test_logger_enabled_true(tmp_path):
    """Logger with enable=True should have enabled=True."""
    config = BenchmarkMetricsConfig({"enable": True})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)
    assert logger.enabled is True
    logger.shutdown()


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


def test_throughput_in_output(tmp_path):
    """Throughput fields should appear when there are 2+ records."""
    config = BenchmarkMetricsConfig({"enable": True, "window_size": 0, "percentiles": "50", "metrics": "ttft"})
    log_dir = str(tmp_path)
    logger = BenchmarkMetricsLogger(config=config, log_dir=log_dir, dp_rank=0)

    now = time.time()
    for i in range(3):
        logger.on_request_completed(_make_record(f"req-{i}", now, i * 0.5))

    time.sleep(0.5)
    logger.shutdown()

    jsonl_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
    with open(jsonl_path) as f:
        lines = f.readlines()

    # First record has no throughput (only 1 sample, duration=0)
    rec1 = json.loads(lines[0])
    assert "request_throughput" not in rec1

    # Last record should have throughput
    last = json.loads(lines[-1])
    assert "request_throughput" in last
    assert "output_throughput" in last
    assert "total_throughput" in last
    assert last["request_throughput"] > 0


# ============================================================
# Validation tests (via FDConfig.check())
# ============================================================


def _make_fd_config_with_benchmark(benchmark_cfg):
    """Create a mock FDConfig with valid base attributes, only benchmark_metrics_config is real."""
    cfg = object.__new__(FDConfig)
    # Mock all attributes accessed by check() before benchmark validation
    cfg.scheduler_config = MagicMock()
    cfg.scheduler_config.max_num_seqs = 128
    cfg.scheduler_config.max_num_batched_tokens = 8192
    cfg.scheduler_config.splitwise_role = "mixed"
    cfg.scheduler_config.check = MagicMock()
    cfg.model_config = MagicMock()
    cfg.model_config.max_model_len = 8192
    cfg.cache_config = MagicMock()
    cfg.cache_config.enable_chunked_prefill = True
    cfg.cache_config.block_size = 64
    cfg.speculative_config = None
    cfg.eplb_config = None
    cfg.structured_outputs_config = None
    cfg.graph_opt_config = MagicMock()
    cfg.graph_opt_config.graph_opt_level = 0
    cfg.nnode = 1
    cfg.max_num_partial_prefills = 1
    cfg.max_long_partial_prefills = 1
    cfg.long_prefill_token_threshold = 0
    cfg.benchmark_metrics_config = benchmark_cfg
    return cfg


@patch("fastdeploy.config.envs")
def test_valid_config_passes_check(mock_envs):
    """Valid configs should pass FDConfig.check() without errors."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    configs = [
        {"enable": True},
        {"enable": True, "window_size": 64, "window_mode": "tumbling"},
        {"enable": False, "window_size": 0, "window_mode": "sliding"},
        {"enable": True, "percentiles": "50,90,99", "metrics": "ttft,e2el,s_decode"},
    ]
    for args in configs:
        benchmark_cfg = BenchmarkMetricsConfig(args)
        fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
        fd_cfg.check()  # Should not raise


@patch("fastdeploy.config.envs")
def test_invalid_enable(mock_envs):
    """enable must be a bool."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": "true"})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="'enable' must be a bool"):
        fd_cfg.check()


@patch("fastdeploy.config.envs")
def test_invalid_window_size_negative(mock_envs):
    """window_size must be non-negative."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": True, "window_size": -1})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="'window_size' must be a non-negative integer"):
        fd_cfg.check()


@patch("fastdeploy.config.envs")
def test_invalid_window_size_type(mock_envs):
    """window_size must be an integer."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": True, "window_size": 3.5})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="'window_size' must be a non-negative integer"):
        fd_cfg.check()


@patch("fastdeploy.config.envs")
def test_invalid_window_mode(mock_envs):
    """window_mode must be 'sliding' or 'tumbling'."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": True, "window_mode": "fixed"})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="'window_mode' must be 'sliding' or 'tumbling'"):
        fd_cfg.check()


@patch("fastdeploy.config.envs")
def test_invalid_percentile_out_of_range(mock_envs):
    """Percentile values must be in [0, 100]."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": True, "percentiles": "50,101"})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="percentile value .* out of range"):
        fd_cfg.check()


@patch("fastdeploy.config.envs")
def test_invalid_percentile_negative(mock_envs):
    """Percentile values must be >= 0."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": True, "percentiles": "-1,50"})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="percentile value .* out of range"):
        fd_cfg.check()


@patch("fastdeploy.config.envs")
def test_invalid_metrics_unknown(mock_envs):
    """Unknown metric names should fail validation."""
    mock_envs.ENABLE_V1_KVCACHE_SCHEDULER = 0
    benchmark_cfg = BenchmarkMetricsConfig({"enable": True, "metrics": "ttft,unknown_metric"})
    fd_cfg = _make_fd_config_with_benchmark(benchmark_cfg)
    with pytest.raises(AssertionError, match="unknown metric"):
        fd_cfg.check()


# ============================================================
# Direct method tests (bypass daemon thread for coverage)
# ============================================================


def test_process_pending_direct(tmp_path):
    """Directly call _process_pending to cover lines 98-109."""
    config = BenchmarkMetricsConfig({"enable": True, "window_size": 0, "metrics": "all", "percentiles": "50,99"})
    logger = BenchmarkMetricsLogger(config=config, log_dir=str(tmp_path), dp_rank=0)

    now = time.time()
    # Add records directly to _pending without relying on background thread
    for i in range(3):
        logger._pending.append(_make_record(f"req-{i}", now, i * 0.5))

    # Call _process_pending directly from main thread (coverage-tracked)
    logger._process_pending()

    assert len(logger._window) == 3
    logger.shutdown()

    jsonl_path = os.path.join(str(tmp_path), "benchmark_metrics.jsonl")
    with open(jsonl_path) as f:
        lines = f.readlines()
    assert len(lines) == 3
    rec = json.loads(lines[-1])
    assert rec["completed"] == 3
    assert "ttft_ms" in rec
    assert "tpot_ms" in rec
    assert "e2el_ms" in rec
    assert "s_ttft_ms" in rec
    assert "s_e2el_ms" in rec
    assert "s_decode" in rec
    assert "input_len" in rec
    assert "s_input_len" in rec
    assert "output_len" in rec
    assert "request_throughput" in rec
    assert "output_throughput" in rec
    assert "total_throughput" in rec


def test_process_pending_tumbling_clear(tmp_path):
    """Tumbling window clears after reaching window_size via direct call."""
    config = BenchmarkMetricsConfig(
        {"enable": True, "window_size": 2, "window_mode": "tumbling", "metrics": "ttft", "percentiles": "50"}
    )
    logger = BenchmarkMetricsLogger(config=config, log_dir=str(tmp_path), dp_rank=0)

    now = time.time()
    for i in range(3):
        logger._pending.append(_make_record(f"req-{i}", now, i * 0.5))

    logger._process_pending()

    # After 3 records with window_size=2: first 2 fill window then clear, 3rd starts fresh
    assert len(logger._window) == 1
    logger.shutdown()


def test_compute_rolling_stats_empty_window(tmp_path):
    """_compute_rolling_stats with empty window returns minimal result."""
    config = BenchmarkMetricsConfig({"enable": True, "window_size": 0, "metrics": "all", "percentiles": "50"})
    logger = BenchmarkMetricsLogger(config=config, log_dir=str(tmp_path), dp_rank=0)

    result = logger._compute_rolling_stats()
    assert result["completed"] == 0
    logger.shutdown()


def test_compute_rolling_stats_single_record(tmp_path):
    """Single record: no throughput, no tpot (needs output_len > 1 check)."""
    config = BenchmarkMetricsConfig({"enable": True, "window_size": 0, "metrics": "all", "percentiles": "50,99"})
    logger = BenchmarkMetricsLogger(config=config, log_dir=str(tmp_path), dp_rank=0)

    now = time.time()
    # output_len=1 means tpot and decode_speed won't be computed
    logger._window.append(
        CompletedRequestRecord(
            request_id="r1",
            completion_time=now,
            arrival_time=now - 0.05,
            inference_start_time=now - 0.04,
            first_token_time=now - 0.02,
            last_token_time=now,
            input_len=100,
            output_len=1,
            itl_samples=[],
        )
    )

    result = logger._compute_rolling_stats()
    assert result["completed"] == 1
    assert "request_throughput" not in result  # duration=0 for single record
    assert result["ttft_ms"]["mean"] > 0
    assert result["tpot_ms"] == {}  # no tpot with output_len=1
    assert result["s_itl_ms"] == {}  # no itl samples
    logger.shutdown()


def test_compute_rolling_stats_multiple_records(tmp_path):
    """Multiple records: throughput and all metrics computed."""
    config = BenchmarkMetricsConfig({"enable": True, "window_size": 0, "metrics": "all", "percentiles": "50,95"})
    logger = BenchmarkMetricsLogger(config=config, log_dir=str(tmp_path), dp_rank=0)

    now = time.time()
    for i in range(3):
        logger._window.append(_make_record(f"req-{i}", now, i * 0.5))

    result = logger._compute_rolling_stats()
    assert result["completed"] == 3
    assert result["request_throughput"] > 0
    assert result["output_throughput"] > 0
    assert result["total_throughput"] > 0
    assert result["ttft_ms"]["mean"] > 0
    assert result["s_ttft_ms"]["mean"] > 0
    assert result["tpot_ms"]["mean"] > 0
    assert result["s_itl_ms"]["mean"] > 0
    assert result["e2el_ms"]["mean"] > 0
    assert result["s_e2el_ms"]["mean"] > 0
    assert result["s_decode"]["mean"] > 0
    assert "p50" in result["ttft_ms"]
    assert "p95" in result["ttft_ms"]
    logger.shutdown()


def test_stats_with_float_percentile():
    """Percentile key uses float format when not integer."""
    stats = BenchmarkMetricsLogger._stats([1.0, 2.0, 3.0], [99.9])
    assert "p99.9" in stats
