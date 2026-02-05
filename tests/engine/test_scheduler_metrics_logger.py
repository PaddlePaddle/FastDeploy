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

import types
from unittest import mock

from fastdeploy.engine.sched.scheduler_metrics_logger import SchedulerMetricsLogger


def test_on_decode_tokens_accumulates():
    logger = SchedulerMetricsLogger(enabled=True, dp_rank=0)
    logger._decode_tokens_since_last = 0

    logger.on_decode_tokens(3)
    logger.on_decode_tokens(0)
    logger.on_decode_tokens(-1)

    assert logger._decode_tokens_since_last == 3


def test_log_prefill_batch_logs_expected_message():
    logger = SchedulerMetricsLogger(enabled=True, dp_rank=2)
    logger._logger = mock.Mock()

    reqs = [
        types.SimpleNamespace(prefill_start_index=0, prefill_end_index=4, num_cached_tokens=2),
        types.SimpleNamespace(prefill_start_index=3, prefill_end_index=3, num_cached_tokens=1),
    ]

    logger.log_prefill_batch(prefill_reqs=reqs, running_cnt=5, queue_cnt=6, tokens_used=10, token_usage=0.75)

    logger._logger.info.assert_called_once()
    message = logger._logger.info.call_args[0][0]
    assert "Prefill batch" in message
    assert "dp_rank: 2" in message
    assert "#new-seq: 2" in message
    assert "#new-token: 4" in message
    assert "#cached-token: 3" in message
    assert "token usage: 0.75" in message
    assert "#running-req: 5" in message
    assert "#queue-req: 6" in message


def test_log_decode_batch_computes_throughput(monkeypatch):
    logger = SchedulerMetricsLogger(enabled=True, dp_rank=1)
    logger._logger = mock.Mock()
    logger._decode_tokens_since_last = 10
    logger._last_decode_tic = 1.0

    monkeypatch.setattr("fastdeploy.engine.sched.scheduler_metrics_logger.time.perf_counter", lambda: 3.0)

    logger.log_decode_batch(running_cnt=4, queue_cnt=7, tokens_used=8, token_usage=0.5, use_cudagraph=True)

    logger._logger.info.assert_called_once()
    message = logger._logger.info.call_args[0][0]
    assert "Decode batch" in message
    assert "dp_rank: 1" in message
    assert "gen throughput (token/s): 5.00" in message
    assert "#queue-req: 7" in message
    assert logger._decode_tokens_since_last == 0
    assert logger._last_decode_tic == 3.0
