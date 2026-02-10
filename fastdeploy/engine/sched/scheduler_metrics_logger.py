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

import logging
import threading
import time
from typing import Iterable

from fastdeploy import envs


class SchedulerMetricsLogger:
    """
    Lightweight console logger for scheduler-level prefill/decode metrics.
    """

    DEFAULT_DECODE_LOG_INTERVAL = 5

    def __init__(self, enabled: bool = True, dp_rank: int = 0) -> None:
        self.enabled = enabled
        self.dp_rank = dp_rank
        decode_log_interval = envs.FD_CONSOLE_DECODE_LOG_INTERVAL
        if decode_log_interval <= 0:
            decode_log_interval = self.DEFAULT_DECODE_LOG_INTERVAL
        self._lock = threading.Lock()
        self._decode_log_interval = decode_log_interval
        self._decode_batch_count = 0
        self._last_decode_tic = time.perf_counter()
        self._decode_tokens_since_last = 0
        self._logger = self._get_logger()

    def _get_logger(self) -> logging.Logger:
        logger = logging.getLogger("fastdeploy.scheduler_metrics")
        if not getattr(logger, "_fd_scheduler_metrics_configured", False):
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s",
                "%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger._fd_scheduler_metrics_configured = True
        return logger

    def on_decode_tokens(self, num_tokens: int) -> None:
        if not self.enabled:
            return
        if num_tokens <= 0:
            return
        with self._lock:
            self._decode_tokens_since_last += num_tokens

    def log_prefill_batch(
        self,
        prefill_reqs: Iterable,
        running_cnt: int,
        queue_cnt: int,
        tokens_used: int,
        token_usage: float,
    ) -> None:
        if not self.enabled:
            return
        prefill_reqs = list(prefill_reqs)
        if not prefill_reqs:
            return

        new_tokens = 0
        cached_tokens = 0
        for req in prefill_reqs:
            start = getattr(req, "prefill_start_index", 0) or 0
            end = getattr(req, "prefill_end_index", 0) or 0
            if end > start:
                new_tokens += end - start
            cached_tokens += getattr(req, "num_cached_tokens", 0) or 0

        msg = (
            "Prefill batch, "
            f"dp_rank: {self.dp_rank}, "
            f"#new-seq: {len(prefill_reqs)}, "
            f"#new-token: {new_tokens}, "
            f"#cached-token: {cached_tokens}, "
            f"token usage: {token_usage:.2f}, "
            f"#running-req: {running_cnt}, "
            f"#queue-req: {queue_cnt}, "
        )
        self._logger.info(msg)

    def log_decode_batch(
        self,
        running_cnt: int,
        queue_cnt: int,
        tokens_used: int,
        token_usage: float,
        use_cudagraph: bool,
    ) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._decode_batch_count += 1
            if self._decode_batch_count % self._decode_log_interval != 0:
                return
            now = time.perf_counter()
            elapsed = now - self._last_decode_tic
            if elapsed > 0:
                throughput = self._decode_tokens_since_last / elapsed
            else:
                throughput = 0.0
            self._decode_tokens_since_last = 0
            self._last_decode_tic = now

        msg = (
            "Decode batch, "
            f"dp_rank: {self.dp_rank}, "
            f"#running-req: {running_cnt}, "
            f"#token: {tokens_used}, "
            f"token usage: {token_usage:.2f}, "
            f"cuda graph: {use_cudagraph}, "
            f"gen throughput (token/s): {throughput:.2f}, "
            f"#queue-req: {queue_cnt}, "
        )
        self._logger.info(msg)
