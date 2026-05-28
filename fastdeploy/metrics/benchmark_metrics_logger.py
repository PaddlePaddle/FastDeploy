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

import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from fastdeploy.config import BenchmarkMetricsConfig


@dataclass(slots=True)
class CompletedRequestRecord:
    """Raw timing data collected when a request completes."""

    request_id: str
    completion_time: float
    arrival_time: float
    inference_start_time: float
    first_token_time: float
    last_token_time: float
    input_len: int
    output_len: int
    num_cached_tokens: int = 0
    itl_samples: list = field(default_factory=list)


class BenchmarkMetricsLogger:
    """
    In-process performance monitoring that produces metrics aligned with
    benchmark_serving.py. Uses a lock-free deque for data collection and
    a background daemon thread for stats computation and file I/O.
    """

    def __init__(self, config: BenchmarkMetricsConfig, log_dir: str, dp_rank: int = 0):
        self.config = config
        self.enabled = config.enable
        self.dp_rank = dp_rank

        if config.window_mode == "sliding" and config.window_size > 0:
            self._window: deque = deque(maxlen=config.window_size)
        else:
            self._window: deque = deque()

        self._pending: deque = deque()
        self._condition = threading.Condition()
        self._stop_event = threading.Event()

        os.makedirs(log_dir, exist_ok=True)
        self._file_path = os.path.join(log_dir, "benchmark_metrics.jsonl")
        self._file = open(self._file_path, "a", encoding="utf-8")

        self._thread = threading.Thread(
            target=self._writer_loop,
            daemon=True,
            name=f"BenchmarkMetricsLogger-dp{dp_rank}",
        )
        self._thread.start()

    def on_request_completed(self, record: CompletedRequestRecord) -> None:
        """Called from token processor on request completion. Lock-free append."""
        self._pending.append(record)
        with self._condition:
            self._condition.notify()

    def _writer_loop(self) -> None:
        """Background thread: wait for new records, compute stats, write JSONL."""
        while not self._stop_event.is_set():
            with self._condition:
                self._condition.wait(timeout=1.0)
            self._process_pending()

    def _process_pending(self) -> None:
        """Process all pending records, write one JSONL line per record."""
        while True:
            try:
                record = self._pending.popleft()
            except IndexError:
                break
            self._window.append(record)
            stats = self._compute_rolling_stats()
            line = json.dumps(stats, ensure_ascii=False)
            self._file.write(line + "\n")
            # Tumbling window: clear after reaching window_size
            if (
                self.config.window_mode == "tumbling"
                and self.config.window_size > 0
                and len(self._window) >= self.config.window_size
            ):
                self._window.clear()
        self._file.flush()

    def _compute_rolling_stats(self) -> dict:
        """Compute aggregate statistics over the current window."""
        records = list(self._window)
        n = len(records)
        if n == 0:
            return {"timestamp": datetime.now().isoformat(), "completed": 0}

        selected = self.config.selected_metrics
        percentile_values = self.config.percentile_values

        ttfts = []
        s_ttfts = []
        tpots = []
        all_itls = []
        e2els = []
        s_e2els = []
        decode_speeds = []
        input_lens = []
        s_input_lens = []
        output_lens = []

        for r in records:
            if r.first_token_time and r.arrival_time:
                ttfts.append((r.first_token_time - r.arrival_time) * 1000)
            if r.first_token_time and r.inference_start_time:
                s_ttfts.append((r.first_token_time - r.inference_start_time) * 1000)
            if r.output_len > 1 and r.first_token_time and r.arrival_time:
                e2el_s = r.last_token_time - r.arrival_time
                ttft_s = r.first_token_time - r.arrival_time
                tpots.append(((e2el_s - ttft_s) / (r.output_len - 1)) * 1000)
            if r.itl_samples:
                all_itls.extend([x * 1000 for x in r.itl_samples])
            if r.last_token_time and r.arrival_time:
                e2els.append((r.last_token_time - r.arrival_time) * 1000)
            if r.last_token_time and r.inference_start_time:
                s_e2els.append((r.last_token_time - r.inference_start_time) * 1000)
            if r.output_len > 1 and r.first_token_time and r.last_token_time:
                decode_time = r.last_token_time - r.first_token_time
                if decode_time > 0:
                    decode_speeds.append((r.output_len - 1) / decode_time)
            input_lens.append(r.num_cached_tokens)
            s_input_lens.append(r.input_len)
            output_lens.append(r.output_len)

        # Throughput: based on window time span
        total_input = sum(s_input_lens)
        total_output = sum(output_lens)
        if n >= 2:
            duration = records[-1].completion_time - records[0].arrival_time
        else:
            duration = 0.0

        result: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "window_size": self.config.window_size,
            "window_mode": self.config.window_mode,
            "completed": n,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
        }

        if duration > 0:
            result["request_throughput"] = round(n / duration, 2)
            result["output_throughput"] = round(total_output / duration, 2)
            result["total_throughput"] = round((total_input + total_output) / duration, 2)

        if "ttft" in selected:
            result["ttft_ms"] = self._stats(ttfts, percentile_values)
        if "s_ttft" in selected:
            result["s_ttft_ms"] = self._stats(s_ttfts, percentile_values)
        if "tpot" in selected:
            result["tpot_ms"] = self._stats(tpots, percentile_values)
        if "s_itl" in selected:
            result["s_itl_ms"] = self._stats(all_itls, percentile_values)
        if "e2el" in selected:
            result["e2el_ms"] = self._stats(e2els, percentile_values)
        if "s_e2el" in selected:
            result["s_e2el_ms"] = self._stats(s_e2els, percentile_values)
        if "s_decode" in selected:
            result["s_decode"] = self._stats(decode_speeds, percentile_values)
        if "input_len" in selected:
            result["input_len"] = self._stats(input_lens, percentile_values)
        if "s_input_len" in selected:
            result["s_input_len"] = self._stats(s_input_lens, percentile_values)
        if "output_len" in selected:
            result["output_len"] = self._stats(output_lens, percentile_values)

        return result

    @staticmethod
    def _stats(values: list, percentiles: list[float]) -> dict:
        """Compute mean/median/percentiles for a list of values."""
        if not values:
            return {}
        arr = np.array(values)
        result = {
            "mean": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
        }
        for p in percentiles:
            key = f"p{int(p)}" if int(p) == p else f"p{p}"
            result[key] = round(float(np.percentile(arr, p)), 2)
        return result

    def shutdown(self) -> None:
        """Stop the writer thread and close the file."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify()
        self._thread.join(timeout=5)
        self._process_pending()
        self._file.close()
