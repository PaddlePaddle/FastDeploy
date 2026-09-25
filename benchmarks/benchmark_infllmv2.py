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

"""Run and compare traceable dense/InfLLM-V2 long-context benchmarks.

The ``run`` command delegates request generation and latency collection to
``benchmark_serving.py`` while sampling total memory used on one GPU.  The
``report`` command only compares run files whose workload fingerprints match.
The ``sparse-diagnostic`` command binds a newly generated selector trace to the
exact random-token workload without treating its trace-instrumented latency as
a performance result.  A timed sparse run requires that bound diagnostic.
The ``operators`` command uses CUDA events to time compressed-K update, Stage 1,
Stage 2, and their complete chain for a configurable context/concurrency matrix.
The ``prefill`` command compares complete sparse and dense Paddle prefill paths.
The ``cuda-impl`` command runs the equivalent decode workload against a local
``infllm_v2`` PyTorch extension checkout.

Selector hit rate is the micro-averaged recall of dense-reference blocks:

    sum(|selected_blocks & reference_blocks|) / sum(|reference_blocks|)

Each selector sample represents one request/query/KV-head tuple.  Unbound raw
samples and summaries cannot be attached to timed sparse results.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCHEMA_VERSION = 2
BENCHMARK_SERVING = Path(__file__).with_name("benchmark_serving.py")
SELECTOR_DEFINITION = "micro recall of dense-reference blocks in selected blocks"
SELECTOR_REFERENCE_DEFINITION = (
    "top blocks by exact dense softmax attention mass, summed over the query heads sharing one KV head"
)
SELECTOR_TRACE_PATH_ENV = "FD_INFLLMV2_SELECTOR_TRACE_PATH"
STAGE2_BLOCKS_PER_SPLIT = 2
SERVING_RESULT_FIELDS = (
    "duration",
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "p99_tpot_ms",
)


@dataclass(frozen=True)
class SelectorMetrics:
    selector_hit_rate: float
    selector_samples: int
    selector_hits: int | None
    selector_targets: int | None
    definition: str = SELECTOR_DEFINITION


@dataclass(frozen=True)
class GPUMemoryMetrics:
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    total_mib: int
    baseline_mib: int
    peak_mib: int
    peak_delta_mib: int
    samples: int
    sample_interval_seconds: float


@dataclass(frozen=True)
class GPUIdentity:
    gpu_index: int
    gpu_uuid: str
    gpu_name: str
    total_mib: int


@dataclass(frozen=True)
class OperatorScenario:
    context_length: int
    concurrency: int


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as json_file:
        value = json.load(json_file)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object, got {type(value).__name__}.")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as binary_file:
        for chunk in iter(lambda: binary_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _loaded_shared_object_metadata(filename: str) -> dict[str, Any]:
    mapped_paths = set()
    with Path("/proc/self/maps").open("r", encoding="utf-8") as maps_file:
        for line in maps_file:
            mapped_path = line.rstrip().split(maxsplit=5)[-1]
            if mapped_path.endswith(filename):
                mapped_paths.add(Path(mapped_path).resolve())
    if len(mapped_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one loaded {filename}, found " f"{[str(path) for path in sorted(mapped_paths)]}."
        )
    path = mapped_paths.pop()
    stat_result = path.stat()
    return {
        "path": str(path),
        "mtime_ns": stat_result.st_mtime_ns,
        "mtime_utc": datetime.datetime.fromtimestamp(stat_result.st_mtime, datetime.timezone.utc).isoformat(),
        "size_bytes": stat_result.st_size,
        "sha256": _sha256_file(path),
    }


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str, os.stat_result]:
    stat_before = path.stat()
    if not path.is_file():
        raise ValueError(f"{path} must be a regular file.")
    with path.open("rb") as binary_file:
        raw_value = binary_file.read()
    stat_after = path.stat()
    if (
        stat_before.st_dev,
        stat_before.st_ino,
        stat_before.st_size,
        stat_before.st_mtime_ns,
    ) != (
        stat_after.st_dev,
        stat_after.st_ino,
        stat_after.st_size,
        stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"{path} changed while it was being read.")
    value = json.loads(raw_value)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object, got {type(value).__name__}.")
    return value, _sha256_bytes(raw_value), stat_after


def _write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as json_file:
        json.dump(value, json_file, ensure_ascii=False, indent=2, sort_keys=True)
        json_file.write("\n")


def _require_int(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}.")
    return value


def _require_number(value: Any, name: str, minimum: float, inclusive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {number}.")
    invalid = number < minimum if inclusive else number <= minimum
    if invalid:
        operator = ">=" if inclusive else ">"
        raise ValueError(f"{name} must be {operator} {minimum}, got {number}.")
    return number


def _require_rate(value: Any, name: str) -> float:
    rate = _require_number(value, name, 0.0)
    if rate > 1.0:
        raise ValueError(f"{name} must be <= 1.0, got {rate}.")
    return rate


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest.")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA256 hex digest.")
    return value


def _validate_block_ids(value: Any, name: str, allow_empty: bool) -> set[int]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array.")
    if not value and not allow_empty:
        raise ValueError(f"{name} must not be empty.")
    block_ids = {_require_int(block_id, f"{name}[]", 0) for block_id in value}
    if len(block_ids) != len(value):
        raise ValueError(f"{name} must not contain duplicate block IDs.")
    return block_ids


def _selector_from_trace_payload(
    payload: dict[str, Any],
    path: Path,
) -> tuple[SelectorMetrics, dict[str, Any]]:
    if payload["schema_version"] != 1:
        raise ValueError(f"{path}: selector trace schema_version must be 1.")
    if payload["kind"] != "infllmv2_selector_samples":
        raise ValueError(f"{path}: selector trace kind must be 'infllmv2_selector_samples', got {payload['kind']!r}.")
    reference_definition = payload["reference_definition"]
    if reference_definition != SELECTOR_REFERENCE_DEFINITION:
        raise ValueError(
            f"{path}: reference_definition must be {SELECTOR_REFERENCE_DEFINITION!r}, got {reference_definition!r}."
        )
    rank = _require_int(payload["rank"], f"{path}: rank", 0)
    layer = _require_int(payload["layer"], f"{path}: layer", 0)
    block_size = _require_int(payload["block_size"], f"{path}: block_size", 1)
    kernel_size = _require_int(payload["kernel_size"], f"{path}: kernel_size", 1)
    kernel_stride = _require_int(payload["kernel_stride"], f"{path}: kernel_stride", 1)
    topk = _require_int(payload["topk"], f"{path}: topk", 1)
    dense_len = _require_int(payload["dense_len"], f"{path}: dense_len", 1)
    init_blocks = _require_int(payload["init_blocks"], f"{path}: init_blocks", 0)
    local_blocks = _require_int(payload["local_blocks"], f"{path}: local_blocks", 0)
    selected_capacity = _require_int(payload["selected_capacity"], f"{path}: selected_capacity", 1)
    if block_size % kernel_stride != 0:
        raise ValueError(f"{path}: block_size must be divisible by kernel_stride.")
    if block_size % (4 * kernel_stride) != 0:
        raise ValueError(f"{path}: block_size must be divisible by 4 * kernel_stride.")
    if dense_len < 4 * kernel_size:
        raise ValueError(f"{path}: dense_len must be at least 4 * kernel_size.")
    if init_blocks >= topk:
        raise ValueError(f"{path}: init_blocks must be smaller than topk.")
    expected_capacity = max(
        topk + local_blocks,
        (dense_len + block_size - 1) // block_size,
    )
    if selected_capacity != expected_capacity:
        raise ValueError(
            f"{path}: selected_capacity must be {expected_capacity} for the "
            f"header configuration, got {selected_capacity}."
        )
    samples = payload["samples"]
    if not isinstance(samples, list):
        raise TypeError(f"{path}: samples must be a JSON array.")
    if not samples:
        raise ValueError(f"{path}: samples must not be empty.")

    hits = 0
    targets = 0
    selected_count_values: set[int] = set()
    query_head_group_sizes: set[int] = set()
    kv_heads: set[int] = set()
    request_indices: set[int] = set()
    query_positions: list[int] = []
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise TypeError(f"{path}: samples[{sample_index}] must be a JSON object.")
        sample_name = f"{path}: samples[{sample_index}]"
        sample_rank = _require_int(sample["rank"], f"{sample_name}.rank", 0)
        sample_layer = _require_int(sample["layer"], f"{sample_name}.layer", 0)
        if sample_rank != rank or sample_layer != layer:
            raise ValueError(
                f"{sample_name}: rank/layer must match the trace header "
                f"({rank}/{layer}), got {sample_rank}/{sample_layer}."
            )
        request_index = _require_int(sample["request_index"], f"{sample_name}.request_index", 0)
        _require_int(sample["query_index"], f"{sample_name}.query_index", 0)
        _require_int(sample["query_offset"], f"{sample_name}.query_offset", 0)
        query_position = _require_int(sample["query_position"], f"{sample_name}.query_position", 0)
        kv_head = _require_int(sample["kv_head"], f"{sample_name}.kv_head", 0)
        query_head_start = _require_int(sample["query_head_start"], f"{sample_name}.query_head_start", 0)
        query_head_end = _require_int(sample["query_head_end"], f"{sample_name}.query_head_end", 1)
        if query_head_end <= query_head_start:
            raise ValueError(f"{sample_name}.query_head_end must be greater than query_head_start.")
        query_head_group_size = query_head_end - query_head_start
        if query_head_start != kv_head * query_head_group_size:
            raise ValueError(f"{sample_name}: query-head range does not match contiguous GQA group {kv_head}.")
        sample_block_size = _require_int(sample["block_size"], f"{sample_name}.block_size", 1)
        sample_topk = _require_int(sample["topk"], f"{sample_name}.topk", 1)
        if sample_block_size != block_size or sample_topk != topk:
            raise ValueError(
                f"{sample_name}: block_size/topk must match the trace header "
                f"({block_size}/{topk}), got {sample_block_size}/{sample_topk}."
            )
        selected_count = _require_int(sample["selected_count"], f"{sample_name}.selected_count", 1)
        if selected_count > selected_capacity:
            raise ValueError(f"{sample_name}.selected_count must not exceed selected_capacity {selected_capacity}.")
        selected = _validate_block_ids(
            sample["selected_blocks"],
            f"{sample_name}.selected_blocks",
            allow_empty=False,
        )
        reference = _validate_block_ids(
            sample["reference_blocks"],
            f"{sample_name}.reference_blocks",
            allow_empty=False,
        )
        if len(selected) != selected_count or len(reference) != selected_count:
            raise ValueError(
                f"{sample_name}: selected_blocks and reference_blocks must both "
                f"contain selected_count={selected_count} entries."
            )
        if sample["selected_blocks"] != sorted(selected):
            raise ValueError(f"{sample_name}.selected_blocks must be strictly increasing.")
        if sample["reference_blocks"] != sorted(reference):
            raise ValueError(f"{sample_name}.reference_blocks must be strictly increasing.")
        if sample["reference_metric"] != reference_definition:
            raise ValueError(f"{sample_name}.reference_metric must match reference_definition.")
        valid_blocks = (query_position + 1 + block_size - 1) // block_size
        if query_position + 1 < dense_len:
            raise ValueError(
                f"{sample_name}.query_position must be in the sparse region starting at visible length {dense_len}."
            )
        if selected_count > valid_blocks:
            raise ValueError(f"{sample_name}.selected_count must not exceed {valid_blocks} visible blocks.")
        if max(selected.union(reference)) >= valid_blocks:
            raise ValueError(
                f"{sample_name}: block IDs must be smaller than the {valid_blocks} blocks visible at query_position."
            )
        hits += len(selected.intersection(reference))
        targets += len(reference)
        selected_count_values.add(selected_count)
        query_head_group_sizes.add(query_head_group_size)
        kv_heads.add(kv_head)
        request_indices.add(request_index)
        query_positions.append(query_position)

    if len(query_head_group_sizes) != 1:
        raise ValueError(f"{path}: query-head group size must be consistent across selector samples.")

    selector = SelectorMetrics(
        selector_hit_rate=hits / targets,
        selector_samples=len(samples),
        selector_hits=hits,
        selector_targets=targets,
    )
    config = {
        "block_size": block_size,
        "kernel_size": kernel_size,
        "kernel_stride": kernel_stride,
        "topk": topk,
        "dense_len": dense_len,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
        "selected_capacity": selected_capacity,
        "query_head_group_size": next(iter(query_head_group_sizes)),
        "selected_count_values": sorted(selected_count_values),
        "kv_heads_observed": sorted(kv_heads),
        "request_slots_observed": sorted(request_indices),
        "query_position_min": min(query_positions),
        "query_position_max": max(query_positions),
    }
    provenance = {
        "rank": rank,
        "layer": layer,
        "config": config,
        "reference_definition": reference_definition,
    }
    return selector, provenance


def _load_selector_trace(
    path: Path,
) -> tuple[SelectorMetrics, dict[str, Any], str, os.stat_result]:
    payload, trace_sha256, trace_stat = _read_json_snapshot(path)
    selector, provenance = _selector_from_trace_payload(payload, path)
    return selector, provenance, trace_sha256, trace_stat


def _validate_workload_args(args: argparse.Namespace) -> None:
    parsed_url = urlparse(args.base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError(f"--base-url must be an absolute HTTP(S) URL, got {args.base_url!r}.")
    if not args.endpoint.startswith("/"):
        raise ValueError(f"--endpoint must start with '/', got {args.endpoint!r}.")
    if not args.model.strip():
        raise ValueError("--model must not be empty.")
    if args.tokenizer is not None and not args.tokenizer.strip():
        raise ValueError("--tokenizer must not be empty when provided.")
    _require_int(args.seed, "--seed", 0)
    _require_int(args.num_prompts, "--num-prompts", 1)
    _require_int(args.input_len, "--input-len", 1)
    _require_int(args.output_len, "--output-len", 2)
    _require_int(args.max_concurrency, "--max-concurrency", 1)
    if math.isnan(args.request_rate) or args.request_rate <= 0:
        raise ValueError(f"--request-rate must be positive, got {args.request_rate}.")


def _validate_output_args(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; pass --overwrite to replace it.")
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {args.output.parent}")


def _validate_run_args(args: argparse.Namespace) -> None:
    _validate_workload_args(args)
    _validate_output_args(args)
    _require_int(args.gpu_index, "--gpu-index", 0)
    _require_number(args.sample_interval, "--sample-interval", 0.0, inclusive=False)
    if args.variant == "dense" and args.selector_diagnostic is not None:
        raise ValueError("--selector-diagnostic applies only to --variant sparse.")
    if args.variant == "sparse" and args.selector_diagnostic is None:
        raise ValueError("--variant sparse requires --selector-diagnostic.")
    if args.variant == "sparse" and os.getenv(SELECTOR_TRACE_PATH_ENV):
        raise ValueError(
            f"Timed sparse runs require tracing to be disabled; unset {SELECTOR_TRACE_PATH_ENV} "
            "and restart the sparse server without selector tracing."
        )


def _validate_diagnostic_args(args: argparse.Namespace) -> None:
    _validate_workload_args(args)
    _validate_output_args(args)
    trace_path = args.trace_path.resolve()
    if not trace_path.parent.is_dir():
        raise FileNotFoundError(f"Selector trace output directory does not exist: {trace_path.parent}")
    if trace_path.exists():
        raise FileExistsError(
            f"Selector trace output already exists: {trace_path}; diagnostic runs require a new trace path."
        )
    configured_trace_path = os.getenv(SELECTOR_TRACE_PATH_ENV)
    if configured_trace_path and Path(configured_trace_path).expanduser().resolve() != trace_path:
        raise ValueError(f"--trace-path must match {SELECTOR_TRACE_PATH_ENV} when that environment variable is set.")


def _canonical_request_rate(request_rate: float) -> float | str:
    return "inf" if math.isinf(request_rate) else request_rate


def _build_workload(
    args: argparse.Namespace,
    prompt_token_ids_sha256: str,
    benchmark_serving_sha256: str,
) -> dict[str, Any]:
    return {
        "backend": "openai-chat",
        "benchmark_serving_sha256": benchmark_serving_sha256,
        "dataset_name": "random_token_ids",
        "endpoint": args.endpoint,
        "ignore_eos": True,
        "input_len": args.input_len,
        "max_concurrency": args.max_concurrency,
        "model": args.model,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "prompt_token_ids_sha256": prompt_token_ids_sha256,
        "random_range_ratio": 0.0,
        "request_rate": _canonical_request_rate(args.request_rate),
        "seed": args.seed,
        "stream": True,
        "tokenizer": args.tokenizer,
    }


def _workload_id(workload: dict[str, Any]) -> str:
    serialized = json.dumps(workload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _prompt_token_ids_sha256(result: dict[str, Any], args: argparse.Namespace) -> str:
    prompts = result["input_texts"]
    if not isinstance(prompts, list):
        raise TypeError("benchmark result: input_texts must be a JSON array.")
    if len(prompts) != args.num_prompts:
        raise RuntimeError(f"Expected {args.num_prompts} prompt token arrays, got {len(prompts)}.")
    for prompt_index, prompt_token_ids in enumerate(prompts):
        if not isinstance(prompt_token_ids, list):
            raise TypeError(f"benchmark result: input_texts[{prompt_index}] must be a JSON array.")
        if len(prompt_token_ids) != args.input_len:
            raise RuntimeError(
                f"Expected input_texts[{prompt_index}] to contain {args.input_len} "
                f"tokens, got {len(prompt_token_ids)}."
            )
        for token_index, token_id in enumerate(prompt_token_ids):
            _require_int(
                token_id,
                f"benchmark result: input_texts[{prompt_index}][{token_index}]",
                0,
            )
    serialized = json.dumps(prompts, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _query_gpu_identity(gpu_index: int) -> GPUIdentity:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = list(csv.reader(completed.stdout.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError(f"Expected one four-field nvidia-smi identity row for GPU {gpu_index}, got {rows!r}.")
    reported_index = _require_int(int(rows[0][0]), f"GPU {gpu_index} reported index", 0)
    if reported_index != gpu_index:
        raise RuntimeError(f"nvidia-smi --id={gpu_index} reported physical GPU index {reported_index}.")
    gpu_uuid = rows[0][1].strip()
    gpu_name = rows[0][2].strip()
    if not gpu_uuid or not gpu_name:
        raise RuntimeError(f"GPU {gpu_index} UUID and name must not be empty.")
    total_mib = _require_int(int(rows[0][3]), f"GPU {gpu_index} memory.total", 1)
    return GPUIdentity(
        gpu_index=reported_index,
        gpu_uuid=gpu_uuid,
        gpu_name=gpu_name,
        total_mib=total_mib,
    )


def _query_gpu_memory_mib(gpu_index: int) -> int:
    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected one nvidia-smi memory value for GPU {gpu_index}, got {lines!r}.")
    return _require_int(int(lines[0]), f"GPU {gpu_index} memory.used", 0)


def _benchmark_command(
    args: argparse.Namespace,
    raw_result: Path,
    *,
    no_warmup: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SERVING),
        "--backend",
        "openai-chat",
        "--base-url",
        args.base_url.rstrip("/"),
        "--endpoint",
        args.endpoint,
        "--model",
        args.model,
        "--dataset-name",
        "random_token_ids",
        "--seed",
        str(args.seed),
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.input_len),
        "--random-output-len",
        str(args.output_len),
        "--random-range-ratio",
        "0",
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.max_concurrency),
        "--ignore-eos",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl",
        "--metric-percentiles",
        "99",
        "--save-result",
        "--result-filename",
        str(raw_result),
    ]
    if args.tokenizer is not None:
        command.extend(["--tokenizer", args.tokenizer])
    if no_warmup:
        command.append("--no-warmup")
    return command


def _run_with_memory_sampling(command: list[str], gpu_index: int, interval: float) -> GPUMemoryMetrics:
    gpu_identity = _query_gpu_identity(gpu_index)
    baseline_mib = _query_gpu_memory_mib(gpu_index)
    memory_samples = [baseline_mib]
    benchmark_process = subprocess.Popen(command)
    try:
        while benchmark_process.poll() is None:
            memory_samples.append(_query_gpu_memory_mib(gpu_index))
            time.sleep(interval)
        memory_samples.append(_query_gpu_memory_mib(gpu_index))
        if benchmark_process.returncode != 0:
            raise subprocess.CalledProcessError(benchmark_process.returncode, command)
    finally:
        if benchmark_process.poll() is None:
            benchmark_process.terminate()
            try:
                benchmark_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                benchmark_process.kill()
                benchmark_process.wait()

    final_gpu_identity = _query_gpu_identity(gpu_index)
    if final_gpu_identity != gpu_identity:
        raise RuntimeError(f"GPU identity changed during benchmark: {gpu_identity} -> {final_gpu_identity}.")
    peak_mib = max(memory_samples)
    if peak_mib > gpu_identity.total_mib:
        raise RuntimeError(
            f"GPU {gpu_index} peak memory {peak_mib} MiB exceeds total memory {gpu_identity.total_mib} MiB."
        )
    return GPUMemoryMetrics(
        gpu_index=gpu_index,
        gpu_uuid=gpu_identity.gpu_uuid,
        gpu_name=gpu_identity.gpu_name,
        total_mib=gpu_identity.total_mib,
        baseline_mib=baseline_mib,
        peak_mib=peak_mib,
        peak_delta_mib=peak_mib - baseline_mib,
        samples=len(memory_samples),
        sample_interval_seconds=interval,
    )


def _validate_serving_result(result: dict[str, Any], args: argparse.Namespace) -> None:
    completed = _require_int(result["completed"], "benchmark result: completed", 0)
    if completed != args.num_prompts:
        raise RuntimeError(f"Expected {args.num_prompts} completed requests, got {completed}.")
    expected_input_tokens = args.num_prompts * args.input_len
    actual_input_tokens = _require_int(result["total_input_tokens"], "benchmark result: total_input_tokens", 0)
    if actual_input_tokens != expected_input_tokens:
        raise RuntimeError(f"Expected {expected_input_tokens} input tokens, got {actual_input_tokens}.")
    expected_output_tokens = args.num_prompts * args.output_len
    actual_output_tokens = _require_int(result["total_output_tokens"], "benchmark result: total_output_tokens", 0)
    if actual_output_tokens != expected_output_tokens:
        raise RuntimeError(f"Expected {expected_output_tokens} output tokens, got {actual_output_tokens}.")

    for metric_name in (
        "duration",
        "request_throughput",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
    ):
        _require_number(result[metric_name], f"benchmark result: {metric_name}", 0.0, inclusive=False)


def _compact_serving_result(result: dict[str, Any]) -> dict[str, Any]:
    return {field: result[field] for field in SERVING_RESULT_FIELDS}


def _verified_workload(
    args: argparse.Namespace,
    serving_result: dict[str, Any],
    benchmark_serving_sha256_before: str,
) -> tuple[dict[str, Any], str]:
    benchmark_serving_sha256_after = _sha256_file(BENCHMARK_SERVING)
    if benchmark_serving_sha256_after != benchmark_serving_sha256_before:
        raise RuntimeError(f"{BENCHMARK_SERVING} changed during the benchmark run.")
    prompt_token_ids_sha256 = _prompt_token_ids_sha256(serving_result, args)
    workload = _build_workload(
        args,
        prompt_token_ids_sha256=prompt_token_ids_sha256,
        benchmark_serving_sha256=benchmark_serving_sha256_before,
    )
    return workload, _workload_id(workload)


def _validate_trace_config(config: Any, name: str) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise TypeError(f"{name} must be a JSON object.")
    block_size = _require_int(config["block_size"], f"{name}.block_size", 1)
    kernel_size = _require_int(config["kernel_size"], f"{name}.kernel_size", 1)
    kernel_stride = _require_int(config["kernel_stride"], f"{name}.kernel_stride", 1)
    topk = _require_int(config["topk"], f"{name}.topk", 1)
    dense_len = _require_int(config["dense_len"], f"{name}.dense_len", 1)
    init_blocks = _require_int(config["init_blocks"], f"{name}.init_blocks", 0)
    local_blocks = _require_int(config["local_blocks"], f"{name}.local_blocks", 0)
    selected_capacity = _require_int(config["selected_capacity"], f"{name}.selected_capacity", 1)
    if block_size % kernel_stride != 0:
        raise ValueError(f"{name}.block_size must be divisible by kernel_stride.")
    if block_size % (4 * kernel_stride) != 0:
        raise ValueError(f"{name}.block_size must be divisible by 4 * kernel_stride.")
    if dense_len < 4 * kernel_size:
        raise ValueError(f"{name}.dense_len must be at least 4 * kernel_size.")
    if init_blocks >= topk:
        raise ValueError(f"{name}.init_blocks must be smaller than topk.")
    expected_capacity = max(
        topk + local_blocks,
        (dense_len + block_size - 1) // block_size,
    )
    if selected_capacity != expected_capacity:
        raise ValueError(f"{name}.selected_capacity must be {expected_capacity}, got {selected_capacity}.")
    query_head_group_size = _require_int(config["query_head_group_size"], f"{name}.query_head_group_size", 1)
    selected_count_values = config["selected_count_values"]
    kv_heads_observed = config["kv_heads_observed"]
    request_slots_observed = config["request_slots_observed"]
    for values, field_name, minimum in (
        (selected_count_values, "selected_count_values", 1),
        (kv_heads_observed, "kv_heads_observed", 0),
        (request_slots_observed, "request_slots_observed", 0),
    ):
        if not isinstance(values, list) or not values:
            raise ValueError(f"{name}.{field_name} must be a non-empty JSON array.")
        normalized_values = [_require_int(value, f"{name}.{field_name}[]", minimum) for value in values]
        if values != sorted(set(normalized_values)):
            raise ValueError(f"{name}.{field_name} must be strictly increasing and unique.")
    query_position_min = _require_int(config["query_position_min"], f"{name}.query_position_min", 0)
    query_position_max = _require_int(config["query_position_max"], f"{name}.query_position_max", 0)
    if query_position_min > query_position_max:
        raise ValueError(f"{name}.query_position_min must not exceed query_position_max.")
    if query_position_min + 1 < dense_len:
        raise ValueError(f"{name}.query_position_min must lie in the sparse region.")
    if any(value > selected_capacity for value in selected_count_values):
        raise ValueError(f"{name}.selected_count_values must not exceed selected_capacity.")
    return {
        "block_size": block_size,
        "kernel_size": kernel_size,
        "kernel_stride": kernel_stride,
        "topk": topk,
        "dense_len": dense_len,
        "init_blocks": init_blocks,
        "local_blocks": local_blocks,
        "selected_capacity": selected_capacity,
        "query_head_group_size": query_head_group_size,
        "selected_count_values": selected_count_values,
        "kv_heads_observed": kv_heads_observed,
        "request_slots_observed": request_slots_observed,
        "query_position_min": query_position_min,
        "query_position_max": query_position_max,
    }


def _load_selector_diagnostic(
    path: Path,
) -> tuple[dict[str, Any], str]:
    diagnostic, diagnostic_sha256, _ = _read_json_snapshot(path)
    if diagnostic["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}: selector diagnostic schema_version must be {SCHEMA_VERSION}.")
    if diagnostic["kind"] != "infllmv2_selector_diagnostic":
        raise ValueError(f"{path}: kind must be 'infllmv2_selector_diagnostic', got {diagnostic['kind']!r}.")
    workload = diagnostic["workload"]
    if not isinstance(workload, dict):
        raise TypeError(f"{path}: workload must be a JSON object.")
    _require_sha256(
        workload["prompt_token_ids_sha256"],
        f"{path}: workload.prompt_token_ids_sha256",
    )
    _require_sha256(
        workload["benchmark_serving_sha256"],
        f"{path}: workload.benchmark_serving_sha256",
    )
    workload_id = _workload_id(workload)
    if diagnostic["workload_id"] != workload_id:
        raise ValueError(f"{path}: workload_id does not match workload contents.")

    trace = diagnostic["trace"]
    if not isinstance(trace, dict):
        raise TypeError(f"{path}: trace must be a JSON object.")
    _require_sha256(trace["sha256"], f"{path}: trace.sha256")
    _require_int(trace["rank"], f"{path}: trace.rank", 0)
    _require_int(trace["layer"], f"{path}: trace.layer", 0)
    _validate_trace_config(trace["config"], f"{path}: trace.config")
    if trace["reference_definition"] != SELECTOR_REFERENCE_DEFINITION:
        raise ValueError(f"{path}: trace.reference_definition must be {SELECTOR_REFERENCE_DEFINITION!r}.")

    trace_window_start_ns = _require_int(diagnostic["trace_window_start_ns"], f"{path}: trace_window_start_ns", 0)
    trace_window_end_ns = _require_int(diagnostic["trace_window_end_ns"], f"{path}: trace_window_end_ns", 0)
    if trace_window_start_ns > trace_window_end_ns:
        raise ValueError(f"{path}: trace request window is inverted.")
    trace_mtime_ns = _require_int(trace["mtime_ns"], f"{path}: trace.mtime_ns", 0)
    if not trace_window_start_ns <= trace_mtime_ns <= trace_window_end_ns:
        raise ValueError(f"{path}: trace mtime falls outside the diagnostic request window.")
    trace_path = Path(trace["path"])
    source_selector, source_provenance, source_trace_sha256, source_trace_stat = _load_selector_trace(trace_path)
    if source_trace_sha256 != trace["sha256"]:
        raise ValueError(f"{path}: trace SHA256 no longer matches {trace_path}.")
    if source_trace_stat.st_mtime_ns != trace_mtime_ns:
        raise ValueError(f"{path}: trace mtime no longer matches {trace_path}.")
    if source_provenance != {
        "rank": trace["rank"],
        "layer": trace["layer"],
        "config": trace["config"],
        "reference_definition": trace["reference_definition"],
    }:
        raise ValueError(f"{path}: embedded trace provenance does not match {trace_path}.")

    selector = _validate_attached_selector(diagnostic["selector"])
    if selector != asdict(source_selector):
        raise ValueError(f"{path}: embedded selector metrics do not match {trace_path}.")
    if selector["selector_hits"] is None or selector["selector_targets"] is None:
        raise ValueError(f"{path}: diagnostic selector must retain hit and target counts.")
    if trace["sample_count"] != selector["selector_samples"]:
        raise ValueError(f"{path}: trace.sample_count must match selector.selector_samples.")
    if trace["metric_definition"] != SELECTOR_DEFINITION:
        raise ValueError(f"{path}: trace.metric_definition must be {SELECTOR_DEFINITION!r}.")
    return diagnostic, diagnostic_sha256


def run_benchmark(args: argparse.Namespace) -> None:
    _validate_run_args(args)
    selector_diagnostic = None
    selector_diagnostic_sha256 = None
    if args.variant == "sparse":
        selector_diagnostic, selector_diagnostic_sha256 = _load_selector_diagnostic(args.selector_diagnostic)

    benchmark_serving_sha256 = _sha256_file(BENCHMARK_SERVING)
    with tempfile.TemporaryDirectory(prefix="infllmv2-benchmark-", dir=args.output.parent) as temporary_dir:
        raw_result = Path(temporary_dir) / "serving.json"
        command = _benchmark_command(args, raw_result, no_warmup=False)
        print(f"Running with interpreter: {sys.executable}")
        gpu_memory = _run_with_memory_sampling(command, args.gpu_index, args.sample_interval)
        serving_result = _read_json(raw_result)

    _validate_serving_result(serving_result, args)
    workload, workload_id = _verified_workload(
        args,
        serving_result,
        benchmark_serving_sha256,
    )
    print(f"Workload ID: {workload_id}")

    selector = None
    selector_source = None
    if selector_diagnostic is not None:
        if selector_diagnostic["workload"] != workload:
            raise ValueError("Selector diagnostic and sparse timing run do not describe the same workload.")
        if selector_diagnostic["workload_id"] != workload_id:
            raise ValueError("Selector diagnostic and sparse timing workload IDs differ.")
        selector = selector_diagnostic["selector"]
        trace = selector_diagnostic["trace"]
        selector_source = {
            "path": str(args.selector_diagnostic.resolve()),
            "sha256": selector_diagnostic_sha256,
            "workload_id": selector_diagnostic["workload_id"],
            "trace_sha256": trace["sha256"],
            "rank": trace["rank"],
            "layer": trace["layer"],
            "config": trace["config"],
            "reference_definition": trace["reference_definition"],
        }

    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "infllmv2_serving_run",
        "variant": args.variant,
        "workload_id": workload_id,
        "workload": workload,
        "gpu_memory": asdict(gpu_memory),
        "selector": selector,
        "selector_source": selector_source,
        "serving": _compact_serving_result(serving_result),
    }
    _write_json(args.output, output, args.overwrite)
    print(f"Saved {args.variant} result to {args.output}")


def run_selector_diagnostic(args: argparse.Namespace) -> None:
    _validate_diagnostic_args(args)
    trace_path = args.trace_path.resolve()
    if trace_path.exists():
        raise FileExistsError(f"Selector trace output already exists immediately before the run: {trace_path}")

    benchmark_serving_sha256 = _sha256_file(BENCHMARK_SERVING)
    trace_window_start_ns = time.time_ns()
    with tempfile.TemporaryDirectory(prefix="infllmv2-diagnostic-", dir=args.output.parent) as temporary_dir:
        raw_result = Path(temporary_dir) / "serving.json"
        command = _benchmark_command(args, raw_result, no_warmup=True)
        print(f"Running selector diagnostic with interpreter: {sys.executable}")
        subprocess.run(command, check=True)
        trace_window_end_ns = time.time_ns()
        serving_result = _read_json(raw_result)

    _validate_serving_result(serving_result, args)
    workload, workload_id = _verified_workload(
        args,
        serving_result,
        benchmark_serving_sha256,
    )
    selector, trace_provenance, trace_sha256, trace_stat = _load_selector_trace(trace_path)
    if trace_stat.st_mtime_ns < trace_window_start_ns:
        raise RuntimeError(f"Selector trace predates the diagnostic request window: {trace_path}")
    if trace_stat.st_mtime_ns > trace_window_end_ns:
        raise RuntimeError(f"Selector trace was modified after the diagnostic request window: {trace_path}")

    trace = {
        "path": str(trace_path),
        "sha256": trace_sha256,
        "mtime_ns": trace_stat.st_mtime_ns,
        "sample_count": selector.selector_samples,
        "metric_definition": SELECTOR_DEFINITION,
        **trace_provenance,
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "infllmv2_selector_diagnostic",
        "workload_id": workload_id,
        "workload": workload,
        "trace_window_start_ns": trace_window_start_ns,
        "trace_window_end_ns": trace_window_end_ns,
        "trace": trace,
        "selector": asdict(selector),
        "serving": _compact_serving_result(serving_result),
    }
    _write_json(args.output, output, args.overwrite)
    print(f"Workload ID: {workload_id}")
    print(f"Selector block hit rate: {selector.selector_hit_rate:.2%} over {selector.selector_samples} samples")
    print(f"Saved selector diagnostic to {args.output}")


def _load_run(path: Path, expected_variant: str) -> dict[str, Any]:
    result = _read_json(path)
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version {result['schema_version']!r}.")
    if result["kind"] != "infllmv2_serving_run":
        raise ValueError(f"{path}: kind must be 'infllmv2_serving_run', got {result['kind']!r}.")
    if result["variant"] != expected_variant:
        raise ValueError(f"{path}: variant must be {expected_variant!r}, got {result['variant']!r}.")
    workload = result["workload"]
    if not isinstance(workload, dict):
        raise TypeError(f"{path}: workload must be a JSON object.")
    _require_sha256(
        workload["prompt_token_ids_sha256"],
        f"{path}: workload.prompt_token_ids_sha256",
    )
    _require_sha256(
        workload["benchmark_serving_sha256"],
        f"{path}: workload.benchmark_serving_sha256",
    )
    calculated_workload_id = _workload_id(workload)
    if result["workload_id"] != calculated_workload_id:
        raise ValueError(f"{path}: workload_id does not match the workload contents.")
    if not isinstance(result["serving"], dict):
        raise TypeError(f"{path}: serving must be a JSON object.")
    if not isinstance(result["gpu_memory"], dict):
        raise TypeError(f"{path}: gpu_memory must be a JSON object.")
    if expected_variant == "dense":
        if result["selector"] is not None or result["selector_source"] is not None:
            raise ValueError(f"{path}: dense runs must not contain selector data.")
    else:
        _validate_attached_selector(result["selector"])
        selector_source = result["selector_source"]
        if not isinstance(selector_source, dict):
            raise TypeError(f"{path}: sparse selector_source must be a JSON object.")
        if selector_source["workload_id"] != calculated_workload_id:
            raise ValueError(f"{path}: selector_source workload_id does not match the run.")
        _require_sha256(selector_source["sha256"], f"{path}: selector_source.sha256")
        _require_sha256(
            selector_source["trace_sha256"],
            f"{path}: selector_source.trace_sha256",
        )
        _require_int(selector_source["rank"], f"{path}: selector_source.rank", 0)
        _require_int(selector_source["layer"], f"{path}: selector_source.layer", 0)
        _validate_trace_config(selector_source["config"], f"{path}: selector_source.config")
        if selector_source["reference_definition"] != SELECTOR_REFERENCE_DEFINITION:
            raise ValueError(f"{path}: selector_source reference definition is invalid.")
        selector_diagnostic_path = Path(selector_source["path"])
        selector_diagnostic, selector_diagnostic_sha256 = _load_selector_diagnostic(selector_diagnostic_path)
        if selector_diagnostic_sha256 != selector_source["sha256"]:
            raise ValueError(f"{path}: selector_source SHA256 no longer matches {selector_diagnostic_path}.")
        if selector_diagnostic["workload"] != workload:
            raise ValueError(f"{path}: selector diagnostic workload does not match the run.")
        if selector_diagnostic["selector"] != result["selector"]:
            raise ValueError(f"{path}: selector diagnostic metrics do not match the run.")
        diagnostic_trace = selector_diagnostic["trace"]
        if {
            "trace_sha256": selector_source["trace_sha256"],
            "rank": selector_source["rank"],
            "layer": selector_source["layer"],
            "config": selector_source["config"],
            "reference_definition": selector_source["reference_definition"],
        } != {
            "trace_sha256": diagnostic_trace["sha256"],
            "rank": diagnostic_trace["rank"],
            "layer": diagnostic_trace["layer"],
            "config": diagnostic_trace["config"],
            "reference_definition": diagnostic_trace["reference_definition"],
        }:
            raise ValueError(f"{path}: selector_source provenance does not match the diagnostic.")
    return result


def _validate_gpu_memory(
    gpu_memory: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    gpu_index = _require_int(gpu_memory["gpu_index"], f"{name}: gpu_index", 0)
    gpu_uuid = gpu_memory["gpu_uuid"]
    gpu_name = gpu_memory["gpu_name"]
    if not isinstance(gpu_uuid, str) or not gpu_uuid:
        raise ValueError(f"{name}: gpu_uuid must be a non-empty string.")
    if not isinstance(gpu_name, str) or not gpu_name:
        raise ValueError(f"{name}: gpu_name must be a non-empty string.")
    total_mib = _require_int(gpu_memory["total_mib"], f"{name}: total_mib", 1)
    baseline_mib = _require_int(gpu_memory["baseline_mib"], f"{name}: baseline_mib", 0)
    peak_mib = _require_int(gpu_memory["peak_mib"], f"{name}: peak_mib", 1)
    peak_delta_mib = _require_int(gpu_memory["peak_delta_mib"], f"{name}: peak_delta_mib", 0)
    if baseline_mib > peak_mib:
        raise ValueError(f"{name}: baseline_mib must not exceed peak_mib.")
    if peak_mib > total_mib:
        raise ValueError(f"{name}: peak_mib must not exceed total_mib.")
    if peak_delta_mib != peak_mib - baseline_mib:
        raise ValueError(f"{name}: peak_delta_mib must equal peak_mib - baseline_mib.")
    _require_int(gpu_memory["samples"], f"{name}: samples", 1)
    _require_number(
        gpu_memory["sample_interval_seconds"],
        f"{name}: sample_interval_seconds",
        0.0,
        inclusive=False,
    )
    return {
        "gpu_index": gpu_index,
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu_name,
        "total_mib": total_mib,
        "baseline_mib": baseline_mib,
        "peak_mib": peak_mib,
        "peak_delta_mib": peak_delta_mib,
    }


def _extract_report_metrics(run: dict[str, Any], name: str) -> dict[str, float | int]:
    serving = run["serving"]
    gpu_memory = _validate_gpu_memory(run["gpu_memory"], name)
    return {
        "completed": _require_int(serving["completed"], f"{name}: completed", 1),
        "request_throughput_requests_per_second": _require_number(
            serving["request_throughput"], f"{name}: request_throughput", 0.0, inclusive=False
        ),
        "output_throughput_tokens_per_second": _require_number(
            serving["output_throughput"], f"{name}: output_throughput", 0.0, inclusive=False
        ),
        "total_throughput_tokens_per_second": _require_number(
            serving["total_token_throughput"], f"{name}: total_token_throughput", 0.0, inclusive=False
        ),
        "mean_ttft_ms": _require_number(serving["mean_ttft_ms"], f"{name}: mean_ttft_ms", 0.0, inclusive=False),
        "median_ttft_ms": _require_number(serving["median_ttft_ms"], f"{name}: median_ttft_ms", 0.0, inclusive=False),
        "p99_ttft_ms": _require_number(serving["p99_ttft_ms"], f"{name}: p99_ttft_ms", 0.0, inclusive=False),
        "mean_tpot_ms": _require_number(serving["mean_tpot_ms"], f"{name}: mean_tpot_ms", 0.0, inclusive=False),
        "median_tpot_ms": _require_number(serving["median_tpot_ms"], f"{name}: median_tpot_ms", 0.0, inclusive=False),
        "p99_tpot_ms": _require_number(serving["p99_tpot_ms"], f"{name}: p99_tpot_ms", 0.0, inclusive=False),
        "baseline_gpu_memory_mib": gpu_memory["baseline_mib"],
        "peak_gpu_memory_mib": gpu_memory["peak_mib"],
        "peak_delta_gpu_memory_mib": gpu_memory["peak_delta_mib"],
    }


def _ratio(sparse: float, dense: float) -> float:
    return sparse / dense


def _reduction_percent(sparse: float, dense: float) -> float:
    return (dense - sparse) / dense * 100.0


def _validate_attached_selector(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Sparse run must contain selector metrics.")
    hit_rate = _require_rate(value["selector_hit_rate"], "sparse selector_hit_rate")
    sample_count = _require_int(value["selector_samples"], "sparse selector_samples", 1)
    hits = value["selector_hits"]
    targets = value["selector_targets"]
    if (hits is None) != (targets is None):
        raise ValueError("sparse selector_hits and selector_targets must either both be integers or both be null.")
    if hits is not None:
        hits = _require_int(hits, "sparse selector_hits", 0)
        targets = _require_int(targets, "sparse selector_targets", 1)
        if targets < sample_count:
            raise ValueError(
                "sparse selector_targets must be at least selector_samples because "
                "every trace sample has a non-empty reference set."
            )
        if hits > targets:
            raise ValueError(f"sparse selector_hits ({hits}) must not exceed selector_targets ({targets}).")
        calculated_hit_rate = hits / targets
        if not math.isclose(hit_rate, calculated_hit_rate, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"sparse selector_hit_rate ({hit_rate}) does not match selector_hits / selector_targets "
                f"({calculated_hit_rate})."
            )
    definition = value["definition"]
    if definition != SELECTOR_DEFINITION:
        raise ValueError(f"Sparse selector definition must be {SELECTOR_DEFINITION!r}.")
    return {
        "selector_hit_rate": hit_rate,
        "selector_samples": sample_count,
        "selector_hits": hits,
        "selector_targets": targets,
        "definition": definition,
    }


def _print_report(report: dict[str, Any]) -> None:
    dense = report["metrics"]["dense"]
    sparse = report["metrics"]["sparse"]
    print(f"Workload ID: {report['workload_id']}")
    print("| Metric | Dense | Sparse | Sparse / Dense |")
    print("|---|---:|---:|---:|")
    rows = (
        (
            "Request throughput (req/s)",
            dense["request_throughput_requests_per_second"],
            sparse["request_throughput_requests_per_second"],
        ),
        (
            "Output throughput (tok/s)",
            dense["output_throughput_tokens_per_second"],
            sparse["output_throughput_tokens_per_second"],
        ),
        (
            "Total token throughput (tok/s)",
            dense["total_throughput_tokens_per_second"],
            sparse["total_throughput_tokens_per_second"],
        ),
        ("Mean TTFT (ms)", dense["mean_ttft_ms"], sparse["mean_ttft_ms"]),
        ("Median TTFT (ms)", dense["median_ttft_ms"], sparse["median_ttft_ms"]),
        ("P99 TTFT (ms)", dense["p99_ttft_ms"], sparse["p99_ttft_ms"]),
        ("Mean TPOT (ms)", dense["mean_tpot_ms"], sparse["mean_tpot_ms"]),
        ("Median TPOT (ms)", dense["median_tpot_ms"], sparse["median_tpot_ms"]),
        ("P99 TPOT (ms)", dense["p99_tpot_ms"], sparse["p99_tpot_ms"]),
        (
            "Baseline GPU memory (MiB)",
            dense["baseline_gpu_memory_mib"],
            sparse["baseline_gpu_memory_mib"],
        ),
        ("Peak GPU memory (MiB)", dense["peak_gpu_memory_mib"], sparse["peak_gpu_memory_mib"]),
        (
            "Peak delta GPU memory (MiB)",
            dense["peak_delta_gpu_memory_mib"],
            sparse["peak_delta_gpu_memory_mib"],
        ),
    )
    for label, dense_value, sparse_value in rows:
        ratio = "N/A" if dense_value == 0 else f"{_ratio(sparse_value, dense_value):.3f}x"
        print(f"| {label} | {dense_value:.3f} | {sparse_value:.3f} | {ratio} |")
    selector = report["selector"]
    print(
        f"Selector block hit rate: {selector['selector_hit_rate']:.2%} "
        f"over {selector['selector_samples']} request/query/KV-head samples"
    )


def create_report(args: argparse.Namespace) -> None:
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}; pass --overwrite to replace it.")
    dense_run = _load_run(args.dense_result, "dense")
    sparse_run = _load_run(args.sparse_result, "sparse")
    if dense_run["workload"] != sparse_run["workload"]:
        raise ValueError("Dense and sparse results do not describe the same workload.")
    if dense_run["workload_id"] != sparse_run["workload_id"]:
        raise ValueError("Dense and sparse workload IDs differ.")

    dense_gpu = _validate_gpu_memory(dense_run["gpu_memory"], "dense")
    sparse_gpu = _validate_gpu_memory(sparse_run["gpu_memory"], "sparse")
    if dense_gpu["gpu_uuid"] != sparse_gpu["gpu_uuid"]:
        raise ValueError(
            "Dense and sparse runs must use the same physical GPU UUID, got "
            f"{dense_gpu['gpu_uuid']!r} and {sparse_gpu['gpu_uuid']!r}."
        )
    if dense_gpu["gpu_name"] != sparse_gpu["gpu_name"] or dense_gpu["total_mib"] != sparse_gpu["total_mib"]:
        raise ValueError("Dense and sparse GPU identity metadata differs.")

    dense_metrics = _extract_report_metrics(dense_run, "dense")
    sparse_metrics = _extract_report_metrics(sparse_run, "sparse")
    selector = _validate_attached_selector(sparse_run["selector"])
    if selector["definition"] != SELECTOR_DEFINITION:
        raise ValueError(f"Sparse selector definition must be {SELECTOR_DEFINITION!r}.")

    comparison = {
        "output_throughput_speedup": _ratio(
            sparse_metrics["output_throughput_tokens_per_second"],
            dense_metrics["output_throughput_tokens_per_second"],
        ),
        "request_throughput_speedup": _ratio(
            sparse_metrics["request_throughput_requests_per_second"],
            dense_metrics["request_throughput_requests_per_second"],
        ),
        "total_token_throughput_speedup": _ratio(
            sparse_metrics["total_throughput_tokens_per_second"],
            dense_metrics["total_throughput_tokens_per_second"],
        ),
        "mean_ttft_reduction_percent": _reduction_percent(
            sparse_metrics["mean_ttft_ms"], dense_metrics["mean_ttft_ms"]
        ),
        "mean_tpot_reduction_percent": _reduction_percent(
            sparse_metrics["mean_tpot_ms"], dense_metrics["mean_tpot_ms"]
        ),
        "peak_gpu_memory_reduction_percent": _reduction_percent(
            sparse_metrics["peak_gpu_memory_mib"], dense_metrics["peak_gpu_memory_mib"]
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "infllmv2_dense_sparse_report",
        "workload_id": dense_run["workload_id"],
        "workload": dense_run["workload"],
        "gpu": {
            "uuid": dense_gpu["gpu_uuid"],
            "name": dense_gpu["gpu_name"],
            "total_mib": dense_gpu["total_mib"],
            "dense_index": dense_gpu["gpu_index"],
            "sparse_index": sparse_gpu["gpu_index"],
        },
        "metrics": {"dense": dense_metrics, "sparse": sparse_metrics},
        "comparison": comparison,
        "selector": selector,
        "selector_source": sparse_run["selector_source"],
    }
    _write_json(args.output, report, args.overwrite)
    _print_report(report)
    print(f"Saved comparison report to {args.output}")


def summarize_selector(args: argparse.Namespace) -> None:
    selector, provenance, trace_sha256, _ = _load_selector_trace(args.samples)
    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "infllmv2_unbound_selector_summary",
        "trace_sha256": trace_sha256,
        "trace": provenance,
        **asdict(selector),
    }
    _write_json(args.output, output, args.overwrite)
    print(f"Selector block hit rate: {selector.selector_hit_rate:.2%} over {selector.selector_samples} samples")
    print(f"Saved selector summary to {args.output}")


def _parse_operator_scenario(value: str) -> OperatorScenario:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("operator scenario must be CONTEXT_LENGTH:CONCURRENCY, for example 32768:1")
    try:
        context_length = int(parts[0])
        concurrency = int(parts[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("operator scenario context length and concurrency must be integers") from exc
    if context_length < 1:
        raise argparse.ArgumentTypeError("operator scenario context length must be positive")
    if concurrency < 1:
        raise argparse.ArgumentTypeError("operator scenario concurrency must be positive")
    return OperatorScenario(context_length=context_length, concurrency=concurrency)


def _percentile(sorted_values: list[float], quantile: float) -> float:
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _time_gpu_operator(paddle, operation, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    paddle.device.synchronize()
    starts = [paddle.device.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [paddle.device.cuda.Event(enable_timing=True) for _ in range(repeats)]
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    paddle.device.synchronize()
    wall_mean_us = (time.perf_counter() - wall_start) * 1e6 / repeats
    elapsed_us = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    return {
        "mean_us": statistics.fmean(elapsed_us),
        "median_us": statistics.median(elapsed_us),
        "p10_us": _percentile(elapsed_us, 0.10),
        "p90_us": _percentile(elapsed_us, 0.90),
        "p99_us": _percentile(elapsed_us, 0.99),
        "min_us": elapsed_us[0],
        "max_us": elapsed_us[-1],
        "wall_mean_us": wall_mean_us,
        "warmup": warmup,
        "repeats": repeats,
    }


def _run_operator_scenario(paddle, sparse_ops, dense_ops, args, scenario: OperatorScenario) -> dict[str, Any]:
    update_compressed_k, select_blocks, attention_forward = sparse_ops
    append_attention, get_block_shape_and_split_kv_block = dense_ops
    block_size = 64
    query_heads = 32
    kv_heads = 2
    head_dim = 128
    kernel_size = 32
    kernel_stride = 16
    topk = 64
    dense_len = 8192
    init_blocks = 1
    local_blocks = 32
    selected_capacity = max(
        topk + local_blocks,
        (dense_len + block_size - 1) // block_size,
    )
    max_blocks_per_seq = (scenario.context_length + block_size - 1) // block_size
    if max_blocks_per_seq > 2048:
        raise ValueError("InfLLM-V2 operator scenarios support at most 2048 blocks (131072 tokens at block_size=64).")
    physical_blocks = scenario.concurrency * max_blocks_per_seq
    kv_splits = (selected_capacity + STAGE2_BLOCKS_PER_SPLIT - 1) // STAGE2_BLOCKS_PER_SPLIT

    paddle.seed(args.seed)
    query = paddle.randn([scenario.concurrency, query_heads, head_dim], dtype="float32").astype(args.dtype)
    key_cache = paddle.zeros([physical_blocks, kv_heads, block_size, head_dim], dtype=args.dtype)
    value_cache = paddle.zeros_like(key_cache)
    compressed_k = paddle.zeros(
        [physical_blocks, kv_heads, block_size // kernel_stride, head_dim],
        dtype=args.dtype,
    )
    compressed_k2 = paddle.zeros(
        [
            physical_blocks,
            kv_heads,
            block_size // (4 * kernel_stride),
            head_dim,
        ],
        dtype=args.dtype,
    )
    block_tables = paddle.arange(physical_blocks, dtype="int32").reshape([scenario.concurrency, max_blocks_per_seq])
    seq_lens_decoder = paddle.full([scenario.concurrency], scenario.context_length - 1, dtype="int32")
    seq_lens_this_time = paddle.ones([scenario.concurrency], dtype="int32")
    batch_id_per_token = paddle.arange(scenario.concurrency, dtype="int32")
    cu_seqlens_q = paddle.arange(scenario.concurrency + 1, dtype="int32")
    topk_indices = paddle.empty([scenario.concurrency, kv_heads, selected_capacity], dtype="int32")
    block_scores = paddle.empty([scenario.concurrency, kv_heads, max_blocks_per_seq], dtype="float32")
    selected_counts = paddle.empty([scenario.concurrency, kv_heads], dtype="int32")
    coarse_lse = paddle.empty([scenario.concurrency, query_heads], dtype="float32")
    max_coarse_windows = max(
        0,
        (max_blocks_per_seq * block_size - 4 * kernel_size) // (4 * kernel_stride) + 1,
    )
    coarse_splits = max(1, (max_coarse_windows + 15) // 16)
    coarse_partial_max = paddle.empty([scenario.concurrency, query_heads, coarse_splits], dtype="float32")
    coarse_partial_sum = paddle.empty([scenario.concurrency, query_heads, coarse_splits], dtype="float32")
    attention_out = paddle.empty(query.shape, dtype=args.dtype)
    partial_acc = paddle.empty(
        [scenario.concurrency, query_heads, kv_splits, head_dim],
        dtype="float32",
    )
    partial_max = paddle.empty([scenario.concurrency, query_heads, kv_splits], dtype="float32")
    partial_sum = paddle.empty([scenario.concurrency, query_heads, kv_splits], dtype="float32")

    if args.dtype == "float32":
        raise ValueError("FastDeploy decode_unified_attention dense baseline does not support float32.")
    group_size = query_heads // kv_heads
    dense_max_lengths = paddle.zeros([6], dtype="int32").cpu()
    seq_lens_encoder = paddle.zeros([scenario.concurrency], dtype="int32")
    max_decoder_tiles = scenario.concurrency * (scenario.context_length * group_size + 15) // 16
    decoder_batch_ids = paddle.zeros([max_decoder_tiles], dtype="int32")
    decoder_tile_ids = paddle.zeros_like(decoder_batch_ids)
    decoder_num_blocks_cpu = paddle.zeros([1], dtype="int32").cpu()
    decoder_num_blocks_device = paddle.zeros([1], dtype="int32")
    decoder_chunk_size_device = paddle.zeros([1], dtype="int32")
    max_encoder_tiles = scenario.concurrency * (scenario.context_length * group_size + 63) // 64
    encoder_batch_ids = paddle.zeros([max_encoder_tiles], dtype="int32")
    encoder_tile_ids = paddle.zeros_like(encoder_batch_ids)
    encoder_num_blocks_cpu = paddle.zeros([1], dtype="int32").cpu()
    kv_batch_ids = paddle.zeros([max_encoder_tiles], dtype="int32")
    kv_tile_ids = paddle.zeros_like(kv_batch_ids)
    kv_num_blocks_cpu = paddle.zeros([1], dtype="int32").cpu()
    get_block_shape_and_split_kv_block(
        seq_lens_encoder,
        seq_lens_decoder,
        seq_lens_this_time,
        decoder_batch_ids,
        decoder_tile_ids,
        decoder_num_blocks_cpu,
        decoder_num_blocks_device,
        decoder_chunk_size_device,
        dense_max_lengths,
        encoder_batch_ids,
        encoder_tile_ids,
        encoder_num_blocks_cpu,
        kv_batch_ids,
        kv_tile_ids,
        kv_num_blocks_cpu,
        64,
        16,
        group_size,
        block_size,
    )
    current_kv = paddle.zeros([scenario.concurrency, 2 * kv_heads * head_dim], dtype=args.dtype)
    dense_qkv = paddle.concat(
        [query.reshape([scenario.concurrency, query_heads * head_dim]), current_kv],
        axis=1,
    )

    def update_op():
        return update_compressed_k(
            query,
            key_cache,
            compressed_k,
            compressed_k2,
            block_tables,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            kernel_size,
            kernel_stride,
        )

    def stage1_op():
        return select_blocks(
            query,
            compressed_k,
            compressed_k2,
            block_tables,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            topk_indices,
            block_scores,
            selected_counts,
            coarse_lse,
            coarse_partial_max,
            coarse_partial_sum,
            block_size,
            kernel_size,
            kernel_stride,
            topk,
            dense_len,
            init_blocks,
            local_blocks,
        )

    def stage2_op():
        return attention_forward(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            topk_indices,
            attention_out,
            partial_acc,
            partial_max,
            partial_sum,
        )

    def sparse_chain():
        update_op()
        stage1_op()
        stage2_op()

    def dense_attention_op():
        return append_attention(
            dense_qkv,
            key_cache,
            value_cache,
            seq_lens_encoder,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_id_per_token,
            cu_seqlens_q,
            block_tables,
            encoder_batch_ids,
            encoder_tile_ids,
            encoder_num_blocks_cpu,
            kv_batch_ids,
            kv_tile_ids,
            kv_num_blocks_cpu,
            decoder_batch_ids,
            decoder_tile_ids,
            decoder_num_blocks_cpu,
            dense_max_lengths,
            rotary_embs=None,
            attn_mask=None,
            qkv_bias=None,
            qkv_scale=None,
            k_quant_scale=None,
            v_quant_scale=None,
            k_dequant_scale=None,
            v_dequant_scale=None,
            cache_k_zp=None,
            cache_v_zp=None,
            linear_shift=None,
            linear_smooth=None,
            mask_offset=None,
            kv_signal_data=None,
            q_norm_weight=None,
            k_norm_weight=None,
            sinks=None,
            rms_norm_eps=1e-6,
            compute_type="bf16" if args.dtype == "bfloat16" else "fp16",
            cache_quant_type="none",
            use_neox_rotary_style=False,
            rope_3d=False,
            max_input_length=scenario.context_length,
            quant_max_bound=0.0,
            quant_min_bound=0.0,
            out_linear_in_scale=-1.0,
            encoder_block_shape_q=64,
            decoder_block_shape_q=16,
            # Match FlashAttentionBackend and the serving runs.  A single
            # 32K partition severely underfills the SMs and is not an
            # equivalent dense-decode baseline.
            max_partition_size=1024,
            encoder_max_partition_size=scenario.context_length,
            speculate_max_draft_token_num=1,
            causal=True,
            speculate_decoder=False,
            sliding_window=0,
            sink_size=0,
            head_wise_full_hidden=0,
            only_do_attn=True,
        )

    update_op()
    stage1_outputs = stage1_op()
    stage2_outputs = stage2_op()
    dense_attention_op()
    paddle.device.synchronize()
    if not isinstance(stage1_outputs, (tuple, list)) or len(stage1_outputs) != 6:
        raise RuntimeError("infllmv2_select_blocks must return six in-place workspace aliases.")
    expected_stage1_workspaces = (
        topk_indices,
        block_scores,
        selected_counts,
        coarse_lse,
        coarse_partial_max,
        coarse_partial_sum,
    )
    for output_index, (returned, workspace) in enumerate(zip(stage1_outputs, expected_stage1_workspaces)):
        if not returned._is_shared_buffer_with(workspace):
            raise RuntimeError(
                f"infllmv2_select_blocks output {output_index} does not alias its persistent input workspace."
            )
    if not isinstance(stage2_outputs, (tuple, list)) or len(stage2_outputs) != 4:
        raise RuntimeError("infllmv2_attention_forward must return four in-place workspace aliases.")
    expected_workspaces = (attention_out, partial_acc, partial_max, partial_sum)
    for output_index, (returned, workspace) in enumerate(zip(stage2_outputs, expected_workspaces)):
        if not returned._is_shared_buffer_with(workspace):
            raise RuntimeError(
                f"infllmv2_attention_forward output {output_index} does not alias its persistent input workspace."
            )

    count_values = selected_counts.numpy().reshape(-1).tolist()
    if len(set(count_values)) != 1:
        raise RuntimeError("Synthetic operator workload must select the same block count for every request/KV head.")
    selected_blocks = int(count_values[0])
    selected_indices = topk_indices.numpy()
    effective_stage2_tokens = 0
    for token_id in range(scenario.concurrency):
        visible_tokens = scenario.context_length
        for kv_head in range(kv_heads):
            for logical_block in selected_indices[token_id, kv_head]:
                if logical_block < 0:
                    break
                block_start = int(logical_block) * block_size
                effective_stage2_tokens += max(0, min(block_size, visible_tokens - block_start))
    visible_stage2_tokens = scenario.concurrency * kv_heads * scenario.context_length
    effective_sparse_kv_fraction = effective_stage2_tokens / visible_stage2_tokens
    preset_sparse_kv_fraction = min(1.0, (topk + local_blocks) * block_size / scenario.context_length)
    selected_tokens = effective_stage2_tokens // (scenario.concurrency * kv_heads)
    workspace_tensors = (
        topk_indices,
        block_scores,
        selected_counts,
        coarse_lse,
        coarse_partial_max,
        coarse_partial_sum,
        attention_out,
        partial_acc,
        partial_max,
        partial_sum,
    )
    stage2_workspace_tensors = (
        attention_out,
        partial_acc,
        partial_max,
        partial_sum,
    )
    workspace_bytes = sum(math.prod(tensor.shape) * tensor.element_size() for tensor in workspace_tensors)
    stage2_workspace_bytes = sum(
        math.prod(tensor.shape) * tensor.element_size() for tensor in stage2_workspace_tensors
    )
    return {
        "context_length": scenario.context_length,
        "concurrency": scenario.concurrency,
        "max_blocks_per_seq": max_blocks_per_seq,
        "selected_blocks": selected_blocks,
        "selected_tokens": selected_tokens,
        "effective_stage2_kv_tokens": effective_stage2_tokens,
        "visible_stage2_kv_tokens": visible_stage2_tokens,
        "effective_sparse_kv_fraction": effective_sparse_kv_fraction,
        "preset_sparse_kv_fraction": preset_sparse_kv_fraction,
        "effective_minus_preset_sparse_kv_fraction": (effective_sparse_kv_fraction - preset_sparse_kv_fraction),
        "configured_topk_only_kv_fraction": min(1.0, topk * block_size / scenario.context_length),
        "stage2_token_sparsity_ideal_speedup": 1.0 / effective_sparse_kv_fraction,
        "persistent_workspace_bytes": workspace_bytes,
        "persistent_stage2_workspace_bytes": stage2_workspace_bytes,
        "stage2_inplace_alias_verified": True,
        "operators": {
            "update_compressed_k": _time_gpu_operator(paddle, update_op, args.warmup, args.repeats),
            "stage1_select_blocks": _time_gpu_operator(paddle, stage1_op, args.warmup, args.repeats),
            "stage2_attention": _time_gpu_operator(paddle, stage2_op, args.warmup, args.repeats),
            "sparse_decode_chain": _time_gpu_operator(paddle, sparse_chain, args.warmup, args.repeats),
            "dense_attention": _time_gpu_operator(paddle, dense_attention_op, args.warmup, args.repeats),
        },
    }


def run_operator_benchmark(args: argparse.Namespace) -> None:
    _validate_output_args(args)
    _require_int(args.seed, "--seed", 0)
    _require_int(args.warmup, "--warmup", 1)
    _require_int(args.repeats, "--repeats", 1)
    _require_int(args.gpu_index, "--gpu-index", 0)
    if args.scenarios is None:
        args.scenarios = [
            OperatorScenario(context_length=32768, concurrency=1),
            OperatorScenario(context_length=131072, concurrency=4),
        ]
    scenario_keys = {(scenario.context_length, scenario.concurrency) for scenario in args.scenarios}
    if len(scenario_keys) != len(args.scenarios):
        raise ValueError("--scenario entries must be unique.")

    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("The InfLLM-V2 operator benchmark requires CUDA Paddle.")
    paddle.set_device(args.device)
    from fastdeploy.model_executor.layers.attention.ops import (
        append_attention,
        get_block_shape_and_split_kv_block,
    )
    from fastdeploy.model_executor.ops.gpu import (
        infllmv2_attention_forward,
        infllmv2_select_blocks,
        infllmv2_update_compressed_k,
    )

    gpu_identity = _query_gpu_identity(args.gpu_index)
    scenarios = []
    for scenario in args.scenarios:
        scenarios.append(
            _run_operator_scenario(
                paddle,
                (
                    infllmv2_update_compressed_k,
                    infllmv2_select_blocks,
                    infllmv2_attention_forward,
                ),
                (append_attention, get_block_shape_and_split_kv_block),
                args,
                scenario,
            )
        )
        paddle.device.cuda.empty_cache()

    output = {
        "schema_version": SCHEMA_VERSION,
        "kind": "infllmv2_operator_benchmark",
        "gpu": asdict(gpu_identity),
        "device": args.device,
        "dtype": args.dtype,
        "seed": args.seed,
        "model_shape": {
            "query_heads": 32,
            "kv_heads": 2,
            "gqa_group_size": 16,
            "head_dim": 128,
            "block_size": 64,
            "selected_capacity": 128,
            "blocks_per_stage2_split": STAGE2_BLOCKS_PER_SPLIT,
        },
        "source_sha256": {
            "launcher": _sha256_file(Path(__file__).parents[1] / "custom_ops/gpu_ops/infllmv2_attention/infllmv2.cu"),
            "kernels": _sha256_file(
                Path(__file__).parents[1] / "custom_ops/gpu_ops/infllmv2_attention/infllmv2_impl.cuh"
            ),
        },
        "loaded_fastdeploy_ops": _loaded_shared_object_metadata("fastdeploy_ops_pd_.so"),
        "scenarios": scenarios,
    }
    _write_json(args.output, output, args.overwrite)
    for scenario in scenarios:
        operators = scenario["operators"]
        print(
            f"{scenario['context_length']} tokens / concurrency "
            f"{scenario['concurrency']}: "
            f"update={operators['update_compressed_k']['median_us']:.3f} us, "
            f"stage1={operators['stage1_select_blocks']['median_us']:.3f} us, "
            f"stage2={operators['stage2_attention']['median_us']:.3f} us, "
            f"chain={operators['sparse_decode_chain']['median_us']:.3f} us, "
            f"dense={operators['dense_attention']['median_us']:.3f} us, "
            f"effective_sparse_kv_fraction={scenario['effective_sparse_kv_fraction']:.6f}"
        )
    print(f"Saved operator benchmark to {args.output}")


def _time_prefill_operator(paddle, operation, warmup: int, repeats: int) -> dict[str, float | int]:
    for _ in range(warmup):
        operation()
    paddle.device.synchronize()
    elapsed = []
    for _ in range(repeats):
        start = paddle.device.cuda.Event(enable_timing=True)
        end = paddle.device.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        elapsed.append(start.elapsed_time(end))
    elapsed.sort()
    return {
        "median_ms": statistics.median(elapsed),
        "mean_ms": statistics.fmean(elapsed),
        "min_ms": elapsed[0],
        "max_ms": elapsed[-1],
        "warmup": warmup,
        "repeats": repeats,
    }


def _make_prefill_backend(backend_type, fine, coarse):
    backend = backend_type.__new__(backend_type)
    backend.block_size = 64
    backend.num_heads = 32
    backend.kv_num_heads = 2
    backend.head_dim = 128
    backend.kernel_size = 32
    backend.kernel_stride = 16
    backend.topk = 64
    backend.local_blocks = 32
    backend.init_blocks = 1
    backend.dense_len = 8192
    backend.selected_capacity = 128
    backend.prefill_query_chunk_size = 4096
    backend._compressed_k = fine
    backend._compressed_k2 = coarse
    for name in (
        "_workspace_key",
        "_topk_indices_ws",
        "_block_scores_ws",
        "_selected_counts_ws",
        "_coarse_lse_ws",
        "_coarse_partial_max_ws",
        "_coarse_partial_sum_ws",
        "_attention_out_ws",
        "_partial_acc_ws",
        "_partial_max_ws",
        "_partial_sum_ws",
    ):
        setattr(backend, name, None)
    return backend


def _run_prefill_context(paddle, context_length: int, dtype: str, warmup: int, repeats: int):
    from fastdeploy.model_executor.layers.attention.flash_attn_backend import (
        flash_attn_func,
    )
    from fastdeploy.model_executor.layers.attention.infllmv2_attention_backend import (
        InfLLMV2AttentionBackend,
    )
    from fastdeploy.model_executor.ops.gpu import infllmv2_update_compressed_k

    block_size = 64
    query_heads = 32
    kv_heads = 2
    head_dim = 128
    dense_len = 8192
    if context_length <= dense_len:
        raise ValueError(f"context length must exceed dense_len={dense_len}")
    blocks = (context_length + block_size - 1) // block_size
    padded_length = blocks * block_size

    query = paddle.randn([context_length, query_heads, head_dim], dtype=dtype)
    key = paddle.randn([context_length, kv_heads, head_dim], dtype=dtype)
    value = paddle.randn(key.shape, dtype=dtype)
    padded_key = paddle.zeros([padded_length, kv_heads, head_dim], dtype=dtype)
    padded_value = paddle.zeros_like(padded_key)
    padded_key[:context_length] = key
    padded_value[:context_length] = value
    key_cache = paddle.transpose(
        padded_key.reshape([blocks, block_size, kv_heads, head_dim]), [0, 2, 1, 3]
    ).contiguous()
    value_cache = paddle.transpose(
        padded_value.reshape([blocks, block_size, kv_heads, head_dim]), [0, 2, 1, 3]
    ).contiguous()
    block_tables = paddle.arange(blocks, dtype="int32").reshape([1, blocks])
    fine = paddle.zeros([blocks, kv_heads, 4, head_dim], dtype=dtype)
    coarse = paddle.zeros([blocks, kv_heads, 1, head_dim], dtype=dtype)
    seq_lens_decoder = paddle.zeros([1], dtype="int32")
    seq_lens_this_time = paddle.to_tensor([context_length], dtype="int32")
    batch_ids = paddle.zeros([context_length], dtype="int32")
    cu_seqlens = paddle.to_tensor([0, context_length], dtype="int32")
    backend = _make_prefill_backend(InfLLMV2AttentionBackend, fine, coarse)

    def update_summaries():
        return infllmv2_update_compressed_k(
            query,
            key_cache,
            fine,
            coarse,
            block_tables,
            seq_lens_decoder,
            seq_lens_this_time,
            batch_ids,
            cu_seqlens,
            32,
            16,
        )

    def sparse_without_update():
        return backend._sparse_prefill_attention(
            query,
            key,
            value,
            key_cache,
            value_cache,
            block_tables,
        )

    def sparse_chain():
        update_summaries()
        return sparse_without_update()

    def dense_attention():
        return flash_attn_func(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens,
            context_length,
            context_length,
            causal=True,
            num_heads=query_heads,
            kv_num_heads=kv_heads,
            head_dim=head_dim,
            version=2,
        )[0]

    update_summaries()
    sparse_without_update()
    dense_attention()
    paddle.device.synchronize()
    operators = {
        "update_compressed_k": _time_prefill_operator(paddle, update_summaries, warmup, repeats),
        "sparse_without_update": _time_prefill_operator(paddle, sparse_without_update, warmup, repeats),
        "sparse_prefill_chain": _time_prefill_operator(paddle, sparse_chain, warmup, repeats),
        "dense_prefill": _time_prefill_operator(paddle, dense_attention, warmup, repeats),
    }
    dense_ms = operators["dense_prefill"]["median_ms"]
    sparse_ms = operators["sparse_prefill_chain"]["median_ms"]
    return {
        "context_length": context_length,
        "dense_len": dense_len,
        "query_tile_size": 128,
        "query_chunk_size": 4096,
        "selected_blocks": 96,
        "operators": operators,
        "dense_over_sparse_speedup": dense_ms / sparse_ms,
    }


def run_prefill_benchmark(args: argparse.Namespace) -> None:
    _validate_output_args(args)
    _require_int(args.seed, "--seed", 0)
    _require_int(args.warmup, "--warmup", 1)
    _require_int(args.repeats, "--repeats", 1)
    context_lengths = args.context_lengths or [16384, 32768]
    if len(set(context_lengths)) != len(context_lengths):
        raise ValueError("--context-length entries must be unique")

    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("CUDA Paddle is required")
    paddle.set_device(args.device)
    paddle.seed(args.seed)
    scenarios = [
        _run_prefill_context(paddle, length, args.dtype, args.warmup, args.repeats) for length in context_lengths
    ]
    root = Path(__file__).parents[1]
    properties = paddle.device.get_device_properties()
    output = {
        "kind": "infllmv2_sparse_prefill_operator_benchmark",
        "device": args.device,
        "gpu_name": properties.name,
        "dtype": args.dtype,
        "seed": args.seed,
        "source_sha256": {
            "backend": _sha256_file(root / "fastdeploy/model_executor/layers/attention/infllmv2_attention_backend.py"),
            "benchmark": _sha256_file(Path(__file__)),
        },
        "scenarios": scenarios,
    }
    _write_json(args.output, output, args.overwrite)
    for scenario in scenarios:
        print(
            f"{scenario['context_length']} tokens: "
            f"dense={scenario['operators']['dense_prefill']['median_ms']:.3f} ms, "
            f"sparse_chain={scenario['operators']['sparse_prefill_chain']['median_ms']:.3f} ms, "
            f"speedup={scenario['dense_over_sparse_speedup']:.3f}x"
        )
    print(f"Saved benchmark to {args.output}")


CUDA_IMPL_BLOCK_SIZE = 64
CUDA_IMPL_QUERY_HEADS = 32
CUDA_IMPL_KV_HEADS = 2
CUDA_IMPL_HEAD_DIM = 128
CUDA_IMPL_KERNEL_SIZE = 32
CUDA_IMPL_KERNEL_STRIDE = 16
CUDA_IMPL_TOPK = 64
CUDA_IMPL_LOCAL_BLOCKS = 32
CUDA_IMPL_INIT_BLOCKS = 1
CUDA_IMPL_SELECTED_BLOCKS = CUDA_IMPL_TOPK + CUDA_IMPL_LOCAL_BLOCKS


def _time_torch_cuda(torch, operation, warmup: int, repeats: int) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    wall_start = time.perf_counter()
    for start, end in zip(starts, ends):
        start.record()
        operation()
        end.record()
    torch.cuda.synchronize()
    wall_mean_us = (time.perf_counter() - wall_start) * 1e6 / repeats
    elapsed_us = sorted(start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends))
    return {
        "mean_us": statistics.fmean(elapsed_us),
        "median_us": statistics.median(elapsed_us),
        "p10_us": _percentile(elapsed_us, 0.10),
        "p90_us": _percentile(elapsed_us, 0.90),
        "p99_us": _percentile(elapsed_us, 0.99),
        "min_us": elapsed_us[0],
        "max_us": elapsed_us[-1],
        "wall_mean_us": wall_mean_us,
        "warmup": warmup,
        "repeats": repeats,
    }


def _torch_cumulative_lengths(torch, sequences: int, sequence_length: int, device) -> Any:
    return torch.arange(sequences + 1, dtype=torch.int32, device=device).mul_(sequence_length)


def _cuda_impl_selected_block_indices(torch, context_length: int, concurrency: int, device) -> Any:
    block_count = context_length // CUDA_IMPL_BLOCK_SIZE
    forced_local_start = block_count - (CUDA_IMPL_LOCAL_BLOCKS + 1)
    dynamic_count = CUDA_IMPL_SELECTED_BLOCKS - (CUDA_IMPL_LOCAL_BLOCKS + 1) - CUDA_IMPL_INIT_BLOCKS
    selected = torch.cat(
        (
            torch.arange(CUDA_IMPL_INIT_BLOCKS + dynamic_count, dtype=torch.int32, device=device),
            torch.arange(forced_local_start, block_count, dtype=torch.int32, device=device),
        )
    )
    if selected.numel() != CUDA_IMPL_SELECTED_BLOCKS or torch.unique(selected).numel() != CUDA_IMPL_SELECTED_BLOCKS:
        raise RuntimeError("Equivalent workload did not produce the required unique block budget.")
    return selected.reshape(1, 1, CUDA_IMPL_SELECTED_BLOCKS).expand(CUDA_IMPL_KV_HEADS, concurrency, -1).contiguous()


def _run_cuda_impl_scenario(
    torch,
    infllm_v2,
    context_length: int,
    concurrency: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    if context_length % CUDA_IMPL_BLOCK_SIZE != 0:
        raise ValueError("context length must be divisible by block size")
    if context_length // CUDA_IMPL_BLOCK_SIZE <= CUDA_IMPL_SELECTED_BLOCKS:
        raise ValueError("context length must contain more blocks than the sparse selection budget")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(2026)
    query = torch.randn(
        concurrency,
        CUDA_IMPL_QUERY_HEADS,
        CUDA_IMPL_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    key = torch.zeros(
        concurrency * context_length,
        CUDA_IMPL_KV_HEADS,
        CUDA_IMPL_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    value = torch.zeros_like(key)
    fine_windows = (context_length - CUDA_IMPL_KERNEL_SIZE) // CUDA_IMPL_KERNEL_STRIDE + 1
    coarse_windows = (context_length - 4 * CUDA_IMPL_KERNEL_SIZE) // (4 * CUDA_IMPL_KERNEL_STRIDE) + 1
    compressed_fine = torch.zeros(
        concurrency * fine_windows,
        CUDA_IMPL_KV_HEADS,
        CUDA_IMPL_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    compressed_coarse = torch.zeros(
        concurrency * coarse_windows,
        CUDA_IMPL_KV_HEADS,
        CUDA_IMPL_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    cu_query = _torch_cumulative_lengths(torch, concurrency, 1, device)
    cu_key = _torch_cumulative_lengths(torch, concurrency, context_length, device)
    cu_fine = _torch_cumulative_lengths(torch, concurrency, fine_windows, device)
    cu_coarse = _torch_cumulative_lengths(torch, concurrency, coarse_windows, device)
    cache_lengths = torch.full((concurrency,), context_length - 1, dtype=torch.int32, device=device)
    stage2_indices = _cuda_impl_selected_block_indices(torch, context_length, concurrency, device)

    def stage1_score():
        return infllm_v2.infllmv2_attn_stage1(
            query,
            compressed_fine,
            compressed_coarse,
            cu_seqlens_q=cu_query,
            cu_seqlens_k=cu_fine,
            cu_seqlens_v=cu_coarse,
            max_seqlen_q=1,
            max_seqlen_k=fine_windows,
            causal=False,
        )

    def stage1_with_topk():
        scores = stage1_score()
        pooled = infllm_v2.max_pooling_1d_varlen(
            scores,
            cu_query,
            cu_fine,
            cache_lengths,
            1,
            fine_windows,
            CUDA_IMPL_LOCAL_BLOCKS,
            CUDA_IMPL_INIT_BLOCKS,
            CUDA_IMPL_BLOCK_SIZE,
            CUDA_IMPL_KERNEL_STRIDE,
        )
        return pooled.topk(CUDA_IMPL_SELECTED_BLOCKS, dim=-1).indices.to(torch.int32).contiguous()

    def stage2_attention(indices=stage2_indices):
        return infllm_v2.infllmv2_attn_varlen_func(
            query,
            key,
            value,
            cu_query,
            cu_key,
            1,
            context_length,
            causal=False,
            topk_idx=indices,
        )

    def sparse_chain():
        return stage2_attention(stage1_with_topk())

    def dense_attention():
        return infllm_v2.infllmv2_attn_varlen_func(
            query,
            key,
            value,
            cu_query,
            cu_key,
            1,
            context_length,
            causal=False,
        )

    scores = stage1_score()
    if scores.shape != (CUDA_IMPL_KV_HEADS, concurrency, fine_windows):
        raise RuntimeError(f"Unexpected Stage 1 score shape: {tuple(scores.shape)}")
    selected = stage1_with_topk()
    if selected.shape != (
        CUDA_IMPL_KV_HEADS,
        concurrency,
        CUDA_IMPL_SELECTED_BLOCKS,
    ):
        raise RuntimeError(f"Unexpected Top-K shape: {tuple(selected.shape)}")
    stage2_output = stage2_attention()
    if stage2_output.shape != query.shape:
        raise RuntimeError(f"Unexpected Stage 2 output shape: {tuple(stage2_output.shape)}")
    torch.cuda.synchronize()

    effective_tokens = concurrency * CUDA_IMPL_KV_HEADS * CUDA_IMPL_SELECTED_BLOCKS * CUDA_IMPL_BLOCK_SIZE
    visible_tokens = concurrency * CUDA_IMPL_KV_HEADS * context_length
    return {
        "context_length": context_length,
        "concurrency": concurrency,
        "fine_windows": fine_windows,
        "coarse_windows": coarse_windows,
        "selected_blocks": CUDA_IMPL_SELECTED_BLOCKS,
        "effective_stage2_kv_tokens": effective_tokens,
        "visible_stage2_kv_tokens": visible_tokens,
        "effective_sparse_kv_fraction": effective_tokens / visible_tokens,
        "operators": {
            "stage1_score": _time_torch_cuda(torch, stage1_score, warmup, repeats),
            "stage1_with_topk": _time_torch_cuda(torch, stage1_with_topk, warmup, repeats),
            "stage2_attention": _time_torch_cuda(torch, stage2_attention, warmup, repeats),
            "sparse_decode_chain": _time_torch_cuda(torch, sparse_chain, warmup, repeats),
            "dense_attention": _time_torch_cuda(torch, dense_attention, warmup, repeats),
        },
    }


def _parse_cuda_impl_scenario(value: str) -> tuple[int, int]:
    fields = value.split(":")
    if len(fields) != 2:
        raise argparse.ArgumentTypeError("scenario must be CONTEXT_LENGTH:CONCURRENCY")
    try:
        context_length, concurrency = (int(field) for field in fields)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("scenario fields must be integers") from exc
    if context_length < 1 or concurrency < 1:
        raise argparse.ArgumentTypeError("scenario fields must be positive")
    return context_length, concurrency


def run_cuda_impl_benchmark(args: argparse.Namespace) -> None:
    _validate_output_args(args)
    _require_int(args.gpu_index, "--gpu-index", 0)
    _require_int(args.warmup, "--warmup", 1)
    _require_int(args.repeats, "--repeats", 1)
    repo = args.repo.resolve()
    if not (repo / "infllm_v2").is_dir():
        raise FileNotFoundError(f"Missing infllm_v2 package under {repo}")

    sys.path.insert(0, str(repo))
    import infllm_v2
    import torch
    from infllm_v2 import C

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    extension_path = Path(C.__file__).resolve()
    extension_stat = extension_path.stat()
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={args.gpu_index}",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    scenarios = args.cuda_impl_scenarios or [(32768, 1), (131072, 4)]
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("--scenario entries must be unique")
    results = [
        _run_cuda_impl_scenario(torch, infllm_v2, context, concurrency, args.warmup, args.repeats)
        for context, concurrency in scenarios
    ]
    output = {
        "kind": "infllmv2_cuda_impl_operator_benchmark",
        "repo": str(repo),
        "git_head": git_head,
        "gpu": gpu_query,
        "torch_version": torch.__version__,
        "dtype": "bfloat16",
        "model_shape": {
            "query_heads": CUDA_IMPL_QUERY_HEADS,
            "kv_heads": CUDA_IMPL_KV_HEADS,
            "gqa_group_size": CUDA_IMPL_QUERY_HEADS // CUDA_IMPL_KV_HEADS,
            "head_dim": CUDA_IMPL_HEAD_DIM,
            "block_size": CUDA_IMPL_BLOCK_SIZE,
            "selected_blocks": CUDA_IMPL_SELECTED_BLOCKS,
        },
        "loaded_extension": {
            "path": str(extension_path),
            "mtime_ns": extension_stat.st_mtime_ns,
            "mtime_utc": datetime.datetime.fromtimestamp(extension_stat.st_mtime, datetime.timezone.utc).isoformat(),
            "size_bytes": extension_stat.st_size,
            "sha256": _sha256_file(extension_path),
        },
        "scenarios": results,
    }
    _write_json(args.output, output, args.overwrite)
    for scenario in results:
        operators = scenario["operators"]
        print(
            f"{scenario['context_length']} tokens / concurrency {scenario['concurrency']}: "
            f"stage1+topk={operators['stage1_with_topk']['median_us']:.3f} us, "
            f"stage2={operators['stage2_attention']['median_us']:.3f} us, "
            f"chain={operators['sparse_decode_chain']['median_us']:.3f} us, "
            f"dense={operators['dense_attention']['median_us']:.3f} us"
        )
    print(f"Saved local cuda_impl benchmark to {args.output}")


def _add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="JSON output path; its parent directory must exist.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace --output if it already exists.")


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        required=True,
        help="OpenAI-compatible server base URL, e.g. http://127.0.0.1:8000.",
    )
    parser.add_argument("--endpoint", default="/v1/chat/completions")
    parser.add_argument("--model", required=True, help="Model name accepted by the server.")
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer name/path forwarded to benchmark_serving.py.",
    )
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--num-prompts", default=1, type=int)
    parser.add_argument("--input-len", required=True, type=int)
    parser.add_argument("--output-len", default=128, type=int)
    parser.add_argument("--request-rate", default=float("inf"), type=float)
    parser.add_argument("--max-concurrency", default=1, type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmark_serving.py against an existing OpenAI-compatible server and sample GPU memory.",
    )
    run_parser.add_argument("--variant", required=True, choices=("dense", "sparse"))
    _add_workload_arguments(run_parser)
    run_parser.add_argument("--gpu-index", required=True, type=int, help="GPU index sampled with nvidia-smi.")
    run_parser.add_argument(
        "--sample-interval",
        default=0.1,
        type=float,
        help="nvidia-smi polling interval in seconds.",
    )
    run_parser.add_argument(
        "--selector-diagnostic",
        type=Path,
        help=("Bound sparse-diagnostic JSON; required for sparse timing runs and rejected for dense runs."),
    )
    _add_output_arguments(run_parser)
    run_parser.set_defaults(handler=run_benchmark)

    diagnostic_parser = subparsers.add_parser(
        "sparse-diagnostic",
        help=(
            "Run an un-warmed sparse request workload and bind a newly generated selector trace to its exact prompts."
        ),
    )
    _add_workload_arguments(diagnostic_parser)
    diagnostic_parser.add_argument(
        "--trace-path",
        required=True,
        type=Path,
        help=(f"New selector trace path configured on the sparse server through {SELECTOR_TRACE_PATH_ENV}."),
    )
    _add_output_arguments(diagnostic_parser)
    diagnostic_parser.set_defaults(handler=run_selector_diagnostic)

    report_parser = subparsers.add_parser("report", help="Compare matching dense and sparse run JSON files.")
    report_parser.add_argument("--dense-result", required=True, type=Path)
    report_parser.add_argument("--sparse-result", required=True, type=Path)
    _add_output_arguments(report_parser)
    report_parser.set_defaults(handler=create_report)

    selector_parser = subparsers.add_parser(
        "selector",
        help="Inspect a backend selector trace without binding it to a serving workload.",
    )
    selector_parser.add_argument(
        "--samples",
        required=True,
        type=Path,
        help="Backend infllmv2_selector_samples trace JSON.",
    )
    _add_output_arguments(selector_parser)
    selector_parser.set_defaults(handler=summarize_selector)

    operator_parser = subparsers.add_parser(
        "operators",
        help=("Time InfLLM-V2 update, Stage 1, Stage 2, and their decode chain with CUDA events."),
    )
    operator_parser.add_argument(
        "--scenario",
        dest="scenarios",
        action="append",
        type=_parse_operator_scenario,
        help=("CONTEXT_LENGTH:CONCURRENCY; repeat for a matrix. Defaults to 32768:1 and 131072:4."),
    )
    operator_parser.add_argument("--device", default="gpu:0", help="Paddle CUDA device visible to this process.")
    operator_parser.add_argument(
        "--gpu-index",
        required=True,
        type=int,
        help="Physical GPU index reported through nvidia-smi.",
    )
    operator_parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
    )
    operator_parser.add_argument("--seed", default=2026, type=int)
    operator_parser.add_argument("--warmup", default=20, type=int)
    operator_parser.add_argument("--repeats", default=100, type=int)
    _add_output_arguments(operator_parser)
    operator_parser.set_defaults(handler=run_operator_benchmark)

    prefill_parser = subparsers.add_parser(
        "prefill",
        help="Compare complete dense and InfLLM-V2 sparse prefill paths with CUDA events.",
    )
    prefill_parser.add_argument(
        "--context-length",
        action="append",
        type=int,
        dest="context_lengths",
        help="Repeat for multiple lengths. Defaults to 16384 and 32768.",
    )
    prefill_parser.add_argument("--device", default="gpu:0")
    prefill_parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    prefill_parser.add_argument("--seed", type=int, default=2026)
    prefill_parser.add_argument("--warmup", type=int, default=10)
    prefill_parser.add_argument("--repeats", type=int, default=30)
    _add_output_arguments(prefill_parser)
    prefill_parser.set_defaults(handler=run_prefill_benchmark)

    cuda_impl_parser = subparsers.add_parser(
        "cuda-impl",
        help="Benchmark a local infllm_v2 PyTorch extension with an equivalent decode workload.",
    )
    cuda_impl_parser.add_argument("--repo", required=True, type=Path)
    cuda_impl_parser.add_argument("--gpu-index", required=True, type=int)
    cuda_impl_parser.add_argument(
        "--scenario",
        dest="cuda_impl_scenarios",
        action="append",
        type=_parse_cuda_impl_scenario,
        help="CONTEXT_LENGTH:CONCURRENCY; defaults to 32768:1 and 131072:4.",
    )
    cuda_impl_parser.add_argument("--warmup", default=20, type=int)
    cuda_impl_parser.add_argument("--repeats", default=100, type=int)
    _add_output_arguments(cuda_impl_parser)
    cuda_impl_parser.set_defaults(handler=run_cuda_impl_benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
