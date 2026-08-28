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

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_URL_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


@dataclass(frozen=True)
class MiniCPM41E2EConfig:
    model_path: str
    request_model: str
    port: int
    metrics_port: int
    queue_port: int
    cache_queue_port: int
    max_model_len: str
    max_num_seqs: str
    tensor_parallel_size: str
    startup_timeout: int
    log_path: Path
    attention_backend: str
    quantization: str | None
    extra_args: tuple[str, ...]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _make_config(log_path: Path) -> MiniCPM41E2EConfig:
    return MiniCPM41E2EConfig(
        model_path=os.getenv("MODEL_PATH") or os.getenv("MINICPM41_MODEL_PATH", "openbmb/MiniCPM4.1-8B"),
        request_model=os.getenv("MINICPM41_E2E_REQUEST_MODEL", "default"),
        port=_env_int("FD_API_PORT", 8188),
        metrics_port=_env_int("FD_METRICS_PORT", 8233),
        queue_port=_env_int("FD_ENGINE_QUEUE_PORT", 8133),
        cache_queue_port=_env_int("FD_CACHE_QUEUE_PORT", 8333),
        max_model_len=os.getenv("MINICPM41_E2E_MAX_MODEL_LEN", "4096"),
        max_num_seqs=os.getenv("MINICPM41_E2E_MAX_NUM_SEQS", "2"),
        tensor_parallel_size=os.getenv("MINICPM41_E2E_TP", "1"),
        startup_timeout=_env_int("MINICPM41_E2E_TIMEOUT", 300),
        log_path=log_path,
        attention_backend=os.getenv("FD_ATTENTION_BACKEND", "FLASH_ATTN"),
        quantization=os.getenv("MINICPM41_E2E_QUANTIZATION") or None,
        extra_args=tuple(shlex.split(os.getenv("MINICPM41_E2E_EXTRA_ARGS", ""))),
    )


def _use_available_default_ports(config: MiniCPM41E2EConfig) -> MiniCPM41E2EConfig:
    """Avoid collisions while preserving explicitly configured E2E ports."""
    port_env_by_field = {
        "port": "FD_API_PORT",
        "metrics_port": "FD_METRICS_PORT",
        "queue_port": "FD_ENGINE_QUEUE_PORT",
        "cache_queue_port": "FD_CACHE_QUEUE_PORT",
    }
    updates: dict[str, int] = {}
    reserved_ports = {
        getattr(config, field) for field, env_name in port_env_by_field.items() if env_name in os.environ
    }
    for field, env_name in port_env_by_field.items():
        if env_name in os.environ:
            continue
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                candidate = sock.getsockname()[1]
            if candidate not in reserved_ports:
                reserved_ports.add(candidate)
                updates[field] = candidate
                break
    return replace(config, **updates)


def _build_server_command(config: MiniCPM41E2EConfig) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        config.model_path,
        "--port",
        str(config.port),
        "--metrics-port",
        str(config.metrics_port),
        "--engine-worker-queue-port",
        str(config.queue_port),
        "--cache-queue-port",
        str(config.cache_queue_port),
        "--tensor-parallel-size",
        config.tensor_parallel_size,
        "--max-model-len",
        config.max_model_len,
        "--max-num-seqs",
        config.max_num_seqs,
    ]
    if config.quantization:
        cmd.extend(["--quantization", config.quantization])
    cmd.extend(config.extra_args)
    if config.attention_backend == "INFLLMV2_ATTN":
        # InfLLM-V2 rejects both CUDA Graph and prefix caching. Keep the E2E
        # command runnable when callers select that backend (as go_run_val.sh
        # does) instead of relying on two unrelated defaults.
        if "--graph-optimization-config" not in config.extra_args:
            cmd.extend(["--graph-optimization-config", '{"use_cudagraph": false}'])
        if "--no-enable-prefix-caching" not in config.extra_args:
            cmd.append("--no-enable-prefix-caching")
    return cmd


def test_make_config_reads_quantization_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/models/minicpm41")
    monkeypatch.setenv("MINICPM41_E2E_QUANTIZATION", "wint4")
    monkeypatch.setenv("MINICPM41_E2E_EXTRA_ARGS", "--served-model-name minicpm41")
    monkeypatch.delenv("FD_ATTENTION_BACKEND", raising=False)

    config = _make_config(tmp_path / "server.log")

    assert config.model_path == "/models/minicpm41"
    assert config.attention_backend == "FLASH_ATTN"
    assert config.quantization == "wint4"
    assert config.extra_args == ("--served-model-name", "minicpm41")


def test_auto_selected_ports_preserve_explicit_configuration(tmp_path, monkeypatch):
    monkeypatch.setenv("FD_API_PORT", "18188")
    monkeypatch.delenv("FD_METRICS_PORT", raising=False)
    monkeypatch.delenv("FD_ENGINE_QUEUE_PORT", raising=False)
    monkeypatch.delenv("FD_CACHE_QUEUE_PORT", raising=False)

    config = _use_available_default_ports(_make_config(tmp_path / "server.log"))

    assert config.port == 18188
    assert len({config.port, config.metrics_port, config.queue_port, config.cache_queue_port}) == 4


@pytest.mark.parametrize("quantization", ["wint4", "wint8"])
def test_build_server_command_includes_quantization(tmp_path, quantization):
    config = MiniCPM41E2EConfig(
        model_path="/models/minicpm41",
        request_model="default",
        port=8188,
        metrics_port=8233,
        queue_port=8133,
        cache_queue_port=8333,
        max_model_len="4096",
        max_num_seqs="1",
        tensor_parallel_size="1",
        startup_timeout=300,
        log_path=tmp_path / "server.log",
        attention_backend="FLASH_ATTN",
        quantization=quantization,
        extra_args=("--served-model-name", "minicpm41"),
    )

    cmd = _build_server_command(config)

    assert cmd[cmd.index("--quantization") + 1] == quantization
    assert cmd[-2:] == ["--served-model-name", "minicpm41"]


def test_build_server_command_makes_infllmv2_configuration_compatible(tmp_path):
    config = MiniCPM41E2EConfig(
        model_path="/models/minicpm41",
        request_model="default",
        port=8188,
        metrics_port=8233,
        queue_port=8133,
        cache_queue_port=8333,
        max_model_len="4096",
        max_num_seqs="2",
        tensor_parallel_size="1",
        startup_timeout=300,
        log_path=tmp_path / "server.log",
        attention_backend="INFLLMV2_ATTN",
        quantization=None,
        extra_args=(),
    )

    cmd = _build_server_command(config)

    assert cmd[cmd.index("--graph-optimization-config") + 1] == '{"use_cudagraph": false}'
    assert "--no-enable-prefix-caching" in cmd


def _read_log(log_path: Path, max_chars: int = 20000) -> str:
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    return content[-max_chars:]


def _http_get(url: str, timeout: int = 2) -> tuple[int, str]:
    with LOCAL_URL_OPENER.open(url, timeout=timeout) as response:
        return response.status, response.read().decode("utf-8")


def _http_post_json(url: str, payload: dict, timeout: int = 60) -> tuple[int, dict]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with LOCAL_URL_OPENER.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = {"error": body}
        return exc.code, payload


def _thinking_token_sequences(model_path: str) -> tuple[list[int], list[int]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=False)

    prefix_ids = encode("x")
    contextual_start = encode("x<think>")
    contextual_end = encode("x\n</think>\n")
    assert contextual_start[: len(prefix_ids)] == prefix_ids
    assert contextual_end[: len(prefix_ids)] == prefix_ids
    return contextual_start[len(prefix_ids) :], contextual_end[len(prefix_ids) :]


def _wait_until_healthy(process: subprocess.Popen, config: MiniCPM41E2EConfig) -> None:
    deadline = time.time() + config.startup_timeout
    health_url = f"{config.base_url}/health"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "MiniCPM4.1 server exited before health check passed.\n"
                f"Command: {' '.join(_build_server_command(config))}\n"
                f"Log tail:\n{_read_log(config.log_path)}"
            )
        try:
            status, _ = _http_get(health_url)
            if status == 200:
                return
        except (TimeoutError, urllib.error.URLError):
            time.sleep(1)
    raise RuntimeError(
        "MiniCPM4.1 server did not become healthy before timeout.\n"
        f"Command: {' '.join(_build_server_command(config))}\n"
        f"Log tail:\n{_read_log(config.log_path)}"
    )


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


@pytest.fixture(scope="session")
def minicpm41_server(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("minicpm41_e2e") / "server.log"
    config = _use_available_default_ports(_make_config(log_path))
    env = os.environ.copy()
    env["FD_ATTENTION_BACKEND"] = config.attention_backend
    env.setdefault("FD_MODEL_SOURCE", "HUGGINGFACE")
    # The engine launches worker_process.py as a script. Add the repository to
    # PYTHONPATH so source-tree E2E runs do not depend on an editable install.
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) if not pythonpath else f"{REPO_ROOT}{os.pathsep}{pythonpath}"

    cmd = _build_server_command(config)
    with config.log_path.open("w", encoding="utf-8") as logfile:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=logfile,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )

    try:
        _wait_until_healthy(process, config)
        yield config
    finally:
        _terminate_process(process)


def _assert_chat_response(payload: dict) -> None:
    assert isinstance(payload.get("id"), str)
    assert payload.get("object") == "chat.completion"
    assert isinstance(payload.get("choices"), list)
    assert payload["choices"]
    choice = payload["choices"][0]
    assert choice["index"] == 0
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert choice["message"]["content"]
    assert choice.get("finish_reason") in {"stop", "length"}
    if "usage" in payload:
        assert payload["usage"]["prompt_tokens"] > 0
        assert payload["usage"]["completion_tokens"] > 0


def _assert_non_thinking_response(payload: dict) -> None:
    _assert_chat_response(payload)
    message = payload["choices"][0]["message"]
    assert "<think>" not in message["content"]
    assert "</think>" not in message["content"]
    assert message.get("reasoning_content") in (None, "")


def _assert_forced_end_after_decode_budget(payload: dict, budget: int, forced_end_ids: list[int]) -> None:
    """Check the request-prefix/decode-budget contract used by this E2E test.

    Request-side ``completion_token_ids`` are appended to ``prompt_token_ids``.
    They can place the state machine inside ``<think>``, but prompt-side
    reasoning tokens do not consume ``reasoning_max_tokens``. The budget counts
    newly decoded reasoning tokens, so the forced ``\n</think>\n`` sequence must
    begin at ``budget`` in the response-side completion token ids.
    """
    generated_ids = payload["choices"][0]["message"]["completion_token_ids"]
    forced_end_slice = generated_ids[budget : budget + len(forced_end_ids)]
    assert forced_end_slice == forced_end_ids, generated_ids


def _max_tokens_with_forced_end(budget: int, forced_end_ids: list[int]) -> int:
    """Leave one slot for the serving layer's terminal EOS token."""
    return budget + len(forced_end_ids) + 1


def test_minicpm41_openai_chat_completion_e2e(minicpm41_server):
    status, payload = _http_post_json(
        minicpm41_server.chat_url,
        {
            "model": minicpm41_server.request_model,
            "messages": [
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Say hello in one short sentence."},
            ],
            "temperature": 0,
            "max_tokens": 16,
        },
    )

    assert status == 200, payload
    _assert_chat_response(payload)


def test_minicpm41_chat_completion_with_history_e2e(minicpm41_server):
    status, payload = _http_post_json(
        minicpm41_server.chat_url,
        {
            "model": minicpm41_server.request_model,
            "messages": [
                {"role": "user", "content": "Remember this marker: FD_MINICPM41_E2E."},
                {"role": "assistant", "content": "I will remember FD_MINICPM41_E2E."},
                {"role": "user", "content": "Reply with the marker only."},
            ],
            "temperature": 0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    assert status == 200, payload
    _assert_chat_response(payload)


def test_minicpm41_forces_multitoken_thinking_end_e2e(minicpm41_server):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(minicpm41_server.model_path, trust_remote_code=True)
    think_start_ids, forced_end_ids = _thinking_token_sequences(minicpm41_server.model_path)
    thinking_payload_ids = tokenizer.encode("careful reasoning needs several steps", add_special_tokens=False)[:3]
    assert len(thinking_payload_ids) == 3
    reasoning_budget = len(thinking_payload_ids)

    status, payload = _http_post_json(
        minicpm41_server.chat_url,
        {
            "model": minicpm41_server.request_model,
            "messages": [{"role": "user", "content": "What is 17 plus 25?"}],
            "completion_token_ids": think_start_ids + thinking_payload_ids,
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_max_tokens": reasoning_budget,
            "return_token_ids": True,
            "temperature": 0,
            "max_tokens": _max_tokens_with_forced_end(reasoning_budget, forced_end_ids),
        },
    )

    assert status == 200, payload
    _assert_forced_end_after_decode_budget(payload, reasoning_budget, forced_end_ids)


def test_minicpm41_disable_thinking_does_not_emit_think_block_e2e(minicpm41_server):
    status, payload = _http_post_json(
        minicpm41_server.chat_url,
        {
            "model": minicpm41_server.request_model,
            "messages": [{"role": "user", "content": "Reply with the number 42 only."}],
            "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0,
            "max_tokens": 16,
        },
    )

    assert status == 200, payload
    _assert_non_thinking_response(payload)


def test_minicpm41_mixed_thinking_modes_e2e(minicpm41_server):
    """Submit enabled and disabled thinking requests concurrently.

    ``max_num_seqs >= 2`` lets the scheduler place both requests in the same
    active batch. The token-level mixed-slot behavior is asserted
    deterministically in tests/model_executor/test_minicpm41.py.
    """
    if int(minicpm41_server.max_num_seqs) < 2:
        pytest.skip("mixed thinking E2E requires MINICPM41_E2E_MAX_NUM_SEQS >= 2")

    think_start_ids, forced_end_ids = _thinking_token_sequences(minicpm41_server.model_path)
    thinking_budget = 1
    thinking_payload = {
        "model": minicpm41_server.request_model,
        "messages": [{"role": "user", "content": "What is 17 plus 25?"}],
        "completion_token_ids": think_start_ids,
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_max_tokens": thinking_budget,
        "return_token_ids": True,
        "temperature": 0,
        "max_tokens": _max_tokens_with_forced_end(thinking_budget, forced_end_ids),
    }
    non_thinking_payload = {
        "model": minicpm41_server.request_model,
        "messages": [{"role": "user", "content": "Reply with the number 42 only."}],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0,
        "max_tokens": 16,
    }

    with ThreadPoolExecutor(max_workers=2) as executor:
        thinking_future = executor.submit(_http_post_json, minicpm41_server.chat_url, thinking_payload)
        non_thinking_future = executor.submit(_http_post_json, minicpm41_server.chat_url, non_thinking_payload)
        thinking_status, thinking_response = thinking_future.result()
        non_thinking_status, non_thinking_response = non_thinking_future.result()

    assert thinking_status == 200, thinking_response
    assert non_thinking_status == 200, non_thinking_response
    _assert_forced_end_after_decode_budget(thinking_response, thinking_budget, forced_end_ids)
    _assert_non_thinking_response(non_thinking_response)


@pytest.mark.skip(reason="Skipping because the long-context MiniCPM4.1 E2E test requires target GPU memory.")
def test_minicpm41_long_context_chat_completion_e2e(minicpm41_server):
    prompt_chars = _env_int("MINICPM41_LONG_PROMPT_CHARS", 20000)
    long_context = ("0123456789abcdef " * ((prompt_chars // 17) + 1))[:prompt_chars]
    status, payload = _http_post_json(
        minicpm41_server.chat_url,
        {
            "model": minicpm41_server.request_model,
            "messages": [
                {
                    "role": "user",
                    "content": f"{long_context}\n\nAnswer with one word: ready.",
                }
            ],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=120,
    )

    assert status == 200, payload
    _assert_chat_response(payload)
