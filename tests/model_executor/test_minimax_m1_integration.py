"""
MiniMax-M1 integration tests.

Validate the full MiniMax-M1 inference pipeline: server startup,
health check, model listing, and generation quality.

Environment variables:
    MINIMAX_M1_MODEL_PATH      Path to model weights (default: MiniMax/MiniMax-M1)
    MINIMAX_M1_PORT            Server port (default: 8190)

Hardware:  Minimum 3× A800-80GB (WINT4), 6× for FP8, 12× for BF16
"""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import pytest

# ── Configuration ──
MODEL_PATH = os.environ.get("MINIMAX_M1_MODEL_PATH", "MiniMax/MiniMax-M1")
PORT = int(os.environ.get("MINIMAX_M1_PORT", "8190"))


def _gpu_count():
    """Return number of visible CUDA GPUs."""
    try:
        import paddle

        return paddle.device.cuda.device_count()
    except Exception:
        return 0


def _wait_for_server(port, timeout=900, interval=5):
    """Poll server health endpoint until ready or timeout."""
    import urllib.request

    elapsed = 0
    while elapsed < timeout:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        if elapsed > 0 and elapsed % 60 == 0:
            print(f"Waiting for server on port {port}... ({elapsed}/{timeout}s)")
        time.sleep(interval)
        elapsed += interval
    return False


def _send_chat(port, model, prompt, max_tokens=64):
    """Send a chat completion request and return parsed JSON."""
    import urllib.request

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


@pytest.fixture(scope="module")
def minimax_server():
    """Start a MiniMax-M1 server for the test module.

    Uses WINT4 quantization (smallest GPU footprint: ~228 GB, 3× A800).
    Yields (port, model_path) when ready, kills on teardown.
    """
    ngpus = _gpu_count()
    if ngpus < 3:
        pytest.skip(f"MiniMax-M1 needs ≥3 GPUs (WINT4). Found: {ngpus}")

    tp = min(ngpus, 8)  # cap at 8-way TP
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--tensor-parallel-size",
        str(tp),
        "--quantization",
        "wint4",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--port",
        str(PORT),
    ]

    log_file = tempfile.NamedTemporaryFile(mode="w", prefix="minimax_m1_server_", suffix=".log", delete=False)
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        if not _wait_for_server(PORT, timeout=900):
            # Dump last output for debugging
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=10)
            log_file.close()
            with open(log_file.name, errors="replace") as f:
                tail = f.read()[-2000:] or "(no output)"
            pytest.fail(f"MiniMax-M1 server did not start within 900s.\nLast output:\n{tail}")

        yield PORT, MODEL_PATH
    finally:
        log_file.close()
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=10)


class TestMiniMaxM1ModelLoad:
    """Tier 1: Model loads and server starts."""

    def test_server_health(self, minimax_server):
        """Server /health endpoint returns 200."""
        port, _ = minimax_server
        import urllib.request

        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200

    def test_model_listed(self, minimax_server):
        """Server /v1/models lists the MiniMax-M1 model."""
        port, model_path = minimax_server
        import urllib.request

        req = urllib.request.Request(f"http://localhost:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        model_ids = [m["id"] for m in data.get("data", [])]
        assert len(model_ids) > 0, f"No models served: {data}"


class TestMiniMaxM1Inference:
    """Tier 2: Inference produces coherent output."""

    def test_arithmetic(self, minimax_server):
        """Model can answer simple arithmetic correctly."""
        port, model_path = minimax_server
        resp = _send_chat(port, model_path, "What is 2+3? Answer with just the number.", 32)
        content = resp["choices"][0]["message"]["content"].strip()
        assert content, "Empty response"
        assert "5" in content, f"Expected '5' in response, got: {content!r}"

    def test_coherent_generation(self, minimax_server):
        """Model generates coherent multi-word output."""
        port, model_path = minimax_server
        resp = _send_chat(port, model_path, "Explain quantum entanglement in one sentence.", 64)
        content = resp["choices"][0]["message"]["content"].strip()
        assert len(content) >= 10, f"Response too short ({len(content)} chars): {content!r}"
        words = [w for w in content.split() if len(w) > 2]
        assert len(words) >= 3, f"Response lacks coherent words: {content!r}"

    def test_multi_turn(self, minimax_server):
        """Model handles multi-turn conversation."""
        port, model_path = minimax_server
        body = json.dumps(
            {
                "model": model_path,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 32,
                "temperature": 0.0,
            }
        ).encode()
        import urllib.request

        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        content = data["choices"][0]["message"]["content"].strip().lower()
        assert "alice" in content, f"Expected 'alice' in response, got: {content!r}"
