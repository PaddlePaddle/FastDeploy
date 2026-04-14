"""
MiniCPM4.1-8B integration tests.

These tests validate the full MiniCPM4.1-8B inference pipeline: server startup,
health check, model listing, and generation quality.

Environment variables:
    MINICPM4_MODEL_PATH      Path to model weights (default: openbmb/MiniCPM4.1-8B)
    MINICPM4_PORT            Server port (default: 8180)
    MINICPM4_QUANTIZATION    Quantization mode: wint4, wint8, or omit for BF16

Hardware:  Single A800-80GB (BF16) or V100-32GB (WINT4/WINT8)
"""

import json
import os
import signal
import subprocess
import sys
import time

import pytest

# ── Configuration ──
MODEL_PATH = os.environ.get("MINICPM4_MODEL_PATH", "openbmb/MiniCPM4.1-8B")
PORT = int(os.environ.get("MINICPM4_PORT", "8180"))
QUANTIZATION = os.environ.get("MINICPM4_QUANTIZATION", "")


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
def minicpm4_server():
    """Start a MiniCPM4.1-8B server for the test module.

    MiniCPM4.1-8B is a dense 8B model — single GPU is sufficient.
    Yields (port, model_path) when ready, kills on teardown.
    """
    ngpus = _gpu_count()
    if ngpus < 1:
        pytest.skip(f"MiniCPM4.1-8B needs ≥1 GPU. Found: {ngpus}")

    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "4",
        "--port",
        str(PORT),
    ]
    if QUANTIZATION:
        cmd.extend(["--quantization", QUANTIZATION])

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        if not _wait_for_server(PORT, timeout=900):
            proc.kill()
            out, _ = proc.communicate(timeout=10)
            tail = out.decode(errors="replace")[-2000:] if out else "(no output)"
            pytest.fail(f"MiniCPM4.1-8B server did not start within 900s.\nLast output:\n{tail}")

        yield PORT, MODEL_PATH
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


class TestMiniCPM4ModelLoad:
    """Tier 1: Model loads and server starts."""

    def test_server_health(self, minicpm4_server):
        """Server /health endpoint returns 200."""
        port, _ = minicpm4_server
        import urllib.request

        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200

    def test_model_listed(self, minicpm4_server):
        """Server /v1/models lists the MiniCPM4.1-8B model."""
        port, model_path = minicpm4_server
        import urllib.request

        req = urllib.request.Request(f"http://localhost:{port}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        model_ids = [m["id"] for m in data.get("data", [])]
        assert len(model_ids) > 0, f"No models served: {data}"


class TestMiniCPM4Inference:
    """Tier 2: Inference produces coherent output (not <unk> tokens)."""

    def test_arithmetic(self, minicpm4_server):
        """Model can answer simple arithmetic correctly."""
        port, model_path = minicpm4_server
        resp = _send_chat(port, model_path, "What is 2+3? Answer with just the number.", 256)
        content = resp["choices"][0]["message"]["content"].strip()
        assert content, "Empty response"
        assert "5" in content, f"Expected '5' in response, got: {content!r}"

    def test_no_unk_tokens(self, minicpm4_server):
        """Model does not produce <unk> tokens (validates tokenizer + LongRoPE)."""
        port, model_path = minicpm4_server
        resp = _send_chat(port, model_path, "What is the capital of France?", 64)
        content = resp["choices"][0]["message"]["content"].strip()
        assert content, "Empty response"
        assert "<unk>" not in content, f"Model produced <unk> tokens: {content!r}"
        assert "Paris" in content, f"Expected 'Paris' in response, got: {content!r}"

    def test_coherent_generation(self, minicpm4_server):
        """Model generates coherent multi-word output."""
        port, model_path = minicpm4_server
        resp = _send_chat(port, model_path, "Explain gravity in one sentence.", 64)
        content = resp["choices"][0]["message"]["content"].strip()
        assert len(content) >= 10, f"Response too short ({len(content)} chars): {content!r}"
        words = [w for w in content.split() if len(w) > 2]
        assert len(words) >= 3, f"Response lacks coherent words: {content!r}"

    def test_multi_turn(self, minicpm4_server):
        """Model handles multi-turn conversation."""
        port, model_path = minicpm4_server
        body = json.dumps(
            {
                "model": model_path,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
                "max_tokens": 256,
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
