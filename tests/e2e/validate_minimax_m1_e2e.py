#!/usr/bin/env python3
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
Standalone MiniMax-M1 end-to-end validation for AI Studio.

Run on AI Studio A800 via SSH:
    ssh aistudio 'python3 /home/aistudio/validate_minimax_m1.py 2>&1 | tee /home/aistudio/output/minimax_m1_e2e.log'

This script:
1. Starts FastDeploy API server with MiniMax-M1 (WINT4/WINT8)
2. Waits for server readiness
3. Runs 6 validation checks (health, models, chat, reasoning, Chinese, multi-turn)
4. Prints structured evidence for PR body
5. Cleans up server process

Requirements:
- AI Studio A800 (80GB) or multiple GPUs for full 456B model
- FastDeploy installed with Triton support
- Model weights downloaded to MODEL_PATH

Environment variables:
    MINIMAX_MODEL_PATH  Path to MiniMax-M1 weights (default: MiniMax/MiniMax-M1-80k)
    MINIMAX_PORT        Server port (default: 8189)
    MINIMAX_QUANT       Quantization type: wint4, wint8, or none (default: wint4)
    MINIMAX_TP          Tensor parallel degree (default: 1)
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

MODEL_PATH = os.environ.get("MINIMAX_MODEL_PATH", "MiniMax/MiniMax-M1-80k")
PORT = int(os.environ.get("MINIMAX_PORT", "8189"))
QUANTIZATION = os.environ.get("MINIMAX_QUANT", "wint4")
TP_DEGREE = int(os.environ.get("MINIMAX_TP", "1"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_for_server(port, timeout=900):
    """Poll server health until ready."""
    log(f"Waiting for server on port {port} (timeout={timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"http://localhost:{port}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    log(f"Server ready in {elapsed:.1f}s")
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def send_chat(prompt, max_tokens=128, temperature=0.0, messages=None):
    """Send a chat completion request."""
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    body = json.dumps(
        {
            "model": MODEL_PATH,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    ).encode()
    req = urllib.request.Request(
        f"http://localhost:{PORT}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def run_validations():
    """Run all validation checks. Returns (passed, failed, results)."""
    results = []
    passed = 0
    failed = 0

    # Test 1: Health endpoint
    log("Test 1/6: Health endpoint")
    try:
        req = urllib.request.Request(f"http://localhost:{PORT}/health")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 200
        results.append(("health", "PASS", "HTTP 200"))
        passed += 1
    except Exception as e:
        results.append(("health", "FAIL", str(e)))
        failed += 1

    # Test 2: Model listing
    log("Test 2/6: Model listing")
    try:
        req = urllib.request.Request(f"http://localhost:{PORT}/v1/models")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        model_ids = [m["id"] for m in data.get("data", [])]
        assert len(model_ids) > 0, f"No models listed: {data}"
        results.append(("models", "PASS", f"Models: {model_ids}"))
        passed += 1
    except Exception as e:
        results.append(("models", "FAIL", str(e)))
        failed += 1

    # Test 3: Simple chat
    log("Test 3/6: Simple chat")
    try:
        resp = send_chat("Hello, what is your name?")
        content = resp["choices"][0]["message"]["content"].strip()
        assert len(content) > 0, "Empty response"
        results.append(("chat", "PASS", f"Response: {content[:100]}..."))
        passed += 1
    except Exception as e:
        results.append(("chat", "FAIL", str(e)))
        failed += 1

    # Test 4: Arithmetic reasoning
    log("Test 4/6: Arithmetic reasoning")
    try:
        resp = send_chat("What is 17 * 23? Just give the number.")
        content = resp["choices"][0]["message"]["content"].strip()
        assert "391" in content, f"Expected 391, got: {content}"
        results.append(("arithmetic", "PASS", f"Response: {content[:100]}"))
        passed += 1
    except Exception as e:
        results.append(("arithmetic", "FAIL", str(e)))
        failed += 1

    # Test 5: Chinese language
    log("Test 5/6: Chinese language")
    try:
        resp = send_chat("用中文解释什么是人工智能，一句话。")
        content = resp["choices"][0]["message"]["content"].strip()
        assert len(content) > 5, f"Response too short: {content}"
        # Verify Chinese characters present
        has_chinese = any("\u4e00" <= c <= "\u9fff" for c in content)
        assert has_chinese, f"No Chinese in response: {content}"
        results.append(("chinese", "PASS", f"Response: {content[:100]}"))
        passed += 1
    except Exception as e:
        results.append(("chinese", "FAIL", str(e)))
        failed += 1

    # Test 6: Multi-turn conversation
    log("Test 6/6: Multi-turn conversation")
    try:
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice! How can I help you?"},
            {"role": "user", "content": "What is my name?"},
        ]
        resp = send_chat("", messages=messages)
        content = resp["choices"][0]["message"]["content"].strip()
        assert "alice" in content.lower(), f"Model forgot name: {content}"
        results.append(("multi_turn", "PASS", f"Response: {content[:100]}"))
        passed += 1
    except Exception as e:
        results.append(("multi_turn", "FAIL", str(e)))
        failed += 1

    return passed, failed, results


def main():
    log("=" * 60)
    log("MiniMax-M1 End-to-End Validation")
    log(f"Model: {MODEL_PATH}")
    log(f"Quantization: {QUANTIZATION}")
    log(f"TP Degree: {TP_DEGREE}")
    log(f"Port: {PORT}")
    log("=" * 60)

    # Build server command
    cmd = [
        sys.executable,
        "-m",
        "fastdeploy.entrypoints.openai.api_server",
        "--model",
        MODEL_PATH,
        "--port",
        str(PORT),
        "--max-model-len",
        "4096",
    ]
    if QUANTIZATION and QUANTIZATION != "none":
        cmd.extend(["--quantization", QUANTIZATION])
    if TP_DEGREE > 1:
        cmd.extend(["--tensor-parallel-size", str(TP_DEGREE)])

    log(f"Starting server: {' '.join(cmd)}")
    server = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    try:
        if not wait_for_server(PORT, timeout=900):
            log("FATAL: Server failed to start within 15 minutes!")
            # Dump last output
            if server.stdout:
                output = server.stdout.read(4096)
                if output:
                    log(f"Server output:\n{output.decode(errors='replace')}")
            sys.exit(1)

        passed, failed, results = run_validations()

        # Print structured evidence
        log("")
        log("=" * 60)
        log(f"RESULTS: {passed}/{passed+failed} passed")
        log("=" * 60)
        for name, status, detail in results:
            icon = "✅" if status == "PASS" else "❌"
            log(f"  {icon} {name}: {detail}")

        if failed > 0:
            log(f"\n❌ {failed} test(s) FAILED")
            sys.exit(1)
        else:
            log("\n✅ All validations passed!")

    finally:
        log("Shutting down server...")
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=15)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception:
                pass
        log("Done.")


if __name__ == "__main__":
    main()
