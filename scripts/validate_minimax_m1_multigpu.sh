#!/usr/bin/env bash
# ===========================================================================
# MiniMax-M1 Multi-GPU Validation Script.
# ===========================================================================
# For Baidu reviewers / CI operators with multi-GPU infrastructure.
# MiniMax-M1 is 456B params — requires multiple GPUs for any model-loading test.
#
# Hardware requirements by quantization:
#   BF16:  12× A800-80GB (~912 GB)
#   FP8:    6× A800-80GB (~456 GB)
#   WINT4:  3× A800-80GB (~228 GB)  ← recommended minimum
#
# Usage:
#   # Minimum (WINT4 on 4 GPUs):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/validate_minimax_m1_multigpu.sh
#
#   # With pre-downloaded weights:
#   MODEL_PATH=/data/models/MiniMax-M1 bash scripts/validate_minimax_m1_multigpu.sh
#
#   # Skip slow tiers:
#   SKIP_TIER3=1 bash scripts/validate_minimax_m1_multigpu.sh
#
# Environment variables:
#   MODEL_PATH          Path to local model weights (default: MiniMax/MiniMax-M1)
#   QUANT_MODE          Quantization mode (default: wint4)
#   MAX_MODEL_LEN       Max sequence length (default: 4096)
#   PORT                Server port (default: 8180)
#   SKIP_TIER1          Skip Tier 1 (model load + server start)
#   SKIP_TIER2          Skip Tier 2 (inference coherence)
#   SKIP_TIER3          Skip Tier 3 (quantization variants)
#   TIMEOUT_SECS        Server startup timeout in seconds (default: 900)
# ===========================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-MiniMax/MiniMax-M1}"
QUANT_MODE="${QUANT_MODE:-wint4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
PORT="${PORT:-8180}"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"
SERVER_PID=""

cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Cleaning up server PID=$SERVER_PID"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap 'cleanup; exit' EXIT INT TERM

# ── Detect GPU count ──
NUM_GPUS=$(python3 -c "import paddle; print(paddle.device.cuda.device_count())" 2>/dev/null || echo 0)
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  MiniMax-M1 Multi-GPU Validation                            ║"
echo "║  456B MoE (45.9B active) — full model load + inference      ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  GPUs detected: $NUM_GPUS                                          ║"
echo "║  Model: $MODEL_PATH"
echo "║  Quantization: $QUANT_MODE"
echo "║  Max seq len: $MAX_MODEL_LEN"
echo "╚═══════════════════════════════════════════════════════════════╝"

if [[ "$NUM_GPUS" -lt 3 ]]; then
    echo "ERROR: MiniMax-M1 requires at least 3 GPUs (WINT4). Found: $NUM_GPUS"
    echo "For single-GPU component tests, use: aistudio/task047_minimax_m1_validate.sh"
    exit 1
fi

# Recommend TP size based on GPU count and quant
TP_SIZE="$NUM_GPUS"
if [[ "$QUANT_MODE" == "wint4" ]] && [[ "$NUM_GPUS" -ge 4 ]]; then
    TP_SIZE=4
elif [[ "$QUANT_MODE" == *"fp8"* ]] && [[ "$NUM_GPUS" -ge 8 ]]; then
    TP_SIZE=8
elif [[ "$QUANT_MODE" == "bf16" ]] || [[ "$QUANT_MODE" == "none" ]]; then
    if [[ "$NUM_GPUS" -lt 12 ]]; then
        echo "WARNING: BF16 needs ~12 GPUs. Using all $NUM_GPUS — may OOM."
    fi
    TP_SIZE="$NUM_GPUS"
fi
echo "Using tensor_parallel_size=$TP_SIZE"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0

# ═══════════════════════════════════════════════════════════════════
# Tier 0: Environment
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "━━━ Tier 0: Environment ━━━"
python3 -c "
import paddle
print(f'Paddle {paddle.__version__}')
print(f'CUDA compiled: {paddle.is_compiled_with_cuda()}')
print(f'GPUs: {paddle.device.cuda.device_count()}')
for i in range(paddle.device.cuda.device_count()):
    props = paddle.device.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_memory // (1024**2)} MiB, SM {props.major}.{props.minor}')
"
echo "✅ Tier 0 PASS"
PASS_COUNT=$((PASS_COUNT + 1))

start_server() {
    local quant="$1"
    local tp="$2"
    local tag="$3"

    echo "Starting server: quant=$quant, tp=$tp ($tag)..."

    local quant_args=""
    if [[ "$quant" != "none" ]] && [[ "$quant" != "bf16" ]]; then
        quant_args="--quantization $quant"
    fi

    python -m fastdeploy.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --tensor-parallel-size "$tp" \
        $quant_args \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs 4 \
        --port "$PORT" &
    SERVER_PID=$!

    local elapsed=0
    local interval=5
    while [[ $elapsed -lt $TIMEOUT_SECS ]]; do
        if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "Server ready after ${elapsed}s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server process died during startup"
            SERVER_PID=""
            return 1
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    echo "ERROR: Server did not start within ${TIMEOUT_SECS}s"
    kill "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
    return 1
}

stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 2
    fi
}

send_chat() {
    local prompt="$1"
    local max_tokens="${2:-64}"
    local body
    body=$(jq -n --arg model "$MODEL_PATH" --arg prompt "$prompt" --argjson mt "$max_tokens" \
        '{model: $model, messages: [{role: "user", content: $prompt}], max_tokens: $mt, temperature: 0.0}')
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "$body"
}

# ═══════════════════════════════════════════════════════════════════
# Tier 1: Model Loads + Server Starts
# ═══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TIER1:-0}" != "1" ]]; then
    echo ""
    echo "━━━ Tier 1: Model Loads + Server Starts ($QUANT_MODE, TP=$TP_SIZE) ━━━"

    if start_server "$QUANT_MODE" "$TP_SIZE" "tier1"; then
        # Verify /health and /v1/models both respond
        HEALTH=$(curl -s "http://localhost:${PORT}/health")
        MODELS=$(curl -s "http://localhost:${PORT}/v1/models")
        echo "$MODELS" | python3 -c "
import json, sys
models = json.load(sys.stdin)
assert 'data' in models, f'No data in /v1/models: {models}'
names = [m['id'] for m in models['data']]
print(f'Models served: {names}')
assert len(names) > 0, 'No models loaded'
print('✅ Tier 1 PASS: Model loaded, server healthy')
"
        PASS_COUNT=$((PASS_COUNT + 1))
        # Keep server running for Tier 2
    else
        echo "❌ Tier 1 FAIL: Server did not start"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
else
    echo "━━━ Tier 1: SKIPPED (SKIP_TIER1=1) ━━━"
    SKIP_COUNT=$((SKIP_COUNT + 1))
fi

# ═══════════════════════════════════════════════════════════════════
# Tier 2: Inference Produces Coherent Output
# ═══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TIER2:-0}" != "1" ]] && [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "━━━ Tier 2: Inference Coherence ━━━"

    RESPONSE=$(send_chat "What is 2+3? Answer with just the number." 32)

    export RESPONSE
    python3 <<'PYEOF'
import json, os, sys

resp = json.loads(os.environ["RESPONSE"])
if "choices" not in resp or len(resp["choices"]) == 0:
    print(f"❌ Tier 2 FAIL: No choices in response: {resp}")
    sys.exit(1)

content = resp["choices"][0]["message"]["content"].strip()
print(f"Model output: {content!r}")

# Basic coherence: response is non-empty, contains digits or words
if not content:
    print("❌ Tier 2 FAIL: Empty response")
    sys.exit(1)

if "5" in content:
    print("✅ Arithmetic correct: '5' found in response")
else:
    print(f"⚠️  Expected '5', got: {content!r} — checking general coherence")

# Second test: open-ended generation
import subprocess, os
resp2_raw = subprocess.check_output([
    "curl", "-s", f"http://localhost:{os.environ.get('PORT', '8180')}/v1/chat/completions",
    "-H", "Content-Type: application/json",
    "-d", json.dumps({
        "model": os.environ.get("MODEL_PATH", "MiniMax/MiniMax-M1"),
        "messages": [{"role": "user", "content": "Explain quantum entanglement in one sentence."}],
        "max_tokens": 64,
        "temperature": 0.0,
    })
])
resp2 = json.loads(resp2_raw)
content2 = resp2["choices"][0]["message"]["content"].strip()
print(f"Open-ended output: {content2!r}")

# Coherence check: at least 10 chars, contains some real words
if len(content2) < 10:
    print("❌ Tier 2 FAIL: Response too short — likely garbage output")
    sys.exit(1)

real_words = sum(1 for w in content2.split() if len(w) > 2)
if real_words < 3:
    print("❌ Tier 2 FAIL: Response lacks coherent words")
    sys.exit(1)

print(f"✅ Tier 2 PASS: Coherent output ({len(content2)} chars, {real_words} words)")
PYEOF
    TIER2_RC=$?
    if [[ $TIER2_RC -eq 0 ]]; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    stop_server
elif [[ "${SKIP_TIER2:-0}" == "1" ]]; then
    echo "━━━ Tier 2: SKIPPED (SKIP_TIER2=1) ━━━"
    SKIP_COUNT=$((SKIP_COUNT + 1))
    stop_server
else
    echo "━━━ Tier 2: SKIPPED (no running server from Tier 1) ━━━"
    SKIP_COUNT=$((SKIP_COUNT + 1))
fi

# ═══════════════════════════════════════════════════════════════════
# Tier 3: Quantization Variants
# ═══════════════════════════════════════════════════════════════════
if [[ "${SKIP_TIER3:-0}" != "1" ]]; then
    echo ""
    echo "━━━ Tier 3: Quantization Variants ━━━"

    # Test each quant mode the model supports (skip the one already tested in Tier 1)
    QUANT_MODES=("wint4" "w4a8" "w4afp8" "block_wise_fp8")
    QUANT_TP_MIN=(3 6 6 6)  # minimum TP for each mode

    TIER3_PASS=0
    TIER3_FAIL=0
    TIER3_SKIP=0

    for i in "${!QUANT_MODES[@]}"; do
        qmode="${QUANT_MODES[$i]}"
        min_tp="${QUANT_TP_MIN[$i]}"

        if [[ "$qmode" == "$QUANT_MODE" ]]; then
            echo "  $qmode: already tested in Tier 1, skipping"
            continue
        fi

        if [[ "$NUM_GPUS" -lt "$min_tp" ]]; then
            echo "  $qmode: needs ${min_tp} GPUs, have ${NUM_GPUS} — SKIP"
            TIER3_SKIP=$((TIER3_SKIP + 1))
            continue
        fi

        echo "  Testing $qmode (TP=$min_tp)..."
        if start_server "$qmode" "$min_tp" "tier3-$qmode"; then
            # Quick smoke: one inference call
            SMOKE_RESP=$(send_chat "Hello" 16)
            HAS_CHOICES=$(echo "$SMOKE_RESP" | python3 -c "import json,sys; r=json.load(sys.stdin); print('ok' if r.get('choices') else 'fail')" 2>/dev/null || echo "fail")
            if [[ "$HAS_CHOICES" == "ok" ]]; then
                echo "    ✅ $qmode: server started + inference OK"
                TIER3_PASS=$((TIER3_PASS + 1))
            else
                echo "    ❌ $qmode: server started but inference failed"
                TIER3_FAIL=$((TIER3_FAIL + 1))
            fi
            stop_server
        else
            echo "    ❌ $qmode: server failed to start"
            TIER3_FAIL=$((TIER3_FAIL + 1))
        fi
    done

    echo "  Tier 3 results: $TIER3_PASS pass, $TIER3_FAIL fail, $TIER3_SKIP skip"
    if [[ $TIER3_FAIL -eq 0 ]] && [[ $TIER3_PASS -gt 0 ]]; then
        echo "✅ Tier 3 PASS"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [[ $TIER3_FAIL -gt 0 ]]; then
        echo "❌ Tier 3 FAIL"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    else
        echo "⚠️  Tier 3 SKIP (not enough GPUs for any untested quant mode)"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
else
    echo "━━━ Tier 3: SKIPPED (SKIP_TIER3=1) ━━━"
    SKIP_COUNT=$((SKIP_COUNT + 1))
fi

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  SUMMARY                                                     ║"
echo "╠═══════════════════════════════════════════════════════════════╣"
echo "║  Passed: $PASS_COUNT  Failed: $FAIL_COUNT  Skipped: $SKIP_COUNT                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo "Date: $(date)"

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
