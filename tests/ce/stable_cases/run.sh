#!/bin/bash

# ================== Configuration Parameters ==================
FD_API_PORT=${FD_API_PORT:-8180}
FD_ENGINE_QUEUE_PORT=${FD_ENGINE_QUEUE_PORT:-8181}
FD_METRICS_PORT=${FD_METRICS_PORT:-8182}
FD_CACHE_QUEUE_PORT=${FD_CACHE_QUEUE_PORT:-8183}


HOST="0.0.0.0"
PORT="${FD_API_PORT}"  # 这里需要配合启动脚本那个URL PORT
BASE_URL="http://$HOST:$PORT"

V0_ROUNDS=3
V1_ROUNDS=3
TOTAL_ROUNDS=$((V0_ROUNDS + V1_ROUNDS))
CHAT_REQUESTS_PER_ROUND=3
export CUDA_VISIBLE_DEVICES=0,1
MAX_MEMORY_MB=10240  # 10GB

# ====================================================
# assert_eq actual expected message
assert_eq() {
    local actual="$1"
    local expected="$2"
    local msg="$3"
    if [ "$actual" != "$expected" ]; then
        echo "Assertion failed: $msg" >&2
        exit 1
    fi
}

# assert_true condition message
assert_true() {
    local condition="$1"
    local msg="$2"
    if [ "$condition" != "1" ] && [ "$condition" != "true" ]; then
        echo "Assertion failed: $msg" >&2
        exit 1
    fi
}

# assert_success exit_code message
assert_success() {
    local code="$1"
    local msg="$2"
    if [ "$code" -ne 0 ]; then
        echo "Assertion failed: $msg" >&2
        exit 1
    fi
}

# curl_get_status(url, options) → returns via global variables http_code and response_body
curl_get_status() {
    local result
    result=$(curl -s -w "%{http_code}" "$@")
    http_code="${result: -3}"
    response_body="${result%???}"
}

post_json_and_assert() {
    local url="$1"
    local payload="${2:-}"
    if [ -n "$payload" ]; then
        curl_get_status -X POST "$url" -H "Content-Type: application/json" -d "$payload"
    else
        curl_get_status -X POST "$url"
    fi
    assert_eq "$http_code" "200" "$url failed with HTTP $http_code, body: $response_body"
}

get_and_assert() {
    local url="$1"
    curl_get_status "$url"
    assert_eq "$http_code" "200" "$url failed with HTTP $http_code, body: $response_body"
}

# ====================================================
# Get visible GPU IDs from CUDA_VISIBLE_DEVICES
# ====================================================

get_visible_gpu_ids() {
    local ids=()
    IFS=',' read -ra ADDR <<< "$CUDA_VISIBLE_DEVICES"
    for i in "${ADDR[@]}"; do
        if [[ "$i" =~ ^[0-9]+$ ]]; then
            ids+=("$i")
        fi
    done
    echo "${ids[@]}"
}

# ====================================================
# Check GPU memory usage (must not exceed MAX_MEMORY_MB)
# ====================================================

check_gpu_memory() {
    local gpu_ids
    gpu_ids=($(get_visible_gpu_ids))

    echo "----------------------------------------"
    echo "       GPU Memory Check (MAX: ${MAX_MEMORY_MB}MB)"
    echo "----------------------------------------"

    if [ ${#gpu_ids[@]} -eq 0 ]; then
        echo "ERROR: No valid GPU IDs in CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" >&2
        exit 1
    fi

    for gpu_id in "${gpu_ids[@]}"; do
        # Query summary
        local summary
        summary=$(nvidia-smi -i "$gpu_id" \
            --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
            --format=csv,noheader,nounits) || {
                echo "ERROR: Failed to query GPU $gpu_id" >&2
                exit 1
            }

        # Parse fields
        IFS=',' read -r idx name mem_total mem_used mem_free util <<< "$summary"
        local used_ratio=$(( mem_used * 100 / mem_total ))

        # Print GPU info (single line)
        printf "  GPU %s: %-35s Total:%5sMB Used:%5sMB (%s%%)\n" \
            "$idx" "$name" "$mem_total" "$mem_used" "$util"

        # Hard assertion
        assert_true "$(( mem_used <= MAX_MEMORY_MB ))" \
            "GPU $gpu_id memory.used ${mem_used} MB > MAX_MEMORY_MB ${MAX_MEMORY_MB} MB"

        # Usage ratio check
        if [ "$used_ratio" -gt 90 ]; then
            echo "ERROR: GPU $gpu_id memory usage > 90% (${used_ratio}%)" >&2
            exit 1
        fi

        # Process info (compact format)
        local proc_info
        proc_info=$(nvidia-smi -i "$gpu_id" \
            --query-compute-apps=pid,process_name,used_memory \
            --format=csv,noheader,nounits)

        if [ -n "$proc_info" ]; then
            echo "$proc_info" | while IFS=',' read -r pid pname pmem; do
                printf "    └─ PID=%-8s %-30s MEM=%4sMB\n" \
                    "$pid" "$pname" "$pmem"
            done
        fi
    done

    echo "----------------------------------------"
}

# ====================================================

send_chat_requests() {
    for i in $(seq 1 $CHAT_REQUESTS_PER_ROUND); do
        printf "  └─ Sending chat request %d/%d...\n" "$i" "$CHAT_REQUESTS_PER_ROUND"
        response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"messages": [{"role": "user", "content": "Hello!"}]}')

        content=$(echo "$response" | \
            grep -o '"content":"[^"]*"' | \
            head -1 | \
            sed 's/^"content":"//' | \
            sed 's/"$//')

        if [ -z "$content" ]; then
            content=$(echo "$response" | \
                sed -n 's/.*"content":"\([^"]*\)".*/\1/p' | \
                head -1)
        fi

        if [ -z "$content" ] || [ "$content" = "null" ]; then
            echo "  ERROR: Empty or null 'content' in response" >&2
            echo "  Raw response:" >&2
            echo "$response" >&2
            exit 1
        fi

        printf "  └─ Response received: %s\n" "$content"
    done
}

run_v0_round() {
    local round="$1"
    echo "========================================"
    printf "Round %d/%d (V0)\n" "$round" "$TOTAL_ROUNDS"
    echo "========================================"
    echo ""

    printf "[Step 1] %-30s " "Clearing load weight via v0..."
    get_and_assert "$BASE_URL/clear_load_weight"
    echo "[OK]"
    sleep 10

    printf "[Step 2] %-30s " "Checking GPU memory..."
    echo ""
    check_gpu_memory
    echo ""

    printf "[Step 3] %-30s " "Updating model weight via v0..."
    get_and_assert "$BASE_URL/update_model_weight"
    echo "[OK]"

    echo "[Step 4] Sending $CHAT_REQUESTS_PER_ROUND chat completions"
    send_chat_requests
    echo ""
}

run_v1_round() {
    local round="$1"
    local sleep_payload='{"tags":"weight,kv_cache"}'
    local wakeup_payload='{"tags":"weight,kv_cache"}'

    echo "========================================"
    printf "Round %d/%d (V1)\n" "$round" "$TOTAL_ROUNDS"
    echo "========================================"
    echo ""

    printf "[Step 1] %-30s " "Pausing engine via v1..."
    post_json_and_assert "$BASE_URL/v1/pause" ""
    echo "[OK]"

    printf "[Step 2] %-30s " "Sleeping via v1..."
    post_json_and_assert "$BASE_URL/v1/sleep" "$sleep_payload"
    echo "[OK]"
    sleep 10

    printf "[Step 3] %-30s " "Checking GPU memory..."
    echo ""
    check_gpu_memory
    echo ""

    printf "[Step 4] %-30s " "Waking up via v1..."
    post_json_and_assert "$BASE_URL/v1/wakeup" "$wakeup_payload"
    echo "[OK]"

    printf "[Step 5] %-30s " "Resuming engine via v1..."
    post_json_and_assert "$BASE_URL/v1/resume" ""
    echo "[OK]"

    echo "[Step 6] Sending $CHAT_REQUESTS_PER_ROUND chat completions"
    send_chat_requests
    echo ""
}

for round in $(seq 1 $V0_ROUNDS); do
    run_v0_round "$round"
done

for round in $(seq 1 $V1_ROUNDS); do
    run_v1_round "$round"
done

echo "========================================"
printf "All %d rounds completed successfully.\n" "$TOTAL_ROUNDS"
echo "========================================"
