#!/bin/bash

# ================== Configuration Parameters ==================
FD_API_PORT=${FD_API_PORT:-8000}
FD_ENGINE_QUEUE_PORT=${FD_ENGINE_QUEUE_PORT:-8001}
FD_METRICS_PORT=${FD_METRICS_PORT:-8002}
FD_CACHE_QUEUE_PORT=${FD_CACHE_QUEUE_PORT:-8003}


HOST="0.0.0.0"
PORT="${FD_API_PORT}"  # 这里需要配合启动脚本那个URL PORT
BASE_URL="http://$HOST:$PORT"

TOTAL_ROUNDS=30
CHAT_REQUESTS_PER_ROUND=1
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

# curl_get_status(url, options...) → returns via global variables http_code and response_body
curl_get_status() {
    local result
    result=$(curl -s -w "%{http_code}" "$@")
    http_code="${result: -3}"
    response_body="${result%???}"
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

    if [ ${#gpu_ids[@]} -eq 0 ]; then
        echo "Assertion failed: No valid GPU IDs in CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" >&2
        exit 1
    fi

    for gpu_id in "${gpu_ids[@]}"; do
        local memory_used
        memory_used=$(nvidia-smi -i "$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null) || \
            assert_success $? "Failed to query GPU $gpu_id memory usage"

        if ! [[ "$memory_used" =~ ^[0-9]+ ]]; then
            echo "Assertion failed: Invalid memory value for GPU $gpu_id: $memory_used" >&2
            exit 1
        fi

        assert_true "$(( memory_used <= MAX_MEMORY_MB ))" \
            "GPU $gpu_id memory $memory_used MB > $MAX_MEMORY_MB MB"
    done
}

# ====================================================

for round in $(seq 1 $TOTAL_ROUNDS); do
    echo "=== Round $round / $TOTAL_ROUNDS ==="

    # Step 1: Clear loaded weights
    echo "[Step 1] Clearing load weight..."
    curl_get_status -i "$BASE_URL/clear_load_weight"
    assert_eq "$http_code" "200" "/clear_load_weight failed with HTTP $http_code"

    # Step 2: Check GPU memory usage
    echo "[Step 2] Checking GPU memory..."
    check_gpu_memory

    # Step 3: Update model weights
    echo "[Step 3] Updating model weight..."
    curl_get_status -i "$BASE_URL/update_model_weight"
    assert_eq "$http_code" "200" "/update_model_weight failed with HTTP $http_code"

    # Step 4: Send chat completion requests
    echo "[Step 4] Sending $CHAT_REQUESTS_PER_ROUND chat completions..."
    for i in $(seq 1 $CHAT_REQUESTS_PER_ROUND); do
        echo "  Request $i / $CHAT_REQUESTS_PER_ROUND"
        # Send request and capture response
        response=$(curl -s -X POST "$BASE_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"messages": [{"role": "user", "content": "Hello!"}]}')

        # Extract the 'content' field from the response
        content=$(echo "$response" | \
            grep -o '"content":"[^"]*"' | \
            head -1 | \
            sed 's/^"content":"//' | \
            sed 's/"$//')

        if [ -z "$content" ]; then
            # Fallback: try extracting content using sed more robustly
            content=$(echo "$response" | \
                sed -n 's/.*"content":"\([^"]*\)".*/\1/p' | \
                head -1)
        fi

        # Check if content is empty or null
        if [ -z "$content" ] || [ "$content" = "null" ]; then
            echo "Failed: Empty or null 'content' in response" >&2
            echo "Raw response:" >&2
            echo "$response" >&2
            exit 1
        fi

        echo "Received non-empty response"
        echo -e "\n---\n"
    done

    echo "Round $round completed."
    echo "==================================\n"
done

echo "All $TOTAL_ROUNDS rounds completed successfully."
