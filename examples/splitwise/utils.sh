#!/bin/bash

check_port() {
    local port=$1
    if ss -tuln | grep -q ":$port "; then
        echo "❌ 端口 $port 已被占用"
        return 1
    else
        echo "✅ 端口 $port 可用"
        return 0
    fi
}

wait_for_health() {
    local server_port=$1
    while true; do
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "http://0.0.0.0:${server_port}/health" || echo "000")
    if [ "$status_code" -eq 200 ]; then
            break
    else
            echo "Service not ready. Retrying in 2s..."
            sleep 2
    fi
    done
}
