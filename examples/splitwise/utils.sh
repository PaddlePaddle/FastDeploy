#!/bin/bash

check_ports() {
    for port in "$@"; do
        if ss -tuln | grep -q ":$port "; then
            echo "❌ Port $port is already in use"
            return 1
        fi
    done
    return 0
}

wait_for_health() {
    local server_port=$1
    while true; do
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "http://0.0.0.0:${server_port}/health" || echo "000")
    if [ "$status_code" -eq 200 ]; then
            break
    else
            echo "Service not ready. Retrying in 4s..."
            sleep 4
    fi
    done
}
