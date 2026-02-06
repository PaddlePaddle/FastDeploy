#!/bin/bash

PID=$(ps -ef | grep "fd-router" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "Killing existing fd-router process (PID: $PID)"
    # Try graceful shutdown first
    kill -15 $PID
    TIMEOUT=10
    while kill -0 $PID 2>/dev/null && [ $TIMEOUT -gt 0 ]; do
        echo "Waiting for fd-router (PID: $PID) to exit gracefully... ($TIMEOUT seconds remaining)"
        sleep 1
        TIMEOUT=$((TIMEOUT - 1))
    done
    # Force kill if still running after timeout
    if kill -0 $PID 2>/dev/null; then
        echo "fd-router (PID: $PID) did not exit gracefully; sending SIGKILL..."
        kill -9 $PID
    fi
fi

echo "Starting new fd-router process..."
nohup /usr/local/bin/fd-router --config_path ./config/config.yaml --splitwise > fd-router.log 2>&1 &
echo "fd-router started with PID: $!"
