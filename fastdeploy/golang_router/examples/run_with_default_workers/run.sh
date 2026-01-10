#!/bin/bash

PID=$(ps -ef | grep "fd-router" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "Killing existing fd-router process (PID: $PID)"
    # First try to terminate gracefully
    kill -15 "$PID"
    sleep 5
    # If still running after timeout, force kill as a last resort
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "Process $PID did not terminate gracefully; force killing..."
        kill -9 "$PID"
    fi
fi

echo "Starting new fd-router process..."
nohup ./fd-router --config_path ./config/config.yaml --splitwise > fd-router.log 2>&1 &
echo "fd-router started with PID: $!"
