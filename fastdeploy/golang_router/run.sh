#!/bin/bash

PID=$(ps -ef | grep "fd-router" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    echo "Killing existing fd-router process (PID: $PID)"
    kill -9 $PID
fi

echo "Starting new fd-router process..."
nohup ./fd-router --port 8080 --splitwise > fd-router.log 2>&1 &
echo "fd-router started with PID: $!"
