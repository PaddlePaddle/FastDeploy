#!/usr/bin/env bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Current directory: ${DIR}"

function stop_processes() {
    ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    lsof -t -i :8188 | xargs kill -9 || true
}

echo "Clean up processes..."
stop_processes
echo "Clean up completed."

export model_path=${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle

echo "pip install requirements"
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "uninstall org"
python -m pip uninstall paddlepaddle -y
python -m pip uninstall paddle-custom-gcu -y
python -m pip install paddlepaddle==3.1.0a0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install --pre paddle-custom-gcu==3.0.0.dev20250801 -i https://www.paddlepaddle.org.cn/packages/nightly/gcu/
echo "build whl"
bash build.sh 1 || exit 1

unset http_proxy
unset https_proxy
unset no_proxy

rm -rf log/*
rm -f core*

# Empty the message queue
ipcrm --all=msg
echo "Start server..."
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --port 8188 \
    --metrics-port 8200 \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 4096 \
    --max-model-len 32768 \
    --max-num-seqs 8 \
    --quantization wint4 > server.log 2>&1 &

echo "Waiting 90 seconds..."
sleep 90

if grep -q "Failed to launch worker processes" server.log; then
    echo "Failed to launch worker processes..."
    stop_processes
    cat server.log
    cat log/workerlog.0
    exit 1
fi

if grep -q "Traceback (most recent call last):" server.log; then
    echo "Some errors occurred..."
    stop_processes
    cat server.log
    cat log/workerlog.0
    exit 1
fi

# Health check
TIMEOUT=$((11 * 60))
INTERVAL=30            # Check interval (seconds)
ENDPOINT="http://0.0.0.0:8188/health"
START_TIME=$(date +%s) # Record the start timestamp
echo "Start the server health check, maximum waiting time: ${TIMEOUT} seconds..."
while true; do
    # Used to calculate the time cost
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    # Timeout
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\nServer start timeout: After $((TIMEOUT/60)) minutes, the service still doesn't start!"
        stop_processes
        cat server.log
        cat log/workerlog.0
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\nThe server was successfully launched! Totally takes $((ELAPSED+90)) seconds."
        break
    else
        sleep $INTERVAL
    fi
done

cat server.log
echo -e "\n"

echo "Start inference..."
python tests/ci_use/GCU/run_ernie.py
exit_code=$?
echo -e "exit_code is ${exit_code}.\n"

echo "Stop server..."
stop_processes
echo "Stop server done."

if [ ${exit_code} -ne 0 ]; then
    echo "Exit with error, please refer to log/workerlog.0"
    cat log/workerlog.0
    exit 1
fi
