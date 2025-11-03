#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

function stop_processes() {
    ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    lsof -t -i :8188 | xargs kill -9 || true
}

echo "Clean up processes..."
stop_processes
echo "Clean up completed."

export model_path=${MODEL_PATH}/paddle/ERNIE-4.5-21B-A3B-Paddle

python -m pip install paddlepaddle_dcu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
python -m pip install https://paddle-whl.bj.bcebos.com/stable/dcu/triton/triton-3.0.0%2Bdas.opt4.0da70a2.dtk2504-cp310-cp310-manylinux_2_28_x86_64.whl

python -m pip install git+https://github.com/zhoutianzi666/UseTritonInPaddle.git
python -c "import use_triton_in_paddle; use_triton_in_paddle.make_triton_compatible_with_paddle()"

echo "pip install requirements_dcu"
python -m pip install -r requirements_dcu.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "build whl"
bash build.sh || exit 1

unset http_proxy
unset https_proxy
unset no_proxy


rm -rf log/*
rm -f core*

# Empty the message queue
ipcrm --all=msg
echo "Start server..."
export FD_ATTENTION_BACKEND="BLOCK_ATTN"
python -m fastdeploy.entrypoints.openai.api_server \
    --model ${model_path} \
    --port 8188 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.8 \
    --quantization wint8   > server.log 2>&1 &

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
TIMEOUT=$((5 * 60))
INTERVAL=10 # Check interval (seconds)
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
python tests/ci_use/DCU/run_ernie.py
exit_code=$?
echo "exit_code is ${exit_code}.\n"

echo "Stop server..."
stop_processes
echo "Stop server done."

if [ ${exit_code} -ne 0 ]; then
    echo "Exit with error, please refer to log/workerlog.0"
    cat log/workerlog.0
    exit 1
fi
