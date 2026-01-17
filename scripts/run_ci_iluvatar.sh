#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#先kill一遍
ps -efww | grep -E 'run_ernie300B_4layer' | grep -v grep | awk '{print $2}' | xargs kill -9 || true

unset http_proxy
unset https_proxy
unset no_proxy

export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
ln -sf /usr/local/bin/python3 /usr/local/bin/python
echo "pip requirements"
python -m pip install -r requirements_iluvatar.txt
echo "install paddle cpu and custom device"
python -m pip install  --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install --pre paddle-iluvatar-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
#python -m pip install paddlepaddle==3.3.0.dev20251219 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
#python -m pip install paddle-iluvatar-gpu==3.0.0.dev20251223 -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/

MODEL_DIR=/model_data
mkdir -p $MODEL_DIR
SOURCE_DIR=/aistudio/paddle_ci
for file in "$SOURCE_DIR"/*; do
    echo "start copy $file into $MODEL_DIR ..."
    cp -r $file $MODEL_DIR
done
echo "copy done"
ls $MODEL_DIR

echo "build whl"
bash build.sh || exit 1

function print_error_message() {
    if [ ! -f "log/workerlog.0" ]; then
        echo "------------------- log/launch_worker.log -----------------"
        cat log/launch_worker.log
    else
        echo "------------------- log/workerlog.0 -----------------"
        cat log/workerlog.0
    fi
    if [ -f "log/fastdeploy_error.log" ]; then
        echo "------------------- log/fastdeploy_error.log -----------------"
        cat log/fastdeploy_error.log
    fi
}

CI_PATH=tests/ci_use/iluvatar_UT
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export FD_SAMPLING_CLASS=rejection

################# Test offline ###################

offline_ci_list=(
    ${CI_PATH}/run_ernie300B_4layer.py
    ${CI_PATH}/run_ernie_vl_28B.py
)
echo "test offline ci files: ${offline_ci_list[@]}"
for cur_test_file in ${offline_ci_list[@]}
do
    echo "============ Offline: start to test ${cur_test_file} ==========="
    rm -rf log/*
    python ${cur_test_file}
    exit_code=$?
    echo exit_code is ${exit_code}

    ps -efww | grep -E '${cur_test_file}' | grep -v grep | awk '{print $2}' | xargs kill -9 || true

    if [ ${exit_code} -ne 0 ]; then
        print_error_message
        exit 1
    fi
done

################# Test Online ###################

function clear_message() {
    # clear the message queue
    ipcrm --all=msg
    rm -rf log/* server.log
}

function stop_processes() {
    ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E '8180' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
}

function check_server_status() {
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
    ENDPOINT="http://0.0.0.0:8180/health"
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

    echo -e "\n server.log:"
    cat server.log
    echo -e "\n"
}

echo "============ Online: start to test ERNIE-4.5-21B-A3B-Paddle ==========="
clear_message
echo "Start server..."
python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_DIR}/ERNIE-4.5-21B-A3B-Paddle \
       --port 8180 \
       --tensor-parallel-size 1 \
       --quantization wint8 \
       --max-model-len 32768 \
       --max-num-seqs 8 \
       --block-size 16 > server.log 2>&1 &

check_server_status

echo "Start inference..."
cp ${CI_PATH}/test.jsonl ./
python3 -u ${CI_PATH}/bench_gsm8k.py --port 8180 --num-questions 10 --num-shots 5 --parallel 8

exit_code=$?
echo -e "\nexit_code is ${exit_code}"

echo -e "\nStop server..."
stop_processes
echo -e "\nStop server done."

if [ ${exit_code} -ne 0 ]; then
    print_error_message
    exit 1
fi

acc=`python3 -c "import json; [print(json.loads(line)['latency']) for line in open('result.jsonl')]"`
latency=`python3 -c "import json; [print(json.loads(line)['latency']) for line in open('result.jsonl')]"`
expected_lowerest_acc=0.8
expected_largest_latency=60
if awk -v a="$acc" -v b="$expected_lowerest_acc" 'BEGIN {exit !(a < b)}'; then
    echo -e "\nExit with Accucary error, current accuracy $acc less than $expected_lowerest_acc "
    exit 1
fi

# if awk -v a="$latency" -v b="$expected_largest_latency" 'BEGIN {exit !(a > b)}'; then
#     echo -e "\nExit with Latency Error, current latency $latency greater than $expected_largest_latency "
#     exit 1
# fi
echo -e "\nPASSED"

echo -e "\n============ Online: start to test ERNIE-4.5-VL-28B-A3B-Paddle ==========="
clear_message
echo "Start server..."
python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_DIR}/ERNIE-4.5-VL-28B-A3B-Paddle \
       --port 8180 \
       --tensor-parallel-size 2 \
       --quantization wint8 \
       --limit-mm-per-prompt '{"image": 100, "video": 100}' \
       --reasoning-parser ernie-45-vl \
       --max-model-len 32768 \
       --max-num-seqs 8 \
       --block-size 16 > server.log 2>&1 &

check_server_status

echo "Start inference..."
result_file="full_response.log"

curl -X POST "http://0.0.0.0:8180/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "https://paddlenlp.bj.bcebos.com/datasets/paddlemix/demo_images/example2.jpg"}},
      {"type": "text", "text": "From which era does the artifact in the image originate?"}
    ]}
  ],
  "chat_template_kwargs":{"enable_thinking": false}
}' >& $result_file

echo -e "\nfull response:"
cat $result_file

exit_code=$?
echo -e "\n\nexit_code is ${exit_code}"

echo -e "\nStop server..."
stop_processes
echo -e "\nStop server done."

if [ ${exit_code} -ne 0 ]; then
    print_error_message
    exit 1
fi

expected_strings="Buddhist statue"
if grep -q "$expected_strings" "$result_file"; then
    echo -e "\nPASSED"
else
    echo -e "\nExit with Accucary error: '$expected_strings' is not existed in generate response."
    exit 1
fi

echo -e "\n============ Online: start to test PaddleOCR-VL ==========="
clear_message
echo "Start server..."
python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_DIR}/PaddleOCR-VL \
       --port 8180 \
       --metrics-port 8471 \
       --engine-worker-queue-port 8472 \
       --cache-queue-port 55660 \
       --max-model-len 16384 \
       --max-num-batched-tokens 16384 \
       --max-num-seqs 64 \
       --workers 2 \
       --block-size 16 > server.log 2>&1 &

check_server_status

echo "Start inference..."
result_file="full_response.log"

paddleocr doc_parser -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/paddleocr_vl_demo.png \
       --vl_rec_backend fastdeploy-server --vl_rec_server_url http://127.0.0.1:8180/v1 >& $result_file

echo -e "\nfull response:"
cat $result_file

exit_code=$?
echo -e "\n\nexit_code is ${exit_code}"

echo -e "\nStop server..."
stop_processes
echo -e "\nStop server done."

if [ ${exit_code} -ne 0 ]; then
    print_error_message
    exit 1
fi

expected_strings="本报记者 沈小晓 任彦 黄培昭"
if grep -q "$expected_strings" "$result_file"; then
    echo -e "\nPASSED"
else
    echo -e "\nExit with Accucary error: '$expected_strings' is not existed in generate response."
    exit 1
fi
