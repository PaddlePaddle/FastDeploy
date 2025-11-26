#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#install dependencies
apt install -y lsof

export FD_API_PORT=8388
export FD_ENGINE_QUEUE_PORT=8902
export FD_METRICS_PORT=8202

#release relative resource
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E $FD_API_PORT | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :$FD_API_PORT | xargs kill -9 || true

echo "pip requirements"
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt

echo "uninstall org"
#to uninstall PaddleCustomDevie (paddle-intel-hpu)
python -m pip uninstall paddle-intel-hpu -y
#to uninstall fastdeploy
python -m pip uninstall fastdeploy_intel_hpu -y
#to install paddlepaddle
pip install paddlepaddle==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
#to install paddlecustomdevice? (paddle-intel-hpu)
pip install https://paddle-qa.bj.bcebos.com/suijiaxin/HPU/paddle_intel_hpu-0.0.1-cp310-cp310-linux_x86_64.whl
pip install https://paddle-qa.bj.bcebos.com/suijiaxin/HPU/paddlenlp_ops-0.0.0-cp310-cp310-linux_x86_64.whl

#to build and install fastdeploy
echo "build whl"
wget -q https://paddle-qa.bj.bcebos.com/suijiaxin/HPU/third-party/DeepGEMM.tar.gz && tar -xzf DeepGEMM.tar.gz -C custom_ops/third_party/
wget -q https://paddle-qa.bj.bcebos.com/suijiaxin/HPU/third-party/cutlass.tar.gz && tar -xzf cutlass.tar.gz -C custom_ops/third_party/
wget -q https://paddle-qa.bj.bcebos.com/suijiaxin/HPU/third-party/json.tar.gz && tar -xzf json.tar.gz -C custom_ops/third_party/ && mv custom_ops/third_party/json custom_ops/third_party/nlohmann_json
chmod +x build.sh
bash build.sh || exit 1
pip install dist/fastdeploy_intel_hpu-2.3.0.dev0-py3-none-any.whl --force-reinstall

#to install dependencies
echo "pip others"
pip install numpy
pip install requests
pip install tqdm
pip install ddt
pip install gradio
pip install aistudio-sdk
pip install pytest

#start serving
rm -rf log/*
rm -f server.log
#clear the message queue
ipcrm --all=msg

#start server
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export GC_KERNEL_PATH=/usr/local/lib/python3.10/dist-packages/paddle_custom_device/intel_hpu/libcustom_tpc_perf_lib.so:$GC_KERNEL_PATH
export PADDLE_DISTRI_BACKEND=xccl
export PADDLE_XCCL_BACKEND=intel_hpu
export FLAGS_intel_hpu_recipe_cache_num=20480
export HABANA_PROFILE=0

#no proxy using
unset http_proxy
unset https_proxy
unset no_proxy

echo "MODEL_PATH=${MODEL_PATH}"
#currently Fastdepoly PR testing is working together with PaddleCostomDevice PR testing on a same Intel HPUs Machine
#ERNIE-4.5-300B-A47B-Paddl will use all HPUS (8HPUs) and will block PaddleCostomDevice PR testing
#so let us to use ERNIE-4.5-21B-A3B-Paddle firstly, which only needs 1 HPU
FD_ATTENTION_BACKEND_NAME="HPU_ATTN"
#ERNIE-4.5-300B-A47B-Paddle (300B)
ENABLE_TESTING_ERNIE45_300B_A47B_Paddle=0
if [  $ENABLE_TESTING_ERNIE45_300B_A47B_Paddle -eq 1 ]; then
    export model_path=${MODEL_PATH}/ERNIE-4.5-300B-A47B-Paddle
    export HPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    echo "CMD Line: HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=3072 FD_ATTENTION_BACKEND=$FD_ATTENTION_BACKEND_NAME python -m fastdeploy.entrypoints.openai.api_server --model $model_path --port $FD_API_PORT --engine-worker-queue-port $FD_ENGINE_QUEUE_PORT --metrics-port $FD_METRICS_PORT --kv-cache-ratio 0.98 --num-gpu-blocks-override 3200 --tensor-parallel-size 8 --max-model-len 32786 --max-num-seqs 128 --block-size 128 --graph-optimization-config '{"use_cudagraph":false}' > server.log 2>&1 &"
    HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=3072 FD_ATTENTION_BACKEND=$FD_ATTENTION_BACKEND_NAME python -m fastdeploy.entrypoints.openai.api_server --model $model_path --port $FD_API_PORT --engine-worker-queue-port $FD_ENGINE_QUEUE_PORT --metrics-port $FD_METRICS_PORT --kv-cache-ratio 0.98 --num-gpu-blocks-override 3200 --tensor-parallel-size 8 --max-model-len 32786 --max-num-seqs 128 --block-size 128 --graph-optimization-config '{"use_cudagraph":false}' > server.log 2>&1 &
fi

#ERNIE-4.5-21B-A3B-Paddle (21B)
ENABLE_TESTING_ERNIE45_21B_A3B_Paddle=1
if [  $ENABLE_TESTING_ERNIE45_21B_A3B_Paddle -eq 1 ]; then
    export model_path=${MODEL_PATH}/ERNIE-4.5-21B-A3B-Paddle/
    export HPU_VISIBLE_DEVICES=3
    echo "CMD Line: HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=$FD_ATTENTION_BACKEND_NAME python -m fastdeploy.entrypoints.openai.api_server --model $model_path --port $FD_API_PORT --engine-worker-queue-port $FD_ENGINE_QUEUE_PORT --metrics-port $FD_METRICS_PORT --tensor-parallel-size 1 --max-model-len 32786 --max-num-seqs 128 --block-size 128 --graph-optimization-config '{"use_cudagraph":false}' > server.log 2>&1 &"
    HPU_PERF_BREAKDOWN_SYNC_MODE=1 HPU_WARMUP_BUCKET=0 HPU_WARMUP_MODEL_LEN=4096 FD_ATTENTION_BACKEND=$FD_ATTENTION_BACKEND_NAME python -m fastdeploy.entrypoints.openai.api_server --model $model_path --port $FD_API_PORT --engine-worker-queue-port $FD_ENGINE_QUEUE_PORT --metrics-port $FD_METRICS_PORT --tensor-parallel-size 1 --max-model-len 32786 --max-num-seqs 128 --block-size 128 --graph-optimization-config '{"use_cudagraph":false}' > server.log 2>&1 &
fi

sleep 60
#checking serving active status
TIMEOUT=$((60 * 60)) #60min
INTERVAL=10 #check each 10s
ENDPOINT="http://0.0.0.0:$FD_API_PORT/health"
START_TIME=$(date +%s) #start time
echo "Start to check the serving active status, waiting total ${TIMEOUT} seconds"
while true; do
    #calculate time
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))

    #to check timeout
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo -e "\nstart serving failed with timeout: $((TIMEOUT/60)) seconds"
        cat server.log
	#ERNIE-4.5-21B-A3B-Paddle only has workerlog.0
        cat log/workerlog.0
	#ERNIE-4.5-300B-A47B-Paddle (300B) will have 8 workerlog
	if [  $ENABLE_TESTING_ERNIE45_300B_A47B_Paddle -eq 1 ]; then
            cat log/workerlog.1
            cat log/workerlog.2
            cat log/workerlog.3
            cat log/workerlog.4
            cat log/workerlog.5
            cat log/workerlog.6
            cat log/workerlog.7
	fi
        exit 1
    fi

    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 2 "$ENDPOINT" || true)

    if [ "$HTTP_CODE" = "200" ]; then
        echo -e "\nserving start successfully! it costs total ${ELAPSED} seconds"
        break
    else
	echo -e "$(date +%F_%H:%M:%S) checking serving start status......"
        sleep $INTERVAL
    fi
done

cat server.log

#to do serving inference
echo "Start inference testing..."
python -m pytest tests/ci_use/HPU/run_ernie.py
exit_code=$?
echo exit_code is ${exit_code}

ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E $FD_API_PORT | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :$FD_API_PORT | xargs kill -9 || true

if [ ${exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    echo "mold testing failed, please help to do check for your PR source codeing"
    exit 1
fi

sleep 5
