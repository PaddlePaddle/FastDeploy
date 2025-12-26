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
mkdir -p /model_data/
cp -r /aistudio/paddle_ci/ERNIE_300B_4L/ /model_data/
cp -r /aistudio/paddle_ci/ERNIE-4.5-VL-28B-A3B-Paddle /model_data/
echo "build whl"
bash build.sh || exit 1

CI_PATH=tests/ci_use/iluvatar_UT
export INFERENCE_MSG_QUEUE_ID=232132
export PADDLE_XCCL_BACKEND=iluvatar_gpu
export FD_SAMPLING_CLASS=rejection

ci_list=(
    ${CI_PATH}/run_ernie300B_4layer.py
    ${CI_PATH}/run_ernie_vl_28B.py
)
echo "test ci files: ${ci_list[@]}"
for cur_test_file in ${ci_list[@]}
do
    echo "============ start to test ${cur_test_file} ==========="
    rm -rf log/*
    python ${cur_test_file}
    exit_code=$?
    echo exit_code is ${exit_code}

    ps -efww | grep -E '${cur_test_file}' | grep -v grep | awk '{print $2}' | xargs kill -9 || true

    if [ ${exit_code} -ne 0 ]; then
        if [ ! -f "./log/workerlog.0" ]; then
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
        exit 1
    fi
done
