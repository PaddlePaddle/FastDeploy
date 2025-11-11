#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#先kill一遍
ps -efww | grep -E 'run_ernie300B_4layer' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ixsmi -r

unset http_proxy
unset https_proxy
unset no_proxy

export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
ln -sf /usr/local/bin/python3 /usr/local/bin/python
echo "pip requirements"
python -m pip install -r requirements_iluvatar.txt
echo "install paddle cpu and custom device"
python -m pip install paddlepaddle==3.3.0.dev20251028 -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
python -m pip install paddle-iluvatar-gpu==3.0.0.dev20251029 -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
echo "build whl"
bash build.sh || exit 1

CI_PATH=tests/ci_use/iluvatar_UT
export INFERENCE_MSG_QUEUE_ID=232132
export FD_DEBUG=1
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
        echo "log/workerlog.0"
        cat log/workerlog.0
        exit 1
    fi
done
