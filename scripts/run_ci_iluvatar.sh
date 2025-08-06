#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#先kill一遍
ps -efww | grep -E 'run_ernie300B_4layer' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ixsmi -r

export LD_PRELOAD=/usr/local/corex/lib64/libcuda.so.1
ln -sf /usr/local/bin/python3 /usr/local/bin/python
echo "pip requirements"
python -m pip install -r requirements_iluvatar.txt
echo "uninstall org"
python -m pip uninstall paddlepaddle -y
python -m pip uninstall paddle-iluvatar-gpu -y
python -m pip install --pre paddlepaddle -i https://www.paddlepaddle.org.cn/packages/nightly/cpu/
# TODO: Change to open access URL
# python -m pip install --pre paddle-iluvatar-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/ixuca/
wget -q https://paddle-whl.bj.bcebos.com/stable/ixuca/fastdeploy-iluvatar-gpu/fastdeploy_iluvatar_gpu-2.1.0.dev0-py3-none-any.whl
python -m pip install fastdeploy_iluvatar_gpu-2.1.0.dev0-py3-none-any.whl

# Patch, remove if image updated
cp /data1/fastdeploy/packages/cusolver.h /usr/local/lib/python3.10/site-packages/paddle/include/paddle/phi/backends/dynload/cusolver.h
echo "build whl"
bash build.sh || exit 1

unset http_proxy
unset https_proxy
unset no_proxy

rm -rf log/*
export INFERENCE_MSG_QUEUE_ID=232132
export FD_DEBUG=1
export PADDLE_XCCL_BACKEND=iluvatar_gpu
python test/ci_use/iluvatar_UT/run_ernie300B_4layer.py
exit_code=$?
echo exit_code is ${exit_code}

ps -efww | grep -E 'run_ernie300B_4layer' | grep -v grep | awk '{print $2}' | xargs kill -9 || true

if [ ${exit_code} -ne 0 ]; then
    echo "log/workerlog.0"
    cat log/workerlog.0
    exit 1
fi
