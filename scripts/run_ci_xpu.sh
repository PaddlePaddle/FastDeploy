#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"
export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com
#安装lsof工具
apt install -y lsof

#先kill一遍
function stop_processes() {
    ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    ps -efww | grep -E "$((8188 + XPU_ID * 100))" | grep -v grep | awk '{print $2}' | xargs kill -9 || true
    lsof -t -i :$((8188 + XPU_ID * 100)) | xargs kill -9 || true
    for port in $(seq $((8188 + XPU_ID * 100 + 10)) $((8188 + XPU_ID * 100 + 40))); do
        lsof -t -i :${port} | xargs kill -9 || true
    done
    netstat -tunlp 2>/dev/null | grep $((8190 + XPU_ID * 100)) | awk '{print $NF}' | awk -F'/' '{print $1}' | xargs -r kill -9
    netstat -tunlp 2>/dev/null | grep $((8190 + XPU_ID * 100)) | awk '{print $(NF-1)}' | cut -d/ -f1 | grep -E '^[0-9]+$' | xargs -r kill -9
}

stop_processes >kill.log 2>&1

# # 由于机器原因，需重启使用的卡，以保障没有问题
# if [[ "$XPU_ID" == "0" ]]; then
#     export XPU_VISIBLE_DEVICES="0,1,2,3"
# else
#     export XPU_VISIBLE_DEVICES="4,5,6,7"
# fi

# mkdir -p /workspace/deps
# cd /workspace/deps
# wget -q https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.21/xre-Linux-x86_64-5.0.21.21.tar.gz
# tar -zxf xre-Linux-x86_64-5.0.21.21.tar.gz && mv xre-Linux-x86_64-5.0.21.21 xre
# cd -
# export PATH=/workspace/deps/xre/bin:$PATH

# xpu-smi -r -i $XPU_VISIBLE_DEVICES
# xpu-smi

# echo "pip requirements"
# python -m pip install -r requirements.txt

# echo "uninstall org"
# python -m pip uninstall paddlepaddle-xpu -y
# python -m pip uninstall fastdeploy-xpu -y

# # python -m pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/
# # 由于ep并行报错暂时锁死paddle版本
# python -m pip install https://paddle-whl.bj.bcebos.com/nightly/xpu-p800/paddlepaddle-xpu/paddlepaddle_xpu-3.3.0.dev20251123-cp310-cp310-linux_x86_64.whl
# echo "build whl"
# bash custom_ops/xpu_ops/download_dependencies.sh develop
# export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
# export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
# bash build.sh || exit 1

# echo "pip others"
# python -m pip install openai -U
# python -m pip uninstall -y triton
# python -m pip install triton==3.3.0
# python -m pip install pytest
# python -m pip install pytest-timeout
# unset http_proxy
# unset https_proxy
# unset no_proxy

stop_processes >kill.log 2>&1

export PYTHONPATH=/work/wq/qq/FastDeploy
export XPU_VISIBLE_DEVICES="0"
python -m fastdeploy.entrypoints.openai.api_server \
--model ../../../models/ERNIE-4.5-0.3B-Paddle \
--port 8188 \
--tensor-parallel-size 1 \
--max-model-len 32768 \
--max-num-seqs 128 \
--quantization "wint8" \
--gpu-memory-utilization 0.9 \
--enable-logprob \
--max-logprobs 5