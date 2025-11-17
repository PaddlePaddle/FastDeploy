#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#安装lsof工具
# apt install -y lsof

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com 

#先kill一遍
ps -efww | grep -E 'python -u' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E 'python -m' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '/usr/bin/python' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
lsof -t -i :8111 | xargs kill -9 || true
ps -efww | grep -E 'cache_transfer_manager.py' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E 'api_server' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
ps -efww | grep -E '8188' | grep -v grep | awk '{print $2}' | xargs kill -9 || true
# arch
lsof -t -i :8188 | xargs kill -9 || true


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
# python -m pip install https://paddle-whl.bj.bcebos.com/nightly/xpu-p800/paddlepaddle-xpu/paddlepaddle_xpu-3.3.0.dev20251112-cp310-cp310-linux_x86_64.whl

# echo "build whl"
# # bash custom_ops/xpu_ops/download_dependencies.sh develop
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

# stop_processes >kill.log 2>&1

# 起服务
rm -rf log/*
rm -f core*
# pkill -9 python #流水线不执行这个
#清空消息队列
ipcrm --all=msg
echo "============================开始W4A8测试!============================"
# if [[ "$XPU_ID" == "0" ]]; then
#     export XPU_VISIBLE_DEVICES="0,1,2,3"
# else
#     export XPU_VISIBLE_DEVICES="4,5,6,7"
# fi

# export XPUAPI_DEBUG=0x1
export PYTHONPATH=/opt/output/work_dir/ssd2/yangshuang/work/yinwei/FD_FOR_KETI9
export XPU_VISIBLE_DEVICES="0,1,2,3"
export port_num=$((8188 + XPU_ID * 100))
python -m fastdeploy.entrypoints.openai.api_server \
    --model /opt/output/work_dir/ssd2/yangshuang/work/PaddlePaddle/ERNIE-4.5-300B-A47B-W4A8C8-TP4-Paddle \
    --port $port_num \
    --engine-worker-queue-port $((port_num + 1)) \
    --metrics-port $((port_num + 2)) \
    --cache-queue-port $((port_num + 47873)) \
    --tensor-parallel-size 4 \
    --num-gpu-blocks-override 16384 \
    --max-model-len 32768 \
    --max-num-seqs 64 \
    --quantization "W4A8" 
    # --graph-optimization-config '{"use_cudagraph":true,  "use_unique_memory_pool":true}'
