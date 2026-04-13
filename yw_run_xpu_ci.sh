#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "$DIR"

#安装ci必要工具
apt install -y lsof
apt-get install -y iproute2

export http_proxy=http://agent.baidu.com:8891
export https_proxy=http://agent.baidu.com:8891
export no_proxy=localhost,bj.bcebos.com,su.bcebos.com,pypi.tuna.tsinghua.edu.cn,paddle-ci.gz.bcebos.com 


export PATH=/workspace/deps/xre/bin:$PATH

# xpu-smi -r -i $XPU_VISIBLE_DEVICES
xpu-smi
# python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# echo "pip requirements"
# python -m pip install -r requirements.txt

echo "uninstall org"


# python -m pip install paddlepaddle-xpu -i https://www.paddlepaddle.org.cn/packages/nightly/xpu-p800/ --force-reinstall
# 由于ep并行报错暂时锁死paddle版本
# python -m pip install https://paddle-whl.bj.bcebos.com/nightly/xpu-p800/paddlepaddle-xpu/paddlepaddle_xpu-3.3.0.dev20251123-cp310-cp310-linux_x86_64.whl
# python -m pip install https://paddle-qa.bj.bcebos.com/paddle-pipeline/Release-TagBuild-Training-Linux-Xpu-P800-SelfBuiltPypiUse/latest/paddlepaddle_xpu-0.0.0-cp310-cp310-linux_x86_64.whl --force-reinstall
echo "build whl"
# bash custom_ops/xpu_ops/download_dependencies.sh develop
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=/opt/output/work_dir/ssd1/yinwei06/FastDeploy_debug_paddle3.3/output
# export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
bash build.sh 


# || exit 1

# echo "pip others"
# python -m pip install openai -U
# python -m pip uninstall -y triton
# python -m pip install triton==3.3.0
# python -m pip install pytest
# python -m pip install pytest-timeout
# unset http_proxy
# unset https_proxy
# unset no_proxy


