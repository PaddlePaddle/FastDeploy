#!/bin/bash
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# XPU CI测试入口脚本 - 基于pytest框架
#
# 使用方法:
#   bash scripts/run_xpu_ci_pytest.sh
#
# 环境变量:
#   XPU_ID: XPU设备ID(0或1)
#   MODEL_PATH: 模型路径

set +e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "脚本目录: $DIR"

# ============ 环境准备阶段 ============

echo "============================环境准备============================"

# 安装lsof工具
echo "安装lsof工具..."
apt install -y lsof

# 设置XPU_VISIBLE_DEVICES
if [[ "$XPU_ID" == "0" ]]; then
    export XPU_VISIBLE_DEVICES="0,1,2,3"
else
    export XPU_VISIBLE_DEVICES="4,5,6,7"
fi
echo "XPU_VISIBLE_DEVICES=$XPU_VISIBLE_DEVICES"

# 下载和安装xre
echo "下载和安装xre..."
mkdir -p /workspace/deps
cd /workspace/deps
if [ ! -d "xre" ]; then
    wget -q https://klx-sdk-release-public.su.bcebos.com/xre/kl3-release/5.0.21.21/xre-Linux-x86_64-5.0.21.21.tar.gz
    tar -zxf xre-Linux-x86_64-5.0.21.21.tar.gz && mv xre-Linux-x86_64-5.0.21.21 xre
fi
cd -
export PATH=/workspace/deps/xre/bin:$PATH

# 重启XPU卡
echo "重启XPU卡..."
xpu-smi -r -i $XPU_VISIBLE_DEVICES
xpu-smi
set -e
# ============ Python环境配置 ============

echo "============================Python环境配置============================"

# 安装Python依赖
echo "安装Python依赖..."
python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -r requirements.txt

# 卸载旧版本
echo "卸载旧版本..."
python -m pip uninstall paddlepaddle-xpu -y
python -m pip uninstall fastdeploy-xpu -y

# 安装PaddlePaddle Release分支安装对应的paddle
echo "安装release分支PaddlePaddle..."
python -m pip install paddlepaddle-xpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/xpu-p800/

# ============ 编译项目 ============

echo "============================编译项目============================"
bash custom_ops/xpu_ops/download_dependencies.sh stable
export CLANG_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xtdk
export XVLLM_PATH=$(pwd)/custom_ops/xpu_ops/third_party/xvllm
bash build.sh || exit 1

# ============ 安装测试依赖 ============

echo "============================安装测试依赖============================"
python -m pip install openai -U
python -m pip uninstall -y triton
python -m pip install triton==3.3.0
python -m pip install pytest
python -m pip install pytest-timeout

# 清除代理设置
unset http_proxy
unset https_proxy
unset no_proxy

# ============ 运行pytest测试 ============

echo "============================开始运行pytest测试============================"

# 切换到项目根目录(如果不在的话)
cd "$(dirname "$DIR")"

# 运行pytest
# -v: 详细输出
# -s: 不捕获输出,直接显示print内容
# --tb=short: 简短的traceback
# --junit-xml: 生成junit格式的测试报告
python -m pytest -v -s --tb=short tests/xpu_ci/

# 获取pytest退出码
exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo "============================所有测试通过!============================"
else
    echo "============================测试失败,请检查日志!============================"
    exit $exit_code
fi
