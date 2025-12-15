#!/bin/bash
# 在服务器上执行的完整测试脚本
# 使用方法：复制这个脚本到服务器，然后执行 bash server_test_commands.sh

set -e

echo "=========================================="
echo "FastDeploy 服务器测试脚本"
echo "=========================================="

# 检查是否在正确的目录
if [ ! -f "dockerfiles/Dockerfile.test" ]; then
    echo "错误: 请在 FastDeploy 项目根目录执行此脚本"
    exit 1
fi

PROJECT_ROOT=$(pwd)
IMAGE_NAME="fastdeploy-test:latest"
TEST_FILE="tests/spec_decode/test_mtp_proposer.py"

echo "项目目录: ${PROJECT_ROOT}"
echo "测试文件: ${TEST_FILE}"
echo ""

# 检查 Docker
if ! command -v docker &> /dev/null; then
    echo "错误: 未安装 Docker"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "错误: Docker 未运行，请启动 Docker"
    exit 1
fi

# 检查并构建镜像
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Docker 镜像不存在，开始构建..."
    echo "这可能需要 10-20 分钟，请耐心等待..."
    docker build -f dockerfiles/Dockerfile.test -t "${IMAGE_NAME}" .
    echo "镜像构建完成！"
else
    echo "使用现有镜像: ${IMAGE_NAME}"
fi

echo ""
echo "开始运行测试..."
echo "=========================================="

# 检查是否有GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "检测到 GPU，使用 GPU 模式"
    docker run --rm --gpus all \
        -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
        -w /workspace/FastDeploy \
        -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "${IMAGE_NAME}" \
        python -m pytest "${TEST_FILE}" -v
else
    echo "未检测到 GPU，使用 CPU 模式"
    docker run --rm \
        -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
        -w /workspace/FastDeploy \
        -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "${IMAGE_NAME}" \
        python -m pytest "${TEST_FILE}" -v
fi

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="


