#!/bin/bash
# Script to run tests on server using Docker
# Usage: bash run_test_on_server.sh [test_file]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
IMAGE_NAME="fastdeploy-test:latest"
TEST_FILE="${1:-tests/spec_decode/test_mtp_proposer.py}"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "FastDeploy Test on Server using Docker"
echo "=========================================="
echo "Test file: ${TEST_FILE}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if image exists, if not, build it
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Docker image not found. Building..."
    echo "This may take 10-20 minutes..."
    if [ -f "dockerfiles/Dockerfile.test" ]; then
        docker build -f dockerfiles/Dockerfile.test -t "${IMAGE_NAME}" .
    else
        echo "Error: Dockerfile.test not found!"
        exit 1
    fi
else
    echo "Using existing Docker image: ${IMAGE_NAME}"
fi

echo ""
echo "Running tests in Docker container..."
echo "=========================================="

# Run tests with GPU support if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected, using GPU support"
    docker run --rm --gpus all \
        -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
        -w /workspace/FastDeploy \
        -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "${IMAGE_NAME}" \
        python -m pytest "${TEST_FILE}" -v
else
    echo "No GPU detected, running in CPU mode"
    docker run --rm \
        -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
        -w /workspace/FastDeploy \
        -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
        "${IMAGE_NAME}" \
        python -m pytest "${TEST_FILE}" -v
fi

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="


