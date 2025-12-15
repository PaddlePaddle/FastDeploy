#!/bin/bash
# Script to test test_mtp_proposer.py on server using Docker
# Usage: bash test_on_server_docker.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
IMAGE_NAME="fastdeploy-test:latest"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "Testing test_mtp_proposer.py using Docker"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Check if image exists, if not, build it
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Docker image not found. Building..."
    if [ -f "dockerfiles/Dockerfile.test" ]; then
        docker build -f dockerfiles/Dockerfile.test -t "${IMAGE_NAME}" .
    else
        echo "Error: Dockerfile.test not found!"
        echo "Please ensure dockerfiles/Dockerfile.test exists"
        exit 1
    fi
else
    echo "Using existing Docker image: ${IMAGE_NAME}"
fi

echo ""
echo "Running tests in Docker container..."
echo "=========================================="

docker run --rm \
    -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "${IMAGE_NAME}" \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="

