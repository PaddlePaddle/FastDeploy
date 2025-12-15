#!/bin/bash
# Quick test script - builds and runs tests in one command

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
IMAGE_NAME="fastdeploy-test:latest"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "FastDeploy Test Docker Environment Setup"
echo "=========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build image if it doesn't exist
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Building Docker image (this may take a while)..."
    bash "${SCRIPT_DIR}/build_test_docker.sh"
else
    echo "Docker image already exists. Skipping build."
    echo "To rebuild: bash dockerfiles/build_test_docker.sh"
    echo ""
fi

# Run tests
echo "Running tests..."
echo "=========================================="
docker run --rm \
    -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "${IMAGE_NAME}" \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v

echo ""
echo "=========================================="
echo "Tests completed!"
echo "=========================================="

