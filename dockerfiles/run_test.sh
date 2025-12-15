#!/bin/bash
# Run tests in Docker container

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE_NAME="fastdeploy-test:latest"

cd "${PROJECT_ROOT}"

# Check if image exists
if ! docker image inspect "${IMAGE_NAME}" > /dev/null 2>&1; then
    echo "Docker image ${IMAGE_NAME} not found. Building..."
    bash "${SCRIPT_DIR}/build_test_docker.sh"
fi

# Run tests
echo "Running tests in Docker container..."
docker run --rm \
    -v "${PROJECT_ROOT}:/workspace/FastDeploy" \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    "${IMAGE_NAME}" \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v

