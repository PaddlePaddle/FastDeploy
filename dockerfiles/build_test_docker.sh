#!/bin/bash
# Build Docker image for testing with protobuf fix

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

IMAGE_NAME="fastdeploy-test:latest"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.test"

echo "Building Docker image: ${IMAGE_NAME}"
docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" .

echo "Docker image built successfully!"
echo ""
echo "To run tests in the container:"
echo "  docker run --rm -v \$(pwd):/workspace/FastDeploy ${IMAGE_NAME} python -m pytest tests/spec_decode/test_mtp_proposer.py -v"
echo ""
echo "Or to get an interactive shell:"
echo "  docker run --rm -it -v \$(pwd):/workspace/FastDeploy ${IMAGE_NAME}"

