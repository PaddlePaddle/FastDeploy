#!/bin/bash
# Script to test test_mtp_proposer.py on server
# Usage: bash test_on_server.sh

set -e

echo "=========================================="
echo "Testing test_mtp_proposer.py on Server"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "tests/spec_decode/test_mtp_proposer.py" ]; then
    echo "Error: test_mtp_proposer.py not found!"
    echo "Please run this script from the FastDeploy root directory"
    exit 1
fi

# Set protobuf environment variable to fix compatibility issues
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

echo "1. Checking Python environment..."
python --version
echo ""

echo "2. Checking required packages..."
python -c "import paddle; print(f'PaddlePaddle version: {paddle.__version__}')" || echo "Warning: PaddlePaddle not found"
python -c "import pytest; print(f'pytest version: {pytest.__version__}')" || echo "Error: pytest not installed"
echo ""

echo "3. Running code style check..."
python -m flake8 tests/spec_decode/test_mtp_proposer.py --max-line-length=120 --ignore=E501,W503 || echo "Warning: flake8 check failed"
echo ""

echo "4. Running unit tests..."
python -m pytest tests/spec_decode/test_mtp_proposer.py -v

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="

