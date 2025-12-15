#!/bin/bash
# Deploy and test on server hj@10.26.43.21
# Usage: bash deploy_to_server.sh

set -e

SERVER="hj@10.26.43.21"
PROJECT_DIR="~/FastDeploy"
BRANCH="12"
REPO_URL="https://github.com/kesmeey/FastDeploy.git"

echo "=========================================="
echo "Deploying to Server: $SERVER"
echo "=========================================="
echo ""

# Execute commands on remote server
ssh "$SERVER" << 'REMOTE_SCRIPT'
set -e

PROJECT_DIR=~/FastDeploy
BRANCH=12
REPO_URL="https://github.com/kesmeey/FastDeploy.git"

echo "1. Checking existing directories..."
ls -la ~ | grep -i fastdeploy || echo "No FastDeploy directory found"

echo ""
echo "2. Cloning or updating repository..."
if [ -d "$PROJECT_DIR" ]; then
    echo "   Directory exists, updating..."
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout $BRANCH 2>/dev/null || git checkout -b $BRANCH origin/$BRANCH
    git pull origin $BRANCH || echo "Pull failed, trying reset..."
    git reset --hard origin/$BRANCH
else
    echo "   Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    git checkout $BRANCH
fi

echo ""
echo "3. Current commit:"
cd "$PROJECT_DIR"
git log --oneline -1
git status

echo ""
echo "4. Setting up environment..."
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Check Python
echo "   Python version:"
python --version 2>/dev/null || python3 --version

# Check and install protobuf if needed
echo ""
echo "5. Checking dependencies..."
python -c "import google.protobuf; print('   protobuf:', google.protobuf.__version__)" 2>/dev/null || {
    echo "   Installing protobuf..."
    pip install "protobuf>=3.20.0,<4.0.0" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 2>/dev/null || \
    pip3 install "protobuf>=3.20.0,<4.0.0" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
}

# Check pytest
python -m pytest --version 2>/dev/null || {
    echo "   Installing pytest..."
    pip install pytest pytest-asyncio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple 2>/dev/null || \
    pip3 install pytest pytest-asyncio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
}

echo ""
echo "6. Verifying test file exists..."
if [ -f "$PROJECT_DIR/tests/spec_decode/test_mtp_proposer.py" ]; then
    echo "   ✓ Test file found"
    ls -lh "$PROJECT_DIR/tests/spec_decode/test_mtp_proposer.py"
else
    echo "   ✗ Test file not found!"
    echo "   Looking for test files..."
    find "$PROJECT_DIR" -name "test_mtp_proposer.py" 2>/dev/null || echo "   No test file found"
    exit 1
fi

echo ""
echo "=========================================="
echo "7. Running tests..."
echo "=========================================="
cd "$PROJECT_DIR"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/spec_decode/test_mtp_proposer.py -v

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
REMOTE_SCRIPT

echo ""
echo "Deployment and testing completed!"

