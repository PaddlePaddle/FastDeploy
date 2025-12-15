#!/bin/bash
# Deploy code and run tests on server
# Usage: bash deploy_and_test.sh

set -e

SERVER="hj@10.26.43.21"
PROJECT_DIR="~/FastDeploy"
BRANCH="12"
REPO_URL="https://github.com/kesmeey/FastDeploy.git"

echo "=========================================="
echo "Deploying and Testing on Server"
echo "=========================================="
echo "Server: $SERVER"
echo "Branch: $BRANCH"
echo ""

# Create deployment script to run on server
cat > /tmp/deploy_test_remote.sh << 'REMOTE_SCRIPT'
#!/bin/bash
set -e

PROJECT_DIR="${1:-~/FastDeploy}"
BRANCH="${2:-12}"
REPO_URL="${3:-https://github.com/kesmeey/FastDeploy.git}"

echo "=========================================="
echo "Deploying FastDeploy Test"
echo "=========================================="
echo "Project Dir: $PROJECT_DIR"
echo "Branch: $BRANCH"
echo ""

# Navigate to project directory or clone
if [ -d "$PROJECT_DIR" ]; then
    echo "Project directory exists, updating..."
    cd "$PROJECT_DIR"
    git fetch origin
    git checkout "$BRANCH" || git checkout -b "$BRANCH" origin/"$BRANCH"
    git pull origin "$BRANCH"
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    git checkout "$BRANCH"
fi

echo ""
echo "Current commit:"
git log --oneline -1

echo ""
echo "=========================================="
echo "Setting up environment"
echo "=========================================="

# Set protobuf environment variable
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Check Python version
echo "Python version:"
python --version || python3 --version

# Check if protobuf needs to be fixed
echo ""
echo "Checking protobuf..."
python -c "import google.protobuf; print(f'protobuf version: {google.protobuf.__version__}')" 2>/dev/null || {
    echo "Installing compatible protobuf version..."
    pip install "protobuf>=3.20.0,<4.0.0" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple || \
    pip3 install "protobuf>=3.20.0,<4.0.0" -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
}

# Check pytest
echo ""
echo "Checking pytest..."
python -m pytest --version 2>/dev/null || {
    echo "Installing pytest..."
    pip install pytest pytest-asyncio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple || \
    pip3 install pytest pytest-asyncio -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
}

echo ""
echo "=========================================="
echo "Running tests"
echo "=========================================="

# Run tests
cd "$PROJECT_DIR"
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/spec_decode/test_mtp_proposer.py -v

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
REMOTE_SCRIPT

# Copy script to server and execute
echo "Copying deployment script to server..."
scp /tmp/deploy_test_remote.sh "$SERVER:/tmp/deploy_test_remote.sh"

echo ""
echo "Executing deployment on server..."
ssh "$SERVER" "bash /tmp/deploy_test_remote.sh $PROJECT_DIR $BRANCH $REPO_URL"

# Cleanup
rm -f /tmp/deploy_test_remote.sh

echo ""
echo "=========================================="
echo "Deployment and testing completed!"
echo "=========================================="

