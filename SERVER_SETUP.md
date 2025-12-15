# 在服务器上运行测试的步骤

## 快速开始

### 1. SSH 连接到服务器

```bash
ssh hj@10.26.43.21
```

### 2. 进入项目目录并切换到12分支

```bash
cd ~/FastDeploy  # 或你的实际项目路径
git checkout 12
git pull  # 确保代码是最新的
```

### 3. 运行测试脚本

```bash
# 方法1: 使用我创建的脚本（推荐）
chmod +x server_test_commands.sh
bash server_test_commands.sh

# 方法2: 使用现有的脚本
chmod +x run_test_on_server.sh
bash run_test_on_server.sh

# 方法3: 使用 dockerfiles 目录下的脚本
chmod +x dockerfiles/run_test.sh
bash dockerfiles/run_test.sh
```

### 4. 或者手动执行 Docker 命令

```bash
# 首次运行需要构建镜像（约10-20分钟）
docker build -f dockerfiles/Dockerfile.test -t fastdeploy-test:latest .

# 运行测试
docker run --rm --gpus all \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v
```

## 交互式调试

如果需要进入容器调试：

```bash
docker run --rm -it --gpus all \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    /bin/bash

# 在容器内
python -m pytest tests/spec_decode/test_mtp_proposer.py -v
# 或运行单个测试
python -m pytest tests/spec_decode/test_mtp_proposer.py::TestMTPProposer::test_cache_type_and_empty_cache -v
```

## 常见问题

### Docker 镜像已存在，想重新构建

```bash
docker rmi fastdeploy-test:latest
docker build -f dockerfiles/Dockerfile.test -t fastdeploy-test:latest .
```

### 只运行特定的测试方法

```bash
docker run --rm --gpus all \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py::TestMTPProposer::test_cache_type_and_empty_cache -v
```

### 查看测试覆盖率

```bash
docker run --rm --gpus all \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py --cov=fastdeploy.spec_decode.mtp -v
```


