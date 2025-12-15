# 在服务器上测试 test_mtp_proposer.py

## 方法一：直接运行（推荐，如果环境已配置好）

### 1. 克隆代码并切换到分支

```bash
git clone https://github.com/kesmeey/FastDeploy.git
cd FastDeploy
git checkout 12
```

### 2. 设置环境变量（解决 protobuf 问题）

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

### 3. 运行测试

```bash
# 使用脚本（推荐）
chmod +x test_on_server.sh
bash test_on_server.sh

# 或直接运行
python -m pytest tests/spec_decode/test_mtp_proposer.py -v
```

### 4. 如果遇到 protobuf 错误

```bash
# 安装兼容版本的 protobuf
pip install "protobuf>=3.20.0,<4.0.0"

# 或设置环境变量
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

## 方法二：使用 Docker（推荐，环境隔离）

### 1. 克隆代码

```bash
git clone https://github.com/kesmeey/FastDeploy.git
cd FastDeploy
git checkout 12
```

### 2. 使用 Docker 测试脚本

```bash
chmod +x test_on_server_docker.sh
bash test_on_server_docker.sh
```

### 3. 或手动使用 Docker

```bash
# 构建镜像（如果 dockerfiles/Dockerfile.test 存在）
docker build -f dockerfiles/Dockerfile.test -t fastdeploy-test:latest .

# 运行测试
docker run --rm \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v
```

## 方法三：快速测试命令

```bash
# 设置环境变量并运行
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/spec_decode/test_mtp_proposer.py -v

# 查看测试覆盖率
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python -m pytest tests/spec_decode/test_mtp_proposer.py --cov=fastdeploy.spec_decode.mtp -v
```

## 常见问题

### 1. protobuf 版本错误

**错误信息：**
```
TypeError: Descriptors cannot be created directly.
```

**解决方法：**
```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
pip install "protobuf>=3.20.0,<4.0.0"
```

### 2. 缺少依赖

```bash
pip install pytest pytest-asyncio
```

### 3. PaddlePaddle 未安装

根据服务器环境安装对应版本的 PaddlePaddle。

## 测试输出说明

- `-v`: 详细输出，显示每个测试用例
- `-s`: 显示 print 输出
- `--tb=short`: 简短的错误追踪信息

示例：
```bash
python -m pytest tests/spec_decode/test_mtp_proposer.py -v -s
```

