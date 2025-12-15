# FastDeploy 测试 Docker 环境配置

## 问题说明

在运行单元测试时可能遇到 protobuf 版本兼容性问题：
```
TypeError: Descriptors cannot be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
```

## 解决方案

本项目提供了专门的测试 Docker 镜像，已配置好 protobuf 兼容环境。

## 快速开始

### 在服务器上使用

1. **克隆代码并切换到分支**
```bash
git clone https://github.com/kesmeey/FastDeploy.git
cd FastDeploy
git checkout 12
```

2. **构建 Docker 镜像**
```bash
bash dockerfiles/build_test_docker.sh
```

3. **运行测试**
```bash
bash dockerfiles/run_test.sh
```

## 详细使用方法

### 1. 构建 Docker 镜像

#### 方式一：使用脚本（推荐）
```bash
cd FastDeploy
chmod +x dockerfiles/build_test_docker.sh
bash dockerfiles/build_test_docker.sh
```

#### 方式二：手动构建
```bash
docker build -f dockerfiles/Dockerfile.test -t fastdeploy-test:latest .
```

### 2. 运行测试

#### 方式一：使用脚本（推荐）
```bash
chmod +x dockerfiles/run_test.sh
bash dockerfiles/run_test.sh
```

#### 方式二：手动运行单个测试文件
```bash
docker run --rm \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v
```

#### 方式三：进入容器交互式运行
```bash
docker run --rm -it \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    /bin/bash
```

然后在容器内运行：
```bash
# 运行单个测试文件
python -m pytest tests/spec_decode/test_mtp_proposer.py -v

# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试类
python -m pytest tests/spec_decode/test_mtp_proposer.py::TestMTPProposer -v
```

### 3. 运行所有测试

```bash
docker run --rm \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/ -v
```

### 4. 使用 GPU（如果需要）

```bash
docker run --rm --gpus all \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py -v
```

## Docker 镜像说明

- **基础镜像**: `paddlepaddle/paddleqa:cuda126-py310-cibase`
- **Python 版本**: 3.10
- **CUDA 版本**: 12.6
- **Protobuf 修复**: 
  - 安装 `protobuf>=3.20.0,<4.0.0`（兼容版本）
  - 设置环境变量 `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

## 环境变量

- `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`: 使用纯 Python 实现 protobuf（解决版本兼容问题）

## 注意事项

1. **Docker 要求**: 确保 Docker 已安装并运行
2. **磁盘空间**: 确保有足够的磁盘空间（镜像约 5-10GB）
3. **网络连接**: 首次构建需要下载基础镜像和依赖（可能需要较长时间）
4. **代码挂载**: 使用 `-v $(pwd):/workspace/FastDeploy` 将代码挂载到容器中
5. **GPU 支持**: 如果需要 GPU，添加 `--gpus all` 参数

## 故障排除

### 如果构建失败
- 检查网络连接（需要访问 Docker Hub 和 PyPI 镜像）
- 检查基础镜像是否可访问：`docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleqa:cuda126-py310-cibase`
- 检查 Docker 版本：`docker --version`（建议 >= 20.10）

### 如果测试仍然失败
- 检查代码是否正确挂载到容器：`docker run --rm -v $(pwd):/workspace/FastDeploy fastdeploy-test:latest ls /workspace/FastDeploy`
- 检查 Python 路径：`docker run --rm fastdeploy-test:latest python --version`
- 检查 protobuf 版本：`docker run --rm fastdeploy-test:latest python -c "import google.protobuf; print(google.protobuf.__version__)"`

### 如果遇到权限问题
```bash
chmod +x dockerfiles/*.sh
```

## 示例：完整测试流程

```bash
# 1. 进入项目目录
cd FastDeploy

# 2. 构建镜像（首次需要，后续可跳过）
bash dockerfiles/build_test_docker.sh

# 3. 运行测试
bash dockerfiles/run_test.sh

# 4. 查看测试覆盖率（可选）
docker run --rm \
    -v $(pwd):/workspace/FastDeploy \
    -w /workspace/FastDeploy \
    -e PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
    fastdeploy-test:latest \
    python -m pytest tests/spec_decode/test_mtp_proposer.py --cov=fastdeploy.spec_decode.mtp --cov-report=html
```

