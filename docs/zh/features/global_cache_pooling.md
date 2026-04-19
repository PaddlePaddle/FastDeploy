[English](../../features/global_cache_pooling.md) | 中文文档

# 全局缓存池化

本文档介绍如何将 MooncakeStore 作为 FastDeploy 的 KV Cache 存储后端，实现多推理实例间的**全局缓存池化**。

## 概述

### 什么是全局缓存池化？

全局缓存池化允许多个 FastDeploy 实例通过分布式存储层共享 KV Cache，具有以下优势：

- **跨实例缓存复用**：一个实例计算的 KV Cache 可被其他实例复用
- **PD 分离架构优化**：Prefill 和 Decode 实例可无缝共享缓存
- **减少重复计算**：避免跨请求的重复前缀计算

### 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mooncake Master 服务                         │
│              (元数据与协调服务)                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  FastDeploy     │ │  FastDeploy     │ │  FastDeploy     │
│  Instance P     │ │  Instance D     │ │  Instance X     │
│  (Prefill)      │ │  (Decode)       │ │  (Standalone)   │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │  MooncakeStore  │
                    │  (共享 KV       │
                    │   Cache 池)     │
                    └─────────────────┘
```

## 示例脚本

开箱即用的示例脚本位于 [examples/cache_storage/](../../../examples/cache_storage/)。

| 脚本 | 场景 | 说明 |
|------|------|------|
| `run.sh` | 多实例缓存共享 | 两个独立实例共享缓存 |
| `run_03b_pd_storage.sh` | PD 分离 | P+D 实例配合全局缓存池 |

## 环境要求

### 硬件要求

- 支持 CUDA 的 NVIDIA GPU
- RDMA 网络（生产环境推荐）或 TCP 网络

### 软件要求

- Python 3.8+
- CUDA 11.8+
- FastDeploy（见下方安装说明）

## 安装步骤

参考 [NVIDIA CUDA GPU 安装指南](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/) 安装 FastDeploy。

```bash
# 方式一：从 PyPI 安装
pip install fastdeploy-gpu

# 方式二：从源码编译
bash build.sh
pip install ./dist/fastdeploy*.whl
```

安装FastDeploy后自动安装了MooncakeStore。

## 配置说明

我们支持两种方式配置MooncakeStore，一是通过配置文件`mooncake_config.json`，二是通过环境变量进行配置。

### Mooncake 配置文件

创建 `mooncake_config.json` 配置文件：

```json
{
    "metadata_server": "http://0.0.0.0:15002/metadata",
    "master_server_addr": "0.0.0.0:15001",
    "global_segment_size": 1000000000,
    "local_buffer_size": 1048576,
    "protocol": "rdma",
    "rdma_devices": ""
}
```

设置MOONCAKE_CONFIG_PATH环境变量后，配置文件生效：
```bash
export MOONCAKE_CONFIG_PATH=path/to/mooncake_config.json
```

配置参数说明：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `metadata_server` | HTTP 元数据服务地址 | 必填 |
| `master_server_addr` | Master 服务地址 | 必填 |
| `global_segment_size` | 每个TP进程给全局共享内存共享的内存空间（字节） | 1GB |
| `local_buffer_size` | 数据传输本地缓冲区大小（字节） | 128MB |
| `protocol` | 传输协议：`rdma` 或 `tcp` | `rdma` |
| `rdma_devices` | RDMA 设备名称（逗号分隔） | 自动检测 |

### 环境变量配置

Mooncake 也支持通过环境变量进行配置：

| 环境变量 | 说明 |
|----------|------|
| `MOONCAKE_MASTER_SERVER_ADDR` | Master 服务地址（如 `10.0.0.1:15001`） |
| `MOONCAKE_METADATA_SERVER` | 元数据服务 URL |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | 每个TP进程给全局共享内存共享的内存空间（字节） |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | 本地缓冲区大小（字节） |
| `MOONCAKE_PROTOCOL` | 传输协议（`rdma` 或 `tcp`） |
| `MOONCAKE_RDMA_DEVICES` | RDMA 设备名称 |

## 使用场景

### 场景一：多实例缓存共享

运行多个 FastDeploy 实例，共享全局 KV Cache 池。

**步骤 1：启动 Mooncake Master**

```bash
mooncake_master \
    --port=15001 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=15002 \
    --metrics_port=15003
```

**步骤 2：启动 FastDeploy 实例**

实例 0：
```bash
export MOONCAKE_CONFIG_PATH="./mooncake_config.json"
export CUDA_VISIBLE_DEVICES=0

python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 52700 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake
```

实例 1：
```bash
export MOONCAKE_CONFIG_PATH="./mooncake_config.json"
export CUDA_VISIBLE_DEVICES=1

python -m fastdeploy.entrypoints.openai.api_server \
       --model ${MODEL_NAME} \
       --port 52800 \
       --max-model-len 32768 \
       --max-num-seqs 32 \
       --kvcache-storage-backend mooncake
```

**步骤 3：测试缓存复用**

向两个实例发送相同的 prompt，第二个实例应能复用第一个实例计算的 KV Cache。

```bash
# 请求实例 0
curl -X POST "http://0.0.0.0:52700/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, world!"}], "max_tokens": 50}'

# 请求实例 1（应命中缓存）
curl -X POST "http://0.0.0.0:52800/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, world!"}], "max_tokens": 50}'
```

### 场景二：PD 分离 + 全局缓存池

此场景将 **PD 分离架构** 与 **全局缓存池化** 结合，实现：

- Prefill 实例可读取 Decode 实例的输出缓存
- 优化多轮对话性能

**架构图：**

```
         ┌──────────────────────────────────────────┐
         │              Router                       │
         │           (负载均衡器)                    │
         └─────────────────┬────────────────────────┘
                           │
           ┌───────────────┴───────────────┐
           │                               │
           ▼                               ▼
    ┌─────────────┐                 ┌─────────────┐
    │   Prefill   │                 │   Decode    │
    │  Instance   │◄───────────────►│  Instance   │
    │             │   KV Transfer   │             │
    └──────┬──────┘                 └──────┬──────┘
           │                               │
           └───────────────┬───────────────┘
                           │
                  ┌────────▼────────┐
                  │  MooncakeStore  │
                  │  (全局缓存池)   │
                  └─────────────────┘
```

**步骤 1：启动 Mooncake Master**

```bash
mooncake_master \
    --port=15001 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=15002
```

**步骤 2：启动 Router**

```bash
python -m fastdeploy.router.launch \
    --port 52700 \
    --splitwise
```

**步骤 3：启动 Prefill 实例**

```bash
export MOONCAKE_MASTER_SERVER_ADDR="127.0.0.1:15001"
export MOONCAKE_METADATA_SERVER="http://127.0.0.1:15002/metadata"
export MOONCAKE_PROTOCOL="rdma"
export CUDA_VISIBLE_DEVICES=0

python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port 52400 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --splitwise-role prefill \
    --cache-transfer-protocol rdma \
    --router "0.0.0.0:52700" \
    --kvcache-storage-backend mooncake
```

**步骤 4：启动 Decode 实例**

```bash
export MOONCAKE_MASTER_SERVER_ADDR="127.0.0.1:15001"
export MOONCAKE_METADATA_SERVER="http://127.0.0.1:15002/metadata"
export MOONCAKE_PROTOCOL="rdma"
export CUDA_VISIBLE_DEVICES=1

python -m fastdeploy.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --port 52500 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --splitwise-role decode \
    --cache-transfer-protocol rdma \
    --router "0.0.0.0:52700" \
    --enable-output-caching \
    --kvcache-storage-backend mooncake
```

**步骤 5：通过 Router 发送请求**

```bash
curl -X POST "http://0.0.0.0:52700/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## FastDeploy Mooncake 相关参数

| 参数 | 说明 |
|------|------|
| `--kvcache-storage-backend mooncake` | 启用 Mooncake 作为 KV Cache 存储后端 |
| `--enable-output-caching` | 启用输出 token 缓存（推荐 Decode 实例开启） |
| `--cache-transfer-protocol rdma` | P 和 D 之间使用 RDMA 进行 KV 传输 |
| `--splitwise-role prefill/decode` | 设置实例在 PD 分离中的角色 |
| `--router` | PD 分离场景下的 Router 地址 |

## 验证方法

### 检查缓存命中

通过日志验证缓存命中情况：

```bash
# 多实例场景
grep -E "storage_cache_token_num" log_*/api_server.log

# PD 分离场景
grep -E "storage_cache_token_num" log_prefill/api_server.log
```

如果 `storage_cache_token_num > 0`，表示实例成功从全局池读取了缓存的 KV 块。

### 监控 Mooncake Master

```bash
# 检查 master 状态
curl http://localhost:15002/metadata

# 检查指标（如配置了 metrics_port）
curl http://localhost:15003/metrics
```

## 故障排查

### 常见问题

**1. 端口被占用**

```bash
# 检查端口使用情况
ss -ltn | grep 15001

# 终止占用进程
kill -9 $(lsof -t -i:15001)
```

**2. RDMA 连接失败**

- 检查 RDMA 设备：`ibv_devices`
- 检查 RDMA 网络：`ibv_devinfo`
- 降级使用 TCP：设置 `MOONCAKE_PROTOCOL=tcp`

**3. 缓存未共享**

- 确认所有实例连接到同一个 Mooncake master
- 检查元数据服务 URL 是否一致
- 确认 `global_segment_size` 足够大

**4. /dev/shm 权限不足**

```bash
# 清理残留的共享内存文件
find /dev/shm -type f -print0 | xargs -0 rm -f
```

### 调试模式

开启调试日志：

```bash
export FD_DEBUG=1
```

## 更多资源

- [Mooncake 官方文档](https://github.com/kvcache-ai/Mooncake)
- [Mooncake 故障排查指南](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/troubleshooting.md)
- [FastDeploy 文档](https://paddlepaddle.github.io/FastDeploy/)
