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
| `run_ha.sh` | 高可用（HA） | etcd + 多 Master 选主，杀掉 leader 后验证 failover |

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

### 场景三：高可用（HA）部署

单 Master 是单点，崩溃后集群操作会暂停。生产环境推荐使用 **etcd + 多 Master** 模式：多个 `mooncake_master` 通过 etcd 进行 leader 选举，leader 故障后由备节点自动重新选主，客户端无感切换。

**架构图：**

```
            ┌──────────────────────────────────────┐
            │            etcd 集群 (3 节点)         │
            │       leader 选举 / 元数据存储        │
            └───────────────────┬──────────────────┘
                                │ 选主 (master_view)
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ┌─────────────┐       ┌─────────────┐       ┌─────────────┐
   │  master1    │       │  master2    │       │  master3    │
   │ rpc:8081    │       │ rpc:8082    │       │ rpc:8083    │
   │ (leader)    │       │ (standby)   │       │ (standby)   │
   └──────┬──────┘       └─────────────┘       └─────────────┘
          │  FastDeploy 客户端通过 etcd 发现当前 leader
   ┌──────┴───────┐
   ▼              ▼
server_0      server_1
```

#### 前置准备

**1. 安装 etcd**

下载并解压 etcd（示例为 v3.5.30），将 `etcd` / `etcdctl` 加入 `PATH`：

```bash
ETCD_VER=v3.5.30
curl -L https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz \
  -o etcd-${ETCD_VER}-linux-amd64.tar.gz
tar -xzf etcd-${ETCD_VER}-linux-amd64.tar.gz
export PATH=$PWD/etcd-${ETCD_VER}-linux-amd64:$PATH
etcd --version
```

**2. 源码编译安装 Mooncake（支持 etcd）**

HA 模式需要 Mooncake 在编译时开启 etcd 支持（`-DSTORE_USE_ETCD=ON -DUSE_ETCD=ON`）。先安装依赖再编译：

```bash
# 下载源码
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake

# 安装系统及第三方依赖
bash dependencies.sh

# 编译 C++ 组件（含 mooncake_master，开启 etcd）
mkdir -p build && cd build
cmake .. -DSTORE_USE_ETCD=ON -DUSE_ETCD=ON
make -j
sudo make install
cd ..

# 编译并安装 Python wheel
./scripts/build_wheel.sh
pip install mooncake-wheel/dist/*.whl
```

若需要 CUDA 13 版本，使用如下方式编译安装：

```bash
cd Mooncake
export CU13_BUILD=1
./scripts/build_wheel.sh
pip install mooncake-wheel/dist/mooncake_transfer_engine_cuda13-*.whl
```

#### HA 客户端配置

HA 模式下，`metadata_server` 与 `master_server_addr` 都使用 `etcd://` 前缀指向 etcd 集群，由客户端通过 etcd 发现当前 leader（`ha_mooncake_config.json`）：

```json
{
  "metadata_server": "etcd://127.0.0.1:12379;127.0.0.1:22379;127.0.0.1:32379",
  "global_segment_size": 1000000000,
  "local_buffer_size": 134217728,
  "protocol": "rdma",
  "rdma_devices": "",
  "master_server_addr": "etcd://127.0.0.1:12379;127.0.0.1:22379;127.0.0.1:32379"
}
```

#### 一键启动与 failover 验证

单个自包含脚本 `examples/cache_storage/run_ha.sh` 负责整个流程 —— 它在脚本内部用循环分别拉起 etcd 集群和 HA master 集群，不再依赖单独的 `start_*.sh`。

直接运行：

```bash
cd examples/cache_storage
bash run_ha.sh
```

`run_ha.sh` 的执行流程：

1. **启动 etcd 集群**：端口检查后，用循环拉起 3 个 etcd 节点（client 端口 12379/22379/32379）组成 raft 集群。
2. **启动 3 个 HA Master**：用循环拉起 3 个 `mooncake_master`（rpc 8081/8082/8083，metrics 9091/9092/9093），每个都带 `--enable_ha --etcd_endpoints ... --rpc_port ...`，通过 etcd 选出一个 leader。leader 地址写入 etcd 的 `mooncake-store/mooncake_cluster/master_view`。
3. **启动 2 个 FastDeploy 实例**，均以 `--kvcache-storage-backend mooncake` 接入同一缓存池。
4. **验证池化（failover 前）**：用 prompt **A** 在 `server_0` 预热，再向 `server_1` 发送相同 prompt，应命中全局缓存。
5. **杀掉 leader**：脚本从 etcd 读取当前 leader 的 `rpc_port`，`kill -9` 对应进程，触发重新选主。
6. **验证池化（failover 后）**：等待 etcd 中 `master_view` 更新为新 leader 后，用一条**全新的** prompt **B**（failover 前从未发过）在 `server_0` 预热，再在 `server_1` 复用。使用新 prompt 可确保 `server_1` 的命中只能来自新 leader 的全局池，而非步骤 4 残留的本地缓存。

> 单独验证选主状态：
>
> ```bash
> # 查看当前 leader（rpc_address:rpc_port）
> etcdctl --endpoints=http://127.0.0.1:12379,http://127.0.0.1:22379,http://127.0.0.1:32379 \
>   get "mooncake-store/mooncake_cluster/master_view" --print-value-only
> ```
>
> 各 Master 角色可在 `log_master_1` / `log_master_2` / `log_master_3` 中查看（`role=leader` / `role=standby`），etcd 日志见 `log_etcd_1` / `log_etcd_2` / `log_etcd_3`。

#### HA Master 关键参数

| 参数 | 说明 |
|------|------|
| `--enable_ha` | 开启 HA 模式 |
| `--etcd_endpoints` | etcd 端点，分号分隔（`ha_backend_type=etcd` 时） |
| `--rpc_address` / `--rpc_port` | 该 Master 对外可达的 RPC 地址与端口（每个实例需唯一） |
| `--cluster_id` | 集群标识，同一集群的 Master 需一致 |
| `--root_fs_dir` | HA 模式下的存储根目录（每个实例独立） |

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
