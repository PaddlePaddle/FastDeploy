# KVTransferManager 中文文档

一个专为Prefill节点和Decode节点传输KV Cache的组件，支持RDMA通信。

## 性能基准测试

### KVTransferManager 与 Mooncake 性能对比

### 测试场景
- **硬件配置**:
  - 单张Mellanox ConnectX-7 400G网卡(单端口)
  - 测试参数: BATCH_SIZE = 1538, 块大小 = 1K - 256K
  - 单压力线程(threads = 1)

- **对比基准**:
  - Mooncake性能使用example目录中的transfer_engine_bench测量
  - KVTransferManager使用相同的硬件配置和测试参数

### 性能结果
| Block Size | KVTransferManager | Mooncake | 性能提升 |
|--------|-----------------|----------|----------|
| 1K     | 10.67 GB/s      | 1.54 GB/s | 6.9倍 |
| 2K     | 17.53 GB/s      | 3.40 GB/s | 5.2倍 |
| 4K     | 28.85 GB/s      | 6.95 GB/s | 4.2倍 |
| 8K     | 36.56 GB/s      | 12.48 GB/s | 2.9倍 |
| 16K    | 41.73 GB/s      | 23.42 GB/s | 1.8倍 |
| 32K    | 43.55 GB/s      | 31.58 GB/s | 1.4倍 |
| 64K    | 44.46 GB/s      | 38.39 GB/s | 1.2倍 |
| 128K   | 44.86 GB/s      | 40.11 GB/s | 1.1倍 |
| 256K   | 45.01 GB/s      | 40.71 GB/s | 1.1倍 |

在多压力线程场景下，KVTransferManager 和 Mooncake 都能够充分利用 400Gb 网卡带宽，达到接近网卡硬件理论极限的传输性能

## 快速开始

### 系统要求

- 支持的架构:
    Hopper GPU
    昆仑XPU
    Ampere GPU (需启用KVCACHE_GDRCOPY_FLUSH_ENABLE)

### 依赖安装

#### Python包

```bash
pip install pyzmq pybind11[global]
```

#### 系统库(Linux)

```bash
# Ubuntu/Debian
sudo apt-get install -y libibverbs-dev librdmacm-dev

# RHEL/CentOS
sudo yum install -y libibverbs-devel librdmacm-devel
```

#### 硬件要求
- 支持RDMA的网络硬件(如Mellanox网卡)
- 支持的GPU架构: Hopper, 昆仑XPU, Ampere

#### Ampere架构注意事项
要支持Ampere GPU，需启用环境变量KVCACHE_GDRCOPY_FLUSH_ENABLE。
- 作用:
    在GDRCopy写操作后强制内存刷新，确保Ampere架构上的数据一致性。启用后会在最后一个RDMA写操作后触发一个RDMA读操作。
- 原因:
    当网卡向CPU发送完成通知时，仅表示数据已到达GPU，但不保证GPU可以立即读取该数据。为确保数据已完全写入GPU内存且可被GPU读取，需要执行读操作。
[NCCL Issue #683](https://github.com/NVIDIA/nccl/issues/683) |
[NCCL Issue #1702](https://github.com/NVIDIA/nccl/issues/1702)
    由于上层通常只在轮询完成队列条目(CQE)后发出缓存到达通知，这避免了应用在数据实际写回内存前收到通知的情况。因此，缓存未刷新但应用认为已完成这种潜在问题在实践中被认为是罕见情况。
- 启用方式:
    export KVCACHE_GDRCOPY_FLUSH_ENABLE=1

### 开发构建

```bash
# 构建并创建SO文件的符号链接
python setup.py bdist_wheel

pip install dist/*.whl
```

## 环境变量配置

### RDMA设置
| 变量 | 默认值 | 描述 |
|------|--------|------|
| `KVCACHE_RDMA_GID_INDEX` | 3 | RDMA GID索引 |
| `KVCACHE_RDMA_NICS` | - | RDMA网卡列表，逗号分隔(如"mlx5_0,mlx5_1")，根据 gpu_index 选取ib device设备, 此环境变量必须设置, 根据gpu_index取模选取网卡 |
| `KVCACHE_IB_TIMEOUT` | 18 | InfiniBand通信超时(14-31)，超时时间=4.096μs * 2^值(默认18≈1.07秒) |
| `KVCACHE_RELAX_ORDERING` | false | 启用RDMA宽松排序以提高多GPU场景性能。当多个GPU共享同一网卡时推荐启用，可缓解TX Pause问题。 |

### 网络设置
| 变量 | 默认值 | 描述 |
|------|--------|------|
| `KVCACHE_SOCKET_IFNAME` | auto | 用于socket通信的网络接口(如"eth0")，如果不设置自动检测第一张可用网卡 |

### 调试
| 变量 | 默认值 | 描述 |
|------|--------|------|
| `KVCACHE_DEBUG` | false | 启用调试日志 |
| `KVCACHE_DEBUG_FILE` | - | 调试日志文件路径 |
| `KVCACHE_ERROR_FILE` | - | 错误日志文件路径 |

### 性能调优
| 变量 | 默认值 | 描述 |
|------|--------|------|
| `KVCACHE_GDRCOPY_FLUSH_ENABLE` | false | 为Ampere GPU启用GDRCopy刷新 |

# 设置RDMA GID索引
export KVCACHE_RDMA_GID_INDEX=3

# 设置RDMA IB设备列表
export KVCACHE_RDMA_NICS=mlx5_0,mlx5_1,mlx5_2

# 指定网络接口
export KVCACHE_SOCKET_IFNAME=eth0

# 启用调试模式
export KVCACHE_DEBUG=1

# 设置日志文件
export KVCACHE_DEBUG_FILE=/var/log/kvcache_debug.log
export KVCACHE_ERROR_FILE=/var/log/kvcache_error.log

## 网络配置

kvcache transfer已通过RDMA over Converged Ethernet (RoCE)网络全面测试。理论上也兼容Infiniband。

完整实现细节和高级用法，请参考源代码。

## Python API 接口

### RDMACommunicator 类

```python
from rdma_comm import RDMACommunicator

# 构造函数
comm = RDMACommunicator(
    role,          # 角色("prefill"或"decode")
    gpu_idx,       # GPU设备索引(0~7)
    port,          # 通信端口
    local_key_cache,   # 本地key缓存指针列表
    local_value_cache, # 本地value缓存指针列表
    block_number,   # 块数量
    block_bytes     # 每块字节数
)

# 方法说明
comm.connect(dst_ip, dst_port)  # 连接到目标IP和端口
comm.is_connected(dst_ip, dst_port)  # 检查是否已连接
comm.write_cache(
    ip,               # 目标服务器IP地址
    port,             # 目标服务器端口号
    local_block_ids,  # 本地缓存块ID列表，指定要传输的本地块
    remote_block_ids, # 远程缓存块ID列表，指定要写入的远程块
    layer_idx         # 模型层索引，用于多层模型场景
)
```

**参数说明**:

1. `role`:
   - "prefill"
   - "decode"

2. `gpu_idx`:
   - 使用的GPU设备索引

3. `port`:
   - RDMA通信端口号

4. `local_key_cache`/`local_value_cache`:
   - 本地KV缓存指针列表

5. `block_number`:
   - 缓存块数量

6. `block_bytes`:
   - 每个缓存块的字节大小

**示例代码**:

```python
import numpy as np
from rdma_comm import RDMACommunicator

# 初始化
local_keys = [np.array([0]*1024, dtype=np.int64).ctypes.data]  # 示例key指针
local_values = [np.array([0]*1024, dtype=np.int64).ctypes.data] # 示例value指针

comm = RDMACommunicator(
    role="decode",
    gpu_idx=0,
    port="12345",
    local_key_cache=local_keys,
    local_value_cache=local_values,
    block_number=1024,
    block_bytes=4096
)

# 客户端初始化
comm = RDMACommunicator(
    role="prefill",
    gpu_idx=0,
    port="12345",
    local_key_cache=local_keys,
    local_value_cache=local_values,
    block_number=1024,
    block_bytes=4096
)

if comm.connect("192.168.1.100", "12345"):
    print("连接成功")

    # 写入缓存
    comm.write_cache(
        ip="192.168.1.100",       # 目标服务器IP
        port="12345",             # 目标服务器端口
        local_block_ids=[0,1,2],  # 要传输的本地块ID列表
        remote_block_ids=[3,4,5], # 要写入的远程块ID列表
        layer_idx=0               # 模型层索引(0表示第一层)
    )
```

## 引用

如果您使用此代码库，或认为我们的工作有价值，请引用:
