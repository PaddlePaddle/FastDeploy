# KVTransferManager

A dedicated component for transferring KV Cache between Prefill and Decode nodes, supporting RDMA communication with ultra-low latency.

## Performance Benchmark

### KVTransferManager vs Mooncake Performance Comparison

### Test Scenario
- **Hardware Configuration**:
  - Single Mellanox ConnectX-7 400G NIC (single port)
  - Tested with BATCH_SIZE = 1538 and block size = 1K - 256K
  - Single pressure thread (threads = 1)

- **Comparison Baseline**:
  - Mooncake performance measured using transfer_engine_bench from example directory
  - Same hardware configuration and test parameters applied to KVTransferManager

### Performance Results
| Block Size | KVTransferManager | Mooncake | Performance Gain |
|------------|-----------------|----------|------------------|
| 1K         | 10.67 GB/s      | 1.54 GB/s | 6.9x |
| 2K         | 17.53 GB/s      | 3.40 GB/s | 5.2x |
| 4K         | 28.85 GB/s      | 6.95 GB/s | 4.2x |
| 8K         | 36.56 GB/s      | 12.48 GB/s | 2.9x |
| 16K        | 41.73 GB/s      | 23.42 GB/s | 1.8x |
| 32K        | 43.55 GB/s      | 31.58 GB/s | 1.4x |
| 64K        | 44.46 GB/s      | 38.39 GB/s | 1.2x |
| 128K       | 44.86 GB/s      | 40.11 GB/s | 1.1x |
| 256K       | 45.01 GB/s      | 40.71 GB/s | 1.1x |

Bandwidth Saturation Capability: Under multi-threaded high-pressure scenarios, both KVTransferManager and Mooncake can fully utilize the 400G network card bandwidth, achieving transmission performance close to the theoretical hardware limit (approximately 50 GB/s).

## Quick start

### Requirements

- Supported Architectures:
    Hopper GPUs
    Kunlun XPU
    Ampere GPUs (supported by enabling KVCACHE_GDRCOPY_FLUSH_ENABLE)
### Dependencies Installation

#### Python Packages

```bash
pip install pyzmq pybind11[global]
```

#### System Libraries (Linux)

```bash
# Ubuntu/Debian
sudo apt-get install -y libibverbs-dev librdmacm-dev

# RHEL/CentOS
sudo yum install -y libibverbs-devel librdmacm-devel
```

#### Hardware Requirements
- RDMA-capable network hardware (e.g. Mellanox NICs)
- Supported GPU architectures: Hopper, Kunlun XPU, Ampere

#### Ampere Architecture Note
To support Ampere GPUs, enable the environment variable KVCACHE_GDRCOPY_FLUSH_ENABLE.
- What it does:
    Forces memory flushing after a GDRCopy write operation to ensure data consistency on the Ampere architecture. Here if KVCACHE_GDRCOPY_FLUSH_ENABLE is enable we trigger an RDMA read operation after the last RDMA write.
- Why it’s needed:
    When the NIC delivers a completion to the CPU, it indicates that the data has reach the GPU. However, it doesn't mean that the GPU can read that data yet. To make sure the data has gone all the way down to the GPU memory and the GPU can read it, we need to perform a read.
[NCCL Issue #683](https://github.com/NVIDIA/nccl/issues/683) |
[NCCL Issue #1702](https://github.com/NVIDIA/nccl/issues/1702)
    Since the upper layer typically issues a cache arrival notification only after polling a Completion Queue Entry (CQE), this prevents the application from being notified before the data is actually written back to memory. Therefore, the potential race condition where the cache has not yet been flushed but the application assumes completion is considered a rare event in practice.
- How to enable:
    export KVCACHE_GDRCOPY_FLUSH_ENABLE=1

### Development

```bash
# Build and make symbolic links for SO files
python setup.py bdist_wheel

pip install dist/*.whl
```

## Environment Variables Configuration

### RDMA Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `KVCACHE_RDMA_GID_INDEX` | 3 | RDMA GID index |
| `KVCACHE_RDMA_NICS` | - | RDMA NIC list, comma-separated (e.g., “mlx5_0,mlx5_1”), selects ib device based on gpu_index. This environment variable must be set. NICs are selected using modulo operation on gpu_index. |
| `KVCACHE_IB_TIMEOUT` | 18 | InfiniBand communication timeout (14-31), where timeout = 4.096μs * 2^value (default 18 ≈ 1.07s).|
| `KVCACHE_RELAX_ORDERING` | false | Enable RDMA relaxed ordering to improve performance in multi-GPU scenarios. Recommended when multiple GPUs share the same NIC to mitigate TX pause issues. |

### Network Settings
| Variable | Default | Description |
|----------|---------|-------------|
| `KVCACHE_SOCKET_IFNAME` | auto | Network interface for socket comm (e.g. "eth0") |

### Debugging
| Variable | Default | Description |
|----------|---------|-------------|
| `KVCACHE_DEBUG` | false | Enable debug logging |
| `KVCACHE_DEBUG_FILE` | - | Debug log file path |
| `KVCACHE_ERROR_FILE` | - | Error log file path |

### Performance Tuning
| Variable | Default | Description |
|----------|---------|-------------|
| `KVCACHE_GDRCOPY_FLUSH_ENABLE` | false | Enable GDRCopy flush for Ampere GPUs |

# Set RDMA GID index
export KVCACHE_RDMA_GID_INDEX=3

# Set RDMA IB Device List
export KVCACHE_RDMA_NICS=mlx5_0,mlx5_1,mlx5_2

# Specify network interface
export KVCACHE_SOCKET_IFNAME=eth0

# Enable debug mode
export KVCACHE_DEBUG=1

# Set log files
export KVCACHE_DEBUG_FILE=/var/log/kvcache_debug.log
export KVCACHE_ERROR_FILE=/var/log/kvcache_error.log

## Network configurations

kvcache transfer is fully tested with RDMA over Converged Ethernet (RoCE) networks. However, it is theoretically compatible with Infiniband as well.

For complete implementation details and advanced usage, please refer to the source code.

## Python API Reference

### RDMACommunicator Class

```python
from rdma_comm import RDMACommunicator

# Constructor
comm = RDMACommunicator(
    role,           # Role ("prefill" or "decode")
    gpu_idx,        # GPU device index
    port,           # Communication port
    local_key_cache,    # List of local key cache pointers
    local_value_cache,  # List of local value cache pointers
    block_number,   # Number of blocks
    block_bytes     # Bytes per block
)

# Methods
comm.connect(dst_ip, dst_port)  # Connect to target IP and port
comm.is_connected(dst_ip, dst_port)  # Check connection status
comm.write_cache(
    ip,               # Target server IP address
    port,             # Target server port number
    local_block_ids,  # List of local block IDs to transfer
    remote_block_ids, # List of remote block IDs to write
    layer_idx         # Model layer index (for multi-layer models)
)
```

**Parameter Details**:

1. `role`:
   - "prefill": Prefill node role
   - "decode": Decode node role

2. `gpu_idx`:
   - GPU device index to use

3. `port`:
   - RDMA communication port number

4. `local_key_cache`/`local_value_cache`:
   - List of local KV cache pointers

5. `block_number`:
   - Number of cache blocks

6. `block_bytes`:
   - Bytes per cache block

**Example Usage**:

```python
import numpy as np
from rdma_comm import RDMACommunicator

# Initialize
local_keys = [np.array([0]*1024, dtype=np.int64).ctypes.data]  # Example key pointer
local_values = [np.array([0]*1024, dtype=np.int64).ctypes.data] # Example value pointer

comm = RDMACommunicator(
    role="prefill",
    gpu_idx=0,
    port="12345",
    local_key_cache=local_keys,
    local_value_cache=local_values,
    block_number=1024,
    block_bytes=4096
)

# Client connection
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
    print("Connection established")

    # Write cache
    comm.write_cache(
        ip="192.168.1.100",       # Target server IP
        port="12345",             # Target server port
        local_block_ids=[0,1,2],  # Local block IDs to transfer
        remote_block_ids=[3,4,5], # Remote block IDs to write
        layer_idx=0               # Model layer index (0 for first layer)
    )
```

## Citation

If you use this codebase, or otherwise found our work valuable, please cite:
