[中文文档](../zh/features/global_cache_pooling.md) | English

# Global Cache Pooling

This document describes how to use MooncakeStore as the KV Cache storage backend for FastDeploy, enabling **Global Cache Pooling** across multiple inference instances.

## Overview

### What is Global Cache Pooling?

Global Cache Pooling allows multiple FastDeploy instances to share KV Cache through a distributed storage layer. This enables:

- **Cross-instance cache reuse**: KV Cache computed by one instance can be reused by another
- **PD Disaggregation optimization**: Prefill and Decode instances can share cache seamlessly
- **Reduced computation**: Avoid redundant prefix computation across requests

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Mooncake Master Server                       │
│              (Metadata & Coordination Service)                   │
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
                    │  (Shared KV     │
                    │   Cache Pool)   │
                    └─────────────────┘
```

## Example Scripts

Ready-to-use example scripts are available in [examples/cache_storage/](../../../examples/cache_storage/).

| Script | Scenario | Description |
|--------|----------|-------------|
| `run.sh` | Multi-Instance | Two standalone instances sharing cache |
| `run_03b_pd_storage.sh` | PD Disaggregation | P+D instances with global cache pooling |

## Prerequisites

### Hardware Requirements

- NVIDIA GPU with CUDA support
- RDMA network (recommended for production) or TCP

### Software Requirements

- Python 3.8+
- CUDA 11.8+
- FastDeploy (see installation below)

## Installation

Refer to [NVIDIA CUDA GPU Installation](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/) for FastDeploy installation.

```bash
# Option 1: Install from PyPI
pip install fastdeploy-gpu

# Option 2: Build from source
bash build.sh
pip install ./dist/fastdeploy*.whl
```

MooncakeStore is automatically installed when you install FastDeploy.

## Configuration

We support two ways to configure MooncakeStore: via configuration file `mooncake_config.json` or via environment variables.

### Mooncake Configuration File

Create a `mooncake_config.json` file:

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

Set the `MOONCAKE_CONFIG_PATH` environment variable to enable the configuration:

```bash
export MOONCAKE_CONFIG_PATH=path/to/mooncake_config.json
```

Configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `metadata_server` | HTTP metadata server URL | Required |
| `master_server_addr` | Master server address | Required |
| `global_segment_size` | Memory space each TP process shares to global shared memory (bytes) | 1GB |
| `local_buffer_size` | Local buffer size for data transfer (bytes) | 128MB |
| `protocol` | Transfer protocol: `rdma` or `tcp` | `rdma` |
| `rdma_devices` | RDMA device names (comma-separated) | Auto-detect |

### Environment Variables

Mooncake can also be configured via environment variables:

| Variable | Description |
|----------|-------------|
| `MOONCAKE_MASTER_SERVER_ADDR` | Master server address (e.g., `10.0.0.1:15001`) |
| `MOONCAKE_METADATA_SERVER` | Metadata server URL |
| `MOONCAKE_GLOBAL_SEGMENT_SIZE` | Memory space each TP process shares to global shared memory (bytes) |
| `MOONCAKE_LOCAL_BUFFER_SIZE` | Local buffer size (bytes) |
| `MOONCAKE_PROTOCOL` | Transfer protocol (`rdma` or `tcp`) |
| `MOONCAKE_RDMA_DEVICES` | RDMA device names |

## Usage Scenarios

### Scenario 1: Multi-Instance Cache Sharing

Run multiple FastDeploy instances sharing a global KV Cache pool.

**Step 1: Start Mooncake Master**

```bash
mooncake_master \
    --port=15001 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=15002 \
    --metrics_port=15003
```

**Step 2: Start FastDeploy Instances**

Instance 0:
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

Instance 1:
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

**Step 3: Test Cache Reuse**

Send the same prompt to both instances. The second instance should reuse the KV Cache computed by the first instance.

```bash
# Request to Instance 0
curl -X POST "http://0.0.0.0:52700/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, world!"}], "max_tokens": 50}'

# Request to Instance 1 (should hit cached KV)
curl -X POST "http://0.0.0.0:52800/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello, world!"}], "max_tokens": 50}'
```

### Scenario 2: PD Disaggregation with Global Cache

This scenario combines **PD Disaggregation** with **Global Cache Pooling**, enabling:

- Prefill instances to read Decode instances' output cache
- Optimal multi-turn conversation performance

**Architecture:**

```
         ┌──────────────────────────────────────────┐
         │              Router                       │
         │         (Load Balancer)                   │
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
                  │  (Global Cache) │
                  └─────────────────┘
```

**Step 1: Start Mooncake Master**

```bash
mooncake_master \
    --port=15001 \
    --enable_http_metadata_server=true \
    --http_metadata_server_host=0.0.0.0 \
    --http_metadata_server_port=15002
```

**Step 2: Start Router**

```bash
python -m fastdeploy.router.launch \
    --port 52700 \
    --splitwise
```

**Step 3: Start Prefill Instance**

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

**Step 4: Start Decode Instance**

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

**Step 5: Send Requests via Router**

```bash
curl -X POST "http://0.0.0.0:52700/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

## FastDeploy Parameters for Mooncake

| Parameter | Description |
|-----------|-------------|
| `--kvcache-storage-backend mooncake` | Enable Mooncake as KV Cache storage backend |
| `--enable-output-caching` | Enable output token caching (Decode instance recommended) |
| `--cache-transfer-protocol rdma` | Use RDMA for KV transfer between P and D |
| `--splitwise-role prefill/decode` | Set instance role in PD disaggregation |
| `--router` | Router address for PD disaggregation |

## Verification

### Check Cache Hit

To verify cache hit in logs:

```bash
# For multi-instance scenario
grep -E "storage_cache_token_num" log_*/api_server.log

# For PD disaggregation scenario
grep -E "storage_cache_token_num" log_prefill/api_server.log
```

If `storage_cache_token_num > 0`, the instance successfully read cached KV blocks from the global pool.

### Monitor Mooncake Master

```bash
# Check master status
curl http://localhost:15002/metadata

# Check metrics (if metrics_port is configured)
curl http://localhost:15003/metrics
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**

```bash
# Check port usage
ss -ltn | grep 15001

# Kill existing process
kill -9 $(lsof -t -i:15001)
```

**2. RDMA Connection Failed**

- Verify RDMA devices: `ibv_devices`
- Check RDMA network: `ibv_devinfo`
- Fallback to TCP: Set `MOONCAKE_PROTOCOL=tcp`

**3. Cache Not Being Shared**

- Verify all instances connect to the same Mooncake master
- Check metadata server URL is consistent
- Verify `global_segment_size` is large enough

**4. Permission Denied on /dev/shm**

```bash
# Clean up stale shared memory files
find /dev/shm -type f -print0 | xargs -0 rm -f
```

### Debug Mode

Enable debug logging:

```bash
export FD_DEBUG=1
```

## More Resources

- [Mooncake Official Documentation](https://github.com/kvcache-ai/Mooncake)
- [Mooncake Troubleshooting Guide](https://github.com/kvcache-ai/Mooncake/blob/main/docs/source/troubleshooting/troubleshooting.md)
- [FastDeploy Documentation](https://paddlepaddle.github.io/FastDeploy/)
