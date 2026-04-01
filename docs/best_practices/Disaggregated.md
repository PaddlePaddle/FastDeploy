[简体中文](../zh/best_practices/Disaggregated.md)

# PD Disaggregated Deployment Best Practices

This document provides a comprehensive guide to FastDeploy's PD (Prefill-Decode) disaggregated deployment solution, covering both single-machine and cross-machine deployment modes with support for Tensor Parallelism (TP), Data Parallelism (DP), and Expert Parallelism (EP).

## 1. Deployment Overview and Environment Preparation

This guide demonstrates deployment practices using the ERNIE-4.5-300B-A47B-Paddle model on H100 80GB GPUs. Below are the minimum GPU requirements for different deployment configurations:

**Single-Machine Deployment (8 GPUs, Single Node)**

| Configuration | TP | DP | EP | GPUs Required |
|---------|----|----|----|---------|
| P：TP4DP1<br>D：TP4DP1 | 4 | 1 | - | 8 |
| P：TP1DP4EP4 <br> D：TP1DP4EP4| 1 | 4 | ✓ | 8 |

**Multi-Machine Deployment (16 GPUs, Cross-Node)**

| Configuration | TP | DP | EP | GPUs Required |
|---------|----|----|----|---------|
| P：TP8DP1<br>D：TP8DP1 | 8 | 1 | - | 16 |
| P：TP4DP2<br>D：TP4DP2 | 4 | 2 | - | 16 |
| P：TP1DP8EP8<br>D：TP1DP8EP8 | 1 | 8 | ✓ | 16 |

**Important Notes**:
1. **Quantization**: All configurations above use WINT4 quantization, specified via `--quantization wint4`
2. **EP Limitations**: When Expert Parallelism (EP) is enabled, only TP=1 is currently supported; multi-TP scenarios are not yet available
3. **Cross-Machine Network**: Cross-machine deployment requires RDMA network support for high-speed KV Cache transmission
4. **GPU Calculation**: Total GPUs = TP × DP × 2, with identical configurations for both Prefill and Decode instances
5. **CUDA Graph Capture**: Decode instances enable CUDA Graph capture by default for inference acceleration, while Prefill instances do not

### 1.1 Installing FastDeploy

Please refer to the [FastDeploy Installation Guide](https://paddlepaddle.github.io/FastDeploy/zh/install/) to set up your environment.

For model downloads, please check the [Supported Models List](https://paddlepaddle.github.io/FastDeploy/zh/model_summary/).

### 1.2 Deployment Topology

**Single-Machine Deployment Topology**

```
┌──────────────────────────────┐
│  Single Machine 8×H100 80GB  │
│  ┌──────────────┐            │
│  │  Router      │            │
│  │  0.0.0.0:8109│            │
│  └──────────────┘            │
│         │                    │
│    ┌────┴────┐               │
│    ▼         ▼               │
│ ┌─────────┐  ┌─────────┐     │
│ │Prefill  │  │Decode   │     │
│ │GPU 0-3  │  │GPU 4-7  │     │
│ └─────────┘  └─────────┘     │
└──────────────────────────────┘
```

**Cross-Machine Deployment Topology**

```
┌─────────────────────┐                      ┌─────────────────────┐
│   Prefill Machine   │      RDMA Network    │   Decode Machine    │
│   8×H100 80GB       │◄────────────────────►│   8×H100 80GB       │
│                     │                      │                     │
│  ┌──────────────┐   │                      │                     │
│  │  Router      │   │                      │                     │
│  │ 0.0.0.0:8109 │───┼──────────────────────┼──────────           │
│  └──────────────┘   │                      │         │           │
│         │           │                      │         │           │
│         ▼           │                      │         ▼           │
│  ┌──────────────┐   │                      │  ┌──────────────┐   │
│  │Prefill Nodes │   │                      │  │Decode Nodes  │   │
│  │GPU 0-7       │   │                      │  │GPU 0-7       │   │
│  └──────────────┘   │                      │  └──────────────┘   │
└─────────────────────┘                      └─────────────────────┘
```

---
## 2. Single-Machine PD Disaggregated Deployment

### 2.1 Test Scenarios and Parallelism Configuration

This chapter demonstrates the **TP4DP1｜D：TP4DP1** configuration test scenario:
- **Tensor Parallelism (TP)**: 4 — Each 4 GPUs independently load complete model parameters
- **Data Parallelism (DP)**: 1 — Each GPU forms a data parallelism group
- **Expert Parallelism (EP)**: Not enabled

**To test other parallelism configurations, adjust parameters as follows:**
1. **TP Adjustment**: Modify `--tensor-parallel-size`
2. **DP Adjustment**: Modify `--data-parallel-size`, ensuring `--ports` and `--num-servers` remain consistent with DP
3. **EP Toggle**: Add or remove `--enable-expert-parallel`
4. **GPU Allocation**: Control GPUs used by Prefill and Decode instances via `CUDA_VISIBLE_DEVICES`

### 2.2 Startup Scripts

#### Start Router

```bash
python -m fastdeploy.router.launch \
    --port 8109 \
    --splitwise
```

Note: This uses the Python version of the router. If needed, you can also use the high-performance [Golang version router](../online_serving/router.md).

#### Start Prefill Nodes

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m fastdeploy.entrypoints.openai.api_server \
    --model /path/to/ERNIE-4.5-300B-A47B-Paddle \
    --port 8188 \
    --splitwise-role "prefill" \
    --cache-transfer-protocol "rdma,ipc" \
    --router "0.0.0.0:8109" \
    --quantization wint4 \
    --tensor-parallel-size 4 \
    --data-parallel-size 1 \
    --max-model-len 8192 \
    --max-num-seqs 64
```

#### Start Decode Nodes

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m fastdeploy.entrypoints.openai.multi_api_server \
    --model /path/to/ERNIE-4.5-300B-A47B-Paddle \
    --ports 8200,8201 \
    --splitwise-role "decode" \
    --cache-transfer-protocol "rdma,ipc" \
    --router "0.0.0.0:8109" \
    --quantization wint4 \
    --tensor-parallel-size 2 \
    --data-parallel-size 2 \
    --max-model-len 8192 \
    --max-num-seqs 64
```

### 2.3 Key Parameter Descriptions

| Parameter | Description |
|-----|------|
| `--splitwise` | Enable PD disaggregated mode |
| `--splitwise-role` | Node role: `prefill` or `decode` |
| `--cache-transfer-protocol` | KV Cache transfer protocol: `rdma` or `ipc` |
| `--router` | Router service address |
| `--quantization` | Quantization strategy (wint4/wint8/fp8, etc.) |
| `--tensor-parallel-size` | Tensor parallelism degree (TP) |
| `--data-parallel-size` | Data parallelism degree (DP) |
| `--max-model-len` | Maximum sequence length |
| `--max-num-seqs` | Maximum concurrent sequences |
| `--num-gpu-blocks-override` | GPU KV Cache block count override |

---

## 3. Cross-Machine PD Disaggregated Deployment

### 3.1 Deployment Principles

Cross-machine PD disaggregation deploys Prefill and Decode instances on different physical machines:
- **Prefill Machine**: Runs the Router and Prefill nodes, responsible for processing input sequence prefill computation
- **Decode Machine**: Runs Decode nodes, communicates with the Prefill machine via RDMA network, responsible for autoregressive decoding generation

### 3.2 Test Scenarios and Parallelism Configuration

This chapter demonstrates the **TP1DP8EP8｜D：TP1DP8EP8** cross-machine configuration (16 GPUs total):
- **Tensor Parallelism (TP)**: 1
- **Data Parallelism (DP)**: 8 — 8 GPUs per machine, totaling 8 Prefill instances and 8 Decode instances
- **Expert Parallelism (EP)**: Enabled — MoE layer shared experts are distributed across 8 GPUs for parallel computation

**To test other cross-machine parallelism configurations, adjust parameters as follows:**
1. **Inter-Machine Communication**: Ensure RDMA network connectivity between machines; Prefill machine needs `KVCACHE_RDMA_NICS` environment variable configured
2. **Router Address**: The `--router` parameter on the Decode machine must point to the actual IP address of the Prefill machine
3. **Port Configuration**: The number of ports in the `--ports` list must match `--num-servers` and `--data-parallel-size`
4. **GPU Visibility**: Each machine specifies its local GPUs via `CUDA_VISIBLE_DEVICES`

### 3.3 Prefill Machine Startup Scripts

#### Start Router

```bash
unset http_proxy && unset https_proxy

python -m fastdeploy.router.launch \
    --port 8109 \
    --splitwise
```

#### Start Prefill Nodes

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m fastdeploy.entrypoints.openai.multi_api_server \
    --ports 8198,8199,8200,8201,8202,8203,8204,8205 \
    --num-servers 8 \
    --args --model /path/to/ERNIE-4.5-300B-A47B-Paddle \
    --splitwise-role "prefill" \
    --cache-transfer-protocol "rdma,ipc" \
    --router "<ROUTER_MACHINE_IP>:8109" \
    --quantization wint4 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --max-num-seqs 64
```

### 3.4 Decode Machine Startup Scripts

#### Start Decode Nodes

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m fastdeploy.entrypoints.openai.multi_api_server \
    --ports 8198,8199,8200,8201,8202,8203,8204,8205 \
    --num-servers 8 \
    --args --model /path/to/ERNIE-4.5-300B-A47B-Paddle \
    --splitwise-role "decode" \
    --cache-transfer-protocol "rdma,ipc" \
    --router "<PREFILL_MACHINE_IP>:8109" \
    --quantization wint4 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --max-num-seqs 64
```

**Note**: Please replace `<PREFILL_MACHINE_IP>` with the actual IP address of the Prefill machine.

## 4. Sending Test Requests

```bash
curl -X POST "http://localhost:8109/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "你好，请介绍一下自己。"}
  ],
  "max_tokens": 100,
  "stream": false
}'
```

## 5. Frequently Asked Questions (FAQ)

If you encounter issues during use, please refer to [FAQ](./FAQ.md) for solutions.
