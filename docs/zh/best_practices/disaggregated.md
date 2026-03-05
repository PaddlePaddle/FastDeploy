[English](../../best_practices/disaggregated.md)

# PD分离部署最佳实践

本文档介绍FastDeploy的PD分离式部署方案，包括单机部署和跨机部署两种模式，支持TP、DP和EP。


## 一、部署方案和环境准备

在 ERNIE-4.5-300B-A47B-Paddle 模型上进行部署实践，硬件使用H100 80GB，不同部署模式下所需的最小GPU卡数如下：

#### 单机部署（8卡单节点）

| 配置方案 | TP | DP | EP | 所需卡数 | 
|---------|----|----|----|---------|
| TP4DP1 | 4 | 1 | - | 8 |
| TP1DP4EP | 1 | 4 | ✓ | 8 |

#### 多机部署（16卡跨节点）

| 配置方案 | TP | DP | EP | 所需卡数 |
|---------|----|----|----|---------|
| TP8DP1 | 8 | 1 | - | 16 | 
| TP4DP2 | 4 | 2 | - | 16 | 
| TP1DP8EP | 1 | 8 | ✓ | 16 | 

**说明**：
1. **量化精度**：以上所有配置均采用 WINT4 量化，通过 `--quantization wint4` 指定
2. **EP限制**：开启 EP（专家并行）后，目前仅支持 TP=1，暂不支持多 TP 场景
3. **跨机网络**：跨机部署需要 RDMA 网络支持，用于 KV Cache 高速传输
4. **卡数计算**：总卡数 = TP × DP × 2，Prefill实例和Decode实例采用一致配置

### 1.3 安装 FastDeploy

安装请参考 [FastDeploy Installation](https://paddlepaddle.github.io/FastDeploy/zh/install/) 完成安装。

模型下载请参考 [支持模型列表](https://paddlepaddle.github.io/FastDeploy/zh/model_summary/)。

### 1.4 部署拓扑

**单机部署拓扑（TP1DP4EP）**

```
┌──────────────────────────────┐
│  单机 8×H100 80GB             │
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

**跨机部署拓扑（TP1DP8EP）**

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
## 单机PD分离部署
### 3.2 启动脚本

#### 启动 Router

```bash
python -m fastdeploy.router.launch \
    --port 8109 \
    --splitwise
```

#### 启动 Prefill 节点

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m fastdeploy.entrypoints.openai.multi_api_server \
    --ports 8188,8189,8190,8191 \
    --num-servers 4 \
    --args --model /path/to/ERNIE-4.5-300B-A47B-Paddle \
    --splitwise-role "prefill" \
    --cache-transfer-protocol "rdma,ipc" \
    --router "0.0.0.0:8109" \
    --quantization wint4 \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --num-gpu-blocks-override 1024
```

#### 启动 Decode 节点

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m fastdeploy.entrypoints.openai.multi_api_server \
    --ports 8198,8199,8200,8201 \
    --num-servers 4 \
    --args --model /path/to/ERNIE-4.5-300B-A47B-Paddle \
    --splitwise-role "decode" \
    --cache-transfer-protocol "rdma,ipc" \
    --router "0.0.0.0:8109" \
    --quantization wint4 \
    --tensor-parallel-size 1 \
    --data-parallel-size 4 \
    --enable-expert-parallel \
    --max-model-len 8192 \
    --max-num-seqs 64 \
    --num-gpu-blocks-override 1024
```

### 3.3 关键参数说明

| 参数 | 说明 |
|-----|------|
| `--splitwise` | 开启 PD 分离模式 |
| `--splitwise-role` | 节点角色：`prefill` 或 `decode` |
| `--cache-transfer-protocol` | KV Cache 传输协议：`rdma` 或 `ipc` |
| `--router` | Router 服务地址 |
| `--quantization` | 量化策略（wint4/wint8/fp8 等） |
| `--tensor-parallel-size` | 张量并行度（TP） |
| `--data-parallel-size` | 数据并行度（DP） |
| `--enable-expert-parallel` | 开启专家并行（EP） |
| `--max-model-len` | 最大序列长度 |
| `--max-num-seqs` | 最大并发序列数 |
| `--num-gpu-blocks-override` | GPU KV Cache 块数量覆盖值 |

---

## 四、跨机PD分离部署

### 4.1 原理

跨机 PD 分离将 Prefill 和 Decode 部署在不同机器上：
- **Prefill 机器**：部署 Router 和 Prefill 节点
- **Decode 机器**：部署 Decode 节点，通过 RDMA 网络与 Prefill 机器通信

### 4.2 网络配置

跨机部署需要 RDMA 网络支持，启动前需配置 RDMA 网卡（Prefill 机器）：

```bash
export $(bash /path/to/get_rdma_nics.sh gpu)
echo "KVCACHE_RDMA_NICS:${KVCACHE_RDMA_NICS}"
if [ -z "${KVCACHE_RDMA_NICS}" ]; then
  echo "KVCACHE_RDMA_NICS is empty, please check the output of get_rdma_nics.sh"
  exit 1
fi
```

### 4.3 Prefill 机器启动脚本

#### 启动 Router

```bash
unset http_proxy && unset https_proxy

python -m fastdeploy.router.launch \
    --port 8109 \
    --splitwise
```

#### 启动 Prefill 节点

```bash
export $(bash /path/to/get_rdma_nics.sh gpu)
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
    --max-num-seqs 64 \
    --num-gpu-blocks-override 1024
```

### 4.4 Decode 机器启动脚本

#### 启动 Decode 节点

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
    --max-num-seqs 64 \
    --num-gpu-blocks-override 1024
```

**注意**：将 `<ROUTER_MACHINE_IP>` 替换为 Prefill 机器的实际 IP 地址。



## 五、发送测试请求

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



## 六、常见问题FAQ
如果您在使用过程中遇到问题，可以在[FAQ](./FAQ.md)中查阅。
