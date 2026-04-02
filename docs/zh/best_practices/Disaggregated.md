[English](../../best_practices/Disaggregated.md)

# PD分离部署最佳实践

本文档详细介绍 FastDeploy 的 PD（Prefill-Decode）分离式部署方案，涵盖单机部署与跨机部署两种模式，支持张量并行（TP）、数据并行（DP）和专家并行（EP）。

## 一、部署方案概览与环境准备

本文以 ERNIE-4.5-300B-A47B-Paddle 模型为例进行部署实践，硬件环境采用 H100 80GB GPU。下面例举了不同部署模式下的最小 GPU 卡数需求：

**单机部署（8卡单节点）**

| 配置方案 | TP | DP | EP | 所需卡数 |
|---------|----|----|----|---------|
| P：TP4DP1<br>D：TP4DP1 | 4 | 1 | - | 8 |
| P：TP1DP4EP4 <br> D：TP1DP4EP4| 1 | 4 | ✓ | 8 |

**多机部署（16卡跨节点）**

| 配置方案 | TP | DP | EP | 所需卡数 |
|---------|----|----|----|---------|
| P：TP8DP1<br>D：TP8DP1 | 8 | 1 | - | 16 |
| P：TP4DP2<br>D：TP4DP2 | 4 | 2 | - | 16 |
| P：TP1DP8EP8<br>D：TP1DP8EP8 | 1 | 8 | ✓ | 16 |

**重要说明**：
1. **量化精度**：以上所有配置均采用 WINT4 量化，通过 `--quantization wint4` 参数指定
2. **EP 限制**：开启专家并行（EP）后，当前仅支持 TP=1，暂不支持多 TP 场景
3. **跨机网络**：跨机部署依赖 RDMA 网络实现 KV Cache 的高速传输
4. **卡数计算**：总卡数 = TP × DP × 2（Prefill 实例与 Decode 实例配置相同）
5. **CUDA Graph 捕获**：Decode 实例默认启用 CUDA Graph 捕获以加速推理，Prefill 实例默认不启用

### 1.1 安装 FastDeploy

请参考 [FastDeploy 安装指南](https://paddlepaddle.github.io/FastDeploy/zh/install/) 完成环境搭建。

模型下载请参考 [支持模型列表](https://paddlepaddle.github.io/FastDeploy/zh/model_summary/)。

### 1.2 部署拓扑结构

**单机部署拓扑**

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

**跨机部署拓扑**

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
## 二、单机 PD 分离部署

### 2.1 测试场景与并行度配置

本节演示的测试场景为 **P：TP4DP1｜D：TP4DP1** 配置：
- **张量并行度（TP）**：4 —— 每4张 GPU 独立加载完整模型参数
- **数据并行度（DP）**：1 —— 每张 GPU 组成一个数据并行组
- **专家并行（EP）**：不启用

**若需测试其他并行度配置，请按以下方式调整参数：**
1. **TP 调整**：修改 `--tensor-parallel-size`
2. **DP 调整**：修改 `--data-parallel-size`，同时确保 `--ports` 和 `--num-servers` 与 DP 保持一致
3. **EP 开关**：添加或移除 `--enable-expert-parallel`
4. **GPU 分配**：通过 `CUDA_VISIBLE_DEVICES` 控制 Prefill 和 Decode 实例使用的 GPU

### 2.2 启动脚本

#### 启动 Router

```bash
python -m fastdeploy.router.launch \
    --port 8109 \
    --splitwise
```

注意：这里使用的是python版本router，如果有需要也可以使用高性能的[Golang版本router](../online_serving/router.md)
#### 启动 Prefill 节点

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

#### 启动 Decode 节点

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

### 2.3 关键参数说明

| 参数 | 说明 |
|-----|------|
| `--splitwise` | 开启 PD 分离模式 |
| `--splitwise-role` | 节点角色：`prefill` 或 `decode` |
| `--cache-transfer-protocol` | KV Cache 传输协议：`rdma` 或 `ipc` |
| `--router` | Router 服务地址 |
| `--quantization` | 量化策略（wint4/wint8/fp8 等） |
| `--tensor-parallel-size` | 张量并行度（TP） |
| `--data-parallel-size` | 数据并行度（DP） |
| `--max-model-len` | 最大序列长度 |
| `--max-num-seqs` | 最大并发序列数 |
| `--num-gpu-blocks-override` | GPU KV Cache 块数量覆盖值 |

---

## 三、跨机 PD 分离部署

### 3.1 部署原理

跨机 PD 分离将 Prefill 和 Decode 实例部署在不同物理机器上：
- **Prefill 机器**：运行 Router 和 Prefill 节点，负责处理输入序列的预填充计算
- **Decode 机器**：运行 Decode 节点，通过 RDMA 网络与 Prefill 机器通信，负责自回归解码生成

### 3.2 测试场景与并行度配置

本章节演示的测试场景为 **P：TP1DP8EP8 ｜ D：P：TP1DP8EP8** 跨机配置（共 16 张 GPU）：
- **张量并行度（TP）**：1
- **数据并行度（DP）**：8 —— 每机 8 张 GPU，共 8 个 Prefill 实例和 8 个 Decode 实例
- **专家并行（EP）**：启用—— MoE 层的共享专家分布在8张 GPU 上并行计算

**若需测试其他跨机并行度配置，请按以下方法调整参数：**
1. **机器间通信**：确保两机之间 RDMA 网络连通，Prefill 机器需配置 `KVCACHE_RDMA_NICS` 环境变量
2. **Router 地址**：Decode 机器的 `--router` 参数需指向 Prefill 机器的实际 IP 地址
3. **端口配置**：`--ports` 列表的端口数量必须与 `--num-servers` 和 `--data-parallel-size` 保持一致
4. **GPU 可见性**：每机通过 `CUDA_VISIBLE_DEVICES` 指定本机使用的 GPU

### 3.3 Prefill 机器启动脚本

#### 启动 Router

```bash
unset http_proxy && unset https_proxy

python -m fastdeploy.router.launch \
    --port 8109 \
    --splitwise
```

#### 启动 Prefill 节点

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

### 3.4 Decode 机器启动脚本

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
    --max-num-seqs 64
```

**注意**：请将 `<PREFILL_MACHINE_IP>` 替换为 Prefill 机器的实际 IP 地址。

## 四、发送测试请求

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

## 五、常见问题 FAQ

如果您在使用过程中遇到问题，可以在 [FAQ](./FAQ.md) 中查阅解决方案。
