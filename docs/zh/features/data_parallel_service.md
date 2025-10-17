[English](../../features/data_parallel_service.md)

# 数据并行
在MOE模型下，开启专家并行（EP）与数据并行（DP）相结合，EP 分摊专家负载，结合 DP 实现请求并行处理。

## 数据分发策略
FastDeploy 通过splitwise scheduler 感知各个DP的负载状态，对接收到数据进行分发。

splitwise scheduler 依赖redis存储各个DP的负载状态，对接收到的数据进行分发。

### 专家并行 + 混合式部署

FastDeploy 提供了splitwise scheduler，可以感知各个DP的负载状态，对接收到的数据进行调度。
具体调度流程如下图，用户随机请求ip 与端口，通过redis获取负载状态，将数据分发到负载较低的DP进行推理。
![数据调度架构图](./images/scheduler_img.png)

#### 离线推理
```python

prompts = [
    "Hello, my name is",
    "你好，请问今天是星期",
    "请写6个以数字开头的成语",
    "写一个300字的小说大纲，内容是李白穿越到现代，最后成为公司文职人员的故事",
    "我要采访一位科幻作家，创建一个包含5个问题的列表"
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=128)

llm = LLM(
    model="ERNIE-4_5-300B-A47B-FP8-Paddle",
    tensor_parallel_size=1,
    data_parallel_size=8,
    max_model_len=8192,
    num_gpu_blocks_override=1024,
    engine_worker_queue_port="6077,6078,6079,6080,6081,6082,6083,6084",
    enable_expert_parallel=True,
    scheduler_name="splitwise",
    scheduler_host="127.0.0.1",
    scheduler_topic="test",
    scheduler_port=6379
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print("generated_text: ", generated_text)
    print("\n")


```

#### 在线推理
```shell
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-4_5-300B-A47B-FP8-Paddle \
       --port 8184 --metrics-port 8185 \
       --engine-worker-queue-port "6077,6078,6079,6080,6081,6082,6083,6084"  \
       --data-parallel-size 8 --tensor-parallel-size 1\
       --enable-expert-parallel \
       --scheduler-name "splitwise" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-topic "test" \
       --scheduler-ttl 9000
```

### 用户自行调度
FastDeploy 提供了multi_api_server，用户可以拉起多个api server，用户自行选择dp 进行请求，在该种情况下用户可以自行添加负载均衡模型进行调度。（目前该种方式只支持在线推理）

#### 在线推理

![数据调度架构图](./images/no_scheduler_img.png)

```shell
export FD_ENABLE_MULTI_API_SERVER=1
python -m fastdeploy.entrypoints.openai.multi_api_server \
  --ports "1811,1822,1833,1844,1855,1866,1877,1888" \
  --num-servers 8 \
  --metrics-ports "3101,3201,3301,3401,3501,3601,3701,3801" \
  --args --model ERNIE-4_5-300B-A47B-FP8-Paddle \
  --engine-worker-queue-port "25611,25621,25631,25641,25651,25661,25671,25681" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-model-len 12288 \
  --max-num-seqs 64 \
  --num-gpu-blocks-override 256 \
  --enable-expert-parallel
```

### 参数说明
- num-servers: 指定拉起的api server 的数量
- ports: 指定拉起的api server 的端口
- args: 指定拉起的api server 的参数

### 数据并行 + 分离式部署

具体可以参考[分离式部署](disaggregated.md#多机分离式部署)

#### 在线推理

多机部署时需要确认当前网卡是否支持RDMA，并且需要集群中所有节点网络互通。

**注意**：
- `KVCACHE_RDMA_NICS` 指定当前机器的RDMA网卡，多个网卡用逗号隔开。
- 仓库中提供了自动检测RDMA网卡的脚本 `bash scripts/get_rdma_nics.sh <device>`, 其中 <device> 可以是 `cpu` 或 `gpu`。

**prefill 实例**

```bash
export FD_LOG_DIR="log_prefill"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "set RDMA NICS"
export $(bash scripts/get_rdma_nics.sh gpu)
echo "KVCACHE_RDMA_NICS ${KVCACHE_RDMA_NICS}"

python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-4_5-300B-A47B-FP8-Paddle \
       --port 8180 --metrics-port 8181 \
       --engine-worker-queue-port "25611,25621,25631,25641,25651,25661,25671,25681" \
       --cache-queue-port 8183 \
       --tensor-parallel-size 1 \
       --data-parallel-size 4 \
       --enable-expert-parallel \
       --cache-transfer-protocol "rdma,ipc" \
       --rdma-comm-ports "7671,7672,7673,7674,7675,7676,7677,7678" \
       --pd-comm-port "2334" \
       --splitwise-role "prefill" \
       --scheduler-name "splitwise" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-topic "test" \
       --scheduler-ttl 9000
```

**decode 实例**

```bash
export FD_LOG_DIR="log_decode"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "set RDMA NICS"
export $(bash scripts/get_rdma_nics.sh gpu)
echo "KVCACHE_RDMA_NICS ${KVCACHE_RDMA_NICS}"
python -m fastdeploy.entrypoints.openai.api_server \
       --model ERNIE-4_5-300B-A47B-FP8-Paddle \
       --port 8184 --metrics-port 8185 \
       --engine-worker-queue-port "25611,25621,25631,25641,25651,25661,25671,25681" \
       --cache-queue-port 8187 \
       --tensor-parallel-size 1 \
       --data-parallel-size 4 \
       --enable-expert-parallel \
       --scheduler-name "splitwise" \
       --cache-transfer-protocol "rdma,ipc" \
       --rdma-comm-ports "7671,7672,7673,7674,7675,7676,7677,7678" \
       --pd-comm-port "2334" \
       --scheduler-host "127.0.0.1" \
       --scheduler-port 6379 \
       --scheduler-ttl 9000
       --scheduler-topic "test" \
       --splitwise-role "decode"
```
