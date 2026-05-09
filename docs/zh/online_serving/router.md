[English](../../online_serving/router.md)

# 负载均衡调度Router

FastDeploy提供Golang版本[Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router)，用于实现请求的调度。Router支持集中式部署和PD分离式部署。

![go-router](images/go-router-workflow.png)

## 安装

### 1. Python 命令行启动（推荐）

`fd-router` 二进制已随 FastDeploy Python wheel 包一同打包发布。安装 FastDeploy 后，无需额外下载或编译，即可通过 Python 命令行直接启动 Router：

```bash
# 启动 mixed 模式 Router
python -m fastdeploy.golang_router.launch --port 9000

# 启动 PD 分离模式 Router
python -m fastdeploy.golang_router.launch --port 9000 --splitwise

# 使用配置文件启动
python -m fastdeploy.golang_router.launch --config_path config.yaml

# 查看版本
python -m fastdeploy.golang_router.launch --version
```

### 2. 下载预编译二进制（可选）

如果需要直接运行 Router 二进制文件（例如不使用 Python 环境时），可以下载预编译二进制：

```bash
wget https://paddle-qa.bj.bcebos.com/paddle-pipeline/FastDeploy_ActionCE/develop/latest/fd-router
chmod +x fd-router
mv fd-router /usr/local/bin/fd-router
```

在 FastDeploy v2.5.0 及之后版本中，官方 Docker 镜像也内置了编译好的 Router 二进制，默认位于 `/usr/local/bin` 目录。相关安装方式可参考 [FastDeploy 安装文档](../get_started/installation/nvidia_gpu.md)。

### 3. 编译安装

在以下场景中，需要从源码编译 Router：

* 需要对 Router 进行定制化修改
* 当前平台没有预编译二进制覆盖

环境要求：

* Go >= 1.21

拉取FastDeploy最新代码，编译安装：
```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/fastdeploy/golang_router
bash build.sh
```

## 集中式部署

启动Router服务，其中`--port`参数指定集中式部署的调度端口.
```bash
python -m fastdeploy.golang_router.launch --port 30000
```

启动mixed实例。对比单机部署，增加`--router`参数指定Router的接口，其他参数和单机部署相同。
```bash
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_mixed"
python -m fastdeploy.entrypoints.openai.api_server \
   --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
   --port 31000 \
   --router "0.0.0.0:30000"
```

## PD分离部署

启动Router服务，其中`--splitwise`参数指定为分离式部署的调度方式.
```bash
python -m fastdeploy.golang_router.launch \
  --port 30000 \
  --splitwise
```

启动Prefill实例。对比单机部署，增加`--splitwise-role`参数指定实例角色为Prefill，增加`--router`参数指定Router的接口，其他参数和单机部署相同。
```bash
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 31000 \
    --splitwise-role prefill \
    --router "0.0.0.0:30000"
```

启动Decode实例。
```bash
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_decode"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 32000 \
    --splitwise-role decode \
    --router "0.0.0.0:30000"
```

Prefill和Decode实例启动成功，并且向Router注册成功后，可以发送请求。
```bash
curl -X POST "http://0.0.0.0:30000/v1/chat/completions" \
-H "Content-Type: application/json" \
-d '{
  "messages": [
    {"role": "user", "content": "hello"}
  ],
  "max_tokens": 100,
  "stream": false
}'
```

具体的 PD 分离式部署方案，请参考[使用文档](../features/disaggregated.md)

## CacheAware

负载均衡调度Router支持CacheAware策略，主要应用于 PD 分离部署，以优化请求分配，提高缓存命中率。

使用CacheAware策略需修改默认配置，可复制配置模板并进行调整（示例可参考[Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router)目录下的examples/run_with_config）:
```bash
pushd examples/run_with_config
cp config/config.example.yaml config/config.yaml
popd
```

在Router启动Router服务，其中`--config_path`参数指定配置路径.
```bash
python -m fastdeploy.golang_router.launch \
  --port 30000 \
  --splitwise \
  --config_path examples/run_with_config/config/config.yaml
```

Prefill和Decode实例启动同PD分离部署。

启动Prefill实例。
```bash
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 31000 \
    --splitwise-role prefill \
    --router "0.0.0.0:30000"
```

启动Decode实例。
```bash
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_decode"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 32000 \
    --splitwise-role decode \
    --router "0.0.0.0:30000"
```

## HTTP服务说明

Router 通过 HTTP 接口对外提供统一的调度服务，同时支持运行状态探测与监控指标暴露，便于集成与运维。

| 方法 | 路径 | 说明 |
|----------|------|------|
| POST | `/v1/chat/completions` | 对外提供基于 Chat 接口的推理请求调度服务 |
| POST | `/v1/completions` | 对外提供通用文本补全请求的调度服务 |
| POST | `/v1/abort_requests` | 中断推理请求，释放 GPU 显存和计算资源。支持传入 `req_ids` 或 `abort_all=true`，返回已中断请求列表及其已生成的 token 数 |
| POST | `/register` | 推理实例向 Router 注册自身信息，用于参与调度 |
| GET | `/registered` | 查询当前已注册的推理实例列表 |
| GET | `/registered_number` | 查询当前已注册的推理实例数量 |
| GET | `/health_generate` | 检查已注册 Prefill / Decode 推理实例的健康状态 |
| GET | `/metrics` | 提供 Prometheus 格式的 Router 运行指标，用于监控与观测 |

## 部署参数说明

### Router启动参数说明

* --port: 指定Router的调度端口。
* --splitwise: 指定为PD分离式部署的调度方式。
* --config_path: 指定Router配置文件路径，用于加载自定义调度与运行参数。

### 配置文件准备

在使用 `--config_path` 参数前，请准备符合 Router 规范的配置文件。配置文件通常以 YAML 形式存在，具体参考[配置参数说明](#配置参数说明)。可复制配置模板并进行调整（示例可参考 examples/run_with_config）：
```bash
cp config/config.example.yaml config/config.yaml
```

负载均衡调度Router还支持通过配置文件在启动阶段注册推理实例（示例可参考 examples/run_with_default_workers）：
```bash
cp config/config.example.yaml config/config.yaml
cp config/register.example.yaml config/register.yaml
```

### 配置参数说明

config.yaml 示例：
```yaml
server:
  port: "8080" # 监听端口
  host: "0.0.0.0" # 监听地址
  mode: "debug" # 启动模式: debug, release, test
  splitwise: true # true代表开启pd分离模式,false代表开启非pd分离模式

scheduler:
  policy: "power_of_two" # 调度策略(可选): random, power_of_two, round_robin, process_tokens, request_num, cache_aware, remote_cache_aware, fd_metrics_score, fd_remote_metrics_score; 默认: request_num
  prefill-policy: "cache_aware" # pd分离模式下prefill节点调度策略; 默认: process_tokens
  decode-policy: "request_num" # pd分离模式下decode节点调度策略; 默认: request_num
  eviction-interval-secs: 60 # cache-aware策略清理过期cache的间隔时间
  eviction-duration-mins: 30 # cache-aware策略radix tree节点驱逐时间(分钟); 默认: 30
  balance-abs-threshold: 1 # cache-aware策略绝对阈值
  balance-rel-threshold: 0.2 # cache-aware策略相对阈值
  hit-ratio-weight: 1.0 # cache-aware策略命中率权重
  load-balance-weight: 0.05 # cache-aware策略负载均衡权重
  cache-block-size: 4 # cache-aware策略cache block大小
  # tokenizer-url: "http://0.0.0.0:8098" # tokenizer服务地址(可选), 不配置时cache_aware策略自动使用字符级分词。
  #                                         注意：配置此项会在每次调度时同步调用远程tokenizer服务，引入额外网络时延，
  #                                         仅在需要精确token级分词以提升cache命中率时再考虑启用。
  # tokenizer-timeout-secs: 2 # tokenizer服务超时时间; 默认: 2
  waiting-weight: 10 # cache-aware策略等待权重
  stats-interval-secs: 5 # 日志统计信息打印间隔时间(秒), 包含负载和缓存命中率等统计数据; 默认: 5

manager:
  health-failure-threshold: 3 # 健康检查失败次数,超过次数后认为节点不健康
  health-success-threshold: 2 # 健康检查成功次数,超过次数后认为节点健康
  health-check-timeout-secs: 5 # 健康检查超时时间
  health-check-interval-secs: 5 # 健康检查间隔时间
  health-check-endpoint: /health # 健康检查接口
  register-path: "config/register.yaml" # 推理实例注册配置文件路径(可选)

log:
  level: "info"  # 日志打印级别: debug / info / warn / error
  output: "file" # 日志输出方式: stdout / file
```

register.yaml 示例：
```yaml
instances:
  - role: "prefill"
    host_ip: 127.0.0.1
    port: 8097
    connector_port: 8001
    engine_worker_queue_port: 8002
    transfer_protocol:
      - ipc
      - rdma
    rdma_ports: [7100, "7101"]
    device_ids: [0, "1"]
    metrics_port: 8003
  - role: "decode"
    host_ip: 127.0.0.1
    port: 8098
    connector_port: 8001
    engine_worker_queue_port: 8002
    transfer_protocol: ["ipc","rdma"]
    rdma_ports: ["7100", "7101"]
    device_ids: ["0", "1"]
```

注册实例参数说明：

* role: 实例角色，可选值 decode / prefill / mixed。
* host_ip: 推理实例所在机器的IP地址。
* port: 推理实例的端口号。
* connector_port: 推理实例指定pd通信的端口。
* engine_worker_queue_port: 推理实例内部的共享队列通信端口。
* transfer_protocol:指定KV Cache传输协议，可选值 ipc / rdma，多个协议用逗号分隔。
* rdma_ports: 指定RDMA通信端口，多个端口用逗号隔开（仅当transfer_protocol包含rdma时生效）。
* device_ids: 推理实例的GPU设备ID，多个设备ID用逗号隔开。
* metrics_port: 推理实例的metrics端口号。

其中 `role`、`host_ip` 和 `port` 为必填参数，其余参数为可选。

## 调度策略说明

Router 支持以下调度策略，可通过配置文件中的 `policy`（mixed 模式）、`prefill-policy` 和 `decode-policy`（PD 分离模式）字段指定。

**默认策略**：不配置时，prefill 节点默认使用 `process_tokens`，mixed 和 decode 节点默认使用 `request_num`。

| 策略名 | 适用场景 | 实现方式 |
|--------|----------|----------|
| `random` | 通用 | 从所有可用实例中随机选择一个，无状态感知，适合轻量场景。 |
| `round_robin` | 通用 | 使用原子计数器对实例列表循环取模，按顺序均匀分发请求。 |
| `power_of_two` | 通用 | 随机选取两个实例，比较其当前并发请求数，选择负载较低的一个。 |
| `process_tokens` | **prefill（默认）** | 遍历所有实例，选择当前正在处理的 token 数最少的实例（内存计数），适合 prefill 阶段的长请求负载均衡。 |
| `request_num` | **mixed / decode（默认）** | 遍历所有实例，选择当前并发请求数最少的实例（内存计数），适合 decode 及 mixed 场景的请求均衡。 |
| `fd_metrics_score` | mixed / decode | 基于内存计数获取 running/waiting 请求数，按 `running + waiting × waitingWeight` 打分，选择得分最低的实例。 |
| `fd_remote_metrics_score` | mixed / decode | 实时从各实例的远程 `/metrics` 接口获取 running/waiting 请求数，按 `running + waiting × waitingWeight` 打分，选择得分最低的实例。需要实例注册时提供 `metrics_port`。**注意：每次调度时会同步发起远程 HTTP 请求，在实例数量较多或网络条件较差时会显著增加调度时延，请结合实际情况评估后再启用。** |
| `cache_aware` | prefill | 基于 Radix Tree 维护各实例的 KV Cache 前缀命中情况，综合命中率与负载打分（内存计数）选择实例；负载严重不均衡时自动回退至 `process_tokens`。 |
| `remote_cache_aware` | prefill | 与 `cache_aware` 相同的缓存感知策略，但使用远程 `/metrics` 接口获取实例负载数据。需要实例注册时提供 `metrics_port`。**注意：每次调度时会同步发起远程 HTTP 请求，在实例数量较多或网络条件较差时会显著增加调度时延，请结合实际情况评估后再启用。** |

## 常见问题排查

在使用 Router 过程中遇到问题时，请参考 [Router 常见问题排查](router_faq.md)，其中涵盖了常见日志分析、返回输出解读及问题排查方法。
