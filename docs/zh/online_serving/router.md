[English](../../online_serving/router.md)

# 负载均衡调度Router

FastDeploy提供Golang版本[Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router)，用于实现请求的调度。Router支持集中式部署和PD分离式部署。

![go-router](images/go-router-workflow.png)

## 安装

### 1. 预编译库下载

在 FastDeploy v2.5.0 及之后版本中，官方 Docker 镜像将内置 Golang Router 编译所需的 Go 语言环境，并提供已编译完成的 Router 二进制文件。该二进制文件默认位于 `/usr/local/bin` 目录下，可直接使用。相关安装方式可参考 [FastDeploy 安装文档](../get_started/installation/nvidia_gpu.md)。

若需单独下载 Golang router 二进制文件，可通过以下方式：
```
wget https://paddle-qa.bj.bcebos.com/paddle-pipeline/FastDeploy_ActionCE/develop/latest/fd-router
mv fd-router /usr/local/bin/fd-router
```

### 2. 编译安装

在以下场景中，需要从源码编译 Router：

* 未使用官方 Docker 镜像
* FastDeploy 版本早于 v2.5.0
* 需要对 Router 进行定制化修改

环境要求：

* Go >= 1.21

拉取FastDeploy最新代码，编译安装：
```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/fastdeploy/golang_router
bash build.sh
cp
```

## 集中式部署

启动Router服务，其中`--port`参数指定集中式部署的调度端口.
```
/usr/local/bin/fd-router \
  --port 30000
```

启动mixed实例。对比单机部署，增加`--router`参数指定Router的接口，其他参数和单机部署相同。
```
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_mixed"
python -m fastdeploy.entrypoints.openai.api_server \
   --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
   --port 31000 \
   --router "0.0.0.0:30000"
```

## PD分离部署

启动Router服务，其中`--splitwise`参数指定为分离式部署的调度方式.
```
/usr/local/bin/fd-router \
  --port 30000 \
  --splitwise
```

启动Prefill实例。对比单机部署，增加`--splitwise-role`参数指定实例角色为Prefill，增加`--router`参数指定Router的接口，其他参数和单机部署相同。
```
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 31000 \
    --splitwise-role prefill \
    --router "0.0.0.0:30000"
```

启动Decode实例。
```
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_decode"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 32000 \
    --splitwise-role decode \
    --router "0.0.0.0:30000"
```

Prefill和Decode实例启动成功，并且向Router注册成功后，可以发送请求。
```
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
```
/usr/local/bin/fd-router \
  --port 30000 \
  --splitwise \
  --config_path examples/run_with_config/config/config.yaml
```

Prefill和Decode实例启动同PD分离部署。

启动Prefill实例。
```
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 31000 \
    --splitwise-role prefill \
    --router "0.0.0.0:30000"
```

启动Decode实例。
```
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
  policy: "power_of_two" # 调度策略(可选): random, power_of_two, round_robin, process_tokens, request_num, cache_aware, fd_metrics_score
  prefill-policy: "cache_aware" # pd分离模式下prefill节点调度策略
  decode-policy: "fd_metrics_score" # pd分离模式下decode节点调度策略
  eviction-interval-secs: 60 # cache-aware策略清理过期cache的间隔时间
  balance-abs-threshold: 1 # cache-aware策略绝对阈值
  balance-rel-threshold: 0.2 # cache-aware策略相对阈值
  hit-ratio-weight: 1.0 # cache-aware策略命中率权重
  load-balance-weight: 0.05 # cache-aware策略负载均衡权重
  cache-block-size: 4 # cache-aware策略cache block大小
  tokenizer-url: "http://0.0.0.0:8098" # tokenizer服务地址(可选)
  tokenizer-timeout-secs: 2 # tokenizer服务超时时间
  waiting-weight: 10 # cache-aware策略等待权重

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
