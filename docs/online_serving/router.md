[简体中文](../zh/online_serving/router.md)

# Load-Balancing Scheduling Router

FastDeploy provides a Golang-based [Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router) for request scheduling. The Router supports both centralized deployment and Prefill/Decode (PD) disaggregated deployment.。

## Installation

### 1. Prebuilt Binaries

Starting from FastDeploy v2.5.0, the official Docker images include the Go language environment required to build the Golang Router and also provide a precompiled Router binary. The Router binary is located by default in the `/usr/local/bin` directory and can be used directly without additional compilation. For installation details, please refer to the [FastDeploy Installation Guide](../get_started/installation/nvidia_gpu.md)

### 2. Build from Source

You need to build the Router from source in the following scenarios:

* The official Docker image is not used
* FastDeploy version is earlier than v2.5.0
* Custom modifications to the Router are required

Environment Requirements:

* Go >= 1.21

Clone the FastDeploy repository and build the Router:
```
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/fastdeploy/golang_router
bash build.sh
```

## Centralized Deployment

Start the Router service. The `--port` parameter specifies the scheduling port for centralized deployment.
```
./fd-router --port 30000
```

Start a mixed inference instance. Compared to standalone deployment, specify the Router endpoint via `--router`. Other parameters remain unchanged.
```
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_mixed"
python -m fastdeploy.entrypoints.openai.api_server \
   --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
   --port 31000 \
   --router "0.0.0.0:30000"
```

## PD Disaggregated Deployment

Start the Router service with PD disaggregation enabled using the `--splitwise` flag.
```
./fd-router \
  --port 30000 \
  --splitwise
```

Launch a prefill instance. Compared with standalone deployment, add the `--splitwise-role` parameter to specify the instance role as Prefill, and add the `--router` parameter to specify the Router endpoint. All other parameters remain the same as in standalone deployment.
```
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 31000 \
    --splitwise-role prefill \
    --router "0.0.0.0:30000"
```

Launch a decode instance.
```
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_decode"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 32000 \
    --splitwise-role decode \
    --router "0.0.0.0:30000"
```

Once both Prefill and Decode instances are successfully launched and registered with the Router, requests can be sent:
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

For more details on PD disaggregated deployment, please refer to the [Usage Guide](../features/disaggregated.md)

## CacheAware

The Load-Balancing Scheduling Router supports the CacheAware strategy, mainly applied to PD separation deployment to optimize request allocation and improve cache hit rate.

To use the CacheAware strategy, default configurations need to be modified. You can copy the configuration template and make adjustments (an example is available at [Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/golang_router) directory under examples/run_with_config):
```bash
pushd examples/run_with_config
cp config/config.example.yaml config/config.yaml
popd
```

Launch the Router with the custom configuration specified via `--config_path`:
```
./fd-router \
  --port 30000 \
  --splitwise \
  --config_path examples/run_with_config/config/config.yaml
```

Prefill and Decode instance startup are the same as PD disaggregated deployment.

Launch the prefill instance.
```
export CUDA_VISIBLE_DEVICES=0
export FD_LOG_DIR="log_prefill"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 31000 \
    --splitwise-role prefill \
    --router "0.0.0.0:30000"
```

Launch the decode instance.
```
export CUDA_VISIBLE_DEVICES=1
export FD_LOG_DIR="log_decode"
python -m fastdeploy.entrypoints.openai.api_server \
    --model "PaddlePaddle/ERNIE-4.5-0.3B-Paddle" \
    --port 32000 \
    --splitwise-role decode \
    --router "0.0.0.0:30000"
```

## HTTP Service Description

The Router exposes a set of HTTP services to provide unified request scheduling, runtime health checking, and monitoring metrics, facilitating integration and operations.

| Method | Path | Description |
|----------|------|------|
| POST | `/v1/chat/completions` | Provide scheduling services for inference requests based on the Chat Completions API |
| POST | `/v1/completions` | Provide scheduling services for general text completion inference requests |
| POST | `/register` | Allow inference instances to register their metadata with the Router for scheduling |
| GET | `/registered` | Query the list of currently registered inference instances |
| GET | `/registered_number` | Query the number of currently registered inference instances |
| GET | `/health_generate` | Check the health status of registered Prefill / Decode inference instances |
| GET | `/metrics` | Provide Prometheus-formatted Router runtime metrics for monitoring and observability |

## Deployment Parameters

### Router Startup Parameters

* --port: Specify the Router scheduling port.
* --splitwise: Enable PD disaggregated scheduling mode.
* --config_path: Specify the Router configuration file path for loading custom scheduling and runtime parameters.

### Configuration File Preparation

Before using `--config_path`, prepare a configuration file that conforms to the Router specification.
The configuration file is typically written in YAML format. For detailed parameters, refer to [Configuration Parameteres](#configuration-parameteres)。You may copy and modify the configuration template (example available at examples/run_with_config)：
```bash
cp config/config.example.yaml config/config.yaml
```

The Load-Balancing Scheduling Router also supports registering inference instances through configuration files at startup (example available at examples/run_with_default_workers):
```bash
cp config/config.example.yaml config/config.yaml
cp config/register.example.yaml config/register.yaml
```

### Configuration Parameteres

config.yaml example:
```yaml
server:
  port: "8080" # Listening port
  host: "0.0.0.0" # Listening address
  mode: "debug" # Startup mode: debug, release, test
  splitwise: true # true enables PD disaggregation; false disables it

scheduler:
  policy: "power_of_two" # Scheduling policy (optional): random, power_of_two, round_robin, process_tokens, request_num, cache_aware, fd_metrics_score
  prefill-policy: "cache_aware" # Prefill scheduling policy in PD mode
  decode-policy: "fd_metrics_score" # Decode scheduling policy in PD mode
  eviction-interval-secs: 60 # Cache eviction interval for CacheAware scheduling
  balance-abs-threshold: 1 # Absolute threshold for CacheAware balancing
  balance-rel-threshold: 0.2 # Relative threshold for CacheAware balancing
  hit-ratio-weight: 1.0 # Cache hit ratio weight
  load-balance-weight: 0.05 # Load balancing weight
  cache-block-size: 4 # Cache block size
  tokenizer-url: "http://0.0.0.0:8098" # Tokenizer service endpoint (optional)
  tokenizer-timeout-secs: 2 # Tokenizer service timeout
  waiting-weight: 10 # Waiting weight for CacheAware scheduling

manager:
  health-failure-threshold: 3 # Number of failed health checks before marking unhealthy
  health-success-threshold: 2 # Number of successful health checks before marking healthy
  health-check-timeout-secs: 5 # Health check timeout
  health-check-interval-secs: 5 # Health check interval
  health-check-endpoint: /health # Health check endpoint
  register-path: "config/register.yaml" # Path to instance registration config (optional)

log:
  level: "info"  # Log level: debug / info / warn / error
  output: "file" # Log output: stdout / file
```

register.yaml example：
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

Instance Registration Parameters：

* role: Instance role, one of: decode, prefill, mixed.
* host_ip: IP address of the inference instance host.
* port: Service port of the inference instance.
* connector_port: Connector port used for PD communication.
* engine_worker_queue_port: Shared queue communication port within the inference instance.
* transfer_protocol: Specify KV Cache transfer protocol, optional values: ipc / rdma, multiple protocols separated by commas
* rdma_ports: Specify RDMA communication ports, multiple ports separated by commas (only takes effect when transfer_protocol contains rdma)
* device_ids: GPU device IDs of the inference instance, multiple IDs separated by commas
* metrics_port: Port number of the inference instance's metrics

Among these, `role`, `host_ip`, and `port` are required; all other parameters are optional.
