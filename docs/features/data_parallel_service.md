[简体中文](../zh/features/data_parallel_service.md)

# Data Parallelism (DP)

Data Parallelism (DP) is a distributed inference approach in which incoming requests are distributed across multiple **identical model replicas**, with each replica independently handling inference for its assigned requests.

In practice, especially when deploying **Mixture-of-Experts (MoE)** models, **Data Parallelism (DP)** is often combined with **Expert Parallelism (EP)**.
Each DP service independently performs the Attention computation, while all DP services collaboratively participate in the MoE computation, thereby improving overall inference performance.

FastDeploy supports DP-based inference and provides the `multi_api_server` interface to launch multiple inference services simultaneously.

![Architecture](./images/no_scheduler_img.png)

---

## Launching FastDeploy Services

Taking the **ERNIE-4.5-300B** model as an example, the following command launches a service with **DP=8, TP=1, EP=8**:

```shell
export FD_ENABLE_MULTI_API_SERVER=1
python -m fastdeploy.entrypoints.openai.multi_api_server \
  --num-servers 8 \
  --ports "1811,1822,1833,1844,1855,1866,1877,1888" \
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

### Parameter Description

* `num-servers`: Number of DP service instances to launch.
* `ports`: API server ports for the DP service instances. The number of ports must match `num-servers`.
* `metrics-ports`: Metrics server ports for the DP service instances. The number must match `num-servers`. If not specified, available ports will be allocated automatically.
* `args`: Arguments passed to each DP service instance. Refer to the [parameter documentation](../parameters.md) for details.

---

## Request Scheduling

After launching multiple DP services using the data parallel strategy, incoming user requests must be distributed across the services by a scheduler to achieve load balancing.

### Web Server–Based Scheduling

Once the IP addresses and ports of the DP service instances are known, common web servers (such as **Nginx**) can be used to implement request scheduling. Details are omitted here.

### FastDeploy Router

FastDeploy provides a Python-based [Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/router) to handle request reception and scheduling.
A high-performance Router implementation is currently under development.

The usage and request scheduling workflow is as follows:

* Start the Router
* Start FastDeploy service instances (either single-DP or multi-DP), which register themselves with the Router
* User requests are sent to the Router
* The Router selects an appropriate service instance based on the global load status
* The Router forwards the request to the selected instance for inference
* The Router receives the generated result from the instance and returns it to the user

---

## Quick Start Example

### Launching the Router

Start the Router service. Logs are written to `log_router/router.log`.

```shell
export FD_LOG_DIR="log_router"
python -m fastdeploy.router.launch \
    --host 0.0.0.0 \
    --port 30000
```

### Launching DP Services with Router

Again using the **ERNIE-4.5-300B** model as an example, the following command launches **DP=8, TP=1, EP=8** services and registers them with the Router via the `--router` argument:

```shell
export FD_ENABLE_MULTI_API_SERVER=1
python -m fastdeploy.entrypoints.openai.multi_api_server \
  --num-servers 8 \
  --ports "1811,1822,1833,1844,1855,1866,1877,1888" \
  --metrics-ports "3101,3201,3301,3401,3501,3601,3701,3801" \
  --args --model ERNIE-4_5-300B-A47B-FP8-Paddle \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --max-model-len 12288 \
  --max-num-seqs 64 \
  --num-gpu-blocks-override 256 \
  --enable-expert-parallel \
  --router "0.0.0.0:30000"
```
