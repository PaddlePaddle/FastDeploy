[English](../../features/data_parallel_service.md)

# DP数据并行
DP数据并行，是分布式推理的一种方式，指在多个“完全相同的模型副本”之间分发不同的请求，每个副本完成请求推理。

通常在部署MOE模型时，数据并行（DP）和 专家并行（EP）相结合，每个DP服务独立完成Attention部分推理，所有DP服务协同完成MOE部分推理，提升整体的推理性能。

Fastdeploy 支持DP数据并行，提供了`multi_api_server`接口可以一次性启动多个推理服务。

![架构图](./images/no_scheduler_img.png)

## 启动Fastdeploy服务

以ERNIE-4.5-300B模型为例，启动DP8、TP1、EP8的服务：
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

参数说明：
- num-servers: 指定拉起的DP服务数量
- ports: 指定拉起DP服务的api server端口，数量需要和num-servers一致
- metrics-ports: 指定拉起DP服务的metrics server端口，数量需要和num-servers一致；如果为空，则内部自行分配可用端口
- args: 指定拉起DP服务的参数，可以参考[文档](../parameters.md)；如果端口（除了`ports`）没有手动设置，会自动分配可用端口

## 请求调度

使用DP数据并行策略启动多个DP服务后，用户的请求需要通过调度器来分发到不同的服务，做到负载均衡。

### Web 服务器

获知了DP服务实例的IP和端口后，大家可以通过常用的Web 服务器（比如Nginx），自行实现请求调度，此处不再赘述。

### FastDeploy Router

FastDeploy提供[Router](https://github.com/PaddlePaddle/FastDeploy/tree/develop/fastdeploy/router)（Python版本）来实现请求收发和请求调度。高性能版本Router正在开发中，敬请期待。

使用方式和请求调度流程如下：
- 启动Router
- 启动FastDeploy服务实例（可以单DP或者多DP的服务），向Router进行注册
- 用户请求发送到Router
- Router根据全局实例的负载情况，为请求选择合适的实例
- Router将请求发给选定的实例进行推理
- Router接收实例的生成结果，返回给用户

上手示例：
- 启动Router服务，日志信息输出在`log_router/router.log`。
```
export FD_LOG_DIR="log_router"
python -m fastdeploy.router.launch \
    --host 0.0.0.0 \
    --port 30000 \
```

- 同样以ERNIE-4.5-300B模型为例，启动DP8、TP1、EP8的服务，通过`--router`指定Router服务：
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
