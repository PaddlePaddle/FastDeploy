[English](../../features/multi-node_deployment.md)

# 多节点部署

## 概述
多节点部署旨在解决单个机器GPU显存不足时，支持跨多台机器的张量并行执行。

## 环境准备
### 网络要求
1. 所有节点必须在同一本地网络中
2. 确保所有节点之间双向连通（可使用`ping`和`nc -zv`测试）

#### 软件要求
1. 所有节点安装相同版本的FastDeploy
2. [建议安装]安装并配置MPI（OpenMPI或MPICH）

## 张量并行部署

### 推荐启动方式
我们推荐使用mpirun进行一键启动，无需手动启动每个节点。

### 使用说明
1. 在所有机器上执行相同的命令
2. `ips`参数中的IP顺序决定了节点启动顺序
3. 第一个IP将被指定为主节点
4. 确保所有节点能够解析彼此的主机名

* 在线推理启动示例：

    ```shell
    python -m fastdeploy.entrypoints.openai.api_server \
    --model baidu/ERNIE-4.5-300B-A47B-Paddle \
    --port 8180 \
    --metrics-port 8181 \
    --engine-worker-queue-port 8182 \
    --max-model-len 32768 \
    --max-num-seqs 32 \
    --tensor-parallel-size 16 \
    --ips 192.168.1.101,192.168.1.102
    ```

* 离线启动示例：

    ```python
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    model_name_or_path = "baidu/ERNIE-4.5-300B-A47B-Paddle"

    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=16, ips="192.168.1.101,192.168.1.102")
    if llm._check_master():
        output = llm.generate(prompts="你是谁?", use_tqdm=True, sampling_params=sampling_params)
        print(output)
    ```

* 注意：
* 只有主节点可以接收完成请求
* 请始终将请求发送到主节点（ips列表中的第一个IP）
* 主节点将在所有节点间分配工作负载

### 参数说明

#### `ips`参数
* **类型**: `字符串`
* **格式**: 逗号分隔的IPv4地址
* **描述**: 指定部署组中所有节点的IP地址
* **必填**: 仅多节点部署时需要
* **示例**: `"192.168.1.101,192.168.1.102,192.168.1.103"`

#### `tensor_parallel_size`参数
* **类型**: `整数`
* **描述**: 所有节点上的GPU总数
* **必填**: 是
* **示例**: 对于2个节点各8个GPU，设置为16
