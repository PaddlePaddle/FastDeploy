[简体中文](../zh/features/multi-node_deployment.md)

# Multi-Node Deployment

## Overview
Multi-node deployment addresses scenarios where a single machine's GPU memory is insufficient to support deployment of large models by enabling tensor parallelism across multiple machines.

## Environment Preparation
### Network Requirements
1. All nodes must be within the same local network
2. Ensure bidirectional connectivity between all nodes (test using `ping` and `nc -zv`)

#### Software Requirements
1. Install the same version of FastDeploy on all nodes
2. [Recommended] Install and configure MPI (OpenMPI or MPICH)

## Tensor Parallel Deployment

### Recommended Launch Method
We recommend using mpirun for one-command startup without manually starting each node.

### Usage Instructions
1. Execute the same command on all machines
2. The IP order in the `ips` parameter determines the node startup sequence
3. The first IP will be designated as the master node
4. Ensure all nodes can resolve each other's hostnames

* Online inference startup example:

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

* Offline startup example:

    ```python
    from fastdeploy.engine.sampling_params import SamplingParams
    from fastdeploy.entrypoints.llm import LLM

    model_name_or_path = "baidu/ERNIE-4.5-300B-A47B-Paddle"

    sampling_params = SamplingParams(temperature=0.1, max_tokens=30)
    llm = LLM(model=model_name_or_path, tensor_parallel_size=16, ips="192.168.1.101,192.168.1.102")
    if llm._check_master():
        output = llm.generate(prompts="Who are you?", use_tqdm=True, sampling_params=sampling_params)
        print(output)
    ```

* Notes:
* Only the master node can receive completion requests
* Always send requests to the master node (the first IP in the ips list)
* The master node will distribute workloads across all nodes

### Parameter Description

#### `ips` Parameter
* **Type**: `string`
* **Format**: Comma-separated IPv4 addresses
* **Description**: Specifies the IP addresses of all nodes in the deployment group
* **Required**: Only for multi-node deployments
* **Example**: `"192.168.1.101,192.168.1.102,192.168.1.103"`

#### `tensor_parallel_size` Parameter
* **Type**: `integer`
* **Description**: Total number of GPUs across all nodes
* **Required**: Yes
* **Example**: For 2 nodes with 8 GPUs each, set to 16
