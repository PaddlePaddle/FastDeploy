# Run the Examples on NVIDIA CUDA GPU

## Prepare the Environment
Refer to [NVIDIA CUDA GPU Installation](https://paddlepaddle.github.io/FastDeploy/get_started/installation/nvidia_gpu/) to pull the docker image, such as:
```
docker pull ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.6:2.3.0
```

In the docker container, the [NVIDIA MLNX_OFED](https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/) and [Redis](https://redis.io/) are pre-installed.

## Build and install FastDeploy

```
git clone https://github.com/PaddlePaddle/FastDeploy
cd FastDeploy

export ENABLE_FD_RDMA=1

# Argument 1: Whether to build wheel package (1 for yes, 0 for compile only)
# Argument 2: Python interpreter path
# Argument 3: Whether to compile CPU inference operators
# Argument 4: Target GPU architectures
bash build.sh 1 python false [80,90]
```

## Run the Examples

Run the shell scripts in this directory, ```bash start_v0_tp1.sh``` or ```bash start_v1_tp1.sh```

Note that, there are two methods for splitwise deployment:
* v0: using splitwise_scheduler or dp_scheduler, in which the requests are scheduled in the engine.
* v1: using router, in which the requests are scheduled in the router.

# Run the Examples on Kunlunxin XPU

Coming soon...
