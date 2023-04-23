English | [中文](../zh_CN/compile.md)
# FastDeploy Serving Deployment Compilation

## Compilation with Docker containers

### GPU Image

The GPU images published by FastDploy are based on version 21.10 of [Triton Inference Server](https://github.com/triton-inference-server/server). If developers need to use other CUDA versions, please refer to [ NVIDIA official website](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) to modify the scripts in Dockerfile and scripts.

```shell
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
bash scripts/build.sh

# Exit to the FastDeploy home directory and create the image
# x.y.z is FastDeploy version, example: 1.0.6
cd ../
docker build -t paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.4-21.10 -f serving/Dockerfile .
```

The default version of TensorRT is 8.5.2.2. If you need to change the version, you can run the following commands.

```
cd serving
bash scripts/build.sh -tv 8.4.1.5

cd ../
docker build -t paddlepaddle/fastdeploy:x.y.z-gpu-cuda11.4-trt8.5-21.10 -f serving/Dockerfile_CUDA_11_4_TRT_8_4 .
```

For example, create an GPU image based on FastDeploy v1.0.6 and ubuntu 20.04，cuda11.2 environment
```
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
bash scripts/build_fd_cuda_11_2.sh

# Exit to the FastDeploy home directory and create the image
cd ../
docker build -t paddlepaddle/fastdeploy:1.0.6-gpu-cuda11.2-trt8.4-21.10 -f serving/Dockerfile_CUDA_11_2 .
```

### CPU Image

```shell
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
cd serving
bash scripts/build.sh OFF

# Exit to the FastDeploy home directory and create the image
# x.y.z is FastDeploy version, example: 1.0.6
cd ../
docker build -t paddlepaddle/fastdeploy:x.y.z-cpu-only-21.10 -f serving/Dockerfile_cpu .
```

### IPU Image

```shell
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
bash scripts/build_fd_ipu.sh

# Exit to the FastDeploy home directory and create the image
# x.y.z is FastDeploy version, example: 1.0.6
cd ../
docker build -t paddlepaddle/fastdeploy:x.y.z-ipu-only-21.10 -f serving/Dockerfile_ipu .
```

## Compilation without Docker containers

- [FastDeploy Serving CentOS Compilation Tutorial](./compile_without_docker_centos-en.md)
