# FastDeploy Serving Deployment Image Compilation

How to create a FastDploy image

## GPU Image

The GPU images published by FastDploy are based on version 21.10 of [Triton Inference Server](https://github.com/triton-inference-server/server). If developers need to use other CUDA versions, please refer to [ NVIDIA official website](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) to modify the scripts in Dockerfile and scripts.

```shell
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
bash scripts/build.sh

# Exit to the FastDeploy home directory and create the image
cd ../
docker build -t paddlepaddle/fastdeploy:0.3.0-gpu-cuda11.4-trt8.4-21.10 -f serving/Dockerfile .
```

## CPU Image

```shell
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
cd serving
bash scripts/build.sh OFF

# Exit to the FastDeploy home directory and create the image
cd ../
docker build -t paddlepaddle/fastdeploy:0.3.0-cpu-only-21.10 -f serving/Dockerfile_cpu .
```

## IPU Image

```shell
# Enter the serving directory and execute the script to compile the FastDeploy and serving backend
cd serving
bash scripts/build_fd_ipu.sh

# Exit to the FastDeploy home directory and create the image
cd ../
docker build -t paddlepaddle/fastdeploy:0.3.0-ipu-only-21.10 -f serving/Dockerfile_ipu .
```
