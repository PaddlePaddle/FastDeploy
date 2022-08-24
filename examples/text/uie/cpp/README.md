# UIE C++部署示例

本目录下提供`infer.cc`快速完成UIE模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/quick_start/requirements.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/compile/prebuilt_libraries.md)

以Linux上uie-base模型推理为例，在本目录执行如下命令即可完成编译测试。

```
#下载SDK，编译模型examples代码（SDK中包含了examples代码）
wget https://bj.bcebos.com/paddlehub/fastdeploy/libs/0.2.0/fastdeploy-linux-x64-gpu-0.2.0.tgz
tar xvf fastdeploy-linux-x64-gpu-0.2.0.tgz
cd fastdeploy-linux-x64-gpu-0.2.0/examples/text/uie/cpp
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../fastdeploy-linux-x64-gpu-0.2.0
make -j

# 下载uie-base模型以及词表
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz


# CPU 推理
./infer_demo uie-base 0

# GPU 推理
./infer_demo uie-base 1
```
