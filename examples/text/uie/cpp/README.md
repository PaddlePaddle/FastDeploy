# 通用信息抽取 UIE C++部署示例

本目录下提供`infer.cc`快速完成[UIE模型](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie)在CPU/GPU的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/quick_start/requirements.md)
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/compile/prebuilt_libraries.md)

以Linux上uie-base模型推理为例，在本目录执行如下命令即可完成编译测试。

```
# UIE目前还未发布，当前需开发者自行编译FastDeploy，通过如下脚本编译得到部署库fastdeploy-linux-x64-dev
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON  \
               -DENABLE_VISION=ON \
               -DENABLE_PADDLE_BACKEND=ON \
               -DENABLE_TEXT=ON \
               -DWITH_GPU=ON \
               -DCMAKE_INSTALL_PREFIX=${PWD}/fastdeploy-linux-x64-gpu-dev

make -j8
make install

# 编译模型examples代码（SDK中包含了examples代码）
cd ../examples/text/uie/cpp
mkdir build
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/../../../../../build/fastdeploy-linux-x64-gpu-dev
make -j

# 下载uie-base模型以及词表
wget https://bj.bcebos.com/fastdeploy/models/uie/uie-base.tgz
tar -xvfz uie-base.tgz


# CPU 推理
./infer_demo uie-base 0

# GPU 推理
./infer_demo uie-base 1
```

## 模型获取
UIE 模型介绍可以参考https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie 。其中，在完成训练后，需要将训练后的模型导出成推理模型。该步骤可参考该文档完成导出：https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie#%E6%A8%A1%E5%9E%8B%E9%83%A8%E7%BD%B2 。
