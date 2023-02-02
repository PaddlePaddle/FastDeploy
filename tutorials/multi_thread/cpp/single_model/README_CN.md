[English](README.md) | 中文

# PaddleClas C++多线程部署示例

本目录下提供`multi_thread.cc`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速多线程部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上ResNet50_vd推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg


# CPU多线程推理
./multi_thread_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 0 1
# GPU多线程推理
./multi_thread_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 1 1
# GPU上TensorRT多线程推理
./multi_thread_demo ResNet50_vd_infer ILSVRC2012_val_00000010.jpeg 2 1
```
>> **注意**: 最后一位数字表示线程数

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../docs/cn/faq/use_sdk_on_windows.md)

运行完成后返回结果如下所示
```
Thread Id: 0
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```
