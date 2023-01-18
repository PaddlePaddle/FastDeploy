[English](README.md) | 简体中文
# YOLOv6量化模型 C++部署示例

本目录下提供的`infer.cc`,可以帮助用户快速完成YOLOv6s量化模型在CPU/GPU上的部署推理加速.

## 部署准备
### FastDeploy环境准备
- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

### 量化模型准备
- 1. 用户可以直接使用由FastDeploy提供的量化模型进行部署.
- 2. 用户可以使用FastDeploy提供的[一键模型自动化压缩工具](../../../../../../tools/common_tools/auto_compression/),自行进行模型量化, 并使用产出的量化模型进行部署.

## 以量化后的YOLOv6s模型为例, 进行部署
在本目录执行如下命令即可完成编译,以及量化模型部署.支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)
```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

#下载FastDeloy提供的yolov6s量化模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s_qat_model_new.tar
tar -xvf yolov6s_qat_model.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# 在CPU上使用ONNX Runtime推理量化模型
./infer_demo yolov6s_qat_model 000000014439.jpg 0
# 在GPU上使用TensorRT推理量化模型
./infer_demo yolov6s_qat_model 000000014439.jpg 1
# 在GPU上使用Paddle-TensorRT推理量化模型
./infer_demo yolov6s_qat_model 000000014439.jpg 2
```
