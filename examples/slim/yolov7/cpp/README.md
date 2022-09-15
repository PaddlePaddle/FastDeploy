# YOLOv7量化模型 C++部署示例

本目录下提供`infer.cc`快速完成YOLOv7量化模型在CPU/GPU上的部署推理加速.

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.0.tgz
tar xvf fastdeploy-linux-x64-0.2.0.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.2.0
make -j

#下载FastDeploy提供的yolov7量化模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# 在CPU上使用ONNXRuntime推理量化模型
./infer_demo yolov7_quant 000000014439.jpg 0
# 在CPU上使用Paddle-Inference推理量化模型
./infer_demo yolov7_quant 000000014439.jpg 1
# 在GPU上使用TensorRT推理量化模型
./infer_demo yolov7_quant 000000014439.jpg 2
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184309358-d803347a-8981-44b6-b589-4608021ad0f4.jpg">

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)
