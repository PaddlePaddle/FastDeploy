# YOLOv5Seg C++部署示例

本目录下提供`infer.cc`快速完成YOLOv5Seg在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.3以上(x.x.x>=1.0.3)

```bash
mkdir build
cd build
# 下载 FastDeploy 预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 1. 下载官方转换好的 YOLOv5Seg ONNX 模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-seg.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU推理
./infer_demo yolov5s-seg.onnx 000000014439.jpg 0
# GPU推理
./infer_demo yolov5s-seg.onnx 000000014439.jpg 1
# GPU上TensorRT推理
./infer_demo yolov5s-seg.onnx 000000014439.jpg 2
```
运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/19977378/209955620-657bdd1d-574c-40a2-b05d-42b9e5a15ae8.png">

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## YOLOv5Seg C++接口

### YOLOv5Seg类

```c++
fastdeploy::vision::detection::YOLOv5Seg(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv5Seg模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

```c++
YOLOv5Seg::Predict(const cv::Mat& img, DetectionResult* result)
```

**参数**

> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
