[English](README.md) | 简体中文
# FaceLandmark1000 C++部署示例

本目录下提供`infer.cc`快速完成FaceLandmark1000在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本1.0.2以上(x.x.x>=1.0.2), 或使用nightly built版本

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

#下载官方转换好的 FaceLandmark1000 模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/FaceLandmark1000.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/facealign_input.png

# CPU推理
./infer_demo --model FaceLandmark1000.onnx --image facealign_input.png --device cpu
# GPU推理
./infer_demo --model FaceLandmark1000.onnx --image facealign_input.png --device gpu
# GPU上TensorRT推理
./infer_demo --model FaceLandmark1000.onnx --image facealign_input.png --device gpu --backend trt
```

运行完成可视化结果如下图所示

<div width="500">
<img width="470" height="384" float="left" src="https://user-images.githubusercontent.com/67993288/200761309-90c096e2-c2f3-4140-8012-32ed84e5f389.jpg">
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## FaceLandmark1000 C++接口

### FaceLandmark1000 类

```c++
fastdeploy::vision::facealign::FaceLandmark1000(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

FaceLandmark1000模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> FaceLandmark1000::Predict(cv::Mat* im, FaceAlignmentResult* result)
> ```
>
> 模型预测接口，输入图像直接输出landmarks结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: landmarks结果, FaceAlignmentResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员变量

用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[128, 128]

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
