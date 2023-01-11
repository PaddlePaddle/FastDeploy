[English](README.md) | 简体中文
# RobustVideoMatting C++部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上 RobustVideoMatting 推理为例，在本目录执行如下命令即可完成编译测试，支持此模型需保证FastDeploy版本0.7.0以上(x.x.x>=0.7.0)

本目录下提供`infer.cc`快速完成RobustVideoMatting在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# 下载RobustVideoMatting模型文件和测试图片以及视频
## 原版ONNX模型
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_fp32.onnx
## 为加载TRT特殊处理ONNX模型
wget https://bj.bcebos.com/paddlehub/fastdeploy/rvm_mobilenetv3_trt.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_input.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/matting_bgr.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/video.mp4

# CPU推理
./infer_demo rvm_mobilenetv3_fp32.onnx matting_input.jpg matting_bgr.jpg 0
# GPU推理
./infer_demo rvm_mobilenetv3_fp32.onnx matting_input.jpg matting_bgr.jpg 1
# TRT推理
./infer_demo rvm_mobilenetv3_trt.onnx matting_input.jpg matting_bgr.jpg 2
```

运行完成可视化结果如下图所示
<div width="840">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852040-759da522-fca4-4786-9205-88c622cd4a39.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852587-48895efc-d24a-43c9-aeec-d7b0362ab2b9.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852116-cf91445b-3a67-45d9-a675-c69fe77c383a.jpg">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/67993288/186852554-6960659f-4fd7-4506-b33b-54e1a9dd89bf.jpg">
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## RobustVideoMatting C++接口

```c++
fastdeploy::vision::matting::RobustVideoMatting(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

RobustVideoMatting模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> RobustVideoMatting::Predict(cv::Mat* im, MattingResult* result)
> ```
>
> 模型预测接口，输入图像直接输出抠图结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 抠图结果， MattingResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)


## 其它文档

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
