[English](README.md) | 简体中文
# YOLOv7Face C++部署示例

本目录下提供`infer.cc`快速完成YOLOv7Face在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```bash
mkdir build
cd build
# 下载FastDeploy预编译库，用户可在上文提到的`FastDeploy预编译库`中自行选择合适的版本使用
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz # x.x.x > 1.0.2
tar xvf fastdeploy-linux-x64-x.x.x.tgz # x.x.x > 1.0.2
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x # x.x.x > 1.0.2
make -j

#下载官方转换好的YOLOv7Face模型文件和测试图片
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-lite-e.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-tiny-face.onnx

#使用yolov7-tiny-face.onnx模型
# CPU推理
./infer_demo yolov7-tiny-face.onnx test_lite_face_detector_3.jpg 0
# GPU推理
./infer_demo yolov7-tiny-face.onnx test_lite_face_detector_3.jpg 1
# GPU上TensorRT推理
./infer_demo yolov7-tiny-face.onnx test_lite_face_detector_3.jpg 2

#使用yolov7-lite-e.onnx模型
# CPU推理
./infer_demo yolov7-lite-e.onnx test_lite_face_detector_3.jpg 0
# GPU推理
./infer_demo yolov7-lite-e.onnx test_lite_face_detector_3.jpg 1
# GPU上TensorRT推理
./infer_demo yolov7-lite-e.onnx test_lite_face_detector_3.jpg 2
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/49013063/206170111-843febb6-67d6-4c46-a121-d87d003bba21.jpg">

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## YOLOv7Face C++接口

### YOLOv7Face类

```c++
fastdeploy::vision::facedet::YOLOv7Face(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv7Face模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> YOLOv7Face::Predict(cv::Mat* im, FaceDetectionResult* result,
>                 float conf_threshold = 0.3,
>                 float nms_iou_threshold = 0.5)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, FaceDetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)
> > * **conf_threshold**: 检测框置信度过滤阈值
> > * **nms_iou_threshold**: NMS处理过程中iou阈值

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
