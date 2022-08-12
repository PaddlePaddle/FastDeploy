# UltraFace C++部署示例

本目录下提供`infer.cc`快速完成UltraFace在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/quick_start/requirements.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/compile/prebuild_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```
mkdir build
cd build
wget https://https://bj.bcebos.com/paddlehub/fastdeploy/cpp/fastdeploy-linux-x64-gpu-0.2.0.tgz
tar xvf fastdeploy-linux-x64-0.2.0.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.2.0
make -j

#下载官方转换好的UltraFace模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320.onnx
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg


# CPU推理
./infer_demo version-RFB-320.onnx test_lite_face_detector_3.jpg 0
# GPU推理
./infer_demo version-RFB-320.onnx test_lite_face_detector_3.jpg 1
# GPU上TensorRT推理
./infer_demo version-RFB-320.onnx test_lite_face_detector_3.jpg 2
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301821-0788483b-a72b-42b0-a566-b6430f184f6e.jpg">

## UltraFace C++接口

### UltraFace类

```
fastdeploy::vision::facedet::UltraFace(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX)
```

UltraFace模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为ONNX格式

#### Predict函数

> ```
> UltraFace::Predict(cv::Mat* im, FaceDetectionResult* result,
>                 float conf_threshold = 0.25,
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

### 类成员变量
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[320, 240]

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
