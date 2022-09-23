# YOLOX C++部署示例

本目录下提供`infer.cc`快速完成YOLOX在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.2.1.tgz
tar xvf fastdeploy-linux-x64-0.2.1.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.2.1
make -j

#下载官方转换好的YOLOX模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# CPU推理
./infer_demo yolox_s.onnx 000000014439.jpg 0
# GPU推理
./infer_demo yolox_s.onnx 000000014439.jpg 1
# GPU上TensorRT推理
./infer_demo yolox_s.onnx 000000014439.jpg 2
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301746-04595d76-454a-4f07-8c7d-6f41418f8ae3.jpg">

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

## YOLOX C++接口

### YOLOX类

```c++
fastdeploy::vision::detection::YOLOX(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOX模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> YOLOX::Predict(cv::Mat* im, DetectionResult* result,
>                 float conf_threshold = 0.25,
>                 float nms_iou_threshold = 0.5)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)
> > * **conf_threshold**: 检测框置信度过滤阈值
> > * **nms_iou_threshold**: NMS处理过程中iou阈值

### 类成员变量
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **size**(vector&lt;int&gt;): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[640, 640]
> > * **padding_value**(vector&lt;float&gt;): 通过此参数可以修改图片在resize时候做填充(padding)的值, 包含三个浮点型元素, 分别表示三个通道的值, 默认值为[114, 114, 114]
> > * **is_no_pad**(bool): 通过此参数让图片是否通过填充的方式进行resize, `is_no_pad=ture` 表示不使用填充的方式，默认值为`is_no_pad=false`
> > * **is_decode_exported**(bool): 表示导出的YOLOX的onnx模型文件是否带坐标反算的decode部分, 默认值为`is_decode_exported=false`，官方默认的导出不带decode部分，如果您导出的模型带了decode，请将此参数设置为true  

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
