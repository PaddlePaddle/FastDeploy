# YOLOv5Lite C++部署示例

本目录下提供`infer.cc`快速完成YOLOv5Lite在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

以Linux上CPU推理为例，在本目录执行如下命令即可完成编译测试

```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-0.3.0.tgz
tar xvf fastdeploy-linux-x64-0.3.0.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-0.3.0
make -j

#下载官方转换好的YOLOv5Lite模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-g-sim-640.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# CPU推理
./infer_demo v5Lite-g-sim-640.onnx 000000014439.jpg 0
# GPU推理
./infer_demo v5Lite-g-sim-640.onnx 000000014439.jpg 1
# GPU上TensorRT推理
./infer_demo v5Lite-g-sim-640.onnx 000000014439.jpg 2
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301943-263c8153-a52a-4533-a7c1-ee86d05d314b.jpg">

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/cn/faq/use_sdk_on_windows.md)

## YOLOv5Lite C++接口

### YOLOv5Lite类

```c++
fastdeploy::vision::detection::YOLOv5Lite(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv5Lite模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> YOLOv5Lite::Predict(cv::Mat* im, DetectionResult* result,
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
> > * **is_mini_pad**(bool): 通过此参数可以将resize之后图像的宽高这是为最接近`size`成员变量的值, 并且满足填充的像素大小是可以被`stride`成员变量整除的。默认值为`is_mini_pad=false`
> > * **stride**(int): 配合`stris_mini_pad`成员变量使用, 默认值为`stride=32`

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
