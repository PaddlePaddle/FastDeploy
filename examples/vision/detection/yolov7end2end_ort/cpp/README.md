# YOLOv7End2EndORT C++部署示例

本目录下提供`infer.cc`快速完成YOLOv7End2EndORT在CPU/GPU部署的示例。

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. 根据开发环境，下载预编译部署库和samples代码，参考[FastDeploy预编译库](../../../../../docs/quick_start)

以Linux上推理为例，在本目录执行如下命令即可完成编译测试

```bash
mkdir build
cd build
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-gpu-0.3.0.tgz
tar xvf fastdeploy-linux-x64-0.3.0.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-gpu-0.2.1
make -j
# 如果预编译库还没有支持该模型，请从develop分支源码编译最新的SDK

#下载官方转换好的yolov7模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-end2end-ort-nms.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# CPU推理
./infer_demo yolov7-end2end-ort-nms.onnx 000000014439.jpg 0
# GPU推理
./infer_demo yolov7-end2end-ort-nms.onnx 000000014439.jpg 1
# TensorRT + GPU 部署 (暂不支持 会回退到 ORT + GPU)
./infer_demo yolov7-end2end-ort-nms.onnx 000000014439.jpg 2
```

运行完成可视化结果如下图所示

<div align='center'>
  <img width="639" alt="image" src="https://user-images.githubusercontent.com/31974251/186369053-1b578d61-ca70-4755-9671-c9fccf6314a0.png">
</div>

以上命令只适用于Linux或MacOS, Windows下SDK的使用方式请参考:  
- [如何在Windows中使用FastDeploy C++ SDK](../../../../../docs/compile/how_to_use_sdk_on_windows.md)

注意，YOLOv7End2EndORT是专门用于推理YOLOv7中导出模型带[ORT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L87) 版本的End2End模型，不带nms的模型推理请使用YOLOv7类，而 [TRT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L111) 版本的End2End模型请使用YOLOv7End2EndTRT进行推理。

## YOLOv7End2EndORT C++接口

### YOLOv7End2EndORT 类

```c++
fastdeploy::vision::detection::YOLOv7End2EndORT(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const ModelFormat& model_format = ModelFormat::ONNX)
```

YOLOv7End2EndORT 模型加载和初始化，其中model_file为导出的ONNX模型格式。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX时，此参数传入空字符串即可
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX格式

#### Predict函数

> ```c++
> YOLOv7End2EndORT::Predict(cv::Mat* im, DetectionResult* result,
>                           float conf_threshold = 0.25)
> ```
>
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度, DetectionResult说明参考[视觉模型预测结果](../../../../../docs/api/vision_results/)
> > * **conf_threshold**: 检测框置信度过滤阈值，但由于YOLOv7 End2End的模型在导出成ONNX时已经指定了score阈值，因此该参数只有在大于已经指定的阈值时才会有效。

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
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
