# YOLOv6 API说明

## Python API

### YOLOv6类
```
fastdeploy.vision.meituan.YOLOv6(model_file, params_file=None, runtime_option=None, model_format=fd.Frontend.ONNX)
```
YOLOv6模型加载和初始化，当model_format为`fd.Frontend.ONNX`时，只需提供model_file，如`yolov6s.onnx`；当model_format为`fd.Frontend.PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### predict函数
> ```
> YOLOv6.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式
> > * **conf_threshold**(float): 检测框置信度过滤阈值
> > * **nms_iou_threshold**(float): NMS处理过程中iou阈值

示例代码参考[yolov6.py](./yolov6.py)


## C++ API

### YOLOv6类
```
fastdeploy::vision::meituan::YOLOv6(
        const string& model_file,
        const string& params_file = "",
        const RuntimeOption& runtime_option = RuntimeOption(),
        const Frontend& model_format = Frontend::ONNX)
```
YOLOv6模型加载和初始化，当model_format为`Frontend::ONNX`时，只需提供model_file，如`yolov6s.onnx`；当model_format为`Frontend::PADDLE`时，则需同时提供model_file和params_file。

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式

#### Predict函数
> ```
> YOLOv6::Predict(cv::Mat* im, DetectionResult* result,
>                 float conf_threshold = 0.25,
>                 float nms_iou_threshold = 0.5)
> ```
> 模型预测接口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **im**: 输入图像，注意需为HWC，BGR格式
> > * **result**: 检测结果，包括检测框，各个框的置信度
> > * **conf_threshold**: 检测框置信度过滤阈值
> > * **nms_iou_threshold**: NMS处理过程中iou阈值

示例代码参考[cpp/yolov6.cc](cpp/yolov6.cc)

## 其它API使用

- [模型部署RuntimeOption配置](../../../docs/api/runtime_option.md)
