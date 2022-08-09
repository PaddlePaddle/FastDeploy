# YOLOv7 Python部署示例

如未安装FastDeploy Python环境，请参考下面文档安装。

- [FastDeploy Python安装](../../../../../docs/quick_start/install.md)

本目录下提供`infer.py`快速完成YOLOv7在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```
#下载yolov7模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000087038.jpg

python infer.py --model yolov7.onnx --image 000000087038.jpg --device cpu
```

运行完成可视化结果如下图所示


## YOLOv7 Python接口

```
fastdeploy.vision.detection.YOLOv7(model_file, params_file=None, runtime_option=None, model_format=Frontend.ONNX)
```

YOLOv7模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为ONNX

### predict函数

> ```
> YOLOv7.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
> 模型预测结口，输入图像直接输出检测结果。
>
> **参数**
>
> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式
> > * **conf_threshold**(float): 检测框置信度过滤阈值
> > * **nms_iou_threshold**(float): NMS处理过程中iou阈值

> **返回**
>
> > 返回`fastdeploy.vision.DetectionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

### 类成员属性

> > * **size**(list | tuple): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[640, 640]


## 其它文档

- [YOLOv7 模型介绍](..)
- [YOLOv7 C++部署（../cpp）
- [模型预测结果说明](../../../../../docs/api/vision_results/)
