# YOLOX Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/quick_start/requirements.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start/install.md)

本目录下提供`infer.py`快速完成YOLOX在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vison/detection/yolox/python/

#下载YOLOX模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU推理
python infer.py --model yolox_s.onnx --image 000000014439.jpg --device cpu
# GPU推理
python infer.py --model yolox_s.onnx --image 000000014439.jpg --device gpu
# GPU上使用TensorRT推理 (TensorRT在SDK中已包含，不需要单独安装)
python infer.py --model yolox_s.onnx --image 000000014439.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301746-04595d76-454a-4f07-8c7d-6f41418f8ae3.jpg">

## YOLOX Python接口

```
fastdeploy.vision.detection.YOLOX(model_file, params_file=None, runtime_option=None, model_format=Frontend.ONNX)
```

YOLOX模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为ONNX

### predict函数

> ```
> YOLOX.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
> ```
>
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

> >* **size**(list[int]): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[640, 640]
> >* **padding_value**(list[float]): 通过此参数可以修改图片在resize时候做填充(padding)的值, 包含三个浮点型元素, 分别表示三个通道的值, 默认值为[114, 114, 114]
> >* **is_decode_exported**(bool): 表示导出的YOLOX的onnx模型文件是否带坐标反算的decode部分, 默认值为`is_decode_exported=False`，官方默认的导出不带decode部分，如果您导出的模型带了decode，请将此参数设置为True  



## 其它文档

- [YOLOX 模型介绍](..)
- [YOLOX C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
