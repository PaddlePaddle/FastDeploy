# YOLOv5Cls Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`infer.py`快速完成YOLOv5Cls在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/classification/yolov5cls/python/

#下载yolov5cls模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5n-cls.onnx
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# CPU推理
python infer.py --model yolov5n-cls.onnx --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1
# GPU推理
python infer.py --model yolov5n-cls.onnx --image ILSVRC2012_val_00000010.jpeg --device gpu --topk 1
# GPU上使用TensorRT推理
python infer.py --model yolov5n-cls.onnx --image ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True
```

运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

## YOLOv5Cls Python接口

```python
fastdeploy.vision.detection.YOLOv5Cls(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv5Cls模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX

### predict函数

> ```python
> YOLOv5Cls.predict(image_data, topk=1)
> ```
>
> 模型预测结口，输入图像直接输出分类topk结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式
> > * **topk**(int):返回预测概率最高的topk个分类结果，默认为1

> **返回**
>
> > 返回`fastdeploy.vision.ClassifyResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)


## 其它文档

- [YOLOv5Cls 模型介绍](..)
- [YOLOv5Cls C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
