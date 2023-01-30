# YOLOv5Seg Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成YOLOv5Seg在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/yolov5seg/python/

#下载yolov5seg模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-seg.onnx
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# CPU推理
python infer.py --model yolov5s-seg.onnx --image 000000014439.jpg --device cpu
# GPU推理
python infer.py --model yolov5s-seg.onnx --image 000000014439.jpg --device gpu
# GPU上使用TensorRT推理
python infer.py --model yolov5s-seg.onnx --image 000000014439.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/19977378/209955620-657bdd1d-574c-40a2-b05d-42b9e5a15ae8.png">

## YOLOv5Seg Python接口

```python
fastdeploy.vision.detection.YOLOv5Seg(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

YOLOv5Seg模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX

### predict函数

```python
YOLOv5Seg.predict(image_data)
```

模型预测结口，输入图像直接输出检测结果。

**参数**

> > * **image_data**(np.ndarray): 输入数据，注意需为HWC，BGR格式

**返回**

> > 返回`fastdeploy.vision.DetectionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

## 其它文档

- [YOLOv5Seg 模型介绍](..)
- [YOLOv5Seg C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
