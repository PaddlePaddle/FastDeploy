# BlazeFace Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成YOLOv7Face在CPU/GPU部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/facedet/yolov7face/python/

#下载YOLOv7Face模型文件和测试图片
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/blazeface-1000e.tgz

#使用blazeface-1000e模型
# CPU推理
python infer.py --model blazeface-1000e/ --image test_lite_face_detector_3.jpg --device cpu
# GPU推理
python infer.py --model blazeface-1000e/ --image test_lite_face_detector_3.jpg --device gpu

#使用blazeface-1000e模型
# CPU推理
python infer.py --model blazeface-1000e/ --image test_lite_face_detector_3.jpg --device cpu
# GPU推理
python infer.py --model blazeface-1000e/ --image test_lite_face_detector_3.jpg --device gpu
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301839-a29aefae-16c9-4196-bf9d-9c6cf694f02d.jpg">

## BlazeFace Python接口

```python
fastdeploy.vision.facedet.BlzaeFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.PADDLE)
```

BlazeFace模型加载和初始化

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为PADDLE

### predict函数

> ```python
> BlazeFace.predict(image_data, conf_threshold=0.3, nms_iou_threshold=0.5)
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
> > 返回`fastdeploy.vision.FaceDetectionResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)

## 其它文档

- [BlazeFace 模型介绍](..)
- [BlazeFace C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
