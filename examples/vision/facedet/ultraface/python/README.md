# UltraFace Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/the%20software%20and%20hardware%20requirements.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`infer.py`快速完成UltraFace在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/facedet/ultraface/python/

#下载ultraface模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320.onnx
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# CPU推理
python infer.py --model version-RFB-320.onnx --image test_lite_face_detector_3.jpg --device cpu
# GPU推理
python infer.py --model version-RFB-320.onnx --image test_lite_face_detector_3.jpg --device gpu
# GPU上使用TensorRT推理
python infer.py --model version-RFB-320.onnx --image test_lite_face_detector_3.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301821-0788483b-a72b-42b0-a566-b6430f184f6e.jpg">

## UltraFace Python接口

```python
fastdeploy.vision.facedet.UltraFace(model_file, params_file=None, runtime_option=None, model_format=Frontend.ONNX)
```

UltraFace模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(Frontend): 模型格式，默认为ONNX

### predict函数

> ```python
> UltraFace.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
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

### 类成员属性
#### 预处理参数
用户可按照自己的实际需求，修改下列预处理参数，从而影响最终的推理和部署效果

> > * **size**(list[int]): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[320, 240]

## 其它文档

- [UltraFace 模型介绍](..)
- [UltraFace C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
