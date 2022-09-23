# RetinaFace Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`infer.py`快速完成RetinaFace在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision//retinaface/python/

#下载retinaface模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-640-640.onnx
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# CPU推理
python infer.py --model Pytorch_RetinaFace_mobile0.25-640-640.onnx --image test_lite_face_detector_3.jpg --device cpu
# GPU推理
python infer.py --model Pytorch_RetinaFace_mobile0.25-640-640.onnx --image test_lite_face_detector_3.jpg --device gpu
# GPU上使用TensorRT推理
python infer.py --model Pytorch_RetinaFace_mobile0.25-640-640.onnx --image test_lite_face_detector_3.jpg --device gpu --use_trt True
```

运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301763-1b950047-c17f-4819-b175-c743b699c3b1.jpg">

## RetinaFace Python接口

```python
fastdeploy.vision.facedet.RetinaFace(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

RetinaFace模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX

### predict函数

> ```python
> RetinaFace.predict(image_data, conf_threshold=0.25, nms_iou_threshold=0.5)
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

> > * **size**(list[int]): 通过此参数修改预处理过程中resize的大小，包含两个整型元素，表示[width, height], 默认值为[640, 640]
> > * **variance**(list[float]): 通过此参数可以指定retinaface中的方差variance值，默认是[0.1,0.2], 一般不用修改.
> > * **min_sizes**(list[list[int]]): retinaface中的anchor的宽高设置，默认是 {{16, 32}, {64, 128}, {256, 512}}，分别和步长8、16和32对应
> > * **downsample_strides**(list[int]): 通过此参数可以修改生成anchor的特征图的下采样倍数, 包含三个整型元素, 分别表示默认的生成anchor的下采样倍数, 默认值为[8, 16, 32]
> > * **landmarks_per_face**(int): 指定当前模型检测的人脸所带的关键点个数，默认为5.



## 其它文档

- [RetinaFace 模型介绍](..)
- [RetinaFace C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
