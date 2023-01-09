[English](README.md) | 简体中文

# PIPNet Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成PIPNet在CPU/GPU，以及GPU上通过TensorRT加速部署的示例，保证 FastDeploy 版本 >= 0.7.0 支持PIPNet模型。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/facealign/pipnet/python

# 下载PIPNet模型文件和测试图片以及视频
## 原版ONNX模型
wget https://bj.bcebos.com/paddlehub/fastdeploy/pipnet_resnet18_10x19x32x256_aflw.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/facealign_input.png

# CPU推理
python infer.py --model pipnet_resnet18_10x19x32x256_aflw.onnx --image facealign_input.png --device cpu
# GPU推理
python infer.py --model pipnet_resnet18_10x19x32x256_aflw.onnx --image facealign_input.png --device gpu
# TRT推理
python infer.py --model pipnet_resnet18_10x19x32x256_aflw.onnx --image facealign_input.png --device gpu --backend trt
```

运行完成可视化结果如下图所示

<div width="500">
<img width="470" height="384" float="left" src="https://user-images.githubusercontent.com/67993288/200761400-08491112-56c3-470f-87ac-87be805d5658.jpg">
</div>

## PIPNet Python接口

```python
fd.vision.facealign.PIPNet(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

PIPNet模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX

### predict函数

> ```python
> PIPNet.predict(input_image)
> ```
>
> 模型预测结口，输入图像直接输出landmarks坐标结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.FaceAlignmentResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)


## 其它文档

- [PIPNet 模型介绍](..)
- [PIPNet C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)