# PFLD Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

本目录下提供`infer.py`快速完成PFLD在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/facealign/pfld/python

# 下载PFLD模型文件和测试图片以及视频
## 原版ONNX模型
wget https://bj.bcebos.com/paddlehub/fastdeploy/pfld-106-lite.onnx
wget https://bj.bcebos.com/paddlehub/fastdeploy/facealign_input.png

# CPU推理
python infer.py --model pfld-106-lite.onnx --image facealign_input.png --device cpu
# GPU推理
python infer.py --model pfld-106-lite.onnx --image facealign_input.png --device gpu
# TRT推理
python infer.py --model pfld-106-lite.onnx --image facealign_input.png --device gpu --use_trt True
```

运行完成可视化结果如下图所示

<div width="240">
<img width="200" height="200" float="left" src="https://user-images.githubusercontent.com/19977378/197931737-c2d8e760-a76d-478a-a6c9-4574fb5c70eb.png">
</div>

## PFLD Python接口

```python
fd.vision.facealign.PFLD(model_file, params_file=None, runtime_option=None, model_format=ModelFormat.ONNX)
```

PFLD模型加载和初始化，其中model_file为导出的ONNX模型格式

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径，当模型格式为ONNX格式时，此参数无需设定
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为ONNX

### predict函数

> ```python
> PFLD.predict(input_image)
> ```
>
> 模型预测结口，输入图像直接输出人脸关键点结果。
>
> **参数**
>
> > * **input_image**(np.ndarray): 输入数据，注意需为HWC，BGR格式

> **返回**
>
> > 返回`fastdeploy.vision.FaceAlignmentResult`结构体，结构体说明参考文档[视觉模型预测结果](../../../../../docs/api/vision_results/)


## 其它文档

- [PFLD 模型介绍](..)
- [PFLD C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/cn/faq/how_to_change_backend.md)
