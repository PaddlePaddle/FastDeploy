# PaddleClas模型 Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/environment.md)  
- 2. FastDeploy Python whl包安装，参考[FastDeploy Python安装](../../../../../docs/quick_start)

本目录下提供`infer.py`快速完成ResNet50_vd在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
#下载部署示例代码
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd  FastDeploy/examples/vision/classification/paddleclas/python

# 下载ResNet50_vd模型文件和测试图片
wget https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz
tar -xvf ResNet50_vd_infer.tgz
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# CPU推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device cpu --topk 1
# GPU推理
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --topk 1
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model ResNet50_vd_infer --image ILSVRC2012_val_00000010.jpeg --device gpu --use_trt True --topk 1
```

运行完成后返回结果如下所示
```bash
ClassifyResult(
label_ids: 153,
scores: 0.686229,
)
```

## PaddleClasModel Python接口

```python
fd.vision.classification.PaddleClasModel(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

PaddleClas模型加载和初始化，其中model_file, params_file为训练模型导出的Paddle inference文件，具体请参考其文档说明[模型导出](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/inference_deployment/export_model.md#2-%E5%88%86%E7%B1%BB%E6%A8%A1%E5%9E%8B%E5%AF%BC%E5%87%BA)

**参数**

> * **model_file**(str): 模型文件路径
> * **params_file**(str): 参数文件路径
> * **config_file**(str): 推理部署配置文件
> * **runtime_option**(RuntimeOption): 后端推理配置，默认为None，即采用默认配置
> * **model_format**(ModelFormat): 模型格式，默认为Paddle格式

### predict函数

> ```python
> PaddleClasModel.predict(input_image, topk=1)
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

- [PaddleClas 模型介绍](..)
- [PaddleClas C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [如何切换模型推理后端引擎](../../../../../docs/runtime/how_to_change_backend.md)
