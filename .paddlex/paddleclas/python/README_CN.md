[English](README.md) | 简体中文
# PaddleClas模型 Python部署示例

本目录下提供`infer.py`快速完成PaddleClas系列模型在CPU/GPU，以及GPU上通过TensorRT加速部署的示例。执行如下脚本即可完成

```bash
# 找到部署包内的模型路径，例如ResNet50

# 准备一张测试图片，例如test.jpg

# CPU推理
python infer.py --model ResNet50 --image test.jpg --device cpu --topk 1
# GPU推理
python infer.py --model ResNet50 --image test.jpg --device gpu --topk 1
# GPU上使用TensorRT推理 （注意：TensorRT推理第一次运行，有序列化模型的操作，有一定耗时，需要耐心等待）
ppython infer.py --model ResNet50 --image test.jpg --device gpu --use_trt True --topk 1
# IPU推理（注意：IPU推理首次运行会有序列化模型的操作，有一定耗时，需要耐心等待）
python infer.py --model ResNet50 --image test.jpg --device ipu --topk 1
# 昆仑芯XPU推理
python infer.py --model ResNet50 --image test.jpg --device kunlunxin --topk 1
# 华为昇腾NPU推理
python infer.py --model ResNet50 --image test.jpg --device ascend --topk 1
```

## PaddleClasModel Python接口

```python
fd.vision.classification.PaddleClasModel(model_file, params_file, config_file, runtime_option=None, model_format=ModelFormat.PADDLE)
```

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
> > 返回`fastdeploy.vision.ClassifyResult`结构体，结构体说明参考文档[视觉模型预测结果](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/classification_result_CN.md)
