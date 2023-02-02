[English](../../../en/faq/rknpu2/export.md) | 中文

# 导出模型指南

## 简介

Fastdeploy已经简单的集成了onnx->rknn的转换过程。
本教程使用tools/rknpu2/export.py文件导出模型，在导出之前需要编写yaml配置文件。

## 环境要求
在进行转换前请根据[rknn_toolkit2安装文档](./environment.md)检查环境是否已经安装成功。


## export.py 配置参数介绍

| 参数名称            | 是否可以为空     | 参数作用               |
|-----------------|------------|--------------------|
| verbose         | 是，默认值为True | 是否在屏幕上输出转换模型时的具体信息 |
| config_path     | 否          | 配置文件路径             |
| target_platform | 否          | cpu型号              |

## config 配置文件介绍

### config yaml文件模版

```yaml
mean:
  -
    - 128.5
    - 128.5
    - 128.5
std:
  -
    - 128.5
    - 128.5
    - 128.5
model_path: "./scrfd_500m_bnkps_shape640x640.onnx"
outputs_nodes:
do_quantization: True
dataset: "./datasets.txt"
output_folder: "./"
```

### config 配置参数介绍
#### model_path
代表需要转换为RKNN的ONNX格式的模型路径
```yaml
model_path: "./scrfd_500m_bnkps_shape640x640.onnx"
```
#### output_folder
代表最后储存RKNN模型文件的文件夹路径
```yaml
output_folder: "./"
```

#### std 和 mean
如果需要在NPU上进行normalize操作需要配置此参数，并且请自行将参数乘以255，例如你的normalize中mean参数为[0.5,0.5,0.5]时，
配置文件中的mean应该配置为[128.5,128.5,128.5]。 请自行将[128.5,128.5,128.5]换成yaml格式,如下:
```yaml
mean:
  -
    - 128.5
    - 128.5
    - 128.5
std:
  -
    - 128.5
    - 128.5
    - 128.5
```
当然如果在外部进行normalize和permute操作，则无需配置这两个参数。

#### outputs_nodes
输出节点的名字。当整个模型导出时，无语配置改参数。
```yaml
outputs_nodes:
```

#### do_quantization 和 dataset
do_quantization代表是否进行静态量化。dataset表示进行静态量化时的图片数据集目录。
这两个参数配套使用，当do_quantization生效时，dataset才生效。
```yaml
do_quantization: True
dataset: "./datasets.txt"
```

## 如何转换模型
根目录下执行以下代码

```bash
python tools/export.py  --config_path=./config.yaml
```

## 模型导出要注意的事项

* 不建议导出softmax以及argmax算子
