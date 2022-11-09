# 导出模型指南

## 简介

Fastdeploy已经简单的集成了onnx->rknn的转换过程。本教程使用tools/export.py文件导出模型，在导出之前需要编写yaml配置文件。
在进行转换前请根据[rknn_toolkit2安装文档](./install_rknn_toolkit2.md)检查环境是否已经安装成功。


## export.py 配置参数介绍

| 参数名称            | 是否可以为空     | 参数作用               |
|-----------------|------------|--------------------|
| verbose         | 是，默认值为True | 是否在屏幕上输出转换模型时的具体信息 |
| config_path     | 否          | 配置文件路径             |

## config 配置文件介绍

### config yaml文件模版

```yaml
model_path: ./portrait_pp_humansegv2_lite_256x144_pretrained.onnx
output_folder: ./
target_platform: RK3588
normalize:
  mean: [[0.5,0.5,0.5]]
  std: [[0.5,0.5,0.5]]
outputs: None
```

### config 配置参数介绍
* model_path: 模型储存路径
* output_folder: 模型储存文件夹名字
* target_platform: 模型跑在哪一个设备上，只能为RK3588或RK3568
* normalize: 配置在NPU上的normalize操作，有std和mean两个参数
  * std: 如果在外部做normalize操作，请配置为[1/255,1/255,1/255]
  * mean: 如果在外部做normalize操作，请配置为[0,0,0]
* outputs: 输出节点列表，如果使用默认输出节点，请配置为None

## 如何转换模型
根目录下执行以下代码

```bash
python tools/export.py  --config_path=./config.yaml
```

## 模型导出要注意的事项

* 请不要导出带softmax和argmax的模型，这两个算子存在bug，请在外部进行运算
