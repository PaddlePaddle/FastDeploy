# ResNet准备部署模型

- ResNet部署实现来自[Torchvision](https://github.com/pytorch/vision/tree/v0.12.0)的代码，和[基于ImageNet2012的预训练模型](https://github.com/pytorch/vision/tree/v0.12.0)。

  - （1）[官方库](https://github.com/pytorch/vision/tree/v0.12.0)提供的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）自己数据训练的ResNet模型，按照[导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)操作后，参考[详细部署文档](#详细部署文档)完成部署。


## 导出ONNX模型


  导入[Torchvision](https://github.com/pytorch/vision/tree/v0.12.0)，加载预训练模型，并进行模型转换，具体转换步骤如下。

  ```python
    import torch
    import torchvision.models as models

    model = models.resnet50(pretrained=True)
    batch_size = 1  #批处理大小
    input_shape = (3, 224, 224)   #输入数据,改成自己的输入shape
    # #set the model to inference mode
    model.eval()
    x = torch.randn(batch_size, *input_shape)	# 生成张量
    export_onnx_file = "ResNet50.onnx"			# 目的ONNX文件名
    torch.onnx.export(model,
                        x,
                        export_onnx_file,
                        opset_version=12,
                        input_names=["input"],	# 输入名
                        output_names=["output"],	# 输出名
                        dynamic_axes={"input":{0:"batch_size"},  # 批处理变量
                                        "output":{0:"batch_size"}})
  ```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了ResNet导出的各系列模型，开发者可直接下载使用。（下表中模型的精度来源于源官方库）
| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [ResNet-18](https://bj.bcebos.com/paddlehub/fastdeploy/resnet18.onnx) | 45MB  | |
| [ResNet-34](https://bj.bcebos.com/paddlehub/fastdeploy/resnet34.onnx) | 84MB | |
| [ResNet-50](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50.onnx) | 98MB | |
| [ResNet-101](https://bj.bcebos.com/paddlehub/fastdeploy/resnet101.onnx) | 170MB | |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)

## 版本说明

- 本版本文档和代码基于[Torchvision v0.12.0](https://github.com/pytorch/vision/tree/v0.12.0) 编写
