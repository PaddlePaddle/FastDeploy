English | [简体中文](README.md)

# ResNet Ready-to-deploy Model

- ResNet Deployment is based on the code of [Torchvision](https://github.com/pytorch/vision/tree/v0.12.0) and [基于ImageNet2012的预训练模型](https://github.com/pytorch/vision/tree/v0.12.0)。

  - （1）Deployment is conducted after [导出ONNX模型](#导出ONNX模型) by the *.pt provided by [官方库](https://github.com/pytorch/vision/tree/v0.12.0)；
  - （2）The ResNet Model trained by personal data should [导出ONNX模型](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B). Please refer to [详细部署文档](#详细部署文档) for deployment.


## Export the ONNX Model


  Import [Torchvision](https://github.com/pytorch/vision/tree/v0.12.0), load the pre-trained model, and conduct model transformation as the following steps.

  ```python
    import torch
    import torchvision.models as models

    model = models.resnet50(pretrained=True)
    batch_size = 1  #Batch size
    input_shape = (3, 224, 224)   #Input data, and change to personal input shape
    # #set the model to inference mode
    model.eval()
    x = torch.randn(batch_size, *input_shape)	# Generate tensor
    export_onnx_file = "ResNet50.onnx"			# Purpose ONNX file name
    torch.onnx.export(model,
                        x,
                        export_onnx_file,
                        opset_version=12,
                        input_names=["input"],	# Input name
                        output_names=["output"],	# Output name
                        dynamic_axes={"input":{0:"batch_size"},  # Batch variables
                                        "output":{0:"batch_size"}})
  ```

## Download Pre-trained ONNX Model

For developers' testing, models exported by ResNet are provided below. Developers can download them directly. (The model accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [ResNet-18](https://bj.bcebos.com/paddlehub/fastdeploy/resnet18.onnx) | 45MB  | |
| [ResNet-34](https://bj.bcebos.com/paddlehub/fastdeploy/resnet34.onnx) | 84MB | |
| [ResNet-50](https://bj.bcebos.com/paddlehub/fastdeploy/resnet50.onnx) | 98MB | |
| [ResNet-101](https://bj.bcebos.com/paddlehub/fastdeploy/resnet101.onnx) | 170MB | |


## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Release Note

- Document and code are based on [Torchvision v0.12.0](https://github.com/pytorch/vision/tree/v0.12.0) 
