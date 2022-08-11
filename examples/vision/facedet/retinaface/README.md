# RetinaFace准备部署模型

## 模型版本说明

- [RetinaFace CommitID:b984b4b](https://github.com/biubug6/Pytorch_Retinaface/commit/b984b4b)
  - （1）[链接中](https://github.com/biubug6/Pytorch_Retinaface/commit/b984b4b)的*.pt通过[导出ONNX模型](#导出ONNX模型)操作后，可进行部署；
  - （2）自己数据训练的RetinaFace CommitID:b984b4b模型，可按照[导出ONNX模型](#导出ONNX模型)后，完成部署。

## 导出ONNX模型

[下载预训练ONNX模型](#下载预训练ONNX模型)已事先转换成ONNX；如果从RetinaFace官方repo下载的模型，需要按如下教程导出ONNX。  

* 下载官方仓库并
```bash
git clone https://github.com/biubug6/Pytorch_Retinaface.git
```
* 下载预训练权重并放在weights文件夹
```text
./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
* 运行convert_to_onnx.py导出ONNX模型文件
```bash
PYTHONPATH=. python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --long_side 640 --cpu
PYTHONPATH=. python convert_to_onnx.py --trained_model ./weights/Resnet50_Final.pth --network resnet50 --long_side 640 --cpu
```
注意：需要先对convert_to_onnx.py脚本中的--long_side参数增加类型约束，type=int.
* 使用onnxsim对模型进行简化
```bash
onnxsim FaceDetector.onnx Pytorch_RetinaFace_mobile0.25-640-640.onnx  # mobilenet
onnxsim FaceDetector.onnx Pytorch_RetinaFace_resnet50-640-640.onnx  # resnet50
```

## 下载预训练ONNX模型

为了方便开发者的测试，下面提供了RetinaFace导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 大小    | 精度    |
|:---------------------------------------------------------------- |:----- |:----- |
| [RetinaFace_mobile0.25-640](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-640-640.onnx) | 1.7MB | - |
| [RetinaFace_mobile0.25-720](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-720-1080.onnx) | 1.7MB | -|
| [RetinaFace_resnet50-640](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_resnet50-720-1080.onnx) | 105MB | - |
| [RetinaFace_resnet50-720](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_resnet50-640-640.onnx) | 105MB | - |





## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
