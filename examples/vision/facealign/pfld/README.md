# PFLD 模型部署

## 模型版本说明

- [PFLD](https://github.com/Hsintao/pfld_106_face_landmarks/commit/e150195)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PFLD 模型](https://github.com/Hsintao/pfld_106_face_landmarks)

## 下载预训练模型

为了方便开发者的测试，下面提供了PFLD导出的各系列模型，开发者可直接下载使用。

| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [pfld-106-v2.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/pfld-106-v2.onnx) | 4.9M | - |
| [pfld-106-v3.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/pfld-106-v3.onnx) | 5.6MB | - |
| [pfld-106-lite.onnx](https://bj.bcebos.com/paddlehub/fastdeploy/pfld-106-lite.onnx) | 1.1MB | - |

## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
