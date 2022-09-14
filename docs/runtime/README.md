# FastDeploy推理后端

FastDeploy当前已集成多种推理后端，如下表格列出FastDeploy集成的各后端，与在FastDeploy中其支持的平台、硬件等信息

| 推理后端 | 支持平台 | 支持硬件 | 支持模型格式 |
| :------- | :------- | :------- | :---- | 
| Paddle Inference | Windows(x64)/Linux(x64) | GPU/CPU | Paddle |
| ONNX Runtime | Windows(x64)/Linux(x64/aarch64) | GPU/CPU | Paddle/ONNX |
| TensorRT | Windows(x64)/Linux(x64/jetson) | GPU | Paddle/ONNX |
| OpenVINO | Windows(x64)/Linux(x64) | CPU | Paddle/ONNX |
| Poros[进行中] | Linux(x64) | CPU/GPU | TorchScript |

FastDeploy中各后端独立，用户在自行编译时可以选择开启其中一种或多种后端，FastDeploy中的`Runtime`模块为所有后端提供了统一的使用API，Runtime使用方式参阅文档[FastDeploy Runtime使用文档](usage.md)


## 其它文档

- [FastDeploy编译](../compile)
