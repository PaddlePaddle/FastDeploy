# FastDeploy编译

本文档说明编译C++预测库、Python预测库两种编译过程，根据编译的平台参考如下文档

- [Linux & Mac 编译](linux_and_mac.md)
- [Windows编译](windows.md)

其中编译过程中，各平台上编译选项如下表所示

| 选项 | 作用 | 备注 |
|:---- | :--- | :--- |
| ENABLE_ORT_BACKEND | 启用ONNXRuntime推理后端，默认ON | 默认支持CPU，开启WITH_GPU后，同时支持GPU |
| ENABLE_PADDLE_BACKEND | 启用Paddle Inference推理后端，默认OFF | 默认支持CPU，开启WITH_GPU后，同时支持GPU |
| ENABLE_TRT_BACKEND | 启用TensorRT推理后端，默认OFF | 仅支持GPU |
| WITH_GPU | 是否开启GPU使用，默认OFF | 当设为TRUE，编译后将支持Nvidia GPU部署 |
| CUDA_DIRECTORY | 指定编译时的CUDA路径，默认为/usr/local/cuda |
| TRT_DIRECTORY | 当启用TensorRT推理后端时，需通过此参数指定TensorRT路径 |
| ENABLE_VISION | 启用视觉模型模块，默认为ON |
