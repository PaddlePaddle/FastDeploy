# FastDeploy编译

本文档说明编译C++预测库、Python预测库两种编译过程，根据编译的平台参考如下文档

- [Linux & Mac 编译](linux_and_mac.md)
- [Windows编译](windows.md)

其中编译过程中，各平台上编译选项如下表所示

| 选项 | 作用 | 备注 |
|:---- | :--- | :--- |
| ENABLE_ORT_BACKEND | 启用ONNXRuntime推理后端，默认ON | - |
| WITH_GPU | 是否开启GPU使用，默认OFF | 当设为TRUE时，须通过CUDA_DIRECTORY指定cuda目录，如/usr/local/cuda; Mac上不支持设为ON |
| ENABLE_TRT_BACKEND | 启用TensorRT推理后端，默认OFF | 当设为TRUE时，需通过TRT_DIRECTORY指定tensorrt目录，如/usr/downloads/TensorRT-8.4.0.1; Mac上不支持设为ON|
| ENABLE_VISION | 编译集成视觉模型模块，包括OpenCV的编译集成，默认OFF | - |
| ENABLE_PADDLE_FRONTEND | 编译集成Paddle2ONNX，默认ON | - |
| ENABLE_DEBUG | 当为ON时，支持输出DEBUG信息，但可能会有性能损耗，默认OFF | - |
