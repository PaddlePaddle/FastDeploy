# FastDeploy安装

- [预编译库下载安装](download_prebuilt_libraries.md)
- [GPU部署环境编译安装](gpu.md)
- [CPU部署环境编译安装](cpu.md)
- [Jetson部署环境编译安装](jetson.md)
- [Android平台部署环境编译安装](android.md)
- [RK平台部署环境编译安装](rk.md)


## FastDeploy编译选项说明

| 选项 | 说明 |
| :--- | :---- |
| ENABLE_ORT_BACKEND | 默认OFF, 是否编译集成ONNX Runtime后端(CPU/GPU上推荐打开) |
| ENABLE_PADDLE_BACKEND | 默认OFF，是否编译集成Paddle Inference后端(CPU/GPU上推荐打开) |
| ENABLE_TRT_BACKEND | 默认OFF，是否编译集成TensorRT后端(GPU上推荐打开) |
| ENABLE_OPENVINO_BACKEND | 默认OFF，是否编译集成OpenVINO后端(CPU上推荐打开) |
| ENABLE_VISION | 默认OFF，是否编译集成视觉模型的部署模块 |
| ENABLE_TEXT | 默认OFF，是否编译集成文本NLP模型的部署模块 |
| WITH_GPU | 默认OFF, 当需要在GPU上部署时，需设置为ON |
| CUDA_DIRECTORY | 默认/usr/local/cuda, 当需要在GPU上部署时，用于指定CUDA(>=11.2)的路径 |
| TRT_DIRECTORY | 当开启TensorRT后端时，必须通过此开关指定TensorRT(>=8.4)的路径 |
| ORT_DIRECTORY | 当开启ONNX Runtime后端时，用于指定用户本地的ONNX Runtime库路径；如果不指定，编译过程会自动下载ONNX Runtime库 |
| OPENCV_DIRECTORY | 当ENABLE_VISION=ON时，用于指定用户本地的OpenCV库路径；如果不指定，编译过程会自动下载OpenCV库 |
| OPENVINO_DIRECTORY | 当开启OpenVINO后端时, 用于指定用户本地的OpenVINO库路径；如果不指定，编译过程会自动下载OpenVINO库 |
