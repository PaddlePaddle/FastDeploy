# FastDeploy安装

- [预编译库下载安装](download_prebuilt_libraries.md)
- [GPU部署环境编译安装](gpu.md)
- [CPU部署环境编译安装](cpu.md)
- [Jetson部署环境编译安装](jetson.md)
- [Android平台部署环境编译安装](android.md)


## FastDeploy编译选项说明

| 选项 | 说明 |
| :--- | :---- |
| ENABLE_ORT_BACKEND | 默认OFF, 是否编译集成ONNX Runtime后端(CPU/GPU上推荐打开) |
| ENABLE_PADDLE_BACKEND | 默认OFF，是否编译集成Paddle Inference后端(CPU/GPU上推荐打开) |
| ENABLE_TRT_BACKEND | 默认OFF，是否编译集成TensorRT后端(GPU上推荐打开) |
| ENABLE_OPENVINO_BACKEND | 默认OFF，是否编译集成OpenVINO后端(CPU上推荐打开) |
| ENABLE_VISION | 默认OFF，是否编译集成视觉模型的部署模块 |
| ENABLE_TEXT | 默认OFF，是否编译集成文本NLP模型的部署模块 |
