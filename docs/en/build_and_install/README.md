English | [中文](../../cn/build_and_install/README.md)

# Install FastDeploy - Tutorials

## Install Prebuilt FastDeploy
- [Install Prebuilt FastDeploy Libraries](download_prebuilt_libraries.md)

## Build FastDeploy and Install

- [Build and Install on GPU Platform](gpu.md)
- [Build and Install on CPU Platform](cpu.md)
- [Build and Install on IPU Platform](ipu.md)
- [Build and Install on Nvidia Jetson Platform](jetson.md)
- [Build and Install on Android Platform](android.md)
- [Build and Install on RV1126 Platform](rv1126.md)
- [Build and Install on RK3588 Platform](rknpu2.md)
- [Build and Install on A311D Platform](a311d.md)
- [Build and Install on KunlunXin XPU Platform](kunlunxin.md)
- [Build and Install on Huawei Ascend Platform](huawei_ascend.md)
- [Build and Install on SOPHGO Platform](sophgo.md)

## Build options

| option | description |
| :--- | :---- |
| ENABLE_ORT_BACKEND | Default OFF, whether to enable ONNX Runtime backend(CPU/GPU) |
| ENABLE_PADDLE_BACKEND | Default OFF，whether to enable Paddle Inference backend(CPU/GPU) |
| ENABLE_TRT_BACKEND | Default OFF，whether to enable TensorRT backend(GPU) |
| ENABLE_OPENVINO_BACKEND | Default OFF，whether to enable OpenVINO backend(CPU) |
| ENABLE_VISION | Default OFF，whether to enable vision models deployment module |
| ENABLE_TEXT | Default OFF，whether to enable text models deployment module |
| WITH_GPU | Default OFF, if build on GPU, this needs to be ON |
| WITH_KUNLUNXIN | Default OFF，if deploy on KunlunXin XPU，this needs to be ON |
| WITH_TIMVX | Default OFF，if deploy on RV1126/RV1109/A311D，this needs to be ON |
| WITH_ASCEND | Default OFF，if deploy on Huawei Ascend，this needs to be ON |
| CUDA_DIRECTORY | Default /usr/local/cuda, if build on GPU, this defines the path of CUDA(>=11.2) |
| TRT_DIRECTORY | If build with ENABLE_TRT_BACKEND=ON, this defines the path of TensorRT(>=8.4) |
| ORT_DIRECTORY | [Optional] If build with ENABLE_ORT_BACKEND=ON, this flag defines the path of ONNX Runtime, but if this flag is not set, it will download ONNX Runtime library automatically |
| OPENCV_DIRECTORY | [Optional] If build with ENABLE_VISION=ON, this flag defines the path of OpenCV, but if this flag is not set, it will download OpenCV library automatically |
| OPENVINO_DIRECTORY | [Optional] If build WITH ENABLE_OPENVINO_BACKEND=ON, this flag defines the path of OpenVINO, but if this flag is not set, it will download OpenVINO library automatically |
