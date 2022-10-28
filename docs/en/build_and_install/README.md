# Install FastDeploy - Tutorials

- [How to Install FastDeploy Prebuilt Libraries](download_prebuilt_libraries.md)
- [How to Build and Install FastDeploy on GPU Platform](gpu.md)
- [How to Build and Install FastDeploy Library on CPU Platform](cpu.md)
- [How to Build and Install FastDeploy Library on Nvidia Jetson Platform](jetson.md)
- [How to Build and Install FastDeploy Library on Android Platform](android.md)


## Build options

| option | description |
| :--- | :---- |
| ENABLE_ORT_BACKEND | Default OFF, whether to enable ONNX Runtime backend(CPU/GPU) |
| ENABLE_PADDLE_BACKEND | Default OFF，whether to enable Paddle Inference backend(CPU/GPU) |
| ENABLE_TRT_BACKEND | Default OFF，whether to enable TensorRT backend(GPU) |
| ENABLE_OPENVINO_BACKEND | Default OFF，whether to enable OpenVINO backend(CPU) |
| ENABLE_VISION | Default OFF，whether to enable vision models deployment module |
| ENABLE_TEXT | Default OFF，whether to enable text models deployment module |
| WITH_GPU | Default OFF, if build on GPU, this need to be ON |
| CUDA_DIRECTORY | Default /usr/local/cuda, if build on GPU, this defines the path of CUDA(>=11.2) |
| TRT_DIRECTORY | If build with ENABLE_TRT_BACKEND=ON, this defines the path of TensorRT(>=8.4) |
| ORT_DIRECTORY | [Optional] If build with ENABLE_ORT_BACKEND=ON, this flag defines the path of ONNX Runtime, but if this flag is not set, it will download ONNX Runtime library automatically |
| OPENCV_DIRECTORY | [Optional] If build with ENABLE_VISION=ON, this flag defines the path of OpenCV, but if this flag is not set, it will download OpenCV library automatically |
| OPENVINO_DIRECTORY | [Optional] If build WITH ENABLE_OPENVINO_BACKEND=ON, this flag defines the path of OpenVINO, but if this flag is not set, it will download OpenVINO library automatically |
