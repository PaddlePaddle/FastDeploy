# FastDeploy Compile

This document outlines the compilation process for C++ predictive libraries and Python predictive libraries. Please refer to the following documentation according to your platform:

- [build on Linux & Mac ](build_linux_and_mac.md)
- [build on Windows](build_windows.md)

The compilation options on each platform are listed in the table below:

| Options               | Function                                                                    | Note                                                                          |
|:--------------------- |:--------------------------------------------------------------------------- |:----------------------------------------------------------------------------- |
| ENABLE_ORT_BACKEND    | Enable ONNXRuntime inference backend, The default is ON.                    | CPU is supported by default, and with WITH_GPU enabled, GPU is also supported |
| ENABLE_PADDLE_BACKEND | Enable Paddle Inference backend. The default is OFF.                        | CPU is supported by default, and with WITH_GPU enabled, GPU is also supported |
| ENABLE_TRT_BACKEND    | Enable TensorRT inference backend. The default is OFF.                      | GPU only                                                                      |
| WITH_GPU              | Whether to enable GPU, The default is OFF.                                  | When set to TRUE, the compilation will support Nvidia GPU deployments         |
| CUDA_DIRECTORY        | Choose the path to CUDA for compiling, the default is /usr/local/cuda       | CUDA 11.2 and above                                                           |
| TRT_DIRECTORY         | When the TensorRT inference backend is enabled, choose the path to TensorRT | TensorRT 8.4 and above                                                        |
| ENABLE_VISION         | Enable the visual model module. The default is ON                           |                                                                               |

FastDeploy allows users to choose their own backend for compilation. Paddle Inference, ONNXRuntime, and TensorRT ( with ONNX format ) are currently being supported. The models supported by FastDeploy have been validated on different backends. It will automatically select the available backend or prompt the user accordingly if no backend is available. E.g. YOLOv7 currently only supports the ONNXRuntime and TensorRT. If developers do not enable these two backends, they will be prompted that no backend is available.
