[English](../../en/build_and_install/third_libraries.md) | 简体中文

# 第三方库依赖

FastDeploy当前根据编译选项，会依赖如下第三方依赖

- OpenCV: 当ENABLE_VISION=ON时，会自动下载预编译OpenCV 3.4.16库
- ONNX Runimte: 当ENABLE_ORT_BACKEND=ON时，会自动下载ONNX Runtime库
- OpenVINO: 当ENABLE_OPENVINO_BACKEND=ON时，会自动下载OpenVINO库

用户在实际编译时，可能会根据自身需求集成环境中已有的第三方库，可通出如下开关来配置


- OPENCV_DIRECTORY: 指定环境中的OpenCV路径，如 `-DOPENCV_DIRECTORY=/usr/lib/aarch64-linux-gnu/cmake/opencv4/`
- ORT_DIRECTORY: 指定环境中的ONNX Runtime路径， 如`-DORT_DIRECTORY=/download/onnxruntime-linux-x64-1.0.0`
- OPENVINO_DIRECTORY: 指定环境中的OpenVINO路径， 如`-DOPENVINO_DIRECTORY=//download/openvino`
