English | [中文](../../cn/build_and_install/third_libraries.md)

# Third Library Dependency

FastDeploy will depend on the following third libraries according to compile options.

- OpenCV: OpenCV 3.4.16 library will be downloaded and pre-compiled automatically while ENABLE_VISION=ON.
- ONNX Runimte: ONNX Runtime library will be downloaded automatically while ENABLE_ORT_BACKEND=ON.
- OpenVINO: OpenVINO library will be downloaded automatically while ENABLE_OPENVINO_BACKEND=ON.

You can decide your own third libraries that exist in the environment by setting the following switches.


- OPENCV_DIRECTORY: Specify the OpenCV path in your environment, e.g. `-DOPENCV_DIRECTORY=/usr/lib/aarch64-linux-gnu/cmake/opencv4/`
- ORT_DIRECTORY: Specify the ONNX Runtime path in your environment, e.g.`-DORT_DIRECTORY=/download/onnxruntime-linux-x64-1.0.0`
- OPENVINO_DIRECTORY: Specify the OpenVINO path in your environment, e.g.`-DOPENVINO_DIRECTORY=//download/openvino`