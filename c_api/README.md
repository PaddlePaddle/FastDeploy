# C API tutorial

This directory is the implementation of FastDeploy C SDK, which provides solutions for scenarios where users need C API.

## How to compile and install

When compiling the fastdeploy library, turn on the compile option --WITH_CAPI=ON to start compiling for this directory. The compiled C library and fastdeploy library are in the same dynamic library file, such as libfastdeploy.a under Linux. For example, to compile the CPU version of fastdeploy and compile the C API into the library, you can use the following command

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -DENABLE_ORT_BACKEND=ON \
          -DENABLE_PADDLE_BACKEND=ON \
          -DENABLE_OPENVINO_BACKEND=ON \
          -DCMAKE_INSTALL_PREFIX=${PWD}/compiled_fastdeploy_sdk \
          -DENABLE_VISION=ON \
          -DWITH_CAPI=ON \
          -DOPENCV_DIRECTORY=/usr/lib/x86_64-linux-gnu/cmake/opencv4 \
          -DENABLE_TEXT=ON
make -j12
make install
```
For more details about compiling the fastdeploy library, please refer to the documentation
- [FastDeploy installation](../docs/en/build_and_install/README.md)

## How to use

 The header files of the provided C API can be used. If it is manually compiled and installed, such as the above command, after `make install`, the header files will be in the directory ${PWD}/compiled_fastdeploy_sdk/include/fastdeploy_capi/. If it is a downloaded precompiled library, after decompression,  include/fastdeploy_capi/ is the header file of the C API. For usage examples, please refer to the use cases under examples

- [paddleclas](../examples/vision/classification/paddleclas/c/README.md)
- [paddledetection](../examples/vision/detection/paddledetection/c/README.md)
- [pp-ocrv2](../examples/vision/ocr/PP-OCRv2/c/README.md)
- [paddleseg](../examples/vision/segmentation/paddleseg/cpu-gpu/c/README.md)

## Other documents

- [How to Develop C API for a New Model](../docs/en/faq/develop_c_api_for_a_new_model.md)
- [Vision Results](../docs/api/vision_results/README.md)
