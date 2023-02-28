# C# API tutorial

This directory is the implementation of FastDeploy C# SDK, which provides solutions for scenarios where users need C# API.

## How to compile and install

When compiling the fastdeploy library, enable the compilation option --WITH_CSHARPAPI=ON to enable compilation for this directory. The compiled C# library is in the csharp_lib directory after installation. For example, to compile the CPU version of the fastdeploy library under Windows and add the C# API library, you can use the following command

```shell
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64 ^
          -DENABLE_ORT_BACKEND=ON ^
          -DENABLE_PADDLE_BACKEND=ON ^
          -DENABLE_OPENVINO_BACKEND=ON ^
          -DENABLE_VISION=ON ^
          -DENABLE_TEXT=ON ^
          -DWITH_CAPI=ON ^
          -DWITH_CSHARPAPI=ON ^
          -DCMAKE_INSTALL_PREFIX="D:\Paddle\compiled_fastdeploy"
nuget restore #（please execute it when WITH_CSHARPAPI=ON to prepare dependencies in C#)
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```
For more details about compiling the fastdeploy library, please refer to the documentation
- [FastDeploy installation](../docs/cn/build_and_install/README.md)

## how to use

Use the compiled C# dynamic library file fastdeploy_csharp.dll as a reference to use the corresponding API. For usage examples, please refer to the use cases under examples

- [paddleclas](../examples/vision/classification/paddleclas/csharp/README.md)
- [paddledetection](../examples/vision/detection/paddledetection/csharp/README.md)
- [pp-ocrv2](../examples/vision/ocr/PP-OCRv2/csharp/README.md)
- [paddleseg](../examples/vision/segmentation/paddleseg/cpu-gpu/csharp/README.md)

## Other documents

- [How to add C# API to a new model](../docs/en/faq/develop_c_sharp_api_for_a_new_model.md)
- [Vision Results](../docs/api/vision_results/README.md)
