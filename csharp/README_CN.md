# C# API指南

该目录下为FastDeploy C# SDK的接口实现，为用户需要C# API的场景提供解决方案。

## 如何编译安装

在编译fastdeploy库的时候，打开编译选项 --WITH_CSHARPAPI=ON，即可开启对于该目录的编译。编译后的C#库安装后在csharp_lib目录下。例如在Windows下编译CPU版本的fastdeploy库并且加上C# API库可以使用如下命令

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
nuget restore  #（please execute it when WITH_CSHARPAPI=ON to prepare dependencies in C#)
msbuild fastdeploy.sln /m /p:Configuration=Release /p:Platform=x64
msbuild INSTALL.vcxproj /m /p:Configuration=Release /p:Platform=x64
```
关于编译fastdeploy库的更多详情信息可以参考文档
- [FastDeploy安装](../docs/cn/build_and_install/README.md)

## 如何使用

将编译出来的C#动态库文件fastdeploy_csharp.dll作为引用即可以使用相应的API。关于使用示例可以参考examples下的用例

- [paddleclas](../examples/vision/classification/paddleclas/csharp/README_CN.md)
- [paddledetection](../examples/vision/detection/paddledetection/csharp/README_CN.md)
- [pp-ocrv2](../examples/vision/ocr/PP-OCRv2/csharp/README_CN.md)
- [paddleseg](../examples/vision/segmentation/paddleseg/cpu-gpu/csharp/README_CN.md)

## 其它文档

- [如何给新模型增加C# API](../docs/cn/faq/develop_c_sharp_api_for_a_new_model.md)
- [Vision Results](../docs/api/vision_results/README_CN.md)
