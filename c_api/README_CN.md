# C API指南

该目录下为FastDeploy C SDK的接口实现，为用户需要C API的场景提供解决方案。

## 如何编译安装

在编译fastdeploy库的时候，打开编译选项 --WITH_CAPI=ON，即可开启对于该目录的编译。编译后的C库和fastdeploy库在同一个动态库文件中，如linux下为libfastdeploy.a。比如，编译CPU版本的fastdeploy并且将C API一同编入库可以使用如下命令

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
关于编译fastdeploy库的更多详情信息可以参考文档
- [FastDeploy安装](../docs/cn/build_and_install/README.md)

## 如何使用

因为提供的C API的头文件即可进行使用，如果是手动编译安装，比如上述命令，`make install`之后，头文件将会在目录${PWD}/compiled_fastdeploy_sdk/include/fastdeploy_capi/下。如果是下载的预编译好的库，解压缩之后include/fastdeploy_capi/下即是C API的头文件。关于使用示例可以参考examples下的用例

- [paddleclas](../examples/vision/classification/paddleclas/c/README_CN.md)
- [paddledetection](../examples/vision/detection/paddledetection/c/README_CN.md)
- [pp-ocrv2](../examples/vision/ocr/PP-OCRv2/c/README_CN.md)
- [paddleseg](../examples/vision/segmentation/paddleseg/cpu-gpu/c/README_CN.md)

## 其它文档

- [如何给新模型增加C API](../docs/cn/faq/develop_c_api_for_a_new_model.md)
- [Vision Results](../docs/api/vision_results/README_CN.md)
