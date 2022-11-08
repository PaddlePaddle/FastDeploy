# SCRFD C++部署示例

本目录下提供`infer.cc`快速完成SCRFD在NPU加速部署的示例。

在部署前，需确认以下两个步骤:

1. 软硬件环境满足要求
2. 根据开发环境，下载预编译部署库或者从头编译FastDeploy仓库

以上步骤请参考[RK2代NPU部署库编译](../../../../../../docs/cn/build_and_install/rknpu2.md)实现

## 生成基本目录文件

该例程由以下几个部分组成
```text
.
├── CMakeLists.txt
├── build  # 编译文件夹
├── image  # 存放图片的文件夹
├── infer_cpu_npu.cc
├── infer_cpu_npu.h
├── main.cc
├── model  # 存放模型文件的文件夹
└── thirdpartys  # 存放sdk的文件夹
```

首先需要先生成目录结构
```bash
mkdir build
mkdir images
mkdir model
mkdir thirdpartys
```

## 编译

### 编译并拷贝SDK到thirdpartys文件夹

请参考[RK2代NPU部署库编译](../../../../../../docs/cn/build_and_install/rknpu2.md)仓库编译SDK，编译完成后，将在build目录下生成
fastdeploy-0.6.0目录，请移动它至thirdpartys目录下.

### 拷贝模型文件，以及配置文件至model文件夹
在Paddle动态图模型 -> Paddle静态图模型 -> ONNX模型的过程中，将生成ONNX文件以及对应的yaml配置文件，请将配置文件存放到model文件夹内。
转换为RKNN后的模型文件也需要拷贝至model。

### 准备测试图片至image文件夹
```bash
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg
cp test_lite_face_detector_3.jpg ./images
```

### 编译example

```bash
cd build
cmake ..
make -j8
make install
```
## 运行例程

```bash
cd ./build/install
./rknpu_test
```
运行完成可视化结果如下图所示

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301789-1981d065-208f-4a6b-857c-9a0f9a63e0b1.jpg">

- [模型介绍](../../)
- [Python部署](../python)
- [视觉模型预测结果](../../../../../../docs/api/vision_results/)
