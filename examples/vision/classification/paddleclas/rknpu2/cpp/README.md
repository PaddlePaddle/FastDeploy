# PaddleClas C++部署示例

本目录下用于展示 ResNet50_vd 模型在RKNPU2上的部署，以下的部署过程以 ResNet50_vd 为例子。

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
├── images  # 存放图片的文件夹
├── infer.cc
├── ppclas_model_dir  # 存放模型文件的文件夹
└── thirdpartys  # 存放sdk的文件夹
```

首先需要先生成目录结构
```bash
mkdir build
mkdir images
mkdir ppclas_model_dir
mkdir thirdpartys
```

## 编译

### 编译并拷贝SDK到thirdpartys文件夹

请参考[RK2代NPU部署库编译](../../../../../../docs/cn/build_and_install/rknpu2.md)仓库编译SDK，编译完成后，将在build目录下生成
fastdeploy-0.0.3目录，请移动它至thirdpartys目录下.

### 拷贝模型文件，以及配置文件至model文件夹
在Paddle动态图模型 -> Paddle静态图模型 -> ONNX模型的过程中，将生成ONNX文件以及对应的yaml配置文件，请将配置文件存放到model文件夹内。
转换为RKNN后的模型文件也需要拷贝至model，转换方案: ([ResNet50_vd RKNN模型](../README.md))。

### 准备测试图片至image文件夹
```bash
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
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
./rknpu_test ./ppclas_model_dir ./images/ILSVRC2012_val_00000010.jpeg
```

## 运行结果展示
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)

## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，
DisablePermute(C++)或`disable_permute(Python)，在预处理阶段禁用数据格式的转换。

## 其它文档
- [ResNet50_vd Python 部署](../python)
- [模型预测结果说明](../../../../../../docs/api/vision_results/)
- [转换ResNet50_vd RKNN模型文档](../README.md)
