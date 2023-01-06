English | [简体中文](README_CN.md)
# PaddleSeg Deployment Examples for C++

This directory demonstrates the deployment of PaddleSeg series models on RKNPU2. The following deployment process takes PHumanSeg as an example.

Before deployment, the following two steps need to be confirmed:

1. Hardware and software environment meets the requirements.
2. Download the pre-compiled deployment repository or compile the FastDeploy repository from scratch according to the development environment.

For the above steps, please refer to [How to Build RKNPU2 Deployment Environment](../../../../../../docs/en/build_and_install/rknpu2.md).

## Generate Basic Directory Files

The routine consists of the following parts:
```text
.
├── CMakeLists.txt
├── build  # Compile Folder
├── image  # Folder for images
├── infer_cpu_npu.cc
├── infer_cpu_npu.h
├── main.cc
├── model  # Folder for models
└── thirdpartys  # Folder for sdk
```

First, please build a directory structure
```bash
mkdir build
mkdir images
mkdir model
mkdir thirdpartys
```

## Compile

### Compile and Copy SDK to folder thirdpartys

Please refer to [How to Build RKNPU2 Deployment Environment](../../../../../../docs/en/build_and_install/rknpu2.md) to compile SDK.After compiling, the fastdeploy-0.0.3 directory will be created in the build directory, please move it to the thirdpartys directory.

### Copy model and configuration files to folder Model
In the process of Paddle dynamic map model -> Paddle static map model -> ONNX mdoel, ONNX file and the corresponding yaml configuration file will be generated. Please move the configuration file to the folder model. 
After converting to RKNN, the model file also needs to be copied to folder model. Run the following command to download and use (the model file is RK3588. RK3568 needs to be [reconverted to PPSeg RKNN model](../README.md)).

### Prepare Test Images to folder image
```bash
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip
unzip -qo images.zip
```

### Compile example

```bash
cd build
cmake ..
make -j8
make install
```

## Running Routines

```bash
cd ./build/install
./rknpu_test model/Portrait_PP_HumanSegV2_Lite_256x144_infer/ images/portrait_heng.jpg
```

## Notes
The input requirement for the model on RKNPU is to use NHWC format, and image normalization will be embedded into the model when converting the RKNN model, so we need to call DisableNormalizeAndPermute(C++) or disable_normalize_and_permute(Python) first when deploying with FastDeploy to disable normalization and data format conversion in the preprocessing stage.

- [Model Description](../../)
- [Python Deployment](../python)
- [Convert PPSeg and RKNN model](../README.md)
