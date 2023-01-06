English | [简体中文](README_CN.md)
# PaddleClas C++ Deployment Example

This directory demonstrates the deployment of ResNet50_vd model on RKNPU2. The following deployment process takes ResNet50_vd as an example.

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
├── images  # Folder for images
├── infer.cc
├── ppclas_model_dir  # Folder for models
└── thirdpartys  # Folder for sdk
```

First, please build a directory structure
```bash
mkdir build
mkdir images
mkdir ppclas_model_dir
mkdir thirdpartys
```

## Compile

### Compile and Copy SDK to folder thirdpartys

Please refer to [How to Build RKNPU2 Deployment Environment](../../../../../../docs/en/build_and_install/rknpu2.md) to compile SDK.After compiling, the fastdeploy-0.0.3 directory will be created in the build directory, please move it to the thirdpartys directory.

### Copy model and configuration files to folder Model
In the process of Paddle dynamic map model -> Paddle static map model -> ONNX mdoel, ONNX file and the corresponding yaml configuration file will be generated. Please move the configuration file to the folder model. 
After converting to RKNN, the model file also needs to be copied to folder model. Please refer to ([ResNet50_vd RKNN model](../README.md))。

### Prepare Test Images to folder image
```bash
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
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
./rknpu_test ./ppclas_model_dir ./images/ILSVRC2012_val_00000010.jpeg
```

## Results
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)

## Notes
The input requirement for the model on RKNPU is to use NHWC format, and image normalization will be embedded into the model when converting the RKNN model, so we need to call DisablePermute(C++) or disable_permute(Python) first when deploying with FastDeploy to disable data format conversion in the preprocessing stage.

## Other Documents
- [ResNet50_vd Python Deployment](../python)
- [Prediction results](../../../../../../docs/api/vision_results/)
- [Converting ResNet50_vd RKNN model](../README.md)
