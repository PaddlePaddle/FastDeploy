English | [简体中文](README_CN.md)
# YOLOv5 C++ Deployment Example

`infer.cc` in this directory provides a quick example of accelerated deployment of the yolov5s model on SOPHGO BM1684x.

Before deployment, the following two steps need to be confirmed:

1. Hardware and software environment meets the requirements.
2. Compile the FastDeploy repository from scratch according to the development environment.

For the above steps, please refer to [How to Build SOPHGO Deployment Environment](../../../../../../docs/en/build_and_install/sophgo.md).

## Generate Basic Directory Files

The routine consists of the following parts:
```text
.
├── CMakeLists.txt
├── build  # Compile Folder
├── image  # Folder for images
├── infer.cc
└── model  # Folder for models
```

## Compile

### Compile and Copy SDK to folder thirdpartys

Please refer to [How to Build SOPHGO Deployment Environment](../../../../../../docs/en/build_and_install/sophgo.md) to compile SDK.After compiling, the fastdeploy-0.0.3 directory will be created in the build directory.

### Copy model and configuration files to folder Model
Convert Paddle model to SOPHGO bmodel model. For the conversion steps, please refer to [Document](../README.md).
Please copy the converted SOPHGO bmodel to folder model.

### Prepare Test Images to folder image
```bash
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp 000000014439.jpg ./images
```

### Compile example

```bash
cd build
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-0.0.3
make
```

## Running Routines

```bash
./infer_demo model images/000000014439.jpg
```


- [Model Description](../../)
- [Model Conversion](../)
