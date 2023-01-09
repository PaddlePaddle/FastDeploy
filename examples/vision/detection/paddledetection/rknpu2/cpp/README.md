English | [简体中文](README_CN.md)
# PaddleDetection Deployment Examples for C++

`infer_picodet.cc` in this directory provides an example of quickly completing the PPDetection model on Rockchip boards for accelerated deployment via second-generation NPUs.

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
├── infer_picodet.cc
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
After converting to RKNN, the model file also needs to be copied to folder model.

### Prepare Test Images to folder image
```bash
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp 000000014439.jpg ./images
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
./infer_picodet model/picodet_s_416_coco_lcnet images/000000014439.jpg
```


- [Model Description](../../)
- [Python Deployment](../python)
- [Vision model prediction results](../../../../../../docs/api/vision_results/)
