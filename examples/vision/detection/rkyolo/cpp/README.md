English | [简体中文](README_CN.md)
# RKYOLO C++ Deployment Example

This directory provides examples that `infer_xxxxx.cc` fast finishes the deployment of RKYOLO models on Rockchip board through 2-nd generation NPU

Two steps before deployment

1. Software and hardware should meet the requirements.
2. Download the precompiled deployment library or deploy FastDeploy repository from scratch according to your development environment.

Refer to [RK2 generation NPU deployment repository compilation](../../../../../docs/cn/build_and_install/rknpu2.md)

## Generate the base directory file

The routine consists of the following parts
```text
.
├── CMakeLists.txt
├── build  # Compile folder
├── image  # Folder to save images
├── infer_rkyolo.cc
├── model  # Folder to save model files
└── thirdpartys  # Folder to save sdk
```

Generate a directory first
```bash
mkdir build
mkdir images
mkdir model
mkdir thirdpartys
```

## Compile

### Compile and copy SDK to the thirdpartys folder

Refer to [RK2 generation NPU deployment repository compilation](../../../../../../docs/cn/build_and_install/rknpu2.md). It will generate fastdeploy-0.0.3 directory in the build directory after compilation. Move it to the thirdpartys directory.

### Copy model files and configuration files to the model folder
In the process of Paddle dynamic graph model -> Paddle static graph model -> ONNX model, the ONNX file and the corresponding yaml configuration file will be generated. Please save the configuration file in the model folder.
Copy onverted RKNN model files to model。

### Prepare test images and image folder
```bash
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp 000000014439.jpg ./images
```

### Compilation example

```bash
cd build
cmake ..
make -j8
make install
```

## Running routine 

```bash
cd ./build/install
./infer_picodet model/ images/000000014439.jpg
```


- [Model Description](../../)
- [Python Deployment](../python)
- [Vision Model Prediction Results](../../../../../../docs/api/vision_results/)
