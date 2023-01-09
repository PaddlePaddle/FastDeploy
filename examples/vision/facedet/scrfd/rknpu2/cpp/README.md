English | [简体中文](README_CN.md)
# SCRFD C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of SCRFD on NPU.

Two steps before deployment:

1. The environment of software and hardware should meet the requirements.
2. Download the precompiled deployment repo or deploy the FastDeploy repository from scratch according to your development environment. 

Refer to [RK2 generation NPU deployment repository compilation](../../../../../../docs/cn/build_and_install/rknpu2.md) for the steps above

## Generate the base directory file

It consists of the following parts
```text
.
├── CMakeLists.txt
├── build  # Compile folder
├── image  # The folder to save images 
├── infer.cc
├── model  # The folder to save model files
└── thirdpartys  # The folder to save sdk
```

Generate the directory first
```bash
mkdir build
mkdir images
mkdir model
mkdir thirdpartys
```

## Compile

### Compile and copy the SDK into the thirdpartys folder

Refer to [RK2 generation NPU deployment repository compilation](../../../../../../docs/cn/build_and_install/rknpu2.md). It will enerate fastdeploy-0.7.0 directory in the build directory after compilation. Move it to the thirdpartys directory.

### Copy the model files to the model folder
Refer to [SCRFD model conversion](../README.md) to convert SCRFD ONNX model to RKNN model and move it to the model folder.

### Prepare test images to the image folder
```bash
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg
cp test_lite_face_detector_3.jpg ./images
```

### Compile example

```bash
cd build
cmake ..
make -j8
make install
```
## Running routines

```bash
cd ./build/install
export LD_LIBRARY_PATH=${PWD}/lib:${LD_LIBRARY_PATH}
./rknpu_test
```
The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301789-1981d065-208f-4a6b-857c-9a0f9a63e0b1.jpg">

- [Model Description](../../README.md)
- [Python Deployment](../python/README.md)
- [Vision Model Prediction Results](../../../../../../docs/api/vision_results/README.md)
