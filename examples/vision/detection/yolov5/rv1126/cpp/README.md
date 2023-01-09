English | [简体中文](README_CN.md)
# YOLOv5 Quantification Model C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of YOLOv5 on RV1126.

## Prepare the deployment
### Prepare FastDeploy cross-compilation environment
1. For the environment of software, hardware and cross-compilation, refer to [FastDeploy cross-compilation environment preparation](../../../../../../docs/cn/build_and_install/rv1126.md#交叉编译环境搭建)  

### Prepare the quantification model
Users can directly deploy quantized models provided by FastDeploy or prepare quantification models as the following steps:
1. Refer to [YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v6.1) to officially convert the ONNX model or use the following command to download it.
```bash
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx
```
2. Prepare 300 or so images for quantization.And we can also download the prepared data using the following command:
```bash
wget https://bj.bcebos.com/fastdeploy/models/COCO_val_320.tar.gz
tar -xf COCO_val_320.tar.gz
```
3. Users can use the [ One-click auto-compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to automatically conduct quantification model for deployment.
```bash
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model_new/'
```
4. The YOLOv5 model requires heterogeneous computing. Refer to [Heterogeneous Computing](./../../../../../../docs/cn/faq/heterogeneous_computing_on_timvx_npu.md).  Since FastDeploy already provides the YOLOv5 model, we can first test the heterogeneous files to verify whether the accuracy meets the requirements.
```bash
# Download the model, unzip it, and copy the subgraph.txt file to your newly quantized model directory
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
```

Refer to [model quantification](../../quantize/README.md) for more information

## Deploy quantized YOLOv5 detection model on RV1126
Refer to the following steps:
1. For cross compiling FastDeploy repo, refer to [cross compiling FastDeploy](../../../../../../docs/cn/build_and_install/rv1126.md#基于-paddlelite-的-fastdeploy-交叉编译库编译)

2. Copy the compiled repo to your current directory through the following command:
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/detection/yolov5/rv1126/cpp
```

3. Download models and images for deployment in the current location:
```bash
cd FastDeploy/examples/vision/detection/yolov5/rv1126/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
cp -r yolov5s_ptq_model models
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. Compile the deployment example through the following command:
```bash
cd FastDeploy/examples/vision/detection/yolov5/rv1126/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=armhf ..
make -j8
make install
# Successful compilation generates the install folder, containing a running demo and repo required for deployment
```

5. Deploy YOLOv5 detection model to Rockchip RV1126 based on adb. Refer to the following command:
```bash
# Enter the install directory
cd FastDeploy/examples/vision/detection/yolov5/rv1126/cpp/build/install/
# The following commands represent: bash run_with_adb.sh running demo  model path  image path   DEVICE_ID
bash run_with_adb.sh infer_demo yolov5s_ptq_model 000000014439.jpg $DEVICE_ID
```

vis_result.jpg after successful deployment:

<img width="640" src="https://user-images.githubusercontent.com/30516196/203706969-dd58493c-6635-4ee7-9421-41c2e0c9524b.png">

Note that the deployment model on RV1126 is the quantized model. Refer to [Model Quantification](../../../../../../docs/cn/quantize.md)
