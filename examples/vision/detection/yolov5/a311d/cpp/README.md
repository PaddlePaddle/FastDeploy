English | [简体中文](README_CN.md)
# YOLOv5 Quantitative Model C++ Deployment Example

`infer.cc` in this directory can help you quickly complete the inference acceleration of YOLOv5 quantization model deployment on A311D.

## Deployment Preparations
### FastDeploy Cross-compile Environment Preparations
1. For the software and hardware environment, and the cross-compile environment, please refer to [FastDeploy Cross-compile environment](../../../../../../docs/en/build_and_install/a311d.md#Cross-compilation-environment-construction).

### Model Preparations
The quantified model can be deployed directly using the model provided by FastDeploy, or you can prepare it as follows:
1. Export ONNX model according to the official [YOLOv5](https://github.com/ultralytics/yolov5/releases/tag/v6.1) export method, or you can download it directly with the following command:
```bash
wget https://paddle-slim-models.bj.bcebos.com/act/yolov5s.onnx
```
2. Prepare about 300 images for quantification, or you can use the following command to download the data we have prepared.
```bash
wget https://bj.bcebos.com/fastdeploy/models/COCO_val_320.tar.gz
tar -xf COCO_val_320.tar.gz
```
3. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.
```bash
fastdeploy compress --config_path=./configs/detection/yolov5s_quant.yaml --method='PTQ' --save_dir='./yolov5s_ptq_model_new/'
```
4. The model requires heterogeneous computation. Please refer to: [Heterogeneous Computation](./../../../../../../docs/en/faq/heterogeneous_computing_on_timvx_npu.md). Since the YOLOv5 model is already provided, you can test the heterogeneous file we provide first to verify whether the accuracy meets the requirements.
```bash
# First download the model we provide, unzip it and copy the subgraph.txt file to the newly quantized model directory.
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
```

For more information, please refer to [Model Quantization](../../quantize/README.md)

## Deploying the Quantized YOLOv5 Detection model on A311D
Please follow these steps to complete the deployment of the YOLOv5 quantization model on A311D.
1. Cross-compile the FastDeploy library as described in [Cross-compile  FastDeploy](../../../../../../docs/en/build_and_install/a311d.md#FastDeploy-cross-compilation-library-compilation-based-on-Paddle-Lite)

2. Copy the compiled library to the current directory. You can run this line:
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/detection/yolov5/a311d/cpp
```

3. Download the model and example images required for deployment in current path.
```bash
cd FastDeploy/examples/vision/detection/yolov5/a311d/cpp
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/yolov5s_ptq_model.tar.gz
tar -xvf yolov5s_ptq_model.tar.gz
cp -r yolov5s_ptq_model models
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. Compile the deployment example. You can run the following lines:
```bash
cd FastDeploy/examples/vision/detection/yolov5/a311d/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=arm64 ..
make -j8
make install
# After success, an install folder will be created with a running demo and libraries required for deployment.
```

5. Deploy the YOLOv5 detection model to A311D based on adb.
```bash
# Go to the install directory.
cd FastDeploy/examples/vision/detection/yolov5/a311d/cpp/build/install/
# The following line represents: bash run_with_adb.sh, demo needed to run, model path, image path, DEVICE ID.
bash run_with_adb.sh infer_demo yolov5s_ptq_model 000000014439.jpg $DEVICE_ID
```

The result vis_result.jpg is saveed as follows:

<img width="640" src="https://user-images.githubusercontent.com/30516196/203706969-dd58493c-6635-4ee7-9421-41c2e0c9524b.png">

Please note that the model deployed on A311D needs to be quantized. You can refer to [Model Quantization](../../../../../../docs/en/quantize.md)
