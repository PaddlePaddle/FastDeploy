English | [简体中文](README_CN.md)
# PaddleClas A311D Development Board C++ Deployment Example
 `infer.cc` in this directory can help you quickly complete the inference acceleration of PaddleClas quantization model deployment on A311D.

## Deployment Preparations
### FastDeploy Cross-compile Environment Preparations
1. For the software and hardware environment, and the cross-compile environment, please refer to [FastDeploy Cross-compile environment](../../../../../../docs/en/build_and_install/a311d.md#Cross-compilation-environment-construction).  

### Quantization Model Preparations
1. You can directly use the quantized model provided by FastDeploy for deployment.
2. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the inference_cls.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)

For more information, please refer to [Model Quantization](../../quantize/README.md)

## Deploying the Quantized ResNet50_Vd Segmentation model on A311D
Please follow these steps to complete the deployment of the ResNet50_Vd quantization model on A311D.
1. Cross-compile the FastDeploy library as described in [Cross-compile  FastDeploy](../../../../../../docs/en/build_and_install/a311d.md#FastDeploy-cross-compilation-library-compilation-based-on-Paddle-Lite)

2. Copy the compiled library to the current directory. You can run this line:
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/classification/paddleclas/a311d/cpp/
```

3. Download the model and example images required for deployment in current path.
```bash
cd FastDeploy/examples/vision/classification/paddleclas/a311d/cpp/
mkdir models && mkdir images
wget https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar
tar -xvf resnet50_vd_ptq.tar
cp -r resnet50_vd_ptq models
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg
cp -r ILSVRC2012_val_00000010.jpeg images
```

4. Compile the deployment example. You can run the following lines:
```bash
cd FastDeploy/examples/vision/classification/paddleclas/a311d/cpp/
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=arm64 ..
make -j8
make install
# After success, an install folder will be created with a running demo and libraries required for deployment.
```

5. Deploy the ResNet50 segmentation model to A311D based on adb. You can run the following lines:
```bash
# Go to the install directory.
cd FastDeploy/examples/vision/classification/paddleclas/a311d/cpp/build/install/
# The following line represents: bash run_with_adb.sh, demo needed to run, model path, image path, DEVICE ID.
bash run_with_adb.sh infer_demo resnet50_vd_ptq ILSVRC2012_val_00000010.jpeg $DEVICE_ID
```

The output is:

<img width="640" src="https://user-images.githubusercontent.com/30516196/200767389-26519e50-9e4f-4fe1-8d52-260718f73476.png">

Please note that the model deployed on A311D needs to be quantized. You can refer to [Model Quantization](../../../../../../docs/en/quantize.md)
