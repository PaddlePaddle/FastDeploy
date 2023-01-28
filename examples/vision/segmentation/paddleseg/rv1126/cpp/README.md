English | [简体中文](README_CN.md)
# PP-LiteSeg Quantitative Model C++ Deployment Example

`infer.cc` in this directory can help you quickly complete the inference acceleration of PP-LiteSeg quantization model deployment on RV1126.

## Deployment Preparations
### FastDeploy Cross-compile Environment Preparations
1. For the software and hardware environment, and the cross-compile environment, please refer to [Preparations for FastDeploy Cross-compile environment](../../../../../../docs/en/build_and_install/rv1126.md#Cross-compilation-environment-construction).  

### Model Preparations
1. You can directly use the quantized model provided by FastDeploy for deployment.
2. You can use one-click automatical compression tool provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the deploy.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)
3. The model requires heterogeneous computation. Please refer to: [Heterogeneous Computation](./../../../../../../docs/en/faq/heterogeneous_computing_on_timvx_npu.md). Since the model is already provided, you can test the heterogeneous file we provide first to verify whether the accuracy meets the requirements.

For more information, please refer to [Model Quantization](../../quantize/README.md).

## Deploying the Quantized PP-LiteSeg Segmentation model on RV1126
Please follow these steps to complete the deployment of the PP-LiteSeg quantization model on RV1126.
1. Cross-compile the FastDeploy library as described in [Cross-compile FastDeploy](../../../../../../docs/en/build_and_install/rv1126.md#FastDeploy-cross-compilation-library-compilation-based-on-Paddle-Lite).

2. Copy the compiled library to the current directory. You can run this line:
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/segmentation/paddleseg/rv1126/cpp
```

3. Download the model and example images required for deployment in current path.
```bash
mkdir models && mkdir images
wget https://bj.bcebos.com/fastdeploy/models/rk1/ppliteseg.tar.gz
tar -xvf ppliteseg.tar.gz
cp -r ppliteseg models
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png
cp -r cityscapes_demo.png images
```

4. Compile the deployment example. You can run the following lines:
```bash
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=armhf ..
make -j8
make install
# After success, an install folder will be created with a running demo and libraries required for deployment.
```

5. Deploy the PP-LiteSeg segmentation model to Rockchip RV1126 based on adb. You can run the following lines:
```bash
# Go to the install directory.
cd FastDeploy/examples/vision/segmentation/paddleseg/rv1126/cpp/build/install/
# The following line represents: bash run_with_adb.sh, demo needed to run, model path, image path, DEVICE ID.
bash run_with_adb.sh infer_demo ppliteseg cityscapes_demo.png $DEVICE_ID
```

The output is:

<img width="640" src="https://user-images.githubusercontent.com/30516196/205544166-9b2719ff-ed82-4908-b90a-095de47392e1.png">

Please note that the model deployed on RV1126 needs to be quantized. You can refer to [Model Quantization](../../../../../../docs/en/quantize.md).
