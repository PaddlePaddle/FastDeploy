English | [简体中文](README_CN.md)
# PP-YOLOE  Quantitative Model C++ Deployment Example

`infer.cc` in this directory can help you quickly complete the inference acceleration of PP-YOLOE and PicoDet quantization model deployment on RV1126.

## Deployment Preparations
### FastDeploy Cross-compile Environment Preparations
1. For the software and hardware environment, and the cross-compile environment, please refer to [Preparations for FastDeploy Cross-compile environment](../../../../../../docs/en/build_and_install/rv1126.md#Cross-compilation-environment-construction).

### Model Preparations
1. You can directly use the quantized model provided by FastDeploy for deployment.
2. You can use PaddleDetection to export Float32 models, note that you need to set the parameter when exporting PP-YOLOE model: use_shared_conv=False. For more information: [PP-YOLOE](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe). For more information when exporting PicoDet model: [PicoDet](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet).
3. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the infer_cfg.yml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)
4. The model requires heterogeneous computation. Please refer to: [Heterogeneous Computation](./../../../../../../docs/en/faq/heterogeneous_computing_on_timvx_npu.md). Since the model is already provided, you can test the heterogeneous file we provide first to verify whether the accuracy meets the requirements.

For more information, please refer to [Model Quantization](../../quantize/README.md)

## Deploying the Quantized PP-YOLOE and PicoDet Detection model on RV1126
Please follow these steps to complete the deployment of the PP-YOLOE and PicoDet quantization model on RV1126.
1. Cross-compile the FastDeploy library as described in [Cross-compile FastDeploy](../../../../../../docs/en/build_and_install/rv1126.md#FastDeploy-cross-compilation-library-compilation-based-on-Paddle-Lite)

2. Copy the compiled library to the current directory. You can run this line:
```bash
cp -r FastDeploy/build/fastdeploy-timvx/ FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp
```

3. Download the model and example images required for deployment in current path.
```bash
cd FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp
mkdir models && mkdir images
# download PP-YOLOE model
wget https://bj.bcebos.com/fastdeploy/models/ppyoloe_noshare_qat.tar.gz
tar -xvf ppyoloe_noshare_qat.tar.gz
cp -r ppyoloe_noshare_qat models
# download PicoDet model
wget https://bj.bcebos.com/fastdeploy/models/picodet_withNMS_quant_qat.tar.gz
tar -xvf picodet_withNMS_quant_qat.tar.gz
cp -r picodet_withNMS_quant_qat models
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
cp -r 000000014439.jpg images
```

4. Compile the deployment example. You can run the following lines:
```bash
cd FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=${PWD}/../fastdeploy-timvx/toolchain.cmake -DFASTDEPLOY_INSTALL_DIR=${PWD}/../fastdeploy-timvx -DTARGET_ABI=armhf ..
make -j8
make install
# After success, an install folder will be created with a running demo and libraries required for deployment.
```

5. Deploy the PP-YOLOE and PicoDet detection model to Rockchip RV1126 based on adb. You can run the following lines:
```bash
# Go to the install directory.
cd FastDeploy/examples/vision/detection/paddledetection/rv1126/cpp/build/install/
# The following line represents: bash run_with_adb.sh, demo needed to run, model path, image path, DEVICE ID.
bash run_with_adb.sh ppyoloe_infer_demo ppyoloe_noshare_qat 000000014439.jpg $DEVICE_ID
bash run_with_adb.sh picodet_infer_demo picodet_withNMS_quant_qat 000000014439.jpg $DEVICE_ID
```

The output is:

<img width="640" src="https://user-images.githubusercontent.com/30516196/203708564-43c49485-9b48-4eb2-8fe7-0fa517979fff.png">

Please note that the model deployed on RV1126 needs to be quantized. You can refer to [Model Quantization](../../../../../../docs/en/quantize.md)
