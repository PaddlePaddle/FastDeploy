English | [简体中文](README_CN.md)
# PaddleSeg Quantitative Model C++ Deployment Example
 `infer.cc` in this directory can help you quickly complete the inference acceleration of PaddleSeg quantization model deployment on CPU.

## Deployment Preparations
### FastDeploy Environment Preparations
- 1. For the software and hardware requirements, please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).
- 2. For the installation of FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

### Quantized Model Preparations
- 1. You can directly use the quantized model provided by FastDeploy for deployment.
- 2. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the deploy.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)

## Take the Quantized PP_LiteSeg_T_STDC1_cityscapes Model as an example for Deployment
Run the following commands in this directory to compile and deploy the quantized model. FastDeploy version 0.7.0 or higher is required (x.x.x>=0.7.0).
```bash
mkdir build
cd build
# Download pre-compiled FastDeploy libraries. You can choose the appropriate version from `pre-compiled FastDeploy libraries` mentioned above.
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the PP_LiteSeg_T_STDC1_cityscapes quantized model and test images provided by FastDeloy.
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar
tar -xvf PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# Use Paddle-Inference inference quantization model on CPU.
./infer_demo PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ cityscapes_demo.png 1
```
