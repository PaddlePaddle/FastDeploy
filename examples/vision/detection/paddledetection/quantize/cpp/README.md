English | [简体中文](README_CN.md)
# PP-YOLOE-l Quantitative Model C++ Deployment Example

`infer_ppyoloe.cc` in this directory can help you quickly complete the inference acceleration of PP-YOLOE-l quantization model deployment on CPU/GPU.

## Deployment Preparations
### FastDeploy Environment Preparations
- 1. For the software and hardware requirements, please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. For the installation of FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

### Quantized Model Preparations
- 1. You can directly use the quantized model provided by FastDeploy for deployment..
- 2. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the infer_cfg.yml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)

## Take the Quantized PP-YOLOE-l Model as an example for Deployment, FastDeploy version 0.7.0 or higher is required (x.x.x>=0.7.0)
Run the following commands in this directory to compile and deploy the quantized model.
```bash
mkdir build
cd build
# Download pre-compiled FastDeploy libraries. You can choose the appropriate version from `pre-compiled FastDeploy libraries` mentioned above.
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download the ppyoloe_crn_l_300e_coco quantized model and test images provided by FastDeloy.
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar
tar -xvf ppyoloe_crn_l_300e_coco_qat.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Use ONNX Runtime inference quantization model on CPU.
./infer_ppyoloe_demo ppyoloe_crn_l_300e_coco_qat 000000014439.jpg 0
# Use TensorRT inference quantization model on GPU.
./infer_ppyoloe_demo ppyoloe_crn_l_300e_coco_qat 000000014439.jpg 1
# Use Paddle-TensorRT inference quantization model on GPU.
./infer_ppyoloe_demo ppyoloe_crn_l_300e_coco_qat 000000014439.jpg 2
```
