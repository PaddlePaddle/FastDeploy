English | [简体中文](README_CN.md)
# YOLOv6 Quantification Model C++ Deployment Example

This directory provides examples that `infer.cc` fast finishes the deployment of YOLOv6 quantification models on CPU/GPU.

## Prepare the deployment
### FastDeploy Environment Preparation
- 1. Software and hardware should meet the requirements. Please refer to  [FastDeploy Environment Requirements](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

### Prepare the quantification model
- 1. Users can directly deploy quantized models provided by FastDeploy.
- 2. ii.	Or users can use the [One-click auto-compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to automatically conduct quantification model for deployment.

## Example: quantized YOLOv6 model
The compilation and deployment can be completed by executing the following command in this directory. FastDeploy version 0.7.0 or above (x.x.x>=0.7.0) is required to support this model.
```bash
mkdir build
cd build
# Download the FastDeploy precompiled library. Users can choose your appropriate version in the `FastDeploy Precompiled Library` mentioned above 
wget https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-x64-x.x.x.tgz
tar xvf fastdeploy-linux-x64-x.x.x.tgz
cmake .. -DFASTDEPLOY_INSTALL_DIR=${PWD}/fastdeploy-linux-x64-x.x.x
make -j

# Download yolov6 quantification model files and test images provided by FastDeploy
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov6s_qat_model_new.tar
tar -xvf yolov6s_qat_model.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg


# Use ONNX Runtime quantification model on CPU
./infer_demo yolov6s_qat_model 000000014439.jpg 0
# Use TensorRT quantification model on GPU
./infer_demo yolov6s_qat_model 000000014439.jpg 1
# Use Paddle-TensorRT quantification model on GPU
./infer_demo yolov6s_qat_model 000000014439.jpg 2
```
