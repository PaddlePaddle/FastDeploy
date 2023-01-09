English | [简体中文](README_CN.md)
# YOLOv7 Quantification Model Python Deployment Example
This directory provides examples that `infer.py` fast finishes the deployment of YOLOv7 quantification models on CPU/GPU.

## Prepare the deployment
### FastDeploy Environment Preparation
- 1. i.	Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)  
- 2. ii.	Install FastDeploy Python whl package. Refer to [FastDeploy Python Installation](../../../../../../docs/cn/build_and_install/download_prebuilt_libraries.md)

### Prepare the quantification model
- 1. Users can directly deploy quantized models provided by FastDeploy.
- 2. Or users can use the [ One-click auto-compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to automatically conduct quantification model for deployment.

## Example: quantized YOLOv7 model
```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/detection/yolov7/quantize/python

# Download yolov7 quantification model files and test images provided by FastDeploy
wget https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_quant.tar
tar -xvf yolov7_quant.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Use ONNX Runtime quantification model on CPU
python infer.py --model yolov7_quant --image 000000014439.jpg --device cpu --backend ort
#  Use TensorRT quantification model on GPU
python infer.py --model yolov7_quant --image 000000014439.jpg --device gpu --backend trt
# Use Paddle-TensorRT quantification model on GPU
python infer.py --model yolov7_quant --image 000000014439.jpg --device gpu --backend pptrt
```
