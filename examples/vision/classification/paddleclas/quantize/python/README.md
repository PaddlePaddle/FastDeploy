English | [简体中文](README_CN.md)
# PaddleClas Quantitative Model Python Deployment Example
 `infer.py` in this directory can help you quickly complete the inference acceleration of PaddleClas quantization model deployment on CPU/GPU.

## Deployment Preparations
### FastDeploy Environment Preparations
- 1. For the software and hardware requirements, please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md). 
- 2. For the installation of FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md).

### Quantized Model Preparations
- 1. You can directly use the quantized model provided by FastDeploy for deployment.
- 2. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the inference_cls.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)


## Take the Quantized ResNet50_Vd Model as an example for Deployment
```bash
# Download sample deployment code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/classification/paddleclas/quantize/python

# Download the ResNet50_Vd quantized model and test images provided by FastDeloy.
wget https://bj.bcebos.com/paddlehub/fastdeploy/resnet50_vd_ptq.tar
tar -xvf resnet50_vd_ptq.tar
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Use ONNX Runtime inference quantization model on CPU.
python infer.py --model resnet50_vd_ptq --image ILSVRC2012_val_00000010.jpeg --device cpu --backend ort
# Use TensorRT inference quantization model on GPU.
python infer.py --model resnet50_vd_ptq --image ILSVRC2012_val_00000010.jpeg --device gpu --backend trt
# Use Paddle-TensorRT inference quantization model on GPU.
python infer.py --model resnet50_vd_ptq --image ILSVRC2012_val_00000010.jpeg --device gpu --backend pptrt
```
