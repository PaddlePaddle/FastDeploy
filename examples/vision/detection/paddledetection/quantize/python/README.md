English | [简体中文](README_CN.md)
# PP-YOLOE-l Quantitative Model Python Deployment Example
 `infer.py` in this directory can help you quickly complete the inference acceleration of PP-YOLOE quantization model deployment on CPU/GPU.

## Deployment Preparations
### FastDeploy Environment Preparations
- 1. For the software and hardware requirements, please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. For the installation of FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

### Quantized Model Preparations
- 1. You can directly use the quantized model provided by FastDeploy for deployment.
- 2. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the infer_cfg.yml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)


## Take the Quantized PP-YOLOE-l Model as an example for Deployment
```bash
# Download sample deployment code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd /examples/vision/detection/paddledetection/quantize/python

# Download the ppyoloe_crn_l_300e_coco quantized model and test images provided by FastDeloy.
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco_qat.tar
tar -xvf ppyoloe_crn_l_300e_coco_qat.tar
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# Use ONNX Runtime inference quantization model on CPU.
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco_qat --image 000000014439.jpg --device cpu --backend ort
# Use TensorRT inference quantization model on GPU.
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco_qat --image 000000014439.jpg --device gpu --backend trt
# Use Paddle-TensorRT inference quantization model on GPU.
python infer_ppyoloe.py --model ppyoloe_crn_l_300e_coco_qat --image 000000014439.jpg --device gpu --backend pptrt
```
