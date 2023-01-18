English | [简体中文](README_CN.md)
# PaddleSeg Quantitative Model Python Deployment Example
 `infer.py` in this directory can help you quickly complete the inference acceleration of PaddleSeg quantization model deployment on CPU/GPU.

## Deployment Preparations
### FastDeploy Environment Preparations
- 1. For the software and hardware requirements, please refer to [FastDeploy Environment Requirements](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)  
- 2. For the installation of FastDeploy Python whl package, please refer to [FastDeploy Python Installation](../../../../../../docs/en/build_and_install/download_prebuilt_libraries.md)

### Quantized Model Preparations
- 1. You can directly use the quantized model provided by FastDeploy for deployment.
- 2. You can use [one-click automatical compression tool](../../../../../../tools/common_tools/auto_compression/) provided by FastDeploy to quantize model by yourself, and use the generated quantized model for deployment.(Note: The quantized classification model still needs the deploy.yaml file in the FP32 model folder. Self-quantized model folder does not contain this yaml file, you can copy it from the FP32 model folder to the quantized model folder.)


## Take the Quantized PP_LiteSeg_T_STDC1_cityscapes Model as an example for Deployment
```bash
# Download sample deployment code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd examples/vision/segmentation/paddleseg/quantize/python

# Download the PP_LiteSeg_T_STDC1_cityscapes quantized model and test images provided by FastDeloy.
wget https://bj.bcebos.com/paddlehub/fastdeploy/PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar
tar -xvf PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_PTQ.tar
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# Use Paddle-Inference inference quantization model on CPU.
python infer.py --model PP_LiteSeg_T_STDC1_cityscapes_without_argmax_infer_QAT --image cityscapes_demo.png --device cpu --backend paddle

```
