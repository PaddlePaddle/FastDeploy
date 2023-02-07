English | [简体中文](README_CN.md)
# PaddleSeg Deployment Examples for Python

Before deployment, the following step need to be confirmed:

- 1. Hardware and software environment meets the requirements, please refer to [Environment Requirements for FastDeploy](../../../../../../docs/en/build_and_install/rknpu2.md).

【Note】If you are deploying **PP-Matting**, **PP-HumanMatting** or **ModNet**, please refer to [Matting Model Deployment](../../../../matting/).

This directory provides `infer.py` for a quick example of PPHumanseg deployment on RKNPU. This can be done by running the following script.

```bash
# Download the deploying demo code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/python

# Download images.
wget https://paddleseg.bj.bcebos.com/dygraph/pp_humanseg_v2/images.zip
unzip images.zip

# Inference.
python3 infer.py --model_file ./Portrait_PP_HumanSegV2_Lite_256x144_infer/Portrait_PP_HumanSegV2_Lite_256x144_infer_rk3588.rknn \
                --config_file ./Portrait_PP_HumanSegV2_Lite_256x144_infer/deploy.yaml \
                --image images/portrait_heng.jpg
```


## Notes
The input requirement for the model on RKNPU is to use NHWC format, and image normalization will be embedded into the model when converting the RKNN model, so we need to call DisableNormalizeAndPermute(C++) or disable_normalize_and_permute(Python) first when deploying with FastDeploy to disable normalization and data format conversion in the preprocessing stage.

## Other Documents

- [PaddleSeg Model Description](..)
- [PaddleSeg C++ Deployment](../cpp)
- [Description of the prediction](../../../../../../docs/api/vision_results/)
- [Convert PPSeg and RKNN model](../README.md)
