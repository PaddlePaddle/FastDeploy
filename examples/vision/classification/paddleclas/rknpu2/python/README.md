English | [简体中文](README_CN.md)
# PaddleClas Python Deployment Example

Before deployment, the following step need to be confirmed:

- 1. Hardware and software environment meets the requirements, please refer to [Environment Requirements for FastDeploy](../../../../../../docs/en/build_and_install/rknpu2.md).

This directory provides `infer.py` for a quick example of ResNet50_vd deployment on RKNPU. This can be done by running the following script.

```bash
# Download the deploying demo code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/classification/paddleclas/rknpu2/python

# Download images.
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Inference.
python3 infer.py --model_file ./ResNet50_vd_infer/ResNet50_vd_infer_rk3588.rknn  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg

# Results
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)
```


## Notes
The input requirement for the model on RKNPU is to use NHWC format, and image normalization will be embedded into the model when converting the RKNN model, so we need to call DisablePermute(C++) or disable_permute(Python) first when deploying with FastDeploy to disable data format conversion in the preprocessing stage.

## Other Documents
- [ResNet50_vd C++ Deployment](../cpp)
- [Prediction Results](../../../../../../docs/api/vision_results/)
- [Converting ResNet50_vd RKNN model](../README.md)
