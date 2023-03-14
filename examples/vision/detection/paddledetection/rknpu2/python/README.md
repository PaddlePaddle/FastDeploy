English | [简体中文](README_CN.md)
# PaddleDetection Deployment Examples for Python

Before deployment, the following step need to be confirmed:

- 1. Hardware and software environment meets the requirements, please refer to [Environment Requirements for FastDeploy](../../../../../../docs/en/build_and_install/rknpu2.md)

This directory provides `infer.py` for a quick example of Picodet deployment on RKNPU. This can be done by running the following script.

```bash
# Download the deploying demo code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/paddledetection/rknpu2/python

# Download images and model
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/picodet_s_416_coco_lcnet.zip
unzip picodet_s_416_coco_lcnet.zip

# Inference.
python3 infer.py --model_file ./picodet_s_416_coco_lcnet/picodet_s_416_coco_lcnet_rk3568.rknn  \
                  --config_file ./picodet_s_416_coco_lcnet/infer_cfg.yml \
                  --image 000000014439.jpg
```


## Notes
The input requirement for the model on RKNPU is to use NHWC format, and image normalization will be embedded into the model when converting the RKNN model, so we need to call DisableNormalizePermute(C++) or `disable_normalize_permute(Python) first when deploying with FastDeploy to disable normalization and data format conversion in the preprocessing stage.
## Other Documents

- [PaddleDetection Model Description](..)
- [PaddleDetection C++ Deployment](../cpp)
- [Description of the prediction](../../../../../../docs/api/vision_results/)
- [Converting PaddleDetection RKNN model](../README.md)
