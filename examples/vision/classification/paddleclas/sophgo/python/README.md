English | [简体中文](README_CN.md)
# PaddleClas Python Deployment Example

Before deployment, the following step need to be confirmed:

- 1. Hardware and software environment meets the requirements. Please refer to [FastDeploy Environment Requirement](../../../../../../docs/en/build_and_install/sophgo.md).

`infer.py` in this directory provides a quick example of deployment of the ResNet50_vd model on SOPHGO TPU. Please run the following script:

```bash
# Download the sample deployment code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/classification/paddleclas/sophgo/python

# Download images.
wget https://gitee.com/paddlepaddle/PaddleClas/raw/release/2.4/deploy/images/ImageNet/ILSVRC2012_val_00000010.jpeg

# Inference. Need to manually set the model, configuration file and image path used for inference.
python3 infer.py --auto False --model_file ./bmodel/resnet50_1684x_f32.bmodel  --config_file ResNet50_vd_infer/inference_cls.yaml  --image ILSVRC2012_val_00000010.jpeg

# Automatic completion of downloading data - model compilation - inference, no need to set up model, configuration file and image paths.
python3 infer.py --auto True --model '' --config_file '' --image ''

# The returned result.
ClassifyResult(
label_ids: 153,
scores: 0.684570,
)
```

## Other Documents
- [ResNet50_vd C++ Deployment](../cpp)
- [Converting ResNet50_vd SOPHGO model](../README.md)
