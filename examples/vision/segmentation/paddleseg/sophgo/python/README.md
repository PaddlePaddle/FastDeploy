English | [简体中文](README_CN.md)
# PaddleSeg Python Deployment Example

Before deployment, the following step need to be confirmed:

- 1. Hardware and software environment meets the requirements. Please refer to [FastDeploy Environment Requirement](../../../../../../docs/en/build_and_install/sophgo.md).

`infer.py` in this directory provides a quick example of deployment of the pp_liteseg model on SOPHGO TPU. Please run the following script:

```bash
# Download the sample deployment code.
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/segmentation/paddleseg/sophgo/python

# Download images.
wget https://paddleseg.bj.bcebos.com/dygraph/demo/cityscapes_demo.png

# Inference.
python3 infer.py --model_file ./bmodel/pp_liteseg_1684x_f32.bmodel --config_file ./bmodel/deploy.yaml --image cityscapes_demo.png

# The returned result.
The result is saved as sophgo_img.png.
```

## Other Documents
- [pp_liteseg C++ Deployment](../cpp)
- [Converting pp_liteseg SOPHGO model](../README.md)
