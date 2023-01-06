English | [简体中文](README_CN.md)
# RKYOLO Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Refer to [FastDeploy Environment Requirements](../../../../../../docs/cn/build_and_install/rknpu2.md)

This directory provides examples that `infer.py` fast finishes the deployment of Picodet on RKNPU. The script is as follows

```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/rkyolo/python

# Download images
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# copy model
cp -r ./model /path/to/FastDeploy/examples/vision/detection/rkyolo/python

# Inference
python3 infer.py --model_file ./model/  \
                  --image 000000014439.jpg
```


## Note
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，

## Other Documents

- [PaddleDetection Model Description](..)
- [PaddleDetection C++ Deployment](../cpp)
- [model prediction Results](../../../../../../docs/api/vision_results/)
- [Convert PaddleDetection RKNN Model Files](../README.md)
