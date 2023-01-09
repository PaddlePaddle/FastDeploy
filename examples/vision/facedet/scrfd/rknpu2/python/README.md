English | [简体中文](README_CN.md)
# SCRFD Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Please refer to [FastDeploy Environment Requirements](../../../../../../docs/cn/build_and_install/rknpu2.md)

This directory provides examples that `infer.py` fast finishes the deployment of SCRFD on RKNPU. The script is as follows

## Copy model files
Refer to [SCRFD model conversion](../README.md) to convert SCRFD ONNX model to RKNN model and move it to this directory.


## Run example
After copying model files, enter the following command to run it: RKNPU2 Python example
```bash
# Download the example code for deployment
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/facedet/scrfd/rknpu2/python

# Download images
wget https://raw.githubusercontent.com/DefTruth/lite.ai.toolkit/main/examples/lite/resources/test_lite_face_detector_3.jpg

# Inference
python3 infer.py --model_file ./scrfd_500m_bnkps_shape640x640_rk3588.rknn \
                 --image test_lite_face_detector_3.jpg
```

## Visualization
The visualized result after running is as follows

<img width="640" src="https://user-images.githubusercontent.com/67993288/184301789-1981d065-208f-4a6b-857c-9a0f9a63e0b1.jpg">



## Note
The model needs to be in NHWC format on RKNPU. The normalized image will be embedded in the RKNN model. Therefore, when we deploy with FastDeploy, 
call DisablePermute(C++) or `disable_permute(Python) to disable normalization and data format conversion during preprocessing.

## Other Documents

- [SCRFD Model Description](../README.md)
- [SCRFD C++ Deployment](../cpp/README.md)
- [Model Prediction Results](../../../../../../docs/api/vision_results/README.md)
- [Convert SCRFD RKNN Model Files](../README.md)
