English | [简体中文](README_CN.md)
# RKYOLO Python Deployment Example

Two steps before deployment

- 1. Software and hardware should meet the requirements. Refer to [FastDeploy Environment Requirements](../../../../../docs/cn/build_and_install/rknpu2.md)

This directory provides examples that `infer.py` fast finishes the deployment of Picodet on RKNPU. The script is as follows

```bash
git clone https://github.com/PaddlePaddle/FastDeploy.git
cd FastDeploy/examples/vision/detection/rkyolo/python

# download picture
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg

# infer yolov5
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolov5-s-relu.zip
unzip yolov5-s-relu.zip
python3 infer_rkyolov5.py --model_file yolov5-s-relu/yolov5s_relu_tk2_RK3588_i8.rknn \
                          --image 000000014439.jpg

# infer yolov7
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolov7-tiny.zip
unzip yolov7-tiny.zip
python3 infer_rkyolov7.py --model_file yolov7-tiny/yolov7-tiny_tk2_RK3588_i8.rknn \
                          --image 000000014439.jpg

# infer yolox
wget https://bj.bcebos.com/paddlehub/fastdeploy/rknpu2/yolox-s.zip
unzip yolox-s.zip
python3 infer_rkyolox.py --model_file yolox-s/yoloxs_tk2_RK3588_i8.rknn \
                          --image 000000014439.jpg
```

## common problem

If you use the YOLOv5 model you have trained, you may encounter the problem of 'segmentation fault' after running the demo of FastDeploy. It is likely that the number of labels is inconsistent. You can use the following solution:

```python
model.postprocessor.class_num = 3
```

## Note
The model needs to be in NHWC format on RKNPU. The normalized image will be embedded in the RKNN model. Therefore, when we deploy with FastDeploy, call DisablePermute(C++) or `disable_permute(Python)` to disable normalization and data format conversion during preprocessing.

## Other Documents

- [PaddleDetection Model Description](..)
- [PaddleDetection C++ Deployment](../cpp)
- [model prediction Results](../../../../../docs/api/vision_results/)
- [Convert PaddleDetection RKNN Model Files](../README.md)
