[English](README.md) | 简体中文
# RKYOLO Python部署示例

在部署前，需确认以下两个步骤

- 1. 软硬件环境满足要求，参考[FastDeploy环境要求](../../../../../docs/cn/build_and_install/rknpu2.md)

本目录下提供`infer.py`快速完成Picodet在RKNPU上部署的示例。执行如下脚本即可完成

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

## 常见问题

如果你使用自己训练的YOLOv5模型，你可能会碰到运行FastDeploy的demo后出现`segmentation fault`的问题，很大概率是label数目不一致，你可以使用以下方案来解决:

```python
model.postprocessor.class_num = 3
```

## 注意事项
RKNPU上对模型的输入要求是使用NHWC格式，且图片归一化操作会在转RKNN模型时，内嵌到模型中，因此我们在使用FastDeploy部署时，需要先调用DisablePermute(C++) `disable_permute(Python)`，在预处理阶段禁用归一化以及数据格式的转换。

## 其它文档

- [PaddleDetection 模型介绍](..)
- [PaddleDetection C++部署](../cpp)
- [模型预测结果说明](../../../../../docs/api/vision_results/)
- [转换PaddleDetection RKNN模型文档](../README.md)
