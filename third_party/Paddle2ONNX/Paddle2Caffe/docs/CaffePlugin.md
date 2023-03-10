# [Contrib] CaffePlugin | 扩充Caffe算子

## 简介

本文列举了部分在Paddle2Caffe中采用了的Caffe扩充算子，通过扩充Caffe算子，能够支持更加广泛的网络结构

例如经典的检测网络SSD、Yolo等等（算子proto可参考paddle2caffe/caffe_helper/caffe.proto）

* 部分参数与参考示例不完全一致，以caffe.proto为准

## 算子参考

### Upsample
https://github.com/ChenYingpeng/darknet2caffe

### PriorBox
https://github.com/weiliu89/caffe/tree/ssd

### DetectionOutput
https://github.com/weiliu89/caffe/tree/ssd

### Yolov3DetectionOutput
https://github.com/wzj5133329/MobileNet_yolo

### Interp
https://github.com/xjqicuhk/3DGNN

### Permute
https://github.com/bairdzhang/smallhardface
