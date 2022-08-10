# PaddleDetection模型部署

## 模型版本说明

- [PaddleDetection Release/2.4](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)

## 支持模型列表

目前FastDeploy支持如下模型的部署

- [PPYOLOE系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)
- [PicoDet系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)
- [PPYOLO系列模型(含v2)](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo)
- [YOLOv3系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3)
- [YOLOX系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolox)
- [FasterRCNN系列模型](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn)

## 导出部署模型

在部署前，需要先将PaddleDetection导出成部署模型，导出步骤参考文档[导出模型](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md)

注意：在导出模型时不要进行NMS的去除操作，正常导出即可。

## 下载预训练模型

为了方便开发者的测试，下面提供了PaddleDetection导出的各系列模型，开发者可直接下载使用。

其中精度指标来源于PaddleDetection中对各模型的介绍，详情各参考PaddleDetection中的说明。


| 模型                                                               | 参数大小    | 精度    | 备注 |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [picodet_l_320_coco_lcnet](https://bj.bcebos.com/paddlehub/fastdeploy/picodet_l_320_coco_lcnet.tgz) |23MB | 42.6% |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz) |200MB | 51.4% |
| [ppyolo_r50vd_dcn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyolo_r50vd_dcn_1x_coco.tgz) | 180MB | 44.8% | 暂不支持TensorRT |
| [ppyolov2_r101vd_dcn_365e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyolov2_r101vd_dcn_365e_coco.tgz) | 282MB | 49.7% | 暂不支持TensorRT |
| [yolov3_darknet53_270e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov3_darknet53_270e_coco.tgz) |237MB | 39.1% | |
| [yolox_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s_300e_coco.tgz) | 35MB | 40.4% | |
| [faster_rcnn_r50_vd_fpn_2x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz) | 160MB | 40.8%| 暂不支持TensorRT |


## 详细部署文档

- [Python部署](python)
- [C++部署](cpp)
