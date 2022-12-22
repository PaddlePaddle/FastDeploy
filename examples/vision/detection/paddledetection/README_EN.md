English | [简体中文](README.md)
# PaddleDetection Model Deployment

## Model Description

- [PaddleDetection Release/2.4](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4)

## List of Supported Models

Now FastDeploy supports the deployment of the following models

- [PP-YOLOE(including PP-YOLOE+) models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyoloe)
- [PicoDet models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/picodet)
- [PP-YOLO models(including v2)](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/ppyolo)
- [YOLOv3 models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolov3)
- [YOLOX models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/yolox)
- [FasterRCNN models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/faster_rcnn)
- [MaskRCNN models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.4/configs/mask_rcnn)
- [SSD models](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.5/configs/ssd)
- [YOLOv5 models](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov5)
- [YOLOv6 models](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov6)
- [YOLOv7 models](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/yolov7)
- [RTMDet models](https://github.com/PaddlePaddle/PaddleYOLO/tree/release/2.5/configs/rtmdet)

## Export Deployment Model

Before deployment, PaddleDetection needs to be exported into the deployment model. Refer to [Export Models](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.4/deploy/EXPORT_MODEL.md) for more details.

**Attention**
- Do not perform NMS removal when exporting the model
- If you are running a native TensorRT backend (not a Paddle Inference backend), do not add the --trt parameter
- Do not add the parameter `fuse_normalize=True` when exporting the model

## Download Pre-trained Model

For developers' testing, models exported by PaddleDetection are provided below. Developers can download them directly. 

The accuracy metric is from model descriptions in PaddleDetection. Refer to them for details.

| Model                                                               | Parameter Size    | Accuracy    | Note |
|:---------------------------------------------------------------- |:----- |:----- | :------ |
| [picodet_l_320_coco_lcnet](https://bj.bcebos.com/paddlehub/fastdeploy/picodet_l_320_coco_lcnet.tgz) |23MB | Box AP 42.6% |
| [ppyoloe_crn_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz) |200MB | Box AP 51.4% |
| [ppyoloe_plus_crn_m_80e_coco](https://bj.bcebos.com/fastdeploy/models/ppyoloe_plus_crn_m_80e_coco.tgz) |83.3MB | Box AP 49.8% |
| [ppyolo_r50vd_dcn_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyolo_r50vd_dcn_1x_coco.tgz) | 180MB | Box AP 44.8% | TensorRT not supported yet |
| [ppyolov2_r101vd_dcn_365e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ppyolov2_r101vd_dcn_365e_coco.tgz) | 282MB | Box AP 49.7% | TensorRT not supported yet |
| [yolov3_darknet53_270e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov3_darknet53_270e_coco.tgz) |237MB | Box AP 39.1% | |
| [yolox_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolox_s_300e_coco.tgz) | 35MB | Box AP 40.4% | |
| [faster_rcnn_r50_vd_fpn_2x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/faster_rcnn_r50_vd_fpn_2x_coco.tgz) | 160MB | Box AP 40.8%| TensorRT not supported yet |
| [mask_rcnn_r50_1x_coco](https://bj.bcebos.com/paddlehub/fastdeploy/mask_rcnn_r50_1x_coco.tgz) | 128M | Box AP 37.4%, Mask AP 32.8%| TensorRT、ORT not supported yet |
| [ssd_mobilenet_v1_300_120e_voc](https://bj.bcebos.com/paddlehub/fastdeploy/ssd_mobilenet_v1_300_120e_voc.tgz) | 24.9M | Box AP 73.8%| TensorRT、ORT not supported yet |
| [ssd_vgg16_300_240e_voc](https://bj.bcebos.com/paddlehub/fastdeploy/ssd_vgg16_300_240e_voc.tgz) | 106.5M | Box AP 77.8%| TensorRT、ORT not supported yet |
| [ssdlite_mobilenet_v1_300_coco](https://bj.bcebos.com/paddlehub/fastdeploy/ssdlite_mobilenet_v1_300_coco.tgz) | 29.1M | | TensorRT、ORT not supported yet|
| [rtmdet_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/rtmdet_l_300e_coco.tgz) | 224M | Box AP 51.2%|  |
| [rtmdet_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/rtmdet_s_300e_coco.tgz) | 42M | Box AP 44.5%|  |
| [yolov5_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5_l_300e_coco.tgz) | 183M | Box AP 48.9%|  |
| [yolov5_s_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5_s_300e_coco.tgz) | 31M | Box AP 37.6%|  |
| [yolov6_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6_l_300e_coco.tgz) | 229M | Box AP 51.0%|  |
| [yolov6_s_400e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov6_s_400e_coco.tgz) | 68M | Box AP 43.4%|  |
| [yolov7_l_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_l_300e_coco.tgz) | 145M | Box AP 51.0%|  |
| [yolov7_x_300e_coco](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7_x_300e_coco.tgz) | 277M | Box AP 53.0%|  |

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)
