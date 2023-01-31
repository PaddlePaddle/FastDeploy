English | [简体中文](README_CN.md)
# YOLOv7End2EndTRT Ready-to-deploy Model

The YOLOv7End2EndTRT deployment is based on [YOLOv7](https://github.com/WongKinYiu/yolov7/tree/v0.1) branch code and [Pre-trained Model Baesd on COCO](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1). Attention: YOLOv7End2EndTRT is designed for the inference of exported End2End models in the [TRT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L111) version in YOLOv7. YOLOv7 class is for the inference of models without nms. YOLOv7End2EndORT is for the inference of End2End models in the [ORT_NMS](https://github.com/WongKinYiu/yolov7/blob/main/models/experimental.py#L87) version.

  - （1）*.pt provided by [Official Repository](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1) should [Export the ONNX Model](#Export-the-ONNX-Model) to complete the deployment. The deployment of *.trt and *.pose models is not supported.
  - （2）The YOLOv7 model  trained by personal data should [Export the ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B). Please refer to [Detailed Deployment Documents](#Detailed-Deployment-Documents) to complete the deployment.
<<<<<<< HEAD


=======
>>>>>>> 30def02a8969f52f40b5e3e305271ef8662126f2

## Export the ONNX Model

```bash
#  Download yolov7 Model Files
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# Export file in onnx format with TRT_NMS (Tips: corresponding to the code of YOLOv7 release v0.1)
python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640
# The commands for exporting other models are similar. Replace yolov7.pt with yolov7x.pt yolov7-d6.pt yolov7-w6.pt ...
# Only onnx files are required to employ YOLOv7End2EndTRT. Additional trt files are not required because automatic switching happens during inference.
```

## Download Pre-trained ONNX Models

For developers' testing, models exported by YOLOv7End2EndTRT are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    | Note |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [yolov7-end2end-trt-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-end2end-trt-nms.onnx) | 141MB | 51.4% | This model file is sourced from [YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7x-end2end-trt-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7x-end2end-trt-nms.onnx) | 273MB | 53.1% | This model file is sourced from [YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-w6-end2end-trt-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-w6-end2end-trt-nms.onnx) | 269MB | 54.9% | This model file is sourced from [YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-e6-end2end-trt-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6-end2end-trt-nms.onnx) | 372MB | 56.0% | This model file is sourced from [YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-d6-end2end-trt-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-d6-end2end-trt-nms.onnx) | 511MB | 56.6% | This model file is sourced from [YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |
| [yolov7-e6e-end2end-trt-nms](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6e-end2end-trt-nms.onnx) | 579MB | 56.8% | This model file is sourced from [YOLOv7](https://github.com/WongKinYiu/yolov7)，GPL-3.0 License |

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployement](cpp)

## Release Note

- Document and code are based on [YOLOv7 0.1](https://github.com/WongKinYiu/yolov7/tree/v0.1)
