English | [简体中文](README_CN.md)

# YOLOv7 Prepare the model for Deployment

- YOLOv7 deployment is based on [YOLOv7](https://github.com/WongKinYiu/yolov7/tree/v0.1) branching code, and [COCO Pre-Trained Models](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1).

  - （1）The *.pt provided by the [Official Library](https://github.com/WongKinYiu/yolov7/releases/tag/v0.1) can be deployed after the [export ONNX model](#Export-ONNX-Model) operation; *.trt and *.pose models do not support deployment.
  - （2）As for YOLOv7 model trained on customized data, please follow the operations guidelines in [Export ONNX model](#Export-ONNX-Model) and then refer to [Detailed Deployment Tutorials](#Detailed-Deployment-Tutorials) to complete the deployment.

## Export ONNX Model

```bash
# Download yolov7 model file
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt

# Export onnx file (Tips: in accordance with YOLOv7 release v0.1 code)
python models/export.py --grid --dynamic --weights PATH/TO/yolov7.pt

# If your code supports exporting ONNX files with NMS, please use the following command to export ONNX files, then refer to the example of `yolov7end2end_ort` or `yolov7end2end_ort`
python models/export.py --grid --dynamic --end2end --weights PATH/TO/yolov7.pt
```

To facilitate testing for developers, we provide below the models exported by YOLOv7, which developers can download and use directly. (The accuracy of the models in the table is sourced from the official library)

| Model                                                                    | Size  | Accuracy | Note |
| ------------------------------------------------------------------------ | ----- | -------- | -------- |
| [YOLOv7](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7.onnx)         | 141MB | 51.4%    | This model file comes from [YOLOv7](https://github.com/WongKinYiu/yolov7), GPL-3.0 License |
| [YOLOv7x](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7x.onnx)       | 273MB | 53.1%    | This model file comes from [YOLOv7](https://github.com/WongKinYiu/yolov7), GPL-3.0 License |
| [YOLOv7-w6](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-w6.onnx)   | 269MB | 54.9%    | This model file comes from [YOLOv7](https://github.com/WongKinYiu/yolov7), GPL-3.0 License |
| [YOLOv7-e6](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6.onnx)   | 372MB | 56.0%    | This model file comes from [YOLOv7](https://github.com/WongKinYiu/yolov7), GPL-3.0 License |
| [YOLOv7-d6](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-d6.onnx)   | 511MB | 56.6%    | This model file comes from [YOLOv7](https://github.com/WongKinYiu/yolov7), GPL-3.0 License |
| [YOLOv7-e6e](https://bj.bcebos.com/paddlehub/fastdeploy/yolov7-e6e.onnx) | 579MB | 56.8%    | This model file comes from [YOLOv7](https://github.com/WongKinYiu/yolov7), GPL-3.0 License |

## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Version

- This tutorial and related code are written based on [YOLOv7 0.1](https://github.com/WongKinYiu/yolov7/tree/v0.1)
