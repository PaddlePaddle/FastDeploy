English | [简体中文](README_CN.md)
# YOLOv5Face Ready-to-deploy Model

- [YOLOv5Face](https://github.com/deepcam-cn/yolov5-face/commit/4fd1ead)
  - （1）The *.pt provided by the [Official Library](https://github.com/deepcam-cn/yolov5-face/) can be deployed after the [Export ONNX Model](#export-onnx-model) to complete the deployment；
  - （2）As for YOLOv5Face model trained on customized data, please follow the operations guidelines in [Export ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B)  to complete the deployment.

## Export ONNX Model

Visit [YOLOv5Face](https://github.com/deepcam-cn/yolov5-face) official github repository, follow the guidelines to download the `yolov5s-face.pt` model, and employ `export.py` to get the file in `onnx` format.

* Download yolov5face model files
  ```
  Link: https://pan.baidu.com/s/1fyzLxZYx7Ja1_PCIWRhxbw Link: eq0q  
  https://drive.google.com/file/d/1zxaHeLDyID9YU4-hqK7KNepXIwbTkRIO/view?usp=sharing
  ```

* Export files in onnx format
  ```bash
  PYTHONPATH=. python export.py --weights weights/yolov5s-face.pt --img_size 640 640 --batch_size 1  
  ```
* onnx model simplification (optional)
  ```bash
  onnxsim yolov5s-face.onnx yolov5s-face.onnx
  ```

## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOv5Face are provided below. Developers can download and use them directly. (The accuracy of the models in the table is sourced from the official library)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [YOLOv5s-Face](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5s-face.onnx) | 30MB | 94.3 |
| [YOLOv5s-Face-bak](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5face-s-640x640.bak.onnx) | 30MB | -|
| [YOLOv5l-Face](https://bj.bcebos.com/paddlehub/fastdeploy/yolov5face-l-640x640.onnx ) | 181MB | 95.8 |


## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [YOLOv5Face CommitID:4fd1ead](https://github.com/deepcam-cn/yolov5-face/commit/4fd1ead) 
