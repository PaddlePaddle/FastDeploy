English | [简体中文](README_CN.md)
# ScaledYOLOv4 Ready-to-deploy Model

- The ScaledYOLOv4 deployment is based on the code of [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) and [Pre-trained Model on COCO](https://github.com/WongKinYiu/ScaledYOLOv4).

  - （1）The *.pt provided by [Official Repository](https://github.com/WongKinYiu/ScaledYOLOv4) should [Export the ONNX Model](#Export-the-ONNX-Model) to complete the deployment；
  - （2）The ScaledYOLOv4 model trained by personal data should [Export the ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B). Refer to [Detailed Deployment Documents](#Detailed-Deployment-Documents) to complete the deployment.


## Export the ONNX Model


  Visit the official [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4) github repository, follow the guidelines to download the `scaledyolov4.pt` model, and employ `models/export.py` to get the file in `onnx` format.  If you have any problems with the exported `onnx` model, refer to [ScaledYOLOv4#401](https://github.com/WongKinYiu/ScaledYOLOv4/issues/401) for solution.


  ```bash
  # Download the ScaledYOLOv4 model file
  Download from the goole drive https://drive.google.com/file/d/1aXZZE999sHMP1gev60XhNChtHPRMH3Fz/view?usp=sharing

  # Export the file in onnx format
  python models/export.py  --weights PATH/TO/scaledyolov4-xx.pt --img-size 640
  ```


## Download Pre-trained ONNX Model

For developers' testing, models exported by ScaledYOLOv4 are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    | Note |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [ScaledYOLOv4-P5-896](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p5-896.onnx) | 271MB | 51.2% | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P5+BoF-896](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p5_-896.onnx) | 271MB | 51.7% | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p6-1280.onnx) | 487MB | 53.9% | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P6+BoF-1280](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p6_-1280.onnx) | 487MB | 54.4% | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P7-1536](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p7-1536.onnx) | 1.1GB | 55.0% | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P5](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p5.onnx) | 271MB | - | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P5+BoF](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p5_.onnx) | 271MB | -| This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P6](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p6.onnx) | 487MB | - | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P6+BoF](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p6_.onnx) | 487MB | - | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |
| [ScaledYOLOv4-P7](https://bj.bcebos.com/paddlehub/fastdeploy/scaled_yolov4-p7.onnx) | 1.1GB | - | This model file is sourced from [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)，GPL-3.0 License |

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- Document and code are based on [ScaledYOLOv4 CommitID: 6768003](https://github.com/WongKinYiu/ScaledYOLOv4/commit/676800364a3446900b9e8407bc880ea2127b3415)
