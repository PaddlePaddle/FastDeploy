English | [简体中文](README_CN.md)
# YOLOR Ready-to-deploy Model

- The YOLOR deployment is based on the code of [YOLOR](https://github.com/WongKinYiu/yolor/releases/tag/weights) and [Pre-trained Model Based on COCO](https://github.com/WongKinYiu/yolor/releases/tag/weights).

  - （1）The *.pt provided by [Official Repository](https://github.com/WongKinYiu/yolor/releases/tag/weights) should [Export the ONNX Model](#Export-the-ONNX-Model) to complete the deployment. The *.pose model’s deployment is not supported；
  - （2）The ScaledYOLOv4 model trained by personal data should [Export the ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B). Please refer to [Detailed Deployment Documents](#Detailed-Deployment-Documents) to complete the deployment.


## Export the ONNX Model


  Visit the official [YOLOR](https://github.com/WongKinYiu/yolor) github repository, follow the guidelines to download the `yolor.pt` model, and employ `models/export.py` to get the file in `onnx` format. If the exported `onnx` model has a substandard accuracy or other problems about data dimension, you can refer to [yolor#32](https://github.com/WongKinYiu/yolor/issues/32) for the solution.

  ```bash
  # Download yolor model file
  wget https://github.com/WongKinYiu/yolor/releases/download/weights/yolor-d6-paper-570.pt

  # Export the file in onnx format
  python models/export.py  --weights PATH/TO/yolor-xx-xx-xx.pt --img-size 640
  ```

## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOR are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy   | Note |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOR-P6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-p6-paper-541-1280-1280.onnx) | 143MB | 54.1% | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-W6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-w6-paper-555-1280-1280.onnx) | 305MB | 55.5% | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-E6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-e6-paper-564-1280-1280.onnx ) | 443MB | 56.4% | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-570-1280-1280.onnx) | 580MB | 57.0% | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6-1280](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-573-1280-1280.onnx) | 580MB | 57.3% | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-P6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-p6-paper-541-640-640.onnx) | 143MB | - | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-W6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-w6-paper-555-640-640.onnx) | 305MB | - | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-E6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-e6-paper-564-640-640.onnx ) | 443MB | - | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-570-640-640.onnx) | 580MB | - | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |
| [YOLOR-D6](https://bj.bcebos.com/paddlehub/fastdeploy/yolor-d6-paper-573-640-640.onnx) | 580MB | - | This model file is sourced from [YOLOR](https://github.com/WongKinYiu/yolor)，GPL-3.0 License |

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)

## Release Note

- Document and code are based on [YOLOR weights](https://github.com/WongKinYiu/yolor/releases/tag/weights)
