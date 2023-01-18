English | [简体中文](README_CN.md)
# YOLOv5Lite Ready-to-deploy Model

- The YOLOv5Lite Deployment is based on the code of [YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)
and [Pre-trained Model Based on COCO](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)。

  - （1）The *.pt provided by [Official Repository](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4) should [Export the ONNX Model](#Export-the-ONNX-Model)to complete the deployment；
  - （2）The YOLOv5Lite model trained by personal data should [Export the ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B). Refer to [Detailed Deployment Documents](#Detailed-Deployment-Documents) to complete the deployment.


## Export the ONNX Model

- Auto-acquisition
  Visit official [YOLOv5Lite](https://github.com/ppogg/YOLOv5-Lite)
github repository, follow the guidelines to download the `yolov5-lite-xx.onnx` model(Tips: The official ONNX files are currently provided without the decode module)
  ```bash
  # Download yolov5-lite model files(.onnx)
  Download from https://drive.google.com/file/d/1bJByk9eoS6pv8Z3N4bcLRCV3i7uk24aU/view
  Official Repo also supports Baidu cloud download
  ```

- Manual Acquisition

  Visit official [YOLOv5Lite](https://github.com/ppogg/YOLOv5-Lite)
github repository,  follow the guidelines to download the `yolov5-lite-xx.pt` model, and employ `export.py` to get files in `onnx` format.

  - Export ONNX files with the decode module

  First refer to [YOLOv5-Lite#189](https://github.com/ppogg/YOLOv5-Lite/pull/189) to modify the code.

  ```bash
  # Download yolov5-lite model files(.pt)
  Download from https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view
  Official Repo also supports Baidu cloud download

  # Export files in onnx format
  python export.py --grid --dynamic --concat --weights PATH/TO/yolov5-lite-xx.pt


  ```
  - Export ONNX files without the docode module(No code changes are required)

  ```bash
  # Download yolov5-lite model files
  Download from https://drive.google.com/file/d/1oftzqOREGqDCerf7DtD5BZp9YWELlkMe/view
  Official Repo also supports Baidu cloud download

  # Export files in onnx format
  python export.py --grid --dynamic --weights PATH/TO/yolov5-lite-xx.pt

  ```

## Download Pre-trained ONNX Model

For developers' testing, models exported by YOLOv5Lite are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    | Note |
|:---------------------------------------------------------------- |:----- |:----- |:----- |
| [YOLOv5Lite-e](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-e-sim-320.onnx) | 3.1MB | 35.1% | This model file is sourced from [YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |
| [YOLOv5Lite-s](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-s-sim-416.onnx) | 6.3MB | 42.0% | This model file is sourced from [YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |
| [YOLOv5Lite-c](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-c-sim-512.onnx) | 18MB | 50.9% | This model file is sourced from[YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |
| [YOLOv5Lite-g](https://bj.bcebos.com/paddlehub/fastdeploy/v5Lite-g-sim-640.onnx) | 21MB | 57.6% | This model file is sourced from [YOLOv5-Lite](https://github.com/ppogg/YOLOv5-Lite)，GPL-3.0 License |

## Detailed Deployment Documents

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- Document and code are based on [YOLOv5-Lite v1.4](https://github.com/ppogg/YOLOv5-Lite/releases/tag/v1.4)
