English | [中文](../../../cn/faq/rknpu2/rknpu2.md) 
# RKNPU2 Model Deployment

## Installation Environment
RKNPU2 model export is only supported on x86 Linux platform, please refer to [RKNPU2 Model Export Environment Configuration](./install_rknn_toolkit2.md).

## Convert ONNX to RKNN
Since the ONNX model cannot directly calculate by calling the NPU, it is necessary to convert the ONNX model to RKNN model. For detailed information, please refer to [RKNPU2 Conversion Document](./export.md).

## Models supported for RKNPU2
The following tests are at end-to-end speed, and the test environment is as follows:
* Device Model: RK3588
* ARM CPU is tested on ONNX
* with single-core NPU


| Mission Scenario                 | Model                                                                                       | Model Version(tested version)          | ARM CPU/RKNN speed(ms) |
|----------------------|------------------------------------------------------------------------------------------|--------------------------|--------------------|
| Detection            | [Picodet](../../../../examples/vision/detection/paddledetection/rknpu2/README.md)        | Picodet-s                | 162/112            |
| Detection            | [RKYOLOV5](../../../../examples/vision/detection/rkyolo/README.md)                       | YOLOV5-S-Relu(int8)      | -/57               |
| Detection            | [RKYOLOX](../../../../examples/vision/detection/rkyolo/README.md)                        | -                        | -/-                |
| Detection            | [RKYOLOV7](../../../../examples/vision/detection/rkyolo/README.md)                       | -                        | -/-                |
| Segmentation         | [Unet](../../../../examples/vision/segmentation/paddleseg/rknpu2/README.md)              | Unet-cityscapes          | -/-                |
| Segmentation         | [PP-HumanSegV2Lite](../../../../examples/vision/segmentation/paddleseg/rknpu2/README.md) | portrait(int8)           | 133/43             |
| Segmentation         | [PP-HumanSegV2Lite](../../../../examples/vision/segmentation/paddleseg/rknpu2/README.md) | human(int8)              | 133/43             |
| Face Detection       | [SCRFD](../../../../examples/vision/facedet/scrfd/rknpu2/README.md)                      | SCRFD-2.5G-kps-640(int8) | 108/42             |
| Face FaceRecognition | [InsightFace](../../../../examples/vision/faceid/insightface/rknpu2/README_CN.md)        | ms1mv3_arcface_r18(int8) | 81/12              |
| Classification       | [ResNet](../../../../examples/vision/classification/paddleclas/rknpu2/README.md)         | ResNet50_vd              | -/33               |

## Download Pre-trained library

For convenience, here we provide the 1.0.2 version of FastDeploy.

- [FastDeploy RK356X c++ SDK](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-rk356X-1.0.2.tgz)
- [FastDeploy RK3588 c++ SDK](https://bj.bcebos.com/fastdeploy/release/cpp/fastdeploy-linux-aarch64-rk3588-1.0.2.tgz)
