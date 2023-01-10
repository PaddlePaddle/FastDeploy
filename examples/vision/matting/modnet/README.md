English | [简体中文](README_CN.md)
# MODNet Ready-to-deploy Model

- [MODNet](https://github.com/ZHKKKe/MODNet/commit/28165a4)
  - （1）The *.pt provided by the [Official Library](https://github.com/ZHKKKe/MODNet/) can be deployed after [Export ONNX Model](#export-onnx-model)；
  - （2）As for MODNet model trained on customized data, please follow the operations guidelines in [Export ONNX Model](#%E5%AF%BC%E5%87%BAONNX%E6%A8%A1%E5%9E%8B) to complete the deployment.

##  Export ONNX Model


Visit [MODNet](https://github.com/ZHKKKe/MODNet) official github repository, follow the guidelines to download model files, and employ `onnx/export_onnx.py` to get files in `onnx` format.

* Export files in onnx format
  ```bash
  python -m onnx.export_onnx \
    --ckpt-path=pretrained/modnet_photographic_portrait_matting.ckpt \
    --output-path=pretrained/modnet_photographic_portrait_matting.onnx
  ```

## Download Pre-trained ONNX Model

For developers' testing, models exported by MODNet are provided below. Developers can download them directly. (The accuracy in the following table is derived from the source official repository)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [modnet_photographic](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting.onnx) | 25MB | - |
| [modnet_webcam](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting.onnx) | 25MB | -|
| [modnet_photographic_256](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting-256x256.onnx) | 25MB | - |
| [modnet_webcam_256](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting-256x256.onnx) | 25MB | - |
| [modnet_photographic_512](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting-512x512.onnx) | 25MB  | - |
| [modnet_webcam_512](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting-512x512.onnx) | 25MB | - |
| [modnet_photographic_1024](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_photographic_portrait_matting-1024x1024.onnx) | 25MB | - |
| [modnet_webcam_1024](https://bj.bcebos.com/paddlehub/fastdeploy/modnet_webcam_portrait_matting-1024x1024.onnx) | 25MB | -|




## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [MODNet CommitID:28165a4](https://github.com/ZHKKKe/MODNet/commit/28165a4) 
