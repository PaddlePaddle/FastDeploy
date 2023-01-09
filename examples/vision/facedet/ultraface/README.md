English | [简体中文](README_CN.md)
# UltraFace Ready-to-deploy Model


- [UltraFace](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/commit/dffdddd)
  - （1）The *.onnx  provided by the [Official Library](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/) can be downloaded directly or through the following model link.
  - （2）As for the model trained on customized data, export the onnx model and then refer to [Detailed Deployment Tutorials](#detailed-deployment-tutorials) to complete the deployment.



## Download Pre-trained ONNX Models

For developers' testing, models exported by UltraFace are provided below. Developers can download and use them directly. (The accuracy of the models in the table is sourced from the official library)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [RFB-320](https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320.onnx) | 1.3MB | 78.7 |
| [RFB-320-sim](https://bj.bcebos.com/paddlehub/fastdeploy/version-RFB-320-sim.onnx) | 1.2MB | 78.7 |



## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [UltraFace CommitID:dffdddd](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/commit/dffdddd) 
