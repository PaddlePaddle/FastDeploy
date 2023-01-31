English | [简体中文](README_CN.md)
# RetinaFace Ready-to-deploy Model

- [RetinaFace](https://github.com/biubug6/Pytorch_Retinaface/commit/b984b4b)
  - （1）The *.pt provided by the[Official Library](https://github.com/biubug6/Pytorch_Retinaface/) can be deployed after the [Export ONNX Model](#Export-ONNX-Model)；
  - （2）As for RetinaFace model trained on customized data, please follow the [Export ONNX Model](#export-onnx-model) to complete the deployment.


##  Export ONNX Model

[Download the pre-trained ONNX model](#download-pre-trained-onnx-models)The model has been converted to ONNX. If you downloaded the model from the RetinaFace official repo, please follow the tutorial below to export ONNX.  


* Download the official repository 
```bash
git clone https://github.com/biubug6/Pytorch_Retinaface.git
```
* Download the pre-trained weights and place them in the weights folder
```text
./weights/
      mobilenet0.25_Final.pth
      mobilenetV1X0.25_pretrain.tar
      Resnet50_Final.pth
```
* run convert_to_onnx.py to export ONNX model files
```bash
PYTHONPATH=. python convert_to_onnx.py --trained_model ./weights/mobilenet0.25_Final.pth --network mobile0.25 --long_side 640 --cpu
PYTHONPATH=. python convert_to_onnx.py --trained_model ./weights/Resnet50_Final.pth --network resnet50 --long_side 640 --cpu
```
Attention: We need to add a type constraint, type=int, to the --long_side parameter in the convert_to_onnx.py script.
* Use onnxsim to simplify the model
```bash
onnxsim FaceDetector.onnx Pytorch_RetinaFace_mobile0.25-640-640.onnx  # mobilenet
onnxsim FaceDetector.onnx Pytorch_RetinaFace_resnet50-640-640.onnx  # resnet50
```

## Download pre-trained ONNX models

For developers' testing, models exported by RetinaFace are provided below. Developers can download and use them directly. (The accuracy of the models in the table is sourced from the official library)
| Model                                                               | Size    | Accuracy    |
|:---------------------------------------------------------------- |:----- |:----- |
| [RetinaFace_mobile0.25-640](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-640-640.onnx) | 1.7MB | - |
| [RetinaFace_mobile0.25-720](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_mobile0.25-720-1080.onnx) | 1.7MB | -|
| [RetinaFace_resnet50-640](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_resnet50-720-1080.onnx) | 105MB | - |
| [RetinaFace_resnet50-720](https://bj.bcebos.com/paddlehub/fastdeploy/Pytorch_RetinaFace_resnet50-640-640.onnx) | 105MB | - |





## Detailed Deployment Tutorials

- [Python Deployment](python)
- [C++ Deployment](cpp)


## Release Note

- This tutorial and related code are written based on [RetinaFace CommitID:b984b4b](https://github.com/biubug6/Pytorch_Retinaface/commit/b984b4b) 
